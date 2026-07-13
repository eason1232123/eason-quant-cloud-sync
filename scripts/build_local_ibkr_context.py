from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.ibkr_readonly import (
    DEFAULT_PRIVATE_SNAPSHOT,
    IbkrReadonlyConfig,
    IbkrReadonlyError,
    ROOT,
    load_private_snapshot,
    validate_config,
)


DEFAULT_DECISION_PACKET = ROOT / "docs" / "decision_packet.json"
DEFAULT_MODEL_GOVERNANCE = ROOT / "docs" / "model_governance.json"
DEFAULT_PRIVATE_CONTEXT = ROOT / "private" / "ibkr" / "chatgpt_account_context.json"
CONTEXT_SCHEMA_VERSION = "v6-private-chatgpt-ibkr-context-v1"
MODEL_CONCENTRATION_SYMBOLS = ("QQQ", "SMH", "MSFT")


class LocalIbkrContextError(IbkrReadonlyError):
    """Raised when public evidence and private account truth cannot be joined safely."""


def _strict_public_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise LocalIbkrContextError(f"missing or empty {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LocalIbkrContextError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise LocalIbkrContextError(f"{label} must be a JSON object")
    _assert_finite(value, label)
    return value


def _assert_finite(value: Any, path: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise LocalIbkrContextError(f"non-finite value at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{path}[{index}]")


def _optional_number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or abs(parsed) >= 1e100:
        return None
    return parsed


def _summary_values(snapshot: dict[str, Any]) -> dict[tuple[str, str, str], float | None]:
    values: dict[tuple[str, str, str], float | None] = {}
    # Account update values are later/more account-specific and intentionally
    # overwrite same-key account-summary values when both exist.
    for collection in ("account_summary", "account_values"):
        for row in snapshot[collection]:
            if not isinstance(row, dict):
                raise LocalIbkrContextError(f"private snapshot {collection} row must be an object")
            account = row.get("account_id")
            tag = row.get("tag")
            currency = row.get("currency") or "BASE"
            if account not in snapshot["managed_accounts"] or not isinstance(tag, str) or not tag:
                raise LocalIbkrContextError(f"invalid private snapshot {collection} identity")
            values[(account, tag, str(currency))] = _optional_number(row.get("value"))
    return values


def _account_currency_metrics(
    snapshot: dict[str, Any],
    account_id: str,
    currency: str,
    summary: dict[tuple[str, str, str], float | None],
) -> dict[str, Any]:
    portfolio_rows = [
        row
        for row in snapshot["portfolio"]
        if row["account_id"] == account_id
        and str(row["contract"].get("currency") or "BASE") == currency
    ]
    net_liquidation = summary.get((account_id, "NetLiquidation", currency))
    cash_value = summary.get((account_id, "TotalCashValue", currency))

    market_values_available = all(row.get("market_value") is not None for row in portfolio_rows)
    denominator_available = (
        net_liquidation is not None
        and math.isfinite(net_liquidation)
        and net_liquidation > 0
    )
    positions: list[dict[str, Any]] = []
    for row in sorted(
        portfolio_rows,
        key=lambda item: (
            str(item["contract"].get("symbol") or ""),
            int(item["contract"].get("con_id") or 0),
        ),
    ):
        market_value = row.get("market_value")
        weight = market_value / net_liquidation if market_value is not None and denominator_available else None
        positions.append(
            {
                "symbol": row["contract"].get("symbol"),
                "local_symbol": row["contract"].get("local_symbol"),
                "security_type": row["contract"].get("security_type"),
                "quantity": row.get("quantity"),
                "average_cost": row.get("average_cost"),
                "market_price": row.get("market_price"),
                "market_value": market_value,
                "unrealized_pnl": row.get("unrealized_pnl"),
                "realized_pnl": row.get("realized_pnl"),
                "net_liquidation_weight": round(weight, 12) if weight is not None else None,
                "weight_status": "AVAILABLE" if weight is not None else "UNAVAILABLE",
            }
        )

    if market_values_available and denominator_available:
        gross_position_exposure = sum(abs(float(row["market_value"])) for row in portfolio_rows)
        net_position_exposure = sum(float(row["market_value"]) for row in portfolio_rows)
        model_concentration = sum(
            abs(float(row["market_value"]))
            for row in portfolio_rows
            if str(row["contract"].get("symbol") or "") in MODEL_CONCENTRATION_SYMBOLS
        )
        gross_exposure_weight = gross_position_exposure / net_liquidation
        net_exposure_weight = net_position_exposure / net_liquidation
        model_concentration_weight = model_concentration / net_liquidation
        exposure_status = "AVAILABLE"
    else:
        gross_exposure_weight = None
        net_exposure_weight = None
        model_concentration_weight = None
        exposure_status = (
            "NO_PORTFOLIO_ROWS"
            if not portfolio_rows
            else "MISSING_MARKET_VALUE_OR_NET_LIQUIDATION"
        )
    return {
        "currency": currency,
        "net_liquidation": net_liquidation,
        "total_cash_value": cash_value,
        "position_count": len(portfolio_rows),
        "positions": positions,
        "gross_position_exposure_weight": (
            round(gross_exposure_weight, 12) if gross_exposure_weight is not None else None
        ),
        "net_position_exposure_weight": (
            round(net_exposure_weight, 12) if net_exposure_weight is not None else None
        ),
        "QQQ_SMH_MSFT_gross_concentration_weight": (
            round(model_concentration_weight, 12)
            if model_concentration_weight is not None
            else None
        ),
        "exposure_status": exposure_status,
        "currency_conversion_applied": False,
    }


def build_private_context(
    snapshot: dict[str, Any],
    decision_packet: dict[str, Any],
    model_governance: dict[str, Any],
    *,
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    if snapshot.get("automatic_order_allowed") is not False:
        raise LocalIbkrContextError("private IBKR snapshot must prohibit automatic orders")
    if snapshot.get("order_methods_invoked") != []:
        raise LocalIbkrContextError("private IBKR snapshot must not invoke order methods")
    if snapshot.get("privacy", {}).get("git_publish_allowed") is not False:
        raise LocalIbkrContextError("private IBKR snapshot must remain local-only")
    if decision_packet.get("decision", {}).get("automatic_order_allowed") is not False:
        raise LocalIbkrContextError("public decision packet must prohibit automatic orders")
    if model_governance.get("automatic_order_allowed") is not False:
        raise LocalIbkrContextError("public model governance must prohibit automatic orders")
    market_data = decision_packet.get("market_data")
    if not isinstance(market_data, dict) or market_data.get("metadata_status") != "COMPLETE":
        raise LocalIbkrContextError("public decision packet market metadata is incomplete")
    required_market_fields = (
        "source",
        "market_timezone",
        "report_generated_at_utc",
        "data_timestamp",
        "price_frequency",
        "price_adjustment_policy",
    )
    if any(not isinstance(market_data.get(field), str) or not market_data[field] for field in required_market_fields):
        raise LocalIbkrContextError("public decision packet market metadata is invalid")

    summary = _summary_values(snapshot)
    accounts = []
    for account_id in snapshot["managed_accounts"]:
        currencies = {
            str(row["contract"].get("currency") or "BASE")
            for row in snapshot["portfolio"]
            if row["account_id"] == account_id
        }
        currencies |= {
            currency
            for (account, tag, currency), value in summary.items()
            if account == account_id
            and tag in {"NetLiquidation", "TotalCashValue"}
            and value is not None
        }
        if not currencies:
            currencies = {"BASE"}
        accounts.append(
            {
                "account_id": account_id,
                "currency_slices": [
                    _account_currency_metrics(snapshot, account_id, currency, summary)
                    for currency in sorted(currencies)
                ],
            }
        )
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.utcoffset() is None:
        raise LocalIbkrContextError("context generation timestamp must include a timezone")
    context = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "status": "PRIVATE_LOCAL_ANALYSIS_CONTEXT_READY",
        "generated_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "privacy": {
            "classification": "PRIVATE_LOCAL_ONLY",
            "git_publish_allowed": False,
            "contains_exact_account_and_position_data": True,
        },
        "sources": {
            "ibkr": {
                "source": snapshot["source"],
                "account_data_timezone": snapshot["account_data_timezone"],
                "data_timestamp": snapshot["collected_at_utc"],
                "portfolio_price_temporality": snapshot["portfolio_price_temporality"],
                "status": snapshot["status"],
            },
            "github_strategy_evidence": {
                "source": market_data["source"],
                "market_timezone": market_data["market_timezone"],
                "report_generated_at_utc": market_data["report_generated_at_utc"],
                "data_timestamp": market_data["data_timestamp"],
                "price_frequency": market_data["price_frequency"],
                "price_adjustment_policy": market_data["price_adjustment_policy"],
                "decision_schema_version": decision_packet.get("schema_version"),
                "model_governance_fingerprint": model_governance.get(
                    "model_governance_fingerprint"
                ),
            },
        },
        "public_decision": decision_packet.get("decision"),
        "public_model_governance": {
            "status": model_governance.get("status"),
            "current_market_regime": model_governance.get("current_market_regime"),
            "allocation_decision": model_governance.get("allocation_decision"),
            "effective_target_weights": model_governance.get("effective_target_weights"),
        },
        "private_accounts": accounts,
        "risk_calculation_policy": {
            "weights_denominator": "same-account same-currency NetLiquidation when available and positive",
            "cross_currency_aggregation": "PROHIBITED_WITHOUT_EXPLICIT_FX_RATES",
            "missing_value_policy": "null with explicit UNAVAILABLE status; never replace with zero",
            "model_concentration_symbols": list(MODEL_CONCENTRATION_SYMBOLS),
        },
        "automatic_order_allowed": False,
        "human_confirmation_required": True,
        "order_payload": None,
    }
    validate_private_context(context)
    return context


def validate_private_context(context: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(context, dict) or context.get("schema_version") != CONTEXT_SCHEMA_VERSION:
        raise LocalIbkrContextError("private ChatGPT context schema mismatch")
    if context.get("status") != "PRIVATE_LOCAL_ANALYSIS_CONTEXT_READY":
        raise LocalIbkrContextError("private ChatGPT context is not ready")
    privacy = context.get("privacy")
    if not isinstance(privacy, dict) or privacy.get("git_publish_allowed") is not False:
        raise LocalIbkrContextError("private ChatGPT context must never be Git-publishable")
    if context.get("automatic_order_allowed") is not False or context.get("order_payload") is not None:
        raise LocalIbkrContextError("private ChatGPT context must not contain an executable order")
    if context.get("human_confirmation_required") is not True:
        raise LocalIbkrContextError("private ChatGPT context requires human confirmation")
    _assert_finite(context, "$context")
    return context


def write_private_context(path: Path, context: dict[str, Any]) -> None:
    validate_config(IbkrReadonlyConfig(private_snapshot_path=path))
    validate_private_context(context)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(context, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def build_local_context(
    *,
    snapshot_path: Path = DEFAULT_PRIVATE_SNAPSHOT,
    decision_path: Path = DEFAULT_DECISION_PACKET,
    governance_path: Path = DEFAULT_MODEL_GOVERNANCE,
    output_path: Path = DEFAULT_PRIVATE_CONTEXT,
    max_snapshot_age_seconds: float = 300.0,
) -> dict[str, Any]:
    snapshot = load_private_snapshot(snapshot_path, max_age_seconds=max_snapshot_age_seconds)
    decision = _strict_public_json(decision_path, "decision_packet")
    governance = _strict_public_json(governance_path, "model_governance")
    context = build_private_context(snapshot, decision, governance)
    write_private_context(output_path, context)
    return context


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Join private IBKR truth with public GitHub model evidence for local ChatGPT analysis"
    )
    parser.add_argument("--max-snapshot-age-seconds", type=float, default=300.0)
    args = parser.parse_args()
    try:
        context = build_local_context(max_snapshot_age_seconds=args.max_snapshot_age_seconds)
        print(
            json.dumps(
                {
                    "status": context["status"],
                    "generated_at_utc": context["generated_at_utc"],
                    "private_account_count": len(context["private_accounts"]),
                    "private_output": str(DEFAULT_PRIVATE_CONTEXT.relative_to(ROOT)),
                    "automatic_order_allowed": False,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 0
    except IbkrReadonlyError as exc:
        print(
            json.dumps(
                {
                    "status": "PRIVATE_LOCAL_ANALYSIS_CONTEXT_UNAVAILABLE",
                    "reason": str(exc),
                    "automatic_order_allowed": False,
                },
                sort_keys=True,
                allow_nan=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
