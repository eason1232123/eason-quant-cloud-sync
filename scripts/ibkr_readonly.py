from __future__ import annotations

import ipaddress
import json
import math
import os
import socket
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRIVATE_SNAPSHOT = ROOT / "private" / "ibkr" / "account_snapshot.json"
SNAPSHOT_SCHEMA_VERSION = "v6-ibkr-private-readonly-snapshot-v1"
PORTFOLIO_PRICE_TEMPORALITY = "UNKNOWN_TWS_ACCOUNT_UPDATE_NOT_VALIDATED_AS_REALTIME"
ACCOUNT_SUMMARY_TAGS = (
    "NetLiquidation",
    "TotalCashValue",
    "SettledCash",
    "BuyingPower",
    "AvailableFunds",
    "ExcessLiquidity",
    "Cushion",
    "MaintMarginReq",
    "GrossPositionValue",
)
ACCOUNT_UPDATE_TAGS = {
    "NetLiquidation",
    "TotalCashValue",
    "SettledCash",
    "BuyingPower",
    "AvailableFunds",
    "ExcessLiquidity",
    "Cushion",
    "MaintMarginReq",
    "GrossPositionValue",
}
INFORMATIONAL_ERROR_CODES = {2104, 2106, 2107, 2108, 2158}
FORBIDDEN_ORDER_METHODS = {
    "placeOrder",
    "cancelOrder",
    "reqOpenOrders",
    "reqAllOpenOrders",
    "reqAutoOpenOrders",
    "reqExecutions",
    "reqIds",
}


class IbkrReadonlyError(RuntimeError):
    """Raised when a private broker snapshot cannot be trusted as current."""


@dataclass(frozen=True)
class IbkrReadonlyConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 71
    timeout_seconds: float = 15.0
    private_snapshot_path: Path = DEFAULT_PRIVATE_SNAPSHOT


class SnapshotAdapter(Protocol):
    def capture(self, config: IbkrReadonlyConfig) -> dict[str, Any]: ...


def _parse_int(value: str, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise IbkrReadonlyError(f"{field} must be an integer") from exc
    return parsed


def _parse_float(value: str, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise IbkrReadonlyError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise IbkrReadonlyError(f"{field} must be finite")
    return parsed


def _is_loopback(host: str) -> bool:
    normalized = host.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def validate_config(config: IbkrReadonlyConfig) -> IbkrReadonlyConfig:
    if not _is_loopback(config.host):
        raise IbkrReadonlyError("IBKR read-only bridge only permits a loopback host")
    if isinstance(config.port, bool) or not 1 <= config.port <= 65535:
        raise IbkrReadonlyError("IBKR port must be in [1, 65535]")
    if isinstance(config.client_id, bool) or not 1 <= config.client_id <= 2_147_483_647:
        raise IbkrReadonlyError("IBKR client_id must be non-zero and positive")
    if (
        isinstance(config.timeout_seconds, bool)
        or not math.isfinite(config.timeout_seconds)
        or not 1 <= config.timeout_seconds <= 120
    ):
        raise IbkrReadonlyError("IBKR timeout must be between 1 and 120 seconds")
    try:
        resolved_output = config.private_snapshot_path.resolve()
        private_root = (ROOT / "private").resolve()
        resolved_output.relative_to(private_root)
    except (OSError, ValueError) as exc:
        raise IbkrReadonlyError("IBKR snapshots must remain under the gitignored private/ directory") from exc
    return config


def config_from_env(env: dict[str, str] | None = None) -> IbkrReadonlyConfig:
    values = os.environ if env is None else env
    raw_path = Path(values.get("IBKR_PRIVATE_SNAPSHOT", str(DEFAULT_PRIVATE_SNAPSHOT)))
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    return validate_config(
        IbkrReadonlyConfig(
            host=values.get("IBKR_HOST", "127.0.0.1"),
            port=_parse_int(values.get("IBKR_PORT", "7497"), "IBKR_PORT"),
            client_id=_parse_int(values.get("IBKR_CLIENT_ID", "71"), "IBKR_CLIENT_ID"),
            timeout_seconds=_parse_float(
                values.get("IBKR_TIMEOUT_SECONDS", "15"),
                "IBKR_TIMEOUT_SECONDS",
            ),
            private_snapshot_path=raw_path,
        )
    )


def probe_endpoint(config: IbkrReadonlyConfig) -> dict[str, Any]:
    validated = validate_config(config)
    try:
        with socket.create_connection(
            (validated.host, validated.port),
            timeout=min(validated.timeout_seconds, 3.0),
        ):
            reachable = True
            reason = None
    except OSError as exc:
        reachable = False
        reason = f"{type(exc).__name__}: {exc}"
    return {
        "status": "IBKR_ENDPOINT_REACHABLE" if reachable else "IBKR_ENDPOINT_OFFLINE",
        "reachable": reachable,
        "host_scope": "LOOPBACK_ONLY",
        "port": validated.port,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "automatic_order_allowed": False,
    }


def _finite_ib_number(value: Any, *, required: bool = False) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        if required:
            raise IbkrReadonlyError("IBKR returned a required non-numeric value")
        return None
    # IBKR uses very large sentinels for some unavailable numeric fields.
    if not math.isfinite(parsed) or abs(parsed) >= 1e100:
        if required:
            raise IbkrReadonlyError("IBKR returned an unavailable required numeric value")
        return None
    return parsed


def _contract_record(contract: Any) -> dict[str, Any]:
    return {
        "con_id": int(getattr(contract, "conId", 0) or 0),
        "symbol": str(getattr(contract, "symbol", "") or ""),
        "local_symbol": str(getattr(contract, "localSymbol", "") or ""),
        "security_type": str(getattr(contract, "secType", "") or ""),
        "currency": str(getattr(contract, "currency", "") or ""),
        "exchange": str(getattr(contract, "exchange", "") or ""),
        "primary_exchange": str(getattr(contract, "primaryExchange", "") or ""),
        "multiplier": str(getattr(contract, "multiplier", "") or ""),
    }


class OfficialIbapiAdapter:
    """Capture account data with the official IBKR Python TWS API only."""

    def capture(self, config: IbkrReadonlyConfig) -> dict[str, Any]:
        validated = validate_config(config)
        endpoint = probe_endpoint(validated)
        if not endpoint["reachable"]:
            raise IbkrReadonlyError(
                f"IBKR_ENDPOINT_OFFLINE: no TWS/IB Gateway listener on loopback port {validated.port}"
            )
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
        except ImportError as exc:
            raise IbkrReadonlyError(
                "IBAPI_NOT_INSTALLED: install the official TWS API Python client from the IBKR API bundle"
            ) from exc

        class ReadonlyApp(EWrapper, EClient):
            def __init__(self) -> None:
                EClient.__init__(self, self)
                self.ready = threading.Event()
                self.managed_accounts_ready = threading.Event()
                self.positions_ready = threading.Event()
                self.account_summary_ready = threading.Event()
                self.account_download_ready = threading.Event()
                self.managed_accounts: list[str] = []
                self.positions: list[dict[str, Any]] = []
                self.portfolio: list[dict[str, Any]] = []
                self.account_summary: list[dict[str, Any]] = []
                self.account_values: list[dict[str, Any]] = []
                self.account_readiness: dict[str, bool | None] = {}
                self.errors: list[dict[str, Any]] = []

            def nextValidId(self, orderId: int) -> None:  # noqa: N802
                # The callback proves the socket handshake is complete. The order
                # identifier is deliberately discarded and never requested/used.
                self.ready.set()

            def managedAccounts(self, accountsList: str) -> None:  # noqa: N802
                self.managed_accounts = sorted(
                    {item.strip() for item in accountsList.split(",") if item.strip()}
                )
                self.managed_accounts_ready.set()

            def position(
                self,
                account: str,
                contract: Any,
                position: Decimal,
                avgCost: float,
            ) -> None:
                self.positions.append(
                    {
                        "account_id": account,
                        "contract": _contract_record(contract),
                        "quantity": _finite_ib_number(position, required=True),
                        "average_cost": _finite_ib_number(avgCost),
                    }
                )

            def positionEnd(self) -> None:  # noqa: N802
                self.positions_ready.set()

            def accountSummary(  # noqa: N802
                self,
                reqId: int,
                account: str,
                tag: str,
                value: str,
                currency: str,
            ) -> None:
                self.account_summary.append(
                    {
                        "account_id": account,
                        "tag": tag,
                        "value": value,
                        "currency": currency,
                    }
                )

            def accountSummaryEnd(self, reqId: int) -> None:  # noqa: N802
                self.account_summary_ready.set()

            def updatePortfolio(  # noqa: N802
                self,
                contract: Any,
                position: Decimal,
                marketPrice: float,
                marketValue: float,
                averageCost: float,
                unrealizedPNL: float,
                realizedPNL: float,
                accountName: str,
            ) -> None:
                self.portfolio.append(
                    {
                        "account_id": accountName,
                        "contract": _contract_record(contract),
                        "quantity": _finite_ib_number(position, required=True),
                        "market_price": _finite_ib_number(marketPrice),
                        "market_value": _finite_ib_number(marketValue),
                        "average_cost": _finite_ib_number(averageCost),
                        "unrealized_pnl": _finite_ib_number(unrealizedPNL),
                        "realized_pnl": _finite_ib_number(realizedPNL),
                    }
                )

            def updateAccountValue(  # noqa: N802
                self,
                key: str,
                value: str,
                currency: str,
                accountName: str,
            ) -> None:
                if key == "accountReady":
                    normalized = value.strip().lower()
                    self.account_readiness[accountName] = {
                        "true": True,
                        "false": False,
                    }.get(normalized)
                    return
                if key in ACCOUNT_UPDATE_TAGS:
                    self.account_values.append(
                        {
                            "account_id": accountName,
                            "tag": key,
                            "value": value,
                            "currency": currency,
                        }
                    )

            def accountDownloadEnd(self, accountName: str) -> None:  # noqa: N802
                self.account_download_ready.set()

            def error(self, reqId: int, errorCode: int, errorString: str, *args: Any) -> None:
                if errorCode not in INFORMATIONAL_ERROR_CODES:
                    self.errors.append(
                        {
                            "request_id": reqId,
                            "error_code": errorCode,
                            "message": errorString,
                        }
                    )

        app = ReadonlyApp()
        thread: threading.Thread | None = None
        summary_request_id = 9001
        try:
            app.connect(validated.host, validated.port, clientId=validated.client_id)
            thread = threading.Thread(target=app.run, daemon=True, name="ibkr-readonly-loop")
            thread.start()
            if not app.ready.wait(validated.timeout_seconds):
                raise IbkrReadonlyError("IBKR connection handshake timed out")
            if not app.managed_accounts_ready.wait(validated.timeout_seconds):
                raise IbkrReadonlyError("IBKR managed-account list timed out")
            if not app.managed_accounts:
                raise IbkrReadonlyError("IBKR returned no accessible accounts")

            app.reqPositions()
            app.reqAccountSummary(summary_request_id, "All", ",".join(ACCOUNT_SUMMARY_TAGS))
            if not app.positions_ready.wait(validated.timeout_seconds):
                raise IbkrReadonlyError("IBKR position snapshot timed out")
            if not app.account_summary_ready.wait(validated.timeout_seconds):
                raise IbkrReadonlyError("IBKR account summary timed out")

            for account_id in app.managed_accounts:
                app.account_download_ready.clear()
                app.account_readiness.pop(account_id, None)
                app.reqAccountUpdates(True, account_id)
                if not app.account_download_ready.wait(validated.timeout_seconds):
                    raise IbkrReadonlyError("IBKR portfolio/account update timed out")
                app.reqAccountUpdates(False, account_id)
                if app.account_readiness.get(account_id) is not True:
                    raise IbkrReadonlyError(
                        "IBKR_ACCOUNT_NOT_READY: account values may be stale or incorrect"
                    )
            if app.errors:
                raise IbkrReadonlyError(
                    "IBKR returned non-informational errors: "
                    + "; ".join(
                        f"{item['error_code']} {item['message']}" for item in app.errors[:5]
                    )
                )
        finally:
            if app.isConnected():
                try:
                    app.cancelPositions()
                    app.cancelAccountSummary(summary_request_id)
                finally:
                    app.disconnect()
            if thread is not None:
                thread.join(timeout=1.0)

        return {
            "managed_accounts": app.managed_accounts,
            "positions": app.positions,
            "portfolio": app.portfolio,
            "account_summary": app.account_summary,
            "account_values": app.account_values,
            "account_readiness": app.account_readiness,
        }


def _parse_aware_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise IbkrReadonlyError(f"{field} must be a timezone-aware timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IbkrReadonlyError(f"{field} must be a valid timestamp") from exc
    if parsed.utcoffset() is None:
        raise IbkrReadonlyError(f"{field} must include a timezone")
    return parsed


def _assert_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise IbkrReadonlyError(f"non-finite number in private snapshot at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"orders", "order_id", "password", "token", "api_key"}:
                raise IbkrReadonlyError(f"prohibited field in private snapshot: {path}.{key}")
            _assert_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{path}[{index}]")


def validate_private_snapshot(
    snapshot: dict[str, Any],
    *,
    max_age_seconds: float = 300.0,
    now: datetime | None = None,
) -> dict[str, Any]:
    if not isinstance(snapshot, dict):
        raise IbkrReadonlyError("private IBKR snapshot must be an object")
    if snapshot.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise IbkrReadonlyError("private IBKR snapshot schema mismatch")
    if snapshot.get("status") != "CONNECTED_READ_ONLY_SNAPSHOT":
        raise IbkrReadonlyError("private IBKR snapshot is not connected/current")
    if snapshot.get("source") != "IBKR official TWS API local socket":
        raise IbkrReadonlyError("private IBKR snapshot source mismatch")
    if snapshot.get("automatic_order_allowed") is not False:
        raise IbkrReadonlyError("private IBKR bridge must never allow automatic orders")
    if snapshot.get("order_methods_invoked") != []:
        raise IbkrReadonlyError("private IBKR bridge invoked or reported an order method")
    collected = _parse_aware_timestamp(snapshot.get("collected_at_utc"), "collected_at_utc")
    reference_now = now or datetime.now(timezone.utc)
    if reference_now.utcoffset() is None:
        raise IbkrReadonlyError("snapshot validation clock must include a timezone")
    age = (reference_now.astimezone(timezone.utc) - collected.astimezone(timezone.utc)).total_seconds()
    if age < -5:
        raise IbkrReadonlyError("private IBKR snapshot timestamp is in the future")
    if not math.isfinite(max_age_seconds) or max_age_seconds <= 0:
        raise IbkrReadonlyError("max_age_seconds must be positive and finite")
    if age > max_age_seconds:
        raise IbkrReadonlyError("private IBKR snapshot is stale")

    accounts = snapshot.get("managed_accounts")
    if not isinstance(accounts, list) or not accounts or any(
        not isinstance(account, str) or not account for account in accounts
    ):
        raise IbkrReadonlyError("private IBKR snapshot has no valid managed accounts")
    if len(accounts) != len(set(accounts)):
        raise IbkrReadonlyError("private IBKR snapshot has duplicate account IDs")
    readiness = snapshot.get("account_readiness")
    if (
        not isinstance(readiness, dict)
        or set(readiness) != set(accounts)
        or any(value is not True for value in readiness.values())
    ):
        raise IbkrReadonlyError("private IBKR snapshot account readiness is not explicitly true")
    if snapshot.get("portfolio_price_temporality") != PORTFOLIO_PRICE_TEMPORALITY:
        raise IbkrReadonlyError("private IBKR snapshot portfolio price temporality is missing")
    for collection in ("positions", "portfolio", "account_summary", "account_values"):
        if not isinstance(snapshot.get(collection), list):
            raise IbkrReadonlyError(f"private IBKR snapshot {collection} must be a list")
    for collection, optional_numeric_fields in (
        ("positions", ("average_cost",)),
        (
            "portfolio",
            (
                "market_price",
                "market_value",
                "average_cost",
                "unrealized_pnl",
                "realized_pnl",
            ),
        ),
    ):
        for row in snapshot[collection]:
            if not isinstance(row, dict) or row.get("account_id") not in accounts:
                raise IbkrReadonlyError("position/portfolio row references an unknown account")
            quantity = row.get("quantity")
            if (
                not isinstance(quantity, (int, float))
                or isinstance(quantity, bool)
                or not math.isfinite(quantity)
            ):
                raise IbkrReadonlyError("position/portfolio quantity must be finite")
            contract = row.get("contract")
            if not isinstance(contract, dict) or not contract.get("symbol"):
                raise IbkrReadonlyError("position/portfolio contract symbol is required")
            for field in optional_numeric_fields:
                value = row.get(field)
                if value is not None and (
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(value)
                ):
                    raise IbkrReadonlyError(
                        f"position/portfolio {field} must be finite or null"
                    )
    for collection in ("account_summary", "account_values"):
        for row in snapshot[collection]:
            if (
                not isinstance(row, dict)
                or row.get("account_id") not in accounts
                or not isinstance(row.get("tag"), str)
                or not row["tag"]
                or not isinstance(row.get("currency"), str)
            ):
                raise IbkrReadonlyError(f"private IBKR snapshot {collection} row is invalid")
    _assert_finite_json(snapshot)
    return snapshot


def capture_private_snapshot(
    config: IbkrReadonlyConfig,
    *,
    adapter: SnapshotAdapter | None = None,
    collected_at: datetime | None = None,
) -> dict[str, Any]:
    validated = validate_config(config)
    provider = adapter or OfficialIbapiAdapter()
    captured = provider.capture(validated)
    timestamp = collected_at or datetime.now(timezone.utc)
    if timestamp.utcoffset() is None:
        raise IbkrReadonlyError("capture timestamp must include a timezone")
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "status": "CONNECTED_READ_ONLY_SNAPSHOT",
        "source": "IBKR official TWS API local socket",
        "account_data_timezone": "UTC",
        "collected_at_utc": timestamp.astimezone(timezone.utc).isoformat(),
        "portfolio_price_temporality": PORTFOLIO_PRICE_TEMPORALITY,
        "connection": {
            "host_scope": "LOOPBACK_ONLY",
            "port": validated.port,
            "client_id": validated.client_id,
            "tws_read_only_setting": "REQUIRED_USER_CONFIGURATION_NOT_PROGRAMMATICALLY_VERIFIABLE",
        },
        "managed_accounts": captured.get("managed_accounts", []),
        "positions": captured.get("positions", []),
        "portfolio": captured.get("portfolio", []),
        "account_summary": captured.get("account_summary", []),
        "account_values": captured.get("account_values", []),
        "account_readiness": captured.get("account_readiness", {}),
        "automatic_order_allowed": False,
        "order_methods_invoked": [],
        "privacy": {
            "classification": "PRIVATE_LOCAL_ONLY",
            "git_publish_allowed": False,
            "contains_account_ids": True,
            "contains_exact_positions": True,
        },
    }
    validate_private_snapshot(snapshot, max_age_seconds=validated.timeout_seconds + 5, now=timestamp)
    _write_private_snapshot(validated.private_snapshot_path, snapshot)
    return snapshot


def _write_private_snapshot(path: Path, snapshot: dict[str, Any]) -> None:
    validate_config(
        IbkrReadonlyConfig(private_snapshot_path=path)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_private_snapshot(
    path: Path = DEFAULT_PRIVATE_SNAPSHOT,
    *,
    max_age_seconds: float = 300.0,
) -> dict[str, Any]:
    validate_config(IbkrReadonlyConfig(private_snapshot_path=path))
    if not path.exists() or path.stat().st_size == 0:
        raise IbkrReadonlyError(f"private IBKR snapshot is missing: {path}")
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IbkrReadonlyError(f"private IBKR snapshot is invalid: {exc}") from exc
    return validate_private_snapshot(snapshot, max_age_seconds=max_age_seconds)
