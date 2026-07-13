from __future__ import annotations

import copy
import ast
import inspect
import json
import math
import textwrap
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.validate_validation_split import canonical_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOVERNANCE_CONFIG = ROOT / "config" / "model_governance.json"
ASSETS = ("QQQ", "SMH", "MSFT", "SPY", "CASH")
REGIMES = ("base", "defensive", "severe_defensive")
WARMUP_REGIME = "warmup_base"
GOVERNANCE_CONTRACT_VERSION = "v6-model-governance-contract-v1"


class ModelGovernanceError(ValueError):
    """Raised when model governance evidence or configuration is ambiguous."""


def _strict_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        raise ModelGovernanceError(f"missing or empty model governance config: {path}")

    def reject_constant(value: str) -> None:
        raise ModelGovernanceError(f"non-finite JSON number is prohibited: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ModelGovernanceError(f"duplicate JSON key is prohibited: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelGovernanceError(f"invalid model governance config {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ModelGovernanceError("model governance config must be a JSON object")
    return value


def _validate_weights(weights: Any, context: str) -> dict[str, float]:
    if not isinstance(weights, dict) or set(weights) != set(ASSETS):
        raise ModelGovernanceError(f"{context} must contain every governed asset exactly once")
    parsed: dict[str, float] = {}
    for asset in ASSETS:
        value = weights[asset]
        if isinstance(value, bool):
            raise ModelGovernanceError(f"{context}.{asset} must be numeric")
        try:
            parsed[asset] = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelGovernanceError(f"{context}.{asset} must be numeric") from exc
        if not math.isfinite(parsed[asset]) or parsed[asset] < 0:
            raise ModelGovernanceError(f"{context}.{asset} must be finite and non-negative")
    if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ModelGovernanceError(f"{context} weights must sum to exactly one")
    return parsed


def validate_governance_config(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema_version") != "v6-model-governance-v1":
        raise ModelGovernanceError("unsupported model governance schema_version")
    if config.get("status") != "FROZEN":
        raise ModelGovernanceError("model governance config must be FROZEN")
    try:
        frozen_on = date.fromisoformat(config["frozen_on_date"])
        boundary = date.fromisoformat(config["evidence_policy"]["strictly_after_market_date"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModelGovernanceError("governance dates must be ISO dates") from exc
    if frozen_on < boundary:
        raise ModelGovernanceError("governance freeze date cannot predate its prospective boundary")

    evidence = config.get("evidence_policy")
    if not isinstance(evidence, dict):
        raise ModelGovernanceError("evidence_policy is required")
    if evidence.get("classification") != "PROSPECTIVE":
        raise ModelGovernanceError("only PROSPECTIVE evidence may govern model allocation")
    if evidence.get("retrospective_evidence_may_promote") is not False:
        raise ModelGovernanceError("retrospective evidence must never promote a model")
    if evidence.get("non_overlapping_samples_required") is not True:
        raise ModelGovernanceError("governance samples must be non-overlapping")
    if evidence.get("paired_incumbent_challenger_samples_required") is not True:
        raise ModelGovernanceError("incumbent/challenger samples must be paired")
    for field in ("evaluation_horizon_bars", "entry_close_offset_bars"):
        value = evidence.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelGovernanceError(f"evidence_policy.{field} must be a positive integer")

    models = config.get("models")
    if not isinstance(models, list) or len(models) < 2:
        raise ModelGovernanceError("one incumbent and at least one challenger are required")
    ids: set[str] = set()
    incumbent_count = 0
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise ModelGovernanceError(f"models[{index}] must be an object")
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id or model_id in ids:
            raise ModelGovernanceError("model_id values must be non-empty and unique")
        ids.add(model_id)
        role = model.get("role")
        if role not in {"INCUMBENT", "CHALLENGER"}:
            raise ModelGovernanceError(f"invalid role for model {model_id}")
        incumbent_count += role == "INCUMBENT"
        regime_weights = model.get("regime_weights")
        if not isinstance(regime_weights, dict) or set(regime_weights) != set(REGIMES):
            raise ModelGovernanceError(f"model {model_id} must define exactly the three regimes")
        for regime in REGIMES:
            _validate_weights(regime_weights[regime], f"models.{model_id}.{regime}")
    if incumbent_count != 1:
        raise ModelGovernanceError("exactly one incumbent model is required")

    allocation = config.get("allocation_gates")
    rollback = config.get("rollback_gates")
    execution = config.get("execution_policy")
    if not isinstance(allocation, dict) or not isinstance(rollback, dict) or not isinstance(execution, dict):
        raise ModelGovernanceError("allocation, rollback, and execution policies are required")
    sample_levels = [
        allocation.get("minimum_paired_samples_for_pilot_blend"),
        allocation.get("minimum_paired_samples_for_limited_blend"),
        allocation.get("minimum_paired_samples_for_promotion_review"),
    ]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in sample_levels):
        raise ModelGovernanceError("sample gates must be positive integers")
    if sample_levels != sorted(sample_levels) or len(set(sample_levels)) != len(sample_levels):
        raise ModelGovernanceError("sample gates must be strictly increasing")
    blend_levels = [
        allocation.get("pilot_challenger_allocation"),
        allocation.get("limited_challenger_allocation"),
        allocation.get("maximum_challenger_allocation"),
    ]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0 < float(value) <= 1
        for value in blend_levels
    ):
        raise ModelGovernanceError("challenger allocation gates must be finite values in (0, 1]")
    if [float(value) for value in blend_levels] != sorted(float(value) for value in blend_levels):
        raise ModelGovernanceError("challenger allocation gates must be non-decreasing")
    minimum_mean_excess = allocation.get("minimum_mean_excess_return")
    minimum_win_rate = allocation.get("minimum_paired_win_rate")
    allocation_drawdown = allocation.get("maximum_candidate_drawdown")
    if (
        isinstance(minimum_mean_excess, bool)
        or not isinstance(minimum_mean_excess, (int, float))
        or not math.isfinite(float(minimum_mean_excess))
    ):
        raise ModelGovernanceError("minimum_mean_excess_return must be finite")
    if (
        isinstance(minimum_win_rate, bool)
        or not isinstance(minimum_win_rate, (int, float))
        or not math.isfinite(float(minimum_win_rate))
        or not 0 <= float(minimum_win_rate) <= 1
    ):
        raise ModelGovernanceError("minimum_paired_win_rate must be in [0, 1]")
    if (
        isinstance(allocation_drawdown, bool)
        or not isinstance(allocation_drawdown, (int, float))
        or not math.isfinite(float(allocation_drawdown))
        or not -1 < float(allocation_drawdown) <= 0
    ):
        raise ModelGovernanceError("allocation maximum_candidate_drawdown must be in (-1, 0]")

    recent_window = rollback.get("recent_paired_sample_window")
    recent_loss = rollback.get("maximum_recent_cumulative_excess_loss")
    rollback_drawdown = rollback.get("maximum_candidate_drawdown")
    if isinstance(recent_window, bool) or not isinstance(recent_window, int) or recent_window <= 0:
        raise ModelGovernanceError("recent_paired_sample_window must be a positive integer")
    for field, value in (
        ("maximum_recent_cumulative_excess_loss", recent_loss),
        ("maximum_candidate_drawdown", rollback_drawdown),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -1 < float(value) <= 0
        ):
            raise ModelGovernanceError(f"rollback {field} must be in (-1, 0]")
    if rollback.get("critical_data_failure_rolls_back") is not True:
        raise ModelGovernanceError("critical data failure must roll governance back")
    maximum_without_human = execution.get("maximum_challenger_allocation_without_human_review")
    if execution.get("automatic_order_allowed") is not False:
        raise ModelGovernanceError("model governance must never allow automatic orders")
    if maximum_without_human != allocation.get("maximum_challenger_allocation"):
        raise ModelGovernanceError("execution and allocation maximum challenger weights must match")
    return copy.deepcopy(config)


def load_governance_config(path: Path = DEFAULT_GOVERNANCE_CONFIG) -> dict[str, Any]:
    return validate_governance_config(_strict_json(path))


def governance_fingerprint(config: dict[str, Any]) -> str:
    validated = validate_governance_config(config)
    functions = (
        validate_governance_config,
        regime_evidence,
        weights_for_model_regime,
        blend_model_weights,
        paired_metrics,
        allocation_decision,
    )
    return canonical_fingerprint(
        {
            "governance_contract_version": GOVERNANCE_CONTRACT_VERSION,
            "config": validated,
            "implementation_ast": {
                function.__name__: ast.dump(
                    ast.parse(textwrap.dedent(inspect.getsource(function))),
                    annotate_fields=True,
                    include_attributes=False,
                )
                for function in functions
            },
        }
    )


def model_fingerprint(model: dict[str, Any]) -> str:
    return canonical_fingerprint(model)


def incumbent_model(config: dict[str, Any]) -> dict[str, Any]:
    validated = validate_governance_config(config)
    return next(model for model in validated["models"] if model["role"] == "INCUMBENT")


def challenger_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    validated = validate_governance_config(config)
    return [model for model in validated["models"] if model["role"] == "CHALLENGER"]


def regime_evidence(row: pd.Series) -> dict[str, Any]:
    required_moving_averages = ("QQQ_ma200", "SMH_ma200", "SPY_ma200")
    if any(pd.isna(row.get(field, np.nan)) for field in required_moving_averages):
        return {
            "regime": WARMUP_REGIME,
            "warmup": True,
            "reason_codes": ["MA200_UNAVAILABLE"],
        }

    qqq_below_200 = bool(row["QQQ"] < row["QQQ_ma200"])
    smh_below_200 = bool(row["SMH"] < row["SMH_ma200"])
    spy_below_200 = bool(row["SPY"] < row["SPY_ma200"])
    qqq_weak = bool(row.get("QQQ_ret20", np.nan) <= -0.08)
    smh_weak = bool(row.get("SMH_ret20", np.nan) <= -0.12)
    reason_codes: list[str] = []
    if spy_below_200 and qqq_below_200 and smh_below_200:
        regime = "severe_defensive"
        reason_codes.append("SPY_QQQ_SMH_BELOW_MA200")
    elif (qqq_below_200 and smh_below_200) or (qqq_weak and smh_weak):
        regime = "defensive"
        if qqq_below_200 and smh_below_200:
            reason_codes.append("QQQ_SMH_BELOW_MA200")
        if qqq_weak and smh_weak:
            reason_codes.append("QQQ_SMH_20D_DRAWDOWN")
    else:
        regime = "base"
        reason_codes.append("NO_DEFENSIVE_TRIGGER")
    return {
        "regime": regime,
        "warmup": False,
        "reason_codes": reason_codes,
        "features": {
            "QQQ_below_ma200": qqq_below_200,
            "SMH_below_ma200": smh_below_200,
            "SPY_below_ma200": spy_below_200,
            "QQQ_ret20_at_or_below_minus_8pct": qqq_weak,
            "SMH_ret20_at_or_below_minus_12pct": smh_weak,
        },
    }


def choose_regime(row: pd.Series) -> str:
    return str(regime_evidence(row)["regime"])


def weights_for_model_regime(
    config: dict[str, Any],
    model_id: str,
    regime: str,
) -> dict[str, float]:
    validated = validate_governance_config(config)
    effective_regime = "base" if regime == WARMUP_REGIME else regime
    if effective_regime not in REGIMES:
        raise ModelGovernanceError(f"unsupported market regime: {regime}")
    model = next((item for item in validated["models"] if item["model_id"] == model_id), None)
    if model is None:
        raise ModelGovernanceError(f"unknown model_id: {model_id}")
    return _validate_weights(
        model["regime_weights"][effective_regime],
        f"models.{model_id}.{effective_regime}",
    )


def blend_model_weights(
    config: dict[str, Any],
    model_allocations: dict[str, float],
    regime: str,
) -> dict[str, float]:
    validated = validate_governance_config(config)
    known_ids = {model["model_id"] for model in validated["models"]}
    if not model_allocations or set(model_allocations) - known_ids:
        raise ModelGovernanceError("model allocations must reference only configured models")
    allocations = _validate_weights_like_allocations(model_allocations)
    blended = {asset: 0.0 for asset in ASSETS}
    for model_id, allocation in allocations.items():
        weights = weights_for_model_regime(validated, model_id, regime)
        for asset in ASSETS:
            blended[asset] += allocation * weights[asset]
    total = sum(blended.values())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ModelGovernanceError("blended asset weights do not sum to one")
    return {asset: round(value, 12) for asset, value in blended.items()}


def _validate_weights_like_allocations(values: dict[str, Any]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            raise ModelGovernanceError("model allocations must be numeric")
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ModelGovernanceError("model allocations must be numeric") from exc
        if not math.isfinite(parsed[key]) or parsed[key] < 0:
            raise ModelGovernanceError("model allocations must be finite and non-negative")
    if not math.isclose(sum(parsed.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ModelGovernanceError("model allocations must sum to one")
    return parsed


def _maximum_drawdown(returns: Iterable[float]) -> float:
    wealth = 1.0
    peak = 1.0
    maximum_drawdown = 0.0
    for value in returns:
        wealth *= 1.0 + value
        peak = max(peak, wealth)
        maximum_drawdown = min(maximum_drawdown, wealth / peak - 1.0)
    return maximum_drawdown


def _compound(returns: Iterable[float]) -> float:
    value = 1.0
    for item in returns:
        value *= 1.0 + item
    return value - 1.0


def paired_metrics(
    incumbent_returns: list[float],
    challenger_returns: list[float],
    *,
    recent_window: int,
) -> dict[str, Any]:
    if isinstance(recent_window, bool) or not isinstance(recent_window, int) or recent_window <= 0:
        raise ModelGovernanceError("recent_window must be a positive integer")
    if len(incumbent_returns) != len(challenger_returns):
        raise ModelGovernanceError("incumbent and challenger sample counts must match")
    if not incumbent_returns:
        return {
            "paired_sample_count": 0,
            "mean_excess_return": None,
            "median_excess_return": None,
            "paired_win_rate": None,
            "cumulative_incumbent_return": None,
            "cumulative_challenger_return": None,
            "cumulative_excess_return": None,
            "recent_cumulative_excess_return": None,
            "challenger_max_drawdown": None,
        }
    values = [*incumbent_returns, *challenger_returns]
    if any(not math.isfinite(value) or value <= -1 for value in values):
        raise ModelGovernanceError("paired returns must be finite and greater than -100%")
    excess = [challenger - incumbent for incumbent, challenger in zip(incumbent_returns, challenger_returns)]
    recent_incumbent = incumbent_returns[-recent_window:]
    recent_challenger = challenger_returns[-recent_window:]
    cumulative_incumbent = _compound(incumbent_returns)
    cumulative_challenger = _compound(challenger_returns)
    return {
        "paired_sample_count": len(excess),
        "mean_excess_return": round(float(np.mean(excess)), 12),
        "median_excess_return": round(float(np.median(excess)), 12),
        "paired_win_rate": round(sum(value > 0 for value in excess) / len(excess), 12),
        "cumulative_incumbent_return": round(cumulative_incumbent, 12),
        "cumulative_challenger_return": round(cumulative_challenger, 12),
        "cumulative_excess_return": round(cumulative_challenger - cumulative_incumbent, 12),
        "recent_cumulative_excess_return": round(
            _compound(recent_challenger) - _compound(recent_incumbent),
            12,
        ),
        "challenger_max_drawdown": round(_maximum_drawdown(challenger_returns), 12),
    }


def allocation_decision(
    config: dict[str, Any],
    challenger_metrics: dict[str, dict[str, Any]],
    *,
    data_integrity_ok: bool = True,
) -> dict[str, Any]:
    validated = validate_governance_config(config)
    incumbent = incumbent_model(validated)
    challengers = challenger_models(validated)
    configured_ids = {model["model_id"] for model in challengers}
    if set(challenger_metrics) - configured_ids:
        raise ModelGovernanceError("metrics contain an unknown challenger")
    gates = validated["allocation_gates"]
    rollback = validated["rollback_gates"]
    incumbent_id = incumbent["model_id"]
    base_result = {
        "incumbent_model_id": incumbent_id,
        "selected_challenger_model_id": None,
        "model_allocations": {incumbent_id: 1.0},
        "challenger_allocation": 0.0,
        "automatic_order_allowed": False,
        "human_review_required": True,
    }
    if not data_integrity_ok:
        return {
            **base_result,
            "status": "ROLLBACK_DATA_INTEGRITY_FAILURE",
            "reason_codes": ["CRITICAL_FORWARD_EVIDENCE_FAILURE"],
        }

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    rollback_candidates: list[str] = []
    for model in challengers:
        model_id = model["model_id"]
        metrics = challenger_metrics.get(model_id, {})
        count = metrics.get("paired_sample_count", 0)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ModelGovernanceError(f"invalid paired sample count for {model_id}")
        if count == 0:
            continue
        required_numeric = (
            "mean_excess_return",
            "paired_win_rate",
            "recent_cumulative_excess_return",
            "challenger_max_drawdown",
        )
        if any(
            not isinstance(metrics.get(field), (int, float))
            or isinstance(metrics.get(field), bool)
            or not math.isfinite(float(metrics[field]))
            for field in required_numeric
        ):
            raise ModelGovernanceError(f"non-finite or missing paired metric for {model_id}")
        if (
            float(metrics["recent_cumulative_excess_return"])
            <= float(rollback["maximum_recent_cumulative_excess_loss"])
            or float(metrics["challenger_max_drawdown"])
            <= float(rollback["maximum_candidate_drawdown"])
        ):
            rollback_candidates.append(model_id)
            continue
        eligible = (
            count >= int(gates["minimum_paired_samples_for_pilot_blend"])
            and float(metrics["mean_excess_return"]) >= float(gates["minimum_mean_excess_return"])
            and float(metrics["paired_win_rate"]) >= float(gates["minimum_paired_win_rate"])
            and float(metrics["challenger_max_drawdown"]) >= float(gates["maximum_candidate_drawdown"])
        )
        if eligible:
            ranked.append((float(metrics["mean_excess_return"]), model_id, metrics))

    if not ranked:
        if rollback_candidates:
            return {
                **base_result,
                "status": "ROLLBACK_TO_INCUMBENT",
                "reason_codes": ["CHALLENGER_ROLLBACK_GATE_BREACHED"],
                "rollback_challenger_model_ids": sorted(rollback_candidates),
            }
        return {
            **base_result,
            "status": "HOLD_INCUMBENT_INSUFFICIENT_OR_WEAK_PROSPECTIVE_EVIDENCE",
            "reason_codes": ["NO_CHALLENGER_PASSED_FROZEN_ALLOCATION_GATES"],
        }

    _, selected_id, selected_metrics = sorted(ranked, key=lambda item: (-item[0], item[1]))[0]
    count = int(selected_metrics["paired_sample_count"])
    if count >= int(gates["minimum_paired_samples_for_promotion_review"]):
        allocation = float(gates["maximum_challenger_allocation"])
        status = "PROMOTION_REVIEW_REQUIRED_MAXIMUM_CAPPED_BLEND"
    elif count >= int(gates["minimum_paired_samples_for_limited_blend"]):
        allocation = float(gates["limited_challenger_allocation"])
        status = "LIMITED_PROSPECTIVE_BLEND"
    else:
        allocation = float(gates["pilot_challenger_allocation"])
        status = "PILOT_PROSPECTIVE_BLEND"
    return {
        **base_result,
        "status": status,
        "selected_challenger_model_id": selected_id,
        "model_allocations": {
            incumbent_id: round(1.0 - allocation, 12),
            selected_id: round(allocation, 12),
        },
        "challenger_allocation": round(allocation, 12),
        "reason_codes": ["CHALLENGER_PASSED_FROZEN_PROSPECTIVE_GATES"],
        "selected_challenger_metrics": selected_metrics,
    }
