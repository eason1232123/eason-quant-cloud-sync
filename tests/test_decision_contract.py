from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

import jsonschema
import pandas as pd

from scripts import build_decision_report as decision
from scripts import build_action_board_v3 as master_board
from scripts import build_dynamic_candidates as dynamic
from scripts import build_report as base_report
from scripts import build_report_safe as safe_report
from scripts import market_clock
from scripts import market_data_contract
from scripts import strategy_contract
from scripts import validate_decision_packet as contract_validator
from scripts import validate_generated_json as generated_json_validator
from scripts import validate_market_universe as universe_validator
from scripts import validate_operational_health as operational_validator


AS_OF = "2026-07-09"
GENERATED_AT = "2026-07-10T00:00:00+00:00"


def technical(latest_date: str = AS_OF, *, high_risk: bool = False) -> dict:
    active_signals = {
        "pullback_reclaim_5dma": False,
        "rsi_oversold_reclaim_40": False,
        "ma20_reclaim_bullish": False,
        "ma50_reclaim_bullish": False,
        "relative_strength_rebound": False,
        "momentum_leader": False,
        "failed_rebound_risk": high_risk,
    }
    if high_risk:
        return {
            "latest_date": latest_date,
            "latest_price": 100.0,
            "trading_days": 500,
            "relative_benchmark": "SMH",
            "above_ma5": False,
            "above_ma20": False,
            "above_ma50": False,
            "above_ma200": True,
            "ret_5d": -0.05,
            "ret_20d": -0.10,
            "drawdown_from_52w_high": -0.20,
            "active_signals": active_signals,
        }
    return {
        "latest_date": latest_date,
        "latest_price": 100.0,
        "trading_days": 500,
        "relative_benchmark": "SMH",
        "above_ma5": True,
        "above_ma20": True,
        "above_ma50": True,
        "above_ma200": True,
        "ret_5d": 0.01,
        "ret_20d": 0.03,
        "drawdown_from_52w_high": -0.02,
        "active_signals": active_signals,
    }


def backtest(latest_date: str = AS_OF, *, active_buy: bool = False) -> dict:
    active = {"relative_strength_rebound": active_buy}
    return {
        "latest_date": latest_date,
        "latest_price": 100.0,
        "active_signals_latest_day": active,
        "relative_strength_rebound": {
            "20d": {
                "samples": 30,
                "valid": True,
                "win_rate": 0.70,
                "avg_return": 0.08,
                "median_return": 0.07,
                "worst_return": -0.10,
                "avg_mae": -0.04,
                "worst_mae": -0.12,
                "avg_alpha_vs_SMH": 0.02,
            }
        },
        "failed_rebound_risk": {"20d": {"samples": 30, "valid": True, "avg_return": -0.01, "median_return": -0.02, "worst_mae": -0.20}},
    }


def report(*, lrcx_date: str = AS_OF, arm_date: str = AS_OF, qqq_date: str = AS_OF) -> dict:
    technicals = {
        "QQQ": technical(qqq_date),
        "SMH": technical(),
        "MSFT": technical(),
        "SPY": technical(),
        "LRCX": technical(lrcx_date),
        "ARM": technical(arm_date, high_risk=True),
    }
    backtests = {ticker: backtest(item["latest_date"]) for ticker, item in technicals.items()}
    backtests["LRCX"] = backtest(lrcx_date, active_buy=True)
    return {
        "generated_at_utc": GENERATED_AT,
        "data_source": "fixture adjusted end-of-day prices",
        "market_timezone": "America/New_York",
        "data_timestamp": max(lrcx_date, arm_date, qqq_date),
        "data_timestamp_granularity": "market_date",
        "data_timestamp_status": "AVAILABLE",
        "price_frequency": "end_of_day_daily",
        "price_adjustment_policy": "adjusted_ohlc_when_available_else_unadjusted",
        "strategy_version": "fixture",
        "universe": {
            "expected_latest_market_date": AS_OF,
            "loaded_ticker_count": len(technicals),
            "configured_ticker_count": len(technicals),
            "fresh_request_count": 1,
        },
        "technicals": technicals,
        "backtests": backtests,
        "rule_evidence_ranking": {
            "LRCX": [{"rule": "relative_strength_rebound", "evidence_score_0_100": 82.0}]
        },
        "errors": {},
    }


def portfolio(*extra_tickers: str) -> dict:
    weights = {"QQQ": 0.30, "SMH": 0.25, "MSFT": 0.20, "SPY": 0.10, "CASH": 0.15}
    for ticker in extra_tickers:
        weights[ticker] = 0.01
    return {
        "available": True,
        "latest_regime": "base",
        "assumptions": {
            "base_weights": weights,
            "defensive_weights": {"QQQ": 0.2, "SPY": 0.2, "CASH": 0.6},
            "severe_defensive_weights": {"SPY": 0.1, "CASH": 0.9},
        },
    }


class DecisionContractTests(unittest.TestCase):
    def compile(self, market_report: dict, portfolio_report: dict) -> dict:
        return decision.compile_decision_outputs(market_report, portfolio_report, GENERATED_AT)

    def test_watchlist_high_risk_does_not_block_fresh_buy(self):
        outputs = self.compile(report(), portfolio())
        signal = outputs["decision"]
        self.assertEqual(signal["final_action"], "BUY_CANDIDATE_REVIEW_REQUIRED")
        self.assertEqual(signal["model_portfolio_high_risks"], [])
        self.assertEqual(signal["watchlist_high_risks_advisory"][0]["ticker"], "ARM")
        self.assertEqual(signal["actionable_buy_candidates"][0]["ticker"], "LRCX")

    def test_model_portfolio_high_risk_blocks_buy(self):
        outputs = self.compile(report(), portfolio("ARM"))
        signal = outputs["decision"]
        self.assertEqual(signal["final_action"], "RISK_REVIEW_REQUIRED")
        self.assertEqual(signal["model_portfolio_high_risks"][0]["ticker"], "ARM")

    def test_stale_buy_is_excluded(self):
        outputs = self.compile(report(lrcx_date="2026-07-07"), portfolio())
        signal = outputs["decision"]
        self.assertEqual(signal["final_action"], "NO_TRADE")
        self.assertEqual(signal["actionable_buy_candidates"], [])
        stale_lrcx = [row for row in signal["all_active_buy_candidates"] if row["ticker"] == "LRCX"]
        self.assertTrue(stale_lrcx)
        self.assertEqual(stale_lrcx[0]["status"], "STALE_DATA_EXCLUDED")

    def test_stale_signal_date_is_excluded_even_when_technical_is_fresh(self):
        market_report = report()
        market_report["backtests"]["LRCX"] = backtest("2026-07-07", active_buy=True)
        outputs = self.compile(market_report, portfolio())
        signal = outputs["decision"]
        self.assertEqual(signal["final_action"], "NO_TRADE")
        self.assertEqual(signal["actionable_buy_candidates"], [])
        candidate = next(row for row in signal["all_active_buy_candidates"] if row["ticker"] == "LRCX")
        self.assertFalse(candidate["signal_data_fresh"])
        self.assertEqual(candidate["status"], "STALE_DATA_EXCLUDED")

    def test_missing_backtest_values_remain_null_and_cannot_authorize_buy(self):
        market_report = report()
        evidence = market_report["backtests"]["LRCX"]["relative_strength_rebound"]["20d"]
        evidence.pop("samples")
        evidence.pop("valid")
        market_report["rule_evidence_ranking"] = {}
        signal = self.compile(market_report, portfolio())["decision"]
        candidate = next(row for row in signal["all_active_buy_candidates"] if row["ticker"] == "LRCX")
        self.assertEqual(signal["final_action"], "NO_TRADE")
        self.assertIsNone(candidate["samples"])
        self.assertIsNone(candidate["valid"])
        self.assertIsNone(candidate["evidence_score_0_100"])
        self.assertIn("samples_missing_or_invalid", candidate["fail_reasons"])
        self.assertIn("evidence_score_missing_or_invalid", candidate["fail_reasons"])

    def test_missing_worst_mae_cannot_authorize_buy(self):
        market_report = report()
        market_report["backtests"]["LRCX"]["relative_strength_rebound"]["20d"].pop("worst_mae")
        signal = self.compile(market_report, portfolio())["decision"]
        candidate = next(row for row in signal["all_active_buy_candidates"] if row["ticker"] == "LRCX")
        self.assertEqual(signal["final_action"], "NO_TRADE")
        self.assertIsNone(candidate["worst_mae_pct"])
        self.assertIn("worst_MAE_missing_or_invalid", candidate["fail_reasons"])

    def test_shadow_candidate_collects_evidence_without_upgrading_no_trade(self):
        market_report = report()
        market_report["backtests"]["LRCX"]["relative_strength_rebound"]["20d"].pop(
            "worst_mae"
        )
        outputs = self.compile(market_report, portfolio())
        packet = outputs["decision_packet"]

        self.assertEqual(packet["decision"]["final_action"], "NO_TRADE")
        self.assertEqual(packet["candidates"]["execution"]["candidate_count"], 0)
        shadow = packet["candidates"]["shadow"]
        self.assertEqual(shadow["candidate_type"], "SHADOW_CANDIDATE")
        self.assertEqual(shadow["candidate_count"], 1)
        self.assertEqual(shadow["top"][0]["ticker"], "LRCX")
        self.assertTrue(shadow["top"][0]["counterfactual_only"])
        self.assertFalse(shadow["top"][0]["execution_eligible"])
        self.assertFalse(shadow["top"][0]["automatic_order_allowed"])
        self.assertFalse(shadow["top"][0]["prospective_evidence_eligible"])

    def test_stale_model_ticker_blocks_on_data_not_technical_risk(self):
        outputs = self.compile(report(qqq_date="2026-07-07"), portfolio())
        signal = outputs["decision"]
        self.assertEqual(signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(signal["buy_permission"], "BLOCKED_BY_MODEL_PORTFOLIO_DATA")
        self.assertEqual(signal["data_status"], "STALE_MODEL_PORTFOLIO_DATA")

    def test_fresh_update_log_cannot_hide_missing_model_analysis(self):
        market_report = report()
        market_report["update_log"] = {"QQQ": {"latest_date": AS_OF}}
        market_report["technicals"].pop("QQQ")
        market_report["backtests"].pop("QQQ")
        outputs = self.compile(market_report, portfolio())
        signal = outputs["decision"]
        self.assertEqual(signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(signal["buy_permission"], "BLOCKED_BY_MODEL_PORTFOLIO_DATA")
        stale_qqq = next(row for row in signal["freshness"]["stale_tickers"] if row["ticker"] == "QQQ")
        self.assertEqual(stale_qqq["reason"], "technical_analysis_missing_or_failed")
        schema = json.loads(Path("schemas/decision_packet.schema.json").read_text(encoding="utf-8"))
        contract_validator.validate_schema(outputs["decision_packet"], schema)
        contract_validator.validate_invariants(outputs["decision_packet"])

        partial_report = report()
        partial_report["technicals"]["QQQ"] = {"latest_date": AS_OF}
        partial_signal = self.compile(partial_report, portfolio())["decision"]
        self.assertEqual(partial_signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(partial_signal["buy_permission"], "BLOCKED_BY_MODEL_PORTFOLIO_DATA")

    def test_non_boolean_active_signal_cannot_trigger_candidate(self):
        market_report = report()
        market_report["backtests"]["LRCX"]["active_signals_latest_day"]["relative_strength_rebound"] = "false"
        signal = self.compile(market_report, portfolio())["decision"]
        self.assertEqual(signal["final_action"], "NO_TRADE")
        self.assertEqual(signal["actionable_buy_candidates"], [])

    def test_business_day_freshness_boundary_and_future_date(self):
        self.assertEqual(decision.business_day_age(AS_OF, AS_OF), 0)
        self.assertEqual(decision.business_day_age("2026-07-07", AS_OF), 2)
        self.assertIsNone(decision.business_day_age("2026-07-10", AS_OF))
        self.assertIsNone(decision.business_day_age("not-a-date", AS_OF))

    def test_runtime_market_clock_prevents_old_report_self_certification(self):
        old_date = "2026-03-17"
        market_report = report(lrcx_date=old_date, arm_date=old_date, qqq_date=old_date)
        for item in market_report["technicals"].values():
            item["latest_date"] = old_date
        for item in market_report["backtests"].values():
            item["latest_date"] = old_date
        market_report["universe"]["expected_latest_market_date"] = old_date
        outputs = self.compile(market_report, portfolio())
        signal = outputs["decision"]
        self.assertEqual(signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(signal["buy_permission"], "BLOCKED_BY_MARKET_DATE_CONTEXT")
        self.assertEqual(signal["freshness"]["reference_market_date_status"], "REPORTED_EXPECTED_DATE_MISMATCH")

    def test_missing_report_expected_date_fails_closed(self):
        market_report = report()
        market_report["universe"].pop("expected_latest_market_date")
        outputs = self.compile(market_report, portfolio())
        self.assertEqual(outputs["decision"]["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(
            outputs["decision"]["freshness"]["reference_market_date_status"],
            "REPORTED_EXPECTED_DATE_MISSING",
        )

    def test_model_scope_unions_regime_weights_and_excludes_cash(self):
        scope = decision.model_risk_scope(portfolio("ARM"))
        self.assertIn("ARM", scope["tickers"])
        self.assertNotIn("CASH", scope["tickers"])
        self.assertFalse(scope["contains_private_shares_or_cash"])

    def test_missing_or_malformed_portfolio_context_blocks_buy(self):
        missing = self.compile(report(), {})["decision"]
        self.assertEqual(missing["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(missing["buy_permission"], "BLOCKED_BY_PORTFOLIO_CONTEXT")

        malformed = portfolio()
        malformed["assumptions"] = {"base_weights": "invalid"}
        malformed_signal = self.compile(report(), malformed)["decision"]
        self.assertEqual(malformed_signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(malformed_signal["buy_permission"], "BLOCKED_BY_PORTFOLIO_CONTEXT")

        partial = portfolio()
        partial["assumptions"].pop("severe_defensive_weights")
        partial_signal = self.compile(report(), partial)["decision"]
        self.assertEqual(partial_signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(partial_signal["buy_permission"], "BLOCKED_BY_PORTFOLIO_CONTEXT")

        missing_available = portfolio()
        missing_available.pop("available")
        missing_available_outputs = self.compile(report(), missing_available)
        self.assertEqual(
            missing_available_outputs["decision"]["buy_permission"],
            "BLOCKED_BY_PORTFOLIO_CONTEXT",
        )
        self.assertFalse(missing_available_outputs["action_board"]["portfolio_backtest"]["available"])

    def test_missing_market_metadata_is_null_and_blocks_instead_of_fabricating(self):
        market_report = report()
        market_report.pop("data_source")
        market_report["universe"].pop("configured_ticker_count")
        outputs = self.compile(market_report, portfolio())
        signal = outputs["decision"]
        packet = outputs["decision_packet"]
        self.assertEqual(signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(signal["buy_permission"], "BLOCKED_BY_SOURCE_METADATA")
        self.assertEqual(packet["market_data"]["source"], None)
        self.assertIsNone(packet["data_quality"]["configured_ticker_count"])
        self.assertEqual(packet["data_quality"]["data_quality_metadata_status"], "MISSING_OR_INVALID")
        schema = json.loads(Path("schemas/decision_packet.schema.json").read_text(encoding="utf-8"))
        contract_validator.validate_schema(packet, schema)
        contract_validator.validate_invariants(packet)

    def test_invalid_portfolio_regime_is_not_silently_substituted(self):
        portfolio_report = portfolio()
        portfolio_report["latest_regime"] = "unknown"
        outputs = self.compile(report(), portfolio_report)
        signal = outputs["decision"]
        packet = outputs["decision_packet"]
        self.assertEqual(signal["final_action"], "DATA_REVIEW_REQUIRED")
        self.assertEqual(signal["buy_permission"], "BLOCKED_BY_PORTFOLIO_CONTEXT")
        self.assertEqual(signal["portfolio_scope"]["portfolio_context_reason"], "portfolio_regime_missing_or_invalid")
        self.assertIsNone(packet["market_context"]["model_regime"])
        schema = json.loads(Path("schemas/decision_packet.schema.json").read_text(encoding="utf-8"))
        contract_validator.validate_schema(packet, schema)
        contract_validator.validate_invariants(packet)

    def test_packet_schema_and_no_automatic_order(self):
        outputs = self.compile(report(), portfolio())
        packet = outputs["decision_packet"]
        schema_path = Path("schemas/decision_packet.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        contract_validator.validate_schema(packet, schema)
        contract_validator.validate_invariants(packet)
        validator = jsonschema.Draft202012Validator(
            schema,
            format_checker=contract_validator.FORMAT_CHECKER,
        )
        self.assertFalse(packet["decision"]["automatic_order_allowed"])
        self.assertFalse(packet["execution_contract"]["automatic_order_allowed"])
        json.dumps(packet, allow_nan=False)

        for path in (
            ("market_data", "data_timestamp"),
            ("data_quality", "data_status"),
            ("data_quality", "reference_market_date"),
            ("portfolio_scope", "source"),
            ("market_context", "model_regime"),
            ("market_context", "live_regime_check_required"),
        ):
            invalid = deepcopy(packet)
            invalid[path[0]][path[1]] = None
            with self.subTest(path=path), self.assertRaises(jsonschema.ValidationError):
                validator.validate(invalid)

    def test_packet_invariants_reject_forged_health_permissions_actions_and_dates(self):
        packet = self.compile(report(), portfolio())["decision_packet"]
        schema = json.loads(Path("schemas/decision_packet.schema.json").read_text(encoding="utf-8"))

        forged_health = deepcopy(packet)
        loaded = forged_health["data_quality"]["loaded_ticker_count"]
        forged_health["data_quality"]["fresh_ticker_count"] = 0
        forged_health["data_quality"]["stale_ticker_count"] = loaded
        forged_health["data_quality"]["data_status"] = "FRESH"
        contract_validator.validate_schema(forged_health, schema)
        with self.assertRaises(AssertionError):
            contract_validator.validate_invariants(forged_health)

        forged_permission = deepcopy(packet)
        forged_permission["decision"]["buy_permission"] = "NO_QUANT_CANDIDATE"
        contract_validator.validate_schema(forged_permission, schema)
        with self.assertRaises(AssertionError):
            contract_validator.validate_invariants(forged_permission)

        conflicting_action = deepcopy(packet)
        conflicting_action["decision"]["default_human_action"] = "NO_TRADE_UNLESS_LIVE_RISK_OVERRIDE"
        contract_validator.validate_schema(conflicting_action, schema)
        with self.assertRaises(AssertionError):
            contract_validator.validate_invariants(conflicting_action)

        bypassed_live_review = deepcopy(packet)
        bypassed_live_review["decision"]["chatgpt_review_required"] = False
        contract_validator.validate_schema(bypassed_live_review, schema)
        with self.assertRaises(AssertionError):
            contract_validator.validate_invariants(bypassed_live_review)

        future_candidate = deepcopy(packet)
        future_candidate["candidates"]["execution"]["top"][0]["latest_date"] = "2026-07-10"
        contract_validator.validate_schema(future_candidate, schema)
        with self.assertRaises(AssertionError):
            contract_validator.validate_invariants(future_candidate)

    def test_master_board_finalizes_packet_without_recursive_fields(self):
        outputs = self.compile(report(), portfolio())
        finalized = master_board.finalize_decision_packet(
            {**outputs["decision_packet"], "available": True},
            {
                "quant_signal": "BUY_CANDIDATE_REVIEW_REQUIRED",
                "recommended_default_action": "WAIT_FOR_CHATGPT_LIVE_REVIEW",
                "warnings": ["paper-trade validation pending"],
                "chatgpt_review_required": True,
            },
        )
        self.assertNotIn("available", finalized)
        self.assertEqual(finalized["evidence_gate"]["warnings"], ["paper-trade validation pending"])
        self.assertFalse(finalized["decision"]["automatic_order_allowed"])
        schema = json.loads(Path("schemas/decision_packet.schema.json").read_text(encoding="utf-8"))
        contract_validator.validate_schema(finalized, schema)
        contract_validator.validate_invariants(finalized)

    def test_zero_coverage_vectorbt_cannot_leave_buy_review_as_default(self):
        gate = master_board.final_gate(
            {"available": True, "final_action": "BUY_CANDIDATE_REVIEW_REQUIRED", "data_status": "FRESH"},
            {"verdict": "PASS_STABILITY_CHECK"},
            {},
            {"available": True, "errors": {}},
            {
                "available": True,
                "loaded_ticker_count": 0,
                "configured_ticker_count": 94,
                "required_evidence_fields": {"sample_count": True},
                "top_strategy_results": [],
                "top_entry_forward_evidence_20d": [],
                "errors": {},
            },
        )
        self.assertEqual(gate["recommended_default_action"], "REFRESH_DATA_BEFORE_DECISION")
        self.assertTrue(any("unavailable or incomplete" in warning for warning in gate["warnings"]))

    def test_independent_validation_failure_forces_refresh_and_blocks_finalized_packet(self):
        signal = {
            "available": True,
            "final_action": "BUY_CANDIDATE_REVIEW_REQUIRED",
            "data_status": "FRESH",
        }
        vectorbt_report = {
            "available": True,
            "strategy_contract_version": strategy_contract.STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": strategy_contract.RULE_FINGERPRINT,
            "strategy_fingerprint": strategy_contract.STRATEGY_FINGERPRINT,
            "data_source": "fixture source",
            "market_timezone": market_clock.MARKET_TIMEZONE,
            "data_timestamp": AS_OF,
            "price_frequency": market_data_contract.PRICE_FREQUENCY,
            "price_adjustment_policy": market_data_contract.PRICE_ADJUSTMENT_POLICY,
            "loaded_ticker_count": 6,
            "configured_ticker_count": 6,
            "required_evidence_fields": {"sample_count": True},
            "top_strategy_results": [{"strategy": "fixture"}],
            "top_entry_forward_evidence_20d": [{"rule": "fixture"}],
            "errors": {},
        }
        valid_validation = {
            "available": True,
            "strategy_contract_version": strategy_contract.STRATEGY_CONTRACT_VERSION,
            "rule_fingerprint": strategy_contract.RULE_FINGERPRINT,
            "strategy_fingerprint": strategy_contract.STRATEGY_FINGERPRINT,
            "data_source": "fixture source",
            "market_timezone": market_clock.MARKET_TIMEZONE,
            "data_timestamp": AS_OF,
            "price_frequency": market_data_contract.PRICE_FREQUENCY,
            "price_adjustment_policy": market_data_contract.PRICE_ADJUSTMENT_POLICY,
            "data": {"rows": 300, "loaded_tickers": ["SPY"]},
            "top_by_sharpe": [{"rule": "fixture"}],
            "errors": {},
        }
        healthy_gate = master_board.final_gate(
            signal,
            {"verdict": "PASS_STABILITY_CHECK"},
            {},
            valid_validation,
            vectorbt_report,
        )
        self.assertEqual(healthy_gate["recommended_default_action"], "WAIT_FOR_CHATGPT_LIVE_REVIEW")

        failed_gate = master_board.final_gate(
            signal,
            {"verdict": "PASS_STABILITY_CHECK"},
            {},
            {"available": False},
            vectorbt_report,
        )
        self.assertEqual(failed_gate["recommended_default_action"], "REFRESH_DATA_BEFORE_DECISION")
        finalized = master_board.finalize_decision_packet(
            self.compile(report(), portfolio())["decision_packet"],
            failed_gate,
        )
        schema = json.loads(Path("schemas/decision_packet.schema.json").read_text(encoding="utf-8"))
        contract_validator.validate_schema(finalized, schema)
        contract_validator.validate_invariants(finalized)
        with self.assertRaisesRegex(AssertionError, "evidence gate requires"):
            operational_validator.validate_operational_state(finalized)

    def test_quality_summary_never_substitutes_expected_date_for_observed_date(self):
        market = report()
        market["data_timestamp"] = "2026-07-08"
        market["universe"].pop("latest_price_date_max", None)
        packet = self.compile(report(), portfolio())["decision_packet"]
        packet["data_quality"]["latest_price_date_max"] = "2026-07-09"
        quality = master_board.score_github_data_quality(
            {"available": True, **market},
            {"available": True, "configured_ticker_count": 6, "loaded_ticker_count": 6},
            packet,
        )
        self.assertEqual(quality["latest_price_date_max"], "2026-07-09")

        packet["data_quality"]["latest_price_date_max"] = None
        quality = master_board.score_github_data_quality(
            {"available": True, **market},
            {"available": True, "configured_ticker_count": 6, "loaded_ticker_count": 6},
            packet,
        )
        self.assertEqual(quality["latest_price_date_max"], "2026-07-08")

        market["data_timestamp"] = None
        quality = master_board.score_github_data_quality(
            {"available": True, **market},
            {"available": True, "configured_ticker_count": 6, "loaded_ticker_count": 6},
            packet,
        )
        self.assertIsNone(quality["latest_price_date_max"])

    def test_dynamic_candidate_stale_gate_and_age(self):
        row = {
            "data_fresh": False,
            "history_rows": 300,
            "avg_dollar_volume_20": 100_000_000,
            "volatility_60d_pct": 30.0,
            "ret_20d_pct": 5.0,
            "alpha_60d_vs_qqq_pct": 1.0,
            "alpha_120d_vs_spy_pct": 1.0,
        }
        self.assertIn("stale_market_data", dynamic.gate_candidate(row, is_core=False))
        self.assertEqual(dynamic.business_day_age("2026-07-07", AS_OF), 2)

    def test_dynamic_price_reader_rejects_out_of_order_future_bar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            dates = list(pd.bdate_range(end=AS_OF, periods=22).strftime("%Y-%m-%d"))
            rows = [
                {"date": market_date, "adjClose": 100 + index, "adjVolume": 1_000_000}
                for index, market_date in enumerate(dates)
            ]
            rows.insert(10, {"date": "2026-07-10", "adjClose": 999, "adjVolume": 1_000_000})
            pd.DataFrame(rows).to_csv(out / "SPY_daily.csv", index=False)
            with self.assertRaisesRegex(ValueError, "future-dated"):
                dynamic.read_prices(out / "SPY_daily.csv", AS_OF)

    def test_dynamic_reference_uses_clock_not_cached_report_date(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old_docs = dynamic.DOCS
            dynamic.DOCS = Path(temp_dir)
            try:
                (dynamic.DOCS / "market_report.json").write_text(
                    json.dumps({"universe": {"expected_latest_market_date": "2026-03-17"}}),
                    encoding="utf-8",
                )
                context = dynamic.reference_market_context(as_of_market_date=AS_OF)
            finally:
                dynamic.DOCS = old_docs
        self.assertEqual(context["reference_market_date"], AS_OF)
        self.assertEqual(context["reference_market_date_status"], "REPORTED_EXPECTED_DATE_MISMATCH")
        age, eligible = dynamic.candidate_freshness(AS_OF, context)
        self.assertEqual(age, 0)
        self.assertFalse(eligible)

    def test_shared_market_clock_handles_after_close_and_weekend(self):
        self.assertEqual(
            market_clock.latest_completed_us_market_weekday("2026-07-10T00:00:00+00:00").isoformat(),
            AS_OF,
        )
        self.assertEqual(
            market_clock.latest_completed_us_market_weekday("2026-07-11T12:00:00+00:00").isoformat(),
            "2026-07-10",
        )

    def test_market_data_metadata_uses_observed_dates_and_price_basis(self):
        self.assertEqual(base_report.clean_float(1.23456), 1.2346)
        self.assertIsNone(base_report.clean_float(float("nan")))
        adjusted = pd.DataFrame({"date": ["2026-07-08", AS_OF], "adjClose": [99.0, 100.0]})
        unadjusted = pd.DataFrame({"date": ["2026-07-07"], "close": [50.0]})
        metadata = base_report.market_data_report_fields(
            {"LRCX": adjusted, "CFLT": unadjusted},
            "fixture source",
        )
        self.assertEqual(metadata["data_timestamp"], AS_OF)
        self.assertEqual(metadata["data_timestamp_by_ticker"]["CFLT"], "2026-07-07")
        self.assertEqual(metadata["price_basis_by_ticker"]["LRCX"], "adjusted")
        self.assertEqual(metadata["price_basis_by_ticker"]["CFLT"], "unadjusted")
        self.assertEqual(metadata["market_timezone"], "America/New_York")

        empty = base_report.market_data_report_fields({}, "fixture source")
        self.assertIsNone(empty["data_timestamp"])
        self.assertEqual(empty["data_timestamp_status"], "MISSING")

    def test_severely_stale_long_tail_forces_refresh(self):
        existing = pd.DataFrame({"date": ["2026-03-17"]})
        should_fetch, reason, status = safe_report.should_fetch_today(
            "CFLT",
            existing,
            date.fromisoformat(AS_OF),
            datetime(2026, 7, 10, 12, tzinfo=timezone.utc),
        )
        self.assertTrue(should_fetch)
        self.assertEqual(status, "fetch")
        self.assertIn("forced refresh", reason)

    def test_future_cache_is_quarantined_without_request(self):
        existing = pd.DataFrame({"date": ["2026-07-10"]})
        should_fetch, reason, status = safe_report.should_fetch_today(
            "CFLT",
            existing,
            date.fromisoformat(AS_OF),
            datetime(2026, 7, 10, 12, tzinfo=timezone.utc),
        )
        self.assertFalse(should_fetch)
        self.assertEqual(status, "cache_future_dated_quarantine")
        self.assertIn("future-dated", reason)

    def test_shared_price_loader_rejects_future_cache_for_all_downstream_engines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir)
            pd.DataFrame({"date": [AS_OF, "2026-07-10"], "adjClose": [100.0, 101.0]}).to_csv(
                out / "SPY_daily.csv",
                index=False,
            )
            with self.assertRaisesRegex(ValueError, "future-dated"):
                market_data_contract.read_checked_daily_csv("SPY", out, AS_OF)
            frame = market_data_contract.read_checked_daily_csv("SPY", out, "2026-07-10")
            self.assertEqual(len(frame), 2)

    def test_tiingo_network_errors_never_expose_api_token(self):
        original_key = base_report.API_KEY
        base_report.API_KEY = "unit-test-secret-token"
        try:
            raw = "timeout https://api.tiingo.com/prices?token=unit-test-secret-token&startDate=2026-01-01"
            self.assertNotIn("unit-test-secret-token", base_report.redact_sensitive_text(raw))
            with patch.object(base_report.requests, "get", side_effect=base_report.requests.Timeout(raw)):
                with self.assertRaises(RuntimeError) as raised:
                    base_report.fetch_tiingo("SPY", "2026-01-01")
            message = str(raised.exception)
            self.assertNotIn("unit-test-secret-token", message)
            self.assertNotIn("token=", message)
            self.assertIn("Timeout", message)
        finally:
            base_report.API_KEY = original_key

    def test_request_order_prioritizes_critical_and_overdue_data(self):
        cached_dates = {
            "SPY": date(2026, 7, 8),
            "CFLT": date(2026, 3, 17),
            "AAPL": date.fromisoformat(AS_OF),
        }
        with patch.object(safe_report.br, "cached_latest_date", side_effect=lambda ticker: cached_dates[ticker]):
            expected = date.fromisoformat(AS_OF)
            ordered = sorted(cached_dates, key=lambda ticker: safe_report.request_order_key(ticker, expected))
        self.assertEqual(ordered, ["SPY", "CFLT", "AAPL"])

    def test_market_universe_validator_rejects_partial_source_failure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "market_universe.csv"
            summary_path = root / "market_universe_summary.json"
            rows = [
                {
                    "ticker": f"T{index:04d}",
                    "security_name": f"Security {index}",
                    "exchange": "NASDAQ",
                    "source": "NASDAQ",
                    "eligibility": "eligible",
                }
                for index in range(universe_validator.MINIMUM_ELIGIBLE_TICKERS)
            ]
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            summary = {
                "version": "market-universe-v1",
                "eligible_ticker_count": len(rows),
                "source_counts": {"NASDAQ": len(rows), "OTHER": 0},
                "sources": {"NASDAQ": "source-a", "OTHER": "source-b"},
                "errors": ["OTHER: Timeout"],
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(AssertionError):
                universe_validator.validate_market_universe(csv_path, summary_path)
            summary["errors"] = []
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with self.assertRaises(AssertionError):
                universe_validator.validate_market_universe(csv_path, summary_path)

            for row in rows[-universe_validator.MINIMUM_SOURCE_TICKERS :]:
                row["source"] = "OTHER"
                row["exchange"] = "NYSE"
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            summary["source_counts"] = {
                "NASDAQ": len(rows) - universe_validator.MINIMUM_SOURCE_TICKERS,
                "OTHER": universe_validator.MINIMUM_SOURCE_TICKERS,
            }
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            result = universe_validator.validate_market_universe(csv_path, summary_path)
            self.assertEqual(result["eligible_ticker_count"], len(rows))

    def test_generated_json_validator_rejects_non_finite_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = Path(temp_dir)
            (docs / "valid.json").write_text('{"status": "ok"}', encoding="utf-8")
            self.assertEqual(
                generated_json_validator.validate_generated_json(docs)["json_file_count"],
                1,
            )
            (docs / "invalid.json").write_text('{"value": NaN}', encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "non-finite JSON number"):
                generated_json_validator.validate_generated_json(docs)

    def test_operational_health_gate_accepts_fresh_and_rejects_stale_data(self):
        fresh_packet = self.compile(report(), portfolio())["decision_packet"]
        health = operational_validator.validate_operational_state(fresh_packet)
        self.assertEqual(health["status"], "HEALTHY")

        stale_packet = self.compile(report(qqq_date="2026-07-07"), portfolio())["decision_packet"]
        with self.assertRaisesRegex(AssertionError, "STALE_MODEL_PORTFOLIO_DATA"):
            operational_validator.validate_operational_state(stale_packet)

    def test_direct_script_entrypoints_keep_repository_import_compatibility(self):
        entrypoints = [
            "scripts/build_decision_report.py",
            "scripts/build_action_board_v3.py",
            "scripts/build_portfolio_backtest.py",
            "scripts/build_trade_review.py",
            "scripts/build_vectorbt_backtest.py",
            "scripts/build_vectorbt_validation.py",
            "scripts/build_walk_forward_report.py",
            "scripts/build_forward_ledger.py",
            "scripts/validate_decision_packet.py",
            "scripts/validate_model_artifacts.py",
            "scripts/validate_operational_health.py",
            "scripts/validate_validation_split.py",
        ]
        code = (
            "import runpy, sys, types\n"
            "sys.modules['vectorbt'] = types.ModuleType('vectorbt')\n"
            f"paths = {entrypoints!r}\n"
            "for index, path in enumerate(paths):\n"
            "    runpy.run_path(path, run_name=f'entrypoint_smoke_{index}')\n"
        )
        result = subprocess.run(
            [sys.executable, "-I", "-c", code],
            cwd=Path.cwd(),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
