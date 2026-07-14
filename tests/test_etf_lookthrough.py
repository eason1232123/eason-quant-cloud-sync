from __future__ import annotations

import csv
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from scripts.build_etf_lookthrough import (
    CACHE_COLUMNS,
    build_etf_lookthrough,
    validate_etf_lookthrough,
)


QQQ_URL = (
    "https://www.invesco.com/us/en/financial-products/etfs/"
    "invesco-qqq-trust-series-1.html"
)
SMH_URL = "https://www.vaneck.com/us/en/investments/semiconductor-etf-smh/"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_cache(
    path: Path,
    *,
    fund: str,
    as_of_date: str,
    provider: str,
    url: str,
    symbols: list[str],
) -> None:
    weight = 100.0 / len(symbols)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_COLUMNS)
        writer.writeheader()
        for symbol in symbols:
            writer.writerow(
                {
                    "fund": fund,
                    "as_of_date": as_of_date,
                    "symbol": symbol,
                    "name": f"{symbol} Corp",
                    "weight_pct": weight,
                    "source_provider": provider,
                    "source_url": url,
                }
            )


class EtfLookthroughTests(unittest.TestCase):
    def _files(self, root: Path, *, holdings_date: str) -> tuple[Path, Path, Path, Path]:
        qqq_cache = root / "qqq.csv"
        smh_cache = root / "smh.csv"
        policy = root / "policy.json"
        report = root / "report.json"
        portfolio = root / "portfolio.json"
        qqq_symbols = [f"Q{index:03d}" for index in range(80)]
        smh_symbols = qqq_symbols[:10] + [f"S{index:03d}" for index in range(10)]
        _write_cache(
            qqq_cache,
            fund="QQQ",
            as_of_date=holdings_date,
            provider="Invesco",
            url=QQQ_URL,
            symbols=qqq_symbols,
        )
        _write_cache(
            smh_cache,
            fund="SMH",
            as_of_date=holdings_date,
            provider="VanEck",
            url=SMH_URL,
            symbols=smh_symbols,
        )
        _write_json(
            policy,
            {
                "schema_version": "official-etf-lookthrough-policy-v1",
                "maximum_staleness_weekdays": 2,
                "funds": {
                    "QQQ": {
                        "provider": "Invesco",
                        "official_url": QQQ_URL,
                        "cache_file": str(qqq_cache),
                    },
                    "SMH": {
                        "provider": "VanEck",
                        "official_url": SMH_URL,
                        "cache_file": str(smh_cache),
                    },
                },
            },
        )
        _write_json(report, {"data_timestamp": "2026-07-15"})
        _write_json(
            portfolio,
            {
                "assumptions": {
                    "base_weights": {"QQQ": 0.30, "SMH": 0.25, "Q000": 0.10, "CASH": 0.35}
                }
            },
        )
        return policy, report, portfolio, root / "status.json"

    def test_current_official_caches_compute_overlap_without_execution_permission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy, report, portfolio, output = self._files(
                Path(tmp), holdings_date="2026-07-14"
            )
            payload = build_etf_lookthrough(
                policy_path=policy,
                report_path=report,
                portfolio_path=portfolio,
                output_path=output,
                generated_at_utc=datetime(2026, 7, 15, 23, tzinfo=timezone.utc),
            )
            validate_etf_lookthrough(payload)

            self.assertEqual(payload["status"], "AVAILABLE")
            self.assertEqual(
                payload["overlap_analysis"]["qqq_smh_shared_constituent_count"], 10
            )
            self.assertFalse(payload["shadow_evidence_collection_blocked"])
            self.assertFalse(payload["human_pilot_release_gate"])
            self.assertFalse(payload["automatic_order_allowed"])

    def test_stale_official_cache_is_unavailable_but_does_not_block_shadow_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            policy, report, portfolio, output = self._files(
                Path(tmp), holdings_date="2026-07-01"
            )
            payload = build_etf_lookthrough(
                policy_path=policy,
                report_path=report,
                portfolio_path=portfolio,
                output_path=output,
                generated_at_utc=datetime(2026, 7, 15, 23, tzinfo=timezone.utc),
            )

            self.assertEqual(payload["status"], "UNAVAILABLE")
            self.assertIsNone(payload["overlap_analysis"])
            self.assertEqual(payload["funds"]["QQQ"]["reason"], "OFFICIAL_HOLDINGS_STALE")
            self.assertFalse(payload["shadow_evidence_collection_blocked"])
            self.assertFalse(payload["human_pilot_release_gate"])


if __name__ == "__main__":
    unittest.main()
