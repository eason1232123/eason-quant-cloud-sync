from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from scripts import validate_model_artifacts as validator


class ModelArtifactValidationTests(unittest.TestCase):
    def test_current_artifacts_share_one_frozen_model_lineage(self) -> None:
        result = validator.validate_model_artifacts()

        self.assertEqual(result["status"], "VALID")
        self.assertEqual(len(result["strategy_fingerprint"]), 64)
        self.assertEqual(len(result["full_model_fingerprint"]), 64)

    def test_fingerprint_drift_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            docs = Path(temp_dir)
            for name in (
                set(validator.METADATA_REPORTS)
                | set(validator.METADATA_CSV_REPORTS)
                | set(validator.LEDGER_REPORTS)
            ):
                shutil.copy2(validator.DOCS / name, docs / name)
            path = docs / "vectorbt_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            report["strategy_fingerprint"] = "0" * 64
            path.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")

            with self.assertRaisesRegex(AssertionError, "strategy contract mismatch"):
                validator.validate_model_artifacts(docs=docs)


if __name__ == "__main__":
    unittest.main()
