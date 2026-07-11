from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCS = ROOT / "docs"


def reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key is not allowed: {key}")
        result[key] = value
    return result


def load_strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_non_finite,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise AssertionError(f"invalid generated JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise AssertionError(f"generated JSON must contain a top-level object: {path}")
    return payload


def validate_generated_json(docs_dir: Path = DEFAULT_DOCS) -> dict[str, Any]:
    paths = sorted(docs_dir.glob("*.json"))
    if not paths:
        raise AssertionError(f"no generated JSON files found in {docs_dir}")
    for path in paths:
        load_strict_json(path)
    return {
        "status": "VALID",
        "json_file_count": len(paths),
        "docs_dir": str(docs_dir),
    }


if __name__ == "__main__":
    print(validate_generated_json())
