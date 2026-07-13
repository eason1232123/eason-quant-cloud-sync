from __future__ import annotations

import argparse
import json

from scripts.ibkr_readonly import (
    ROOT,
    IbkrReadonlyError,
    capture_private_snapshot,
    config_from_env,
    load_private_snapshot,
    probe_endpoint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a private local read-only IBKR account snapshot"
    )
    parser.add_argument("--probe", action="store_true", help="check only the local socket")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate the existing private snapshot without connecting",
    )
    parser.add_argument("--max-age-seconds", type=float, default=300.0)
    args = parser.parse_args()
    try:
        config = config_from_env()
        if args.probe:
            result = probe_endpoint(config)
            print(json.dumps(result, sort_keys=True, allow_nan=False))
            return 0 if result["reachable"] else 2
        if args.validate_existing:
            snapshot = load_private_snapshot(
                config.private_snapshot_path,
                max_age_seconds=args.max_age_seconds,
            )
        else:
            snapshot = capture_private_snapshot(config)
        # Exact account IDs, balances, positions, and values are deliberately
        # never printed. ChatGPT reads the gitignored private file locally.
        print(
            json.dumps(
                {
                    "status": "PRIVATE_IBKR_SNAPSHOT_READY",
                    "collected_at_utc": snapshot["collected_at_utc"],
                    "account_count": len(snapshot["managed_accounts"]),
                    "position_record_count": len(snapshot["positions"]),
                    "portfolio_record_count": len(snapshot["portfolio"]),
                    "private_path": str(config.private_snapshot_path.relative_to(ROOT)),
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
                    "status": "PRIVATE_IBKR_SNAPSHOT_UNAVAILABLE",
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
