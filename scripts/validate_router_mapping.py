"""CLI: verify router.yaml feature_mapping against raw CSV headers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.router_mapping_validate import (
    RouterMappingValidationResult,
    validate_router_mapping_from_disk,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Check router feature_mapping vs first IoT/Cloud CSV headers in data/raw.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 if any mapped column is missing.",
    )
    return p.parse_args()


def _print_report(r: RouterMappingValidationResult) -> None:
    """Print human-readable validation output."""
    print("Router feature_mapping validation")
    print("-" * 40)
    if r.iot_sample:
        print(f"IoT sample:    {r.iot_sample} ({r.iot_columns} columns)")
    else:
        print("IoT sample:    (no CSV found)")
    if r.cloud_sample:
        print(f"Cloud sample: {r.cloud_sample} ({r.cloud_columns} columns)")
    else:
        print("Cloud sample: (no CSV found)")

    if r.missing_iot:
        print("\nMissing in IoT CSV:")
        for line in r.missing_iot:
            print(f"  - {line}")
    else:
        print("\nIoT:   all mapped columns present.")

    if r.missing_cloud:
        print("\nMissing in Cloud CSV:")
        for line in r.missing_cloud:
            print(f"  - {line}")
    else:
        print("Cloud: all mapped columns present.")

    print("-" * 40)
    print("Status:", "OK" if r.ok else "FAILED")


def main() -> None:
    """Run validation and optionally exit non-zero."""
    args = parse_args()
    try:
        result = validate_router_mapping_from_disk()
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(2)

    _print_report(result)

    if args.strict and not result.ok:
        sys.exit(1)
    if result.iot_sample is None or result.cloud_sample is None:
        sys.exit(2)


if __name__ == "__main__":
    main()
