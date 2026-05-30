"""CLI: train the IoT vs Cloud traffic router on raw CIC datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset_raw, train_router
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for router training."""
    p = argparse.ArgumentParser(
        description="Train LightGBM/RF router on IoT + Cloud raw CSVs (see configs/router.yaml).",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Max rows per domain (for quick tests). Default: load all.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override save path (default: artifacts/models/router/router.pkl).",
    )
    return p.parse_args()


def main() -> None:
    """Load IoT and Cloud raw data and train the domain router."""
    args = parse_args()
    logger.info("Loading IoT raw data (nrows=%s)...", args.nrows)
    df_iot = load_dataset_raw("iot", nrows=args.nrows)
    logger.info("Loading Cloud raw data (nrows=%s)...", args.nrows)
    df_cloud = load_dataset_raw("cloud", nrows=args.nrows)
    train_router(df_iot, df_cloud, save_path=args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
