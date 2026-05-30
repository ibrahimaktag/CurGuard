"""CLI: train VAE + Isolation Forest anomaly detector on normal traffic only."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset_raw
from src.data.cloud_preprocessor import CloudPreprocessor
from src.data.iot_preprocessor import IoTPreprocessor
from src.models.anomaly_detector import AnomalyDetector
from src.training.splits import split_train_val_test_df
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for anomaly detector training."""
    p = argparse.ArgumentParser(
        description="Train VAE+IF anomaly model on NORMAL samples only (see anomaly_detector.yaml).",
    )
    p.add_argument(
        "--domain",
        choices=("iot", "cloud"),
        default="iot",
        help="Which dataset defines feature space and normal class (default: iot).",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Max rows per CSV file (debug). Default: load all.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: datasets.<domain>.random_state).",
    )
    p.add_argument(
        "--normal-class",
        type=str,
        default=None,
        help='Label string for normal traffic (default: first entry in datasets attack_classes, e.g. "Normal" / "Benign").',
    )
    return p.parse_args()


def main() -> None:
    """Load data, split, fit preprocessor on train, fit anomaly detector, save artifacts."""
    args = parse_args()
    ds_cfg = load_config("datasets")[args.domain]
    rs = args.seed if args.seed is not None else ds_cfg.get("random_state", 42)

    logger.info("Loading %s raw data (nrows=%s)...", args.domain, args.nrows)
    df = load_dataset_raw(args.domain, nrows=args.nrows)

    target = ds_cfg["target_column"]
    test_r = float(ds_cfg.get("test_split", 0.2))
    val_r = float(ds_cfg.get("val_split", 0.1))

    df_train, _df_val, _df_test = split_train_val_test_df(
        df,
        target_column=target,
        test_size=test_r,
        val_size=val_r,
        random_state=rs,
    )

    classes = list(ds_cfg.get("attack_classes", []))
    normal_name = args.normal_class or (classes[0] if classes else "Normal")
    logger.info("Normal class label: %s", normal_name)

    prep: IoTPreprocessor | CloudPreprocessor
    if args.domain == "iot":
        prep = IoTPreprocessor()
    else:
        prep = CloudPreprocessor()

    prep.fit(df_train)
    X_train, y_train = prep.transform(df_train, include_target=True)

    names = prep.class_names
    if str(normal_name) not in names:
        raise ValueError(f"normal-class '{normal_name}' not in label set {names}")
    normal_idx = names.index(str(normal_name))

    ens = load_config("ensemble")
    name = ens.get("anomaly_models", {}).get(args.domain, f"anomaly_detector_{args.domain}")
    out_dir = get_project_root() / "artifacts" / "models" / name

    detector = AnomalyDetector()
    detector.fit(X_train, y_train, normal_label=normal_idx)
    detector.save(out_dir)
    logger.info("Anomaly detector saved to %s (input_dim=%s).", out_dir, X_train.shape[1])


if __name__ == "__main__":
    main()
