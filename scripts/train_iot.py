"""CLI: train the IoT specialist on CIC-IoT2023 (preprocess → CNN+LSTM+Attention)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.iot_specialist import IoTSpecialist
from src.data import load_dataset_raw
from src.data.iot_preprocessor import IoTPreprocessor
from src.training.splits import split_train_val_test_df
from src.training.trainer import train_specialist
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for IoT specialist training."""
    p = argparse.ArgumentParser(description="Train IoT traffic specialist (configs/iot_model.yaml).")
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Max rows per CSV file (debug). Default: load all.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override epochs for quick tests (default: configs/training.yaml).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for quick tests (default: configs/training.yaml).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: datasets.iot.random_state).",
    )
    return p.parse_args()


def main() -> None:
    """Load IoT data, split, fit preprocessor, train specialist, save model."""
    args = parse_args()
    ds_cfg = load_config("datasets")["iot"]
    rs = args.seed if args.seed is not None else ds_cfg.get("random_state", 42)

    logger.info("Loading IoT raw data (nrows=%s)...", args.nrows)
    df = load_dataset_raw("iot", nrows=args.nrows)

    target = ds_cfg["target_column"]
    test_r = float(ds_cfg.get("test_split", 0.2))
    val_r = float(ds_cfg.get("val_split", 0.1))

    df_train, df_val, _df_test = split_train_val_test_df(
        df,
        target_column=target,
        test_size=test_r,
        val_size=val_r,
        random_state=rs,
    )

    prep = IoTPreprocessor()
    prep.fit(df_train)
    X_train, y_train = prep.transform(df_train, include_target=True)
    X_val, y_val = prep.transform(df_val, include_target=True)

    specialist = IoTSpecialist()
    model = specialist.build(
        input_shape=(X_train.shape[1],),
        n_classes=prep.n_classes,
    )

    train_specialist(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        model_name="iot_specialist",
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    specialist.save()
    prep_path = specialist.get_checkpoint_path() / "preprocessor.joblib"
    joblib.dump(prep, prep_path)
    logger.info("Preprocessor saved to %s", prep_path)
    logger.info("Classes: %s", prep.class_names)
    logger.info("IoT specialist training finished.")


if __name__ == "__main__":
    main()
