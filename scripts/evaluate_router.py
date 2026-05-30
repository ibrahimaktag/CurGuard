"""CLI: evaluate saved router on held-out IoT vs Cloud test splits."""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_dataset_raw
from src.data.router import load_router, route
from src.training.splits import split_train_val_test_df
from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _suppress_noisy_loggers() -> None:
    """Reduce per-file loader spam when not using --verbose."""
    import logging

    # Loader emits hundreds of WARNING lines (null counts per file); hide unless --verbose.
    logging.getLogger("src.data.loader").setLevel(logging.ERROR)
    for name in ("src.data.router", "src.training.splits"):
        logging.getLogger(name).setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Evaluate saved router on test portions of IoT and Cloud raw data.",
    )
    p.add_argument(
        "--nrows",
        type=int,
        default=None,
        help="Her bir .csv dosyası için satır üst sınırı (tüm dosyalar birleştirilir; 50k × çok dosya = milyonlarca satır). Örn: 5000",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Loader/router INFO loglarını göster (varsayılan: sadece özet).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: datasets.iot.random_state).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for router_metrics.json (default: artifacts/evaluation).",
    )
    return p.parse_args()


def _router_domain_correct(true_is_iot: bool, pred: dict) -> bool:
    """Return True if prediction matches the true domain (strict; 'both' is incorrect)."""
    dom = pred["domain"]
    if true_is_iot:
        return dom == "iot"
    return dom == "cloud"


def main() -> None:
    """Load data, split test sets, run saved router, save accuracy and timing."""
    args = parse_args()
    # LightGBM/sklearn: noisy FutureWarning about force_all_finite (sklearn 1.6+)
    warnings.filterwarnings(
        "ignore",
        message=".*force_all_finite.*",
        category=FutureWarning,
    )
    if not args.verbose:
        _suppress_noisy_loggers()

    ds = load_config("datasets")
    rs = args.seed if args.seed is not None else ds["iot"].get("random_state", 42)
    test_r = float(ds["iot"].get("test_split", 0.2))
    val_r = float(ds["iot"].get("val_split", 0.1))

    if args.nrows is None:
        print(
            "[evaluate_router] UYARI: --nrows yok; her CSV tam okunuyor (çok uzun sürebilir).",
            flush=True,
        )
    else:
        print(
            f"[evaluate_router] Not: --nrows={args.nrows} → her .csv için ayrı limit; "
            "toplam satır = dosya sayısı × (en fazla nrows).",
            flush=True,
        )
    print("[evaluate_router] IoT verisi yükleniyor...", flush=True)
    logger.info("Loading IoT + Cloud raw (nrows=%s)...", args.nrows)
    df_iot = load_dataset_raw("iot", nrows=args.nrows)
    print(f"[evaluate_router] IoT: {len(df_iot)} satır. Cloud yükleniyor...", flush=True)
    df_cloud = load_dataset_raw("cloud", nrows=args.nrows)
    print(f"[evaluate_router] Cloud: {len(df_cloud)} satır. Train/val/test ayrılıyor...", flush=True)

    t_iot = ds["iot"]["target_column"]
    t_cloud = ds["cloud"]["target_column"]

    _tr, _va, df_iot_test = split_train_val_test_df(
        df_iot,
        target_column=t_iot,
        test_size=test_r,
        val_size=val_r,
        random_state=rs,
    )
    _tr2, _va2, df_cloud_test = split_train_val_test_df(
        df_cloud,
        target_column=t_cloud,
        test_size=test_r,
        val_size=val_r,
        random_state=rs,
    )

    print(
        f"[evaluate_router] Test: IoT n={len(df_iot_test)}, Cloud n={len(df_cloud_test)}. Router çalışıyor...",
        flush=True,
    )
    model, _ = load_router()

    t0 = time.perf_counter()
    res_iot = route(df_iot_test, model=model)
    res_cloud = route(df_cloud_test, model=model)
    elapsed = time.perf_counter() - t0

    n_iot = len(res_iot)
    n_cloud = len(res_cloud)
    correct_iot = sum(_router_domain_correct(True, r) for r in res_iot)
    correct_cloud = sum(_router_domain_correct(False, r) for r in res_cloud)
    total = n_iot + n_cloud
    acc = (correct_iot + correct_cloud) / total if total else 0.0

    both_iot = sum(1 for r in res_iot if r["domain"] == "both")
    both_cloud = sum(1 for r in res_cloud if r["domain"] == "both")

    summary = {
        "task": "router_iot_vs_cloud",
        "n_test_iot": n_iot,
        "n_test_cloud": n_cloud,
        "accuracy_strict": float(acc),
        "correct_iot": int(correct_iot),
        "correct_cloud": int(correct_cloud),
        "fallback_both_count_iot": int(both_iot),
        "fallback_both_count_cloud": int(both_cloud),
        "inference_seconds_total": float(elapsed),
        "inference_ms_per_sample": float(1000.0 * elapsed / total) if total else 0.0,
        "note": "Eğitim tüm veriyle yapıldıysa bu test örnekleri eğitimde görülmüş olabilir; "
        "gerçek genelleme için router'ı yalnızca train split üzerinde eğitin.",
    }

    out_dir = args.output_dir or (get_project_root() / "artifacts" / "evaluation")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "router_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved %s", path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
