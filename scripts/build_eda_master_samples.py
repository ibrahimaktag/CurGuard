"""Build stratified EDA master samples for cloud and IoT datasets.

Outputs Parquet + JSON metadata under data/eda_samples/.
Run from project root: python scripts/build_eda_master_samples.py [--domain all|cloud|iot]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.seed import get_seed


def find_project_root(start: Path) -> Path:
    """Locate repo root via configs/datasets.yaml."""
    for candidate in [start, *start.parents]:
        if (candidate / "configs" / "datasets.yaml").exists():
            return candidate
    raise FileNotFoundError("Project root not found.")


def load_eda_config(root: Path) -> dict[str, Any]:
    """Load EDA sample YAML config."""
    path = root / "configs" / "eda_samples.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _cap_class_quotas(
    counts: dict[str, int],
    n_target: int,
    min_per_class: int,
    max_per_class: int,
) -> dict[str, int]:
    """Compute per-class row quotas for stratified sampling."""
    total = sum(counts.values())
    if total == 0:
        return {}

    n_take = min(n_target, total)
    raw = {label: max(min_per_class, int(n_take * count / total)) for label, count in counts.items()}
    for label, count in counts.items():
        raw[label] = min(raw[label], max_per_class, count)

    excess = sum(raw.values()) - n_take
    if excess <= 0:
        return raw

    sorted_labels = sorted(raw.keys(), key=lambda label: raw[label], reverse=True)
    idx = 0
    while excess > 0 and sorted_labels:
        label = sorted_labels[idx % len(sorted_labels)]
        if raw[label] > min_per_class:
            raw[label] -= 1
            excess -= 1
        idx += 1
        if idx > len(sorted_labels) * 1000:
            break
    return raw


def _collect_stratified_chunks(
    path: Path,
    label_col: str,
    quotas: dict[str, int],
    chunk_size: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample rows per class from a CSV using chunked reads."""
    collected: dict[str, list[pd.DataFrame]] = {label: [] for label in quotas}
    filled = {label: 0 for label in quotas}

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if label_col not in chunk.columns:
            raise KeyError(f"'{label_col}' missing in {path.name}")

        for label, quota in quotas.items():
            if filled[label] >= quota:
                continue
            subset = chunk[chunk[label_col] == label]
            if subset.empty:
                continue
            need = quota - filled[label]
            take = subset.sample(n=min(len(subset), need), random_state=random_state)
            collected[label].append(take)
            filled[label] += len(take)

        if all(filled[label] >= quotas[label] for label in quotas):
            break

    parts = [pd.concat(frames, ignore_index=True) for frames in collected.values() if frames]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["source_file"] = path.name
    return out


def _count_labels_in_csv(path: Path, label_col: str, chunk_size: int) -> dict[str, int]:
    """Count label frequencies in a CSV without loading it fully."""
    counts: dict[str, int] = defaultdict(int)
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        chunk.columns = chunk.columns.str.strip()
        if label_col not in chunk.columns:
            raise KeyError(f"'{label_col}' missing in {path.name}")
        for label, count in chunk[label_col].value_counts().items():
            counts[str(label)] += int(count)
    return dict(counts)


def sample_cloud_file(
    path: Path,
    label_col: str,
    n_target: int,
    min_per_class: int,
    max_per_class: int,
    chunk_size: int,
    large_file_mb: int,
    random_state: int,
) -> pd.DataFrame:
    """Stratified sample from one CIC-IDS2018 CSV."""
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb >= large_file_mb:
        counts = _count_labels_in_csv(path, label_col, chunk_size)
        quotas = _cap_class_quotas(counts, n_target, min_per_class, max_per_class)
        return _collect_stratified_chunks(
            path, label_col, quotas, chunk_size, random_state
        )

    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    if label_col not in df.columns:
        raise KeyError(f"'{label_col}' missing in {path.name}")

    counts = df[label_col].astype(str).value_counts().to_dict()
    quotas = _cap_class_quotas(counts, min(n_target, len(df)), min_per_class, max_per_class)
    parts: list[pd.DataFrame] = []
    for label, quota in quotas.items():
        subset = df[df[label_col] == label]
        if subset.empty:
            continue
        parts.append(subset.sample(n=min(len(subset), quota), random_state=random_state))

    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["source_file"] = path.name
    return out


def _iot_label_from_path(path: Path) -> str:
    """Derive IoT label from PCAP-style CSV filename."""
    return path.stem.replace(".pcap", "")


def sample_iot_file(
    path: Path,
    label_col: str,
    max_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Sample one IoT CSV and attach filename-derived label."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    label_value = _iot_label_from_path(path)

    lower_map = {column.lower(): column for column in df.columns}
    if label_col.lower() not in {column.lower() for column in df.columns}:
        df[label_col] = label_value
    else:
        actual = lower_map[label_col.lower()]
        if df[actual].nunique(dropna=True) <= 1:
            df[actual] = label_value

    n_take = min(max_rows, len(df))
    sampled = df.sample(n=n_take, random_state=random_state)
    sampled["source_file"] = path.name
    return sampled


def downsample_stratified(
    df: pd.DataFrame,
    label_col: str,
    max_rows: int,
    min_per_class: int,
    max_per_class: int,
    random_state: int,
) -> pd.DataFrame:
    """Downsample combined frame to max_rows with class caps."""
    if len(df) <= max_rows:
        return df

    counts = df[label_col].astype(str).value_counts().to_dict()
    quotas = _cap_class_quotas(counts, max_rows, min_per_class, max_per_class)
    parts: list[pd.DataFrame] = []
    for label, quota in quotas.items():
        subset = df[df[label_col].astype(str) == label]
        if subset.empty:
            continue
        parts.append(subset.sample(n=min(len(subset), quota), random_state=random_state))
    return pd.concat(parts, ignore_index=True)


def build_cloud_master(root: Path, cfg: dict[str, Any], random_state: int) -> pd.DataFrame:
    """Build cloud EDA master sample across all daily CSV files."""
    cloud_cfg = cfg["cloud"]
    raw_dir = root / "data" / "raw" / cloud_cfg["raw_subdir"]
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No cloud CSV files in {raw_dir}")

    label_col = cloud_cfg["label_column"]
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        print(f"  cloud: sampling {path.name} ...")
        part = sample_cloud_file(
            path=path,
            label_col=label_col,
            n_target=cloud_cfg["max_rows_per_file"],
            min_per_class=cloud_cfg["min_rows_per_class"],
            max_per_class=cloud_cfg["max_rows_per_class"],
            chunk_size=cloud_cfg["chunk_size"],
            large_file_mb=cloud_cfg["large_file_mb"],
            random_state=random_state,
        )
        if not part.empty:
            frames.append(part)

    combined = pd.concat(frames, ignore_index=True)
    return downsample_stratified(
        combined,
        label_col=label_col,
        max_rows=cloud_cfg["max_total_rows"],
        min_per_class=cloud_cfg["min_rows_per_class"],
        max_per_class=cloud_cfg["max_rows_per_class"],
        random_state=random_state,
    )


def build_iot_master(root: Path, cfg: dict[str, Any], random_state: int) -> pd.DataFrame:
    """Build IoT EDA master sample across PCAP-derived CSV files."""
    iot_cfg = cfg["iot"]
    raw_dir = root / "data" / "raw" / iot_cfg["raw_subdir"]
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No IoT CSV files in {raw_dir}")

    label_col = iot_cfg["label_column"]
    frames: list[pd.DataFrame] = []
    for path in csv_files:
        part = sample_iot_file(
            path=path,
            label_col=label_col,
            max_rows=iot_cfg["max_rows_per_file"],
            random_state=random_state,
        )
        frames.append(part)

    combined = pd.concat(frames, ignore_index=True)
    return downsample_stratified(
        combined,
        label_col=label_col,
        max_rows=iot_cfg["max_total_rows"],
        min_per_class=iot_cfg["min_rows_per_class"],
        max_per_class=iot_cfg["max_rows_per_class"],
        random_state=random_state,
    )


def _coerce_for_parquet(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Coerce object columns to numeric where possible (CIC-IDS2018 quirk)."""
    out = df.copy()
    skip = {label_col, "source_file"}
    for col in out.columns:
        if col in skip:
            continue
        if out[col].dtype == object:
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().mean() >= 0.5:
                out[col] = converted
    return out


def write_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    meta: dict[str, Any],
    label_col: str,
) -> None:
    """Persist parquet and metadata JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _coerce_for_parquet(df, label_col)
    parquet_path = out_dir / f"{name}_master.parquet"
    meta_path = out_dir / f"{name}_master_meta.json"
    df.to_parquet(parquet_path, index=False)
    meta["rows"] = int(len(df))
    meta["columns"] = int(df.shape[1])
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {parquet_path} ({len(df):,} rows)")
    print(f"Wrote {meta_path}")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Build EDA master samples.")
    parser.add_argument(
        "--domain",
        choices=["all", "cloud", "iot"],
        default="all",
        help="Which master sample to build.",
    )
    args = parser.parse_args()

    root = find_project_root(Path.cwd())
    cfg = load_eda_config(root)
    random_state = int(cfg.get("random_state", get_seed()))
    out_dir = root / cfg["output_dir"]

    if args.domain in ("all", "cloud"):
        print("Building cloud master sample...")
        cloud_df = build_cloud_master(root, cfg, random_state)
        label_col = cfg["cloud"]["label_column"]
        meta = {
            "domain": "cloud",
            "label_column": label_col,
            "class_counts": cloud_df[label_col].astype(str).value_counts().to_dict(),
            "source_files": cloud_df["source_file"].value_counts().to_dict(),
            "random_state": random_state,
        }
        write_outputs(cloud_df, out_dir, "cloud", meta, label_col)

    if args.domain in ("all", "iot"):
        print("Building IoT master sample...")
        iot_df = build_iot_master(root, cfg, random_state)
        label_col = cfg["iot"]["label_column"]
        meta = {
            "domain": "iot",
            "label_column": label_col,
            "class_counts": iot_df[label_col].astype(str).value_counts().to_dict(),
            "source_files": int(iot_df["source_file"].nunique()),
            "random_state": random_state,
        }
        write_outputs(iot_df, out_dir, "iot", meta, label_col)


if __name__ == "__main__":
    main()
