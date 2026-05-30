"""Shared helpers for professional EDA notebooks (exploration only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.feature_selection import mutual_info_classif

RANDOM_STATE = 42
TOP_CORR_FEATURES = 15
HIGH_CORR_THRESHOLD = 0.95
TOP_MI_FEATURES = 20
TOP_OUTLIER_FEATURES = 12

_NON_FEATURE_COLS = {
    "Source IP",
    "Src IP",
    "Destination IP",
    "Dst IP",
    "Timestamp",
    "Flow ID",
    "Unnamed: 0",
    "id",
    "ID",
    "index",
    "source_file",
}


def find_project_root(start_path: Path) -> Path:
    """Find repository root via configs/datasets.yaml."""
    for candidate in [start_path, *start_path.parents]:
        if (candidate / "configs" / "datasets.yaml").exists():
            return candidate
    raise FileNotFoundError("Project root bulunamadi.")


def load_eda_config(root: Path) -> dict[str, Any]:
    """Load configs/eda_samples.yaml."""
    path = root / "configs" / "eda_samples.yaml"
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def master_sample_paths(root: Path) -> tuple[Path, Path, Path, Path]:
    """Return cloud/iot parquet paths and metadata JSON paths."""
    out_dir = root / load_eda_config(root)["output_dir"]
    return (
        out_dir / "cloud_master.parquet",
        out_dir / "cloud_master_meta.json",
        out_dir / "iot_master.parquet",
        out_dir / "iot_master_meta.json",
    )


def load_master_sample(root: Path, domain: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load EDA master parquet and metadata for cloud or iot."""
    cloud_pq, cloud_meta, iot_pq, iot_meta = master_sample_paths(root)
    if domain == "cloud":
        parquet_path, meta_path = cloud_pq, cloud_meta
    elif domain == "iot":
        parquet_path, meta_path = iot_pq, iot_meta
    else:
        raise ValueError("domain must be 'cloud' or 'iot'")

    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Master sample yok: {parquet_path}. "
            "Once: python scripts/build_eda_master_samples.py"
        )

    df = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return df, meta


def sanitize_labels(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Drop rows with corrupted label values (e.g. literal column name)."""
    if label_col not in df.columns:
        return df
    labels = df[label_col].astype(str).str.strip()
    bad = labels.str.lower() == label_col.lower()
    if bad.any():
        print(f"Uyari: {int(bad.sum())} satir bozuk etiket ({label_col}) atildi.")
        return df.loc[~bad].copy()
    return df


def get_numeric_feature_columns(df: pd.DataFrame, label_col: str) -> list[str]:
    """Return numeric model-feature columns (exclude IDs and label)."""
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    drop = _NON_FEATURE_COLS | {label_col}
    return [column for column in numeric if column not in drop]


def prepare_numeric_matrix(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Numeric feature matrix safe for sklearn (no inf/NaN)."""
    cols = get_numeric_feature_columns(df, label_col)
    if not cols:
        return pd.DataFrame()

    matrix = df[cols].replace([np.inf, -np.inf], np.nan)
    matrix = matrix.fillna(matrix.median(numeric_only=True))
    matrix = matrix.fillna(0)
    return matrix


def basic_clean_report(df: pd.DataFrame, label_col: str) -> dict[str, Any]:
    """Summarize missing values, infinities, duplicates."""
    work = df.copy()
    work.columns = work.columns.str.strip()
    work = work.replace([np.inf, -np.inf], np.nan)

    missing = work.isna().sum().sort_values(ascending=False)
    missing_top = missing[missing > 0].head(15)

    dup_rows = int(work.duplicated().sum())
    dup_cols = int(work.columns.duplicated().sum())

    return {
        "shape": work.shape,
        "missing_columns_with_nulls": int((missing > 0).sum()),
        "missing_top": missing_top,
        "duplicate_rows": dup_rows,
        "duplicate_column_names": dup_cols,
        "label_column": label_col,
        "n_classes": int(work[label_col].nunique()) if label_col in work.columns else 0,
    }


def imbalance_metrics(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Per-class counts and share for imbalance assessment."""
    counts = df[label_col].astype(str).value_counts()
    total = len(df)
    report = pd.DataFrame(
        {
            "count": counts,
            "share_pct": (counts / total * 100).round(3),
        }
    )
    report["imbalance_ratio_to_max"] = (report["count"].max() / report["count"]).round(2)
    return report.sort_values("count", ascending=False)


def constant_and_low_variance_features(
    df: pd.DataFrame,
    label_col: str,
    low_variance_threshold: float = 1e-8,
) -> tuple[list[str], list[str]]:
    """List constant and near-constant numeric feature columns."""
    feature_cols = get_numeric_feature_columns(df, label_col)
    nunique = df[feature_cols].nunique()
    constant = nunique[nunique <= 1].index.tolist()
    variances = df[feature_cols].var(numeric_only=True)
    low_var = variances[variances <= low_variance_threshold].index.tolist()
    return constant, low_var


def high_correlation_pairs(
    df: pd.DataFrame,
    label_col: str,
    top_features: int = TOP_CORR_FEATURES,
    threshold: float = HIGH_CORR_THRESHOLD,
) -> pd.DataFrame:
    """Find highly correlated numeric feature pairs."""
    matrix = prepare_numeric_matrix(df, label_col)
    if matrix.shape[1] < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "correlation"])

    selected = matrix.var().sort_values(ascending=False).head(top_features).index
    corr = matrix[selected].corr().abs()
    pairs: list[dict[str, Any]] = []
    cols = list(corr.columns)
    for i, col_a in enumerate(cols):
        for col_b in cols[i + 1 :]:
            value = float(corr.loc[col_a, col_b])
            if value >= threshold:
                pairs.append(
                    {"feature_a": col_a, "feature_b": col_b, "correlation": round(value, 4)}
                )
    return pd.DataFrame(pairs).sort_values("correlation", ascending=False)


def calculate_outlier_ratios(df: pd.DataFrame, label_col: str) -> pd.Series:
    """IQR-based outlier ratio per numeric feature."""
    numeric_df = prepare_numeric_matrix(df, label_col)
    if numeric_df.empty:
        return pd.Series(dtype=float)
    q1 = numeric_df.quantile(0.25)
    q3 = numeric_df.quantile(0.75)
    iqr = q3 - q1
    valid_cols = iqr[iqr > 0].index
    if len(valid_cols) == 0:
        return pd.Series(dtype=float)

    lower = q1[valid_cols] - 1.5 * iqr[valid_cols]
    upper = q3[valid_cols] + 1.5 * iqr[valid_cols]
    mask = (numeric_df[valid_cols] < lower) | (numeric_df[valid_cols] > upper)
    return mask.mean().sort_values(ascending=False)


def build_binary_target(df: pd.DataFrame, label_col: str, benign_aliases: set[str]) -> pd.Series:
    """Map labels to binary benign=0 attack=1."""
    labels = df[label_col].astype(str).str.strip().str.lower()
    
    is_benign = pd.Series(False, index=df.index)
    for alias in benign_aliases:
        is_benign = is_benign | labels.str.startswith(alias)
        
    return (~is_benign).astype(int)


def mutual_information_scores(
    df: pd.DataFrame,
    binary_target: pd.Series,
    label_col: str,
) -> pd.Series:
    """Mutual information vs binary target (EDA snapshot, not final XAI)."""
    numeric_df = prepare_numeric_matrix(df, label_col)
    if numeric_df.empty:
        return pd.Series(dtype=float)

    scores = mutual_info_classif(
        numeric_df,
        binary_target,
        random_state=RANDOM_STATE,
        discrete_features=False,
    )
    return pd.Series(scores, index=numeric_df.columns).sort_values(ascending=False)


def per_source_file_coverage(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Rows and distinct labels contributed by each source CSV."""
    if "source_file" not in df.columns:
        return pd.DataFrame()

    rows = df.groupby("source_file").size().rename("rows")
    classes = df.groupby("source_file")[label_col].nunique().rename("n_labels")
    return pd.concat([rows, classes], axis=1).sort_values("rows", ascending=False)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


def plot_class_distribution(
    df: pd.DataFrame,
    label_col: str,
    title: str,
    max_classes: int = 20,
) -> None:
    """Horizontal count plot for top classes."""
    counts = df[label_col].astype(str).value_counts().head(max_classes)
    plot_df = pd.DataFrame({"label": counts.index, "count": counts.values})
    plt.figure(figsize=(11, max(5, 0.35 * len(plot_df))))
    sns.barplot(data=plot_df, y="label", x="count", hue="label", palette="viridis", legend=False)
    plt.title(title)
    plt.xlabel("Ornek Sayisi")
    plt.tight_layout()
    plt.show()


def plot_binary_balance(binary_target: pd.Series, title: str, labels: tuple[str, str]) -> None:
    """Binary class balance bar chart."""
    mapped = binary_target.map({0: labels[0], 1: labels[1]})
    plot_df = pd.DataFrame({"class": mapped.value_counts().index, "count": mapped.value_counts().values})
    plt.figure(figsize=(7, 4))
    sns.barplot(data=plot_df, x="class", y="count", hue="class", palette="Set2", legend=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, label_col: str, title: str) -> None:
    """Heatmap for top-variance numeric features."""
    matrix = prepare_numeric_matrix(df, label_col)
    if matrix.shape[1] < 2:
        print("Uyari: Korelasyon icin yeterli sayisal feature yok.")
        return

    top_features = matrix.var().sort_values(ascending=False).head(TOP_CORR_FEATURES).index
    plt.figure(figsize=(12, 9))
    sns.heatmap(matrix[top_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_top_outliers(
    df: pd.DataFrame,
    outlier_ratios: pd.Series,
    title: str,
    label_col: str = "Label",
) -> None:
    """Boxplots for top outlier-heavy features."""
    if outlier_ratios.empty:
        print("Uyari: Outlier analizi icin uygun feature yok.")
        return

    top_cols = outlier_ratios.head(4).index.tolist()
    matrix = prepare_numeric_matrix(df, label_col)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes_flat = axes.flatten()
    for idx, col in enumerate(top_cols):
        if col not in matrix.columns:
            continue
        sns.boxplot(x=matrix[col], ax=axes_flat[idx], color="skyblue")
        ratio = outlier_ratios[col] * 100
        axes_flat[idx].set_title(f"{col} (outlier: %{ratio:.2f})")
    for idx in range(len(top_cols), 4):
        axes_flat[idx].axis("off")
    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_mi_importance(scores: pd.Series, title: str) -> None:
    """Horizontal bar chart of top MI features."""
    if scores.empty:
        print("Uyari: MI skorlari hesaplanamadi.")
        return

    top_scores = scores.head(TOP_MI_FEATURES).sort_values(ascending=True)
    plt.figure(figsize=(10, 7))
    plt.barh(top_scores.index, top_scores.values, color="teal")
    plt.title(title)
    plt.xlabel("Mutual Information")
    plt.tight_layout()
    plt.show()


def plot_source_file_coverage(coverage: pd.DataFrame, title: str) -> None:
    """Bar chart of rows per source file."""
    if coverage.empty:
        return
    top = coverage.head(15)
    plt.figure(figsize=(11, 5))
    sns.barplot(
        data=top.reset_index(),
        x="source_file",
        y="rows",
        hue="source_file",
        palette="crest",
        legend=False,
    )
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def run_cloud_professional_eda(root: Path) -> None:
    """Execute full cloud EDA pipeline on master sample."""
    sns.set(style="whitegrid")
    label_col = "Label"
    benign_aliases = {"benign", "normal"}

    df, meta = load_master_sample(root, "cloud")
    df = sanitize_labels(df, label_col)
    print_section("CIC-IDS2018 Profesyonel EDA (Master Sample)")
    print(f"Satir: {len(df):,} | Sutun: {df.shape[1]} | Metadata: {meta.get('rows', 'n/a')}")

    clean = basic_clean_report(df, label_col)
    print_section("1) Veri Kalitesi")
    print(f"Boyut: {clean['shape']}")
    print(f"Eksik iceren sutun: {clean['missing_columns_with_nulls']}")
    print(f"Duplicate satir: {clean['duplicate_rows']}")
    if not clean["missing_top"].empty:
        print(clean["missing_top"])

    print_section("2) Sinif Dengesi ve Imbalance")
    imbalance = imbalance_metrics(df, label_col)
    print(imbalance.head(20))
    print(f"\nEn buyuk / en kucuk sinif orani: {imbalance['count'].max() / imbalance['count'].min():.1f}x")

    print_section("3) Kaynak Dosya Kapsami")
    coverage = per_source_file_coverage(df, label_col)
    print(coverage)

    print_section("4) Sabit / Dusuk Varyansli Ozellikler")
    constant, low_var = constant_and_low_variance_features(df, label_col)
    print(f"Sabit sutun sayisi: {len(constant)}")
    if constant:
        print(constant[:20])
    print(f"Dusuk varyansli sutun sayisi: {len(low_var)}")

    print_section("5) Yuksek Korelasyonlu Ciftler")
    corr_pairs = high_correlation_pairs(df, label_col)
    print(corr_pairs.head(15) if not corr_pairs.empty else "Esik ustu cift yok.")

    binary = build_binary_target(df, label_col, benign_aliases)
    print_section("6) Outlier (IQR)")
    outliers = calculate_outlier_ratios(df, label_col)
    print(outliers.head(TOP_OUTLIER_FEATURES))

    print_section("7) Binary Ayrim Gucu (MI — imbalance uyarısı ile)")
    mi_scores = mutual_information_scores(df, binary, label_col)
    print(mi_scores.head(TOP_MI_FEATURES))

    plot_class_distribution(df, label_col, "CIC-IDS2018 — Sinif Dagilimi (Master)")
    plot_binary_balance(binary, "CIC-IDS2018 — Binary Dengesi", ("Benign", "Attack"))
    plot_source_file_coverage(coverage, "Kaynak CSV Basina Ornek Sayisi")
    plot_correlation_heatmap(df, label_col, "Yuksek Varyansli Ozellikler — Korelasyon")
    plot_top_outliers(df, outliers, "Outlier Orani En Yuksek Ozellikler", label_col)
    plot_mi_importance(mi_scores, "Binary Hedef icin MI (Ilk Bakis)")


def run_iot_professional_eda(root: Path) -> None:
    """Execute full IoT EDA pipeline on master sample."""
    sns.set(style="whitegrid")
    label_col = "label"
    benign_aliases = {"benigntraffic", "normal"}

    df, meta = load_master_sample(root, "iot")
    df = sanitize_labels(df, label_col)
    print_section("CIC-IoT2023 Profesyonel EDA (Master Sample)")
    print(f"Satir: {len(df):,} | Sinif: {df[label_col].nunique()} | Dosya: {meta.get('source_files', 'n/a')}")

    clean = basic_clean_report(df, label_col)
    print_section("1) Veri Kalitesi")
    print(f"Boyut: {clean['shape']}")
    print(f"Duplicate satir: {clean['duplicate_rows']}")

    print_section("2) Sinif Dengesi (33 sinif — capped stratified sample)")
    imbalance = imbalance_metrics(df, label_col)
    print(imbalance)
    print(
        f"\nNot: IoT dogal olarak cok dengesiz; EDA'da capped sampling kullanildi. "
        f"Oran (max/min): {imbalance['count'].max() / imbalance['count'].min():.1f}x"
    )

    print_section("3) Kaynak Dosya Kapsami")
    coverage = per_source_file_coverage(df, label_col)
    print(coverage.head(20))

    constant, low_var = constant_and_low_variance_features(df, label_col)
    print_section("4) Sabit / Dusuk Varyansli Ozellikler")
    print(f"Sabit: {len(constant)} | Dusuk varyans: {len(low_var)}")

    corr_pairs = high_correlation_pairs(df, label_col)
    print_section("5) Yuksek Korelasyonlu Ciftler")
    print(corr_pairs.head(15) if not corr_pairs.empty else "Esik ustu cift yok.")

    binary = build_binary_target(df, label_col, benign_aliases)
    outliers = calculate_outlier_ratios(df, label_col)
    mi_scores = mutual_information_scores(df, binary, label_col)

    print_section("6) Outlier & MI")
    print(outliers.head(TOP_OUTLIER_FEATURES))
    print(mi_scores.head(TOP_MI_FEATURES))

    plot_class_distribution(df, label_col, "CIC-IoT2023 — Sinif Dagilimi (Master)", max_classes=33)
    plot_binary_balance(binary, "CIC-IoT2023 — Normal vs Attack", ("Normal", "Attack"))
    plot_correlation_heatmap(df, label_col, "IoT Ozellik Korelasyonu")
    plot_top_outliers(df, outliers, "IoT — Outlier Yogun Ozellikler", label_col)
    plot_mi_importance(mi_scores, "IoT Binary Hedef — MI (Ilk Bakis)")
