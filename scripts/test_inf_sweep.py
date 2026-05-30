"""Integration test for Adim 3.6: post-coercion inf sweep in _coerce_numeric_features."""
import sys
import warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from src.data.preprocessing import CloudPreprocessor

# ---- Build a DataFrame that mimics the exact CIC-IDS2018 failure scenario ----
# The bug: "Infinity" strings pass clean_data(), become real numeric inf
# after pd.to_numeric(), then blow up RobustScaler.
rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame(
    rng.standard_normal((n, 6)).astype("float64"),
    columns=[f"feat_{i}" for i in range(6)],
)

# Inject EXACTLY the patterns seen in CIC-IDS2018 CSVs:
df["feat_1"] = df["feat_1"].astype(object)
df.loc[0,  "feat_1"] = "Infinity"       # capital-I — pd.to_numeric -> +inf
df.loc[1,  "feat_1"] = "-Infinity"      # negative
df.loc[2,  "feat_1"] = "inf"            # lowercase
df.loc[3,  "feat_1"] = "corrupted"     # non-numeric -> NaN

df["feat_3"] = df["feat_3"].astype(object)
df.loc[4,  "feat_3"] = "Infinity"
df.loc[5,  "feat_3"] = "1.2.3.4"       # IP string -> NaN

df["Label"] = (rng.random(n) > 0.5).astype(int)

# ---- Verify "Infinity" -> inf conversion (reproducing the bug) ----
test_series = pd.to_numeric(pd.Series(["Infinity", "-Infinity", "inf"]), errors="coerce")
assert np.isinf(test_series).all(), "pd.to_numeric should produce inf from 'Infinity' strings"
print("[INFO] Confirmed: pd.to_numeric('Infinity') -> numeric inf  (this was the bug)")

# ---- Test _coerce_numeric_features in isolation ----
preproc = CloudPreprocessor(target_col="Label", task_type="binary")
df_coerced = preproc._coerce_numeric_features(df.copy())

feature_cols = [c for c in df_coerced.columns if c != "Label"]

# No inf values should remain
n_inf_remaining = int(np.isinf(df_coerced[feature_cols].values).sum())
assert n_inf_remaining == 0, f"inf values still present after coercion sweep: {n_inf_remaining}"
print(f"[PASS] Post-coercion inf sweep: 0 inf values remain in feature columns")

# No object columns
obj_cols = [c for c in feature_cols if df_coerced[c].dtype == object]
assert not obj_cols, f"Object columns remain: {obj_cols}"
print("[PASS] _coerce_numeric_features: 0 object columns remain")

# ---- Verify RobustScaler no longer blows up ----
preproc2 = CloudPreprocessor(target_col="Label", task_type="binary")
df_train = preproc2.fit_transform(df.copy(), save_sample=False)

feat_train = [c for c in df_train.columns if c != "Label"]
X_np = df_train[feat_train].values

# No NaN
assert not np.isnan(X_np).any(), "NaN values remain after fit_transform"
# No inf
assert not np.isinf(X_np).any(), "inf values remain after fit_transform"
print("[PASS] fit_transform: no NaN and no inf after full pipeline")

# RobustScaler would have raised on inf — torch.tensor confirms safe output
try:
    tensor = torch.tensor(X_np, dtype=torch.float32)
    assert tensor.shape == (len(df_train), len(feat_train))
    print(f"[PASS] torch.tensor() succeeds: shape={tuple(tensor.shape)}")
except Exception as e:
    raise AssertionError(f"torch.tensor failed: {e}")

# ---- transform() path also safe ----
df_test = preproc2.transform(df.copy())
feat_test = [c for c in df_test.columns if c != "Label"]
X_test = df_test[feat_test].values
assert not np.isinf(X_test).any(), "inf in transform() output"
assert not np.isnan(X_test).any(), "NaN in transform() output"
tensor_test = torch.tensor(X_test, dtype=torch.float32)
print(f"[PASS] transform() also safe: shape={tuple(tensor_test.shape)}")

# ---- Summary of the complete guard chain ----
print()
print("Full inf guard chain:")
print("  clean_data           -> replace numeric inf/NaN (misses string 'Infinity')")
print("  _coerce_numeric_features ->")
print("    Pass 1: pd.to_numeric  => 'Infinity' / 'inf' / stray strings -> NaN or inf")
print("    Pass 2: inf sweep      => new numeric inf -> NaN  [Adim 3.6 FIX]")
print("  handle_missing       -> fills ALL NaN with training medians")
print("  scale_features       -> RobustScaler sees clean float data, no crash")
print()
print("=" * 56)
print("ALL TESTS PASSED  -  Adim 3.6 dogrulamasi basarili!")
print("=" * 56)
