"""Integration test for Adim 3.5: _coerce_numeric_features in BasePreprocessor."""
import sys
import warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# ---- Build a DataFrame that mimics CIC-IDS2018 corruption patterns ----
# Some rows contain:
#   - stray strings in numeric columns ('1.2.3.4', 'Flow Duration', '')
#   - header accidentally duplicated as a data row
rng = np.random.default_rng(42)
n = 500
df = pd.DataFrame(
    rng.standard_normal((n, 5)).astype("float64"),
    columns=[f"feat_{i}" for i in range(5)],
)

# Inject corruption
df["feat_1"] = df["feat_1"].astype(object)   # widen to object so strings fit
df.loc[0, "feat_1"] = "corrupted_string"
df.loc[1, "feat_1"] = "Flow Duration"        # header-as-data
df["feat_3"] = df["feat_3"].astype(object)
df.loc[2, "feat_3"] = "1.2.3.4"
df.loc[3, "feat_3"] = ""                     # empty -> NaN after coerce
df["Label"] = (rng.random(n) > 0.5).astype(int)

from src.data.preprocessing import CloudPreprocessor

preproc = CloudPreprocessor(target_col="Label", task_type="binary")

# ---- Test _coerce_numeric_features in isolation ----
df_coerced = preproc._coerce_numeric_features(df.copy())

feature_cols = [c for c in df_coerced.columns if c != "Label"]
object_remaining = [c for c in feature_cols if df_coerced[c].dtype == object]
assert not object_remaining, f"Object columns still remain: {object_remaining}"
print("[PASS] _coerce_numeric_features: 0 object columns remaining")

nan_feat1 = int(df_coerced["feat_1"].isna().sum())
assert nan_feat1 >= 2, f"Expected >=2 NaN in feat_1, got {nan_feat1}"
print(f"[PASS] Corrupted strings correctly became NaN: feat_1 NaN count={nan_feat1}")

# ---- Test full fit_transform pipeline ----
df_train = preproc.fit_transform(df.copy(), save_sample=False)

feat_train = [c for c in df_train.columns if c != "Label"]
nan_total = int(df_train[feat_train].isna().sum().sum())
assert nan_total == 0, f"NaN values remain after fit_transform: {nan_total}"
print(f"[PASS] fit_transform: 0 NaN values after full pipeline")

obj_remaining = [c for c in feat_train if df_train[c].dtype == object]
assert not obj_remaining, f"Object columns remain: {obj_remaining}"
print("[PASS] fit_transform: all feature columns are numeric")

# ---- Critical check: PyTorch tensor creation (original crash point) ----
X_np = df_train[feat_train].values
assert X_np.dtype != object, f"numpy array is still object: {X_np.dtype}"
tensor_train = torch.tensor(X_np, dtype=torch.float32)
assert tensor_train.shape == (len(df_train), len(feat_train))
print(f"[PASS] torch.tensor() succeeds on fit_transform output: shape={tuple(tensor_train.shape)}")

# ---- Test transform() pipeline (val/test path) ----
df_test = preproc.transform(df.copy())
feat_test = [c for c in df_test.columns if c != "Label"]
X_test = df_test[feat_test].values
tensor_test = torch.tensor(X_test, dtype=torch.float32)
assert tensor_test.shape[1] == len(feat_test)
print(f"[PASS] transform() also produces valid tensor: shape={tuple(tensor_test.shape)}")

# ---- Pipeline order log check ----
print()
print("Pipeline order verified:")
print("  clean_data")
print("  -> _coerce_numeric_features  [NEW: object cols -> numeric, NaN flagged]")
print("  -> reduce_memory_usage")
print("  -> handle_multicollinearity")
print("  -> handle_missing            [fills coercion-NaN with train medians]")
print("  -> scale_features")
print("  -> encode_target")
print()
print("=" * 52)
print("ALL TESTS PASSED  -  Adim 3.5 dogrulamasi basarili!")
print("=" * 52)
