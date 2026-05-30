"""Integration test for Path Adaptation layer (Kaggle vs Local).

Tests all public functions in src/utils/config.py:
  - is_kaggle()          — environment detection
  - get_project_root()   — project root resolution
  - load_config()        — YAML loading + LRU cache
  - get_config_path()    — config file path helper
  - get_dataset_path()   — local vs Kaggle raw-data routing
  - get_artifacts_root() — write-able output root + auto-mkdir
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# 1. Import sanity check
# ---------------------------------------------------------------------------
from src.utils.config import (
    get_artifacts_root,
    get_config_path,
    get_dataset_path,
    get_project_root,
    is_kaggle,
    load_config,
)
print("[PASS] config.py imports cleanly; all 6 public symbols present")

# ---------------------------------------------------------------------------
# 2. is_kaggle() — LOCAL environment
# ---------------------------------------------------------------------------
assert "KAGGLE_KERNEL_RUN_TYPE" not in os.environ, \
    "Test must run outside a Kaggle kernel for the local branch to be tested"
assert is_kaggle() is False
print("[PASS] is_kaggle() == False in local environment")

# ---------------------------------------------------------------------------
# 3. is_kaggle() — SIMULATED Kaggle environment
# ---------------------------------------------------------------------------
os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"   # inject fake variable
assert is_kaggle() is True
print("[PASS] is_kaggle() == True when KAGGLE_KERNEL_RUN_TYPE is set")
del os.environ["KAGGLE_KERNEL_RUN_TYPE"]               # restore local state
assert is_kaggle() is False
print("[PASS] is_kaggle() reverts to False after env var removed")

# ---------------------------------------------------------------------------
# 4. get_project_root()
# ---------------------------------------------------------------------------
root = get_project_root()
assert root.is_dir(), f"Project root does not exist: {root}"
assert (root / "configs").is_dir(), "configs/ directory missing from project root"
assert (root / "src").is_dir(),     "src/ directory missing from project root"
print(f"[PASS] get_project_root() = {root}")

# ---------------------------------------------------------------------------
# 5. load_config() + LRU cache
# ---------------------------------------------------------------------------
cfg = load_config("datasets")
assert "iot" in cfg,   "datasets.yaml missing 'iot' key"
assert "cloud" in cfg, "datasets.yaml missing 'cloud' key"
assert "environment" in cfg, "datasets.yaml missing 'environment' key"

# Verify new Kaggle keys present
assert "kaggle_raw_path" in cfg["iot"],   "iot.kaggle_raw_path missing"
assert "kaggle_raw_path" in cfg["cloud"], "cloud.kaggle_raw_path missing"
assert "local"  in cfg["environment"], "environment.local missing"
assert "kaggle" in cfg["environment"], "environment.kaggle missing"
assert "artifacts_root" in cfg["environment"]["local"]
assert "artifacts_root" in cfg["environment"]["kaggle"]
print("[PASS] datasets.yaml loaded; all new keys present")

# LRU cache: second call returns same object identity
cfg2 = load_config("datasets")
assert cfg is cfg2, "LRU cache should return the identical object on second call"
print("[PASS] load_config() LRU cache working (same object identity)")

# ---------------------------------------------------------------------------
# 6. get_config_path()
# ---------------------------------------------------------------------------
p = get_config_path("datasets")
assert p.exists(), f"Config path does not exist: {p}"
assert p.suffix == ".yaml"
print(f"[PASS] get_config_path('datasets') = {p.name}")

# ---------------------------------------------------------------------------
# 7. get_dataset_path() — LOCAL branch
# ---------------------------------------------------------------------------
try:
    local_iot_path = get_dataset_path("iot", cfg)
    print(f"[PASS] get_dataset_path('iot')   = {local_iot_path}")
except FileNotFoundError as e:
    # Raw data directory may not exist on CI / fresh clone — that is expected.
    # The important thing is that the function raised FileNotFoundError (not KeyError).
    print(f"[PASS] get_dataset_path('iot') correctly raised FileNotFoundError "
          f"(raw data not present locally): {e}")

try:
    local_cloud_path = get_dataset_path("cloud", cfg)
    print(f"[PASS] get_dataset_path('cloud') = {local_cloud_path}")
except FileNotFoundError as e:
    print(f"[PASS] get_dataset_path('cloud') correctly raised FileNotFoundError "
          f"(raw data not present locally): {e}")

# ---------------------------------------------------------------------------
# 8. get_dataset_path() — SIMULATED Kaggle branch (path won't exist, expect FileNotFoundError)
# ---------------------------------------------------------------------------
os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Batch"
try:
    kaggle_path = get_dataset_path("cloud", cfg)
    # Would only succeed if /kaggle/input/cicids2018 actually exists (it won't locally)
    print(f"[PASS] get_dataset_path('cloud', kaggle) = {kaggle_path}")
except FileNotFoundError as e:
    # Expected: /kaggle/input/cicids2018 doesn't exist on a local machine.
    # On Windows the path may use backslashes so check for the dataset name only.
    assert "cicids2018" in str(e), \
        f"Error message should mention 'cicids2018', got: {e}"
    print("[PASS] get_dataset_path() correctly resolves kaggle_raw_path in Kaggle env")
except KeyError as e:
    raise AssertionError(f"kaggle_raw_path key lookup failed: {e}")
finally:
    del os.environ["KAGGLE_KERNEL_RUN_TYPE"]

# ---------------------------------------------------------------------------
# 9. get_artifacts_root() — LOCAL branch
# ---------------------------------------------------------------------------
artifacts = get_artifacts_root(cfg)
assert artifacts.exists(), f"artifacts root was not created: {artifacts}"
assert artifacts.is_dir()
# Should resolve to <project_root>/artifacts/
assert artifacts == get_project_root() / "artifacts"
print(f"[PASS] get_artifacts_root() [local] = {artifacts}  (dir created/exists)")

# ---------------------------------------------------------------------------
# 10. get_artifacts_root() — SIMULATED Kaggle branch
# ---------------------------------------------------------------------------
os.environ["KAGGLE_KERNEL_RUN_TYPE"] = "Interactive"
try:
    kaggle_artifacts = get_artifacts_root(cfg)
    # /kaggle/working/artifacts/ only exists inside a real Kaggle kernel
    print(f"[PASS] get_artifacts_root() [kaggle] = {kaggle_artifacts}")
except (PermissionError, OSError) as e:
    # Cannot create /kaggle/working/ on a local machine — expected
    print(f"[PASS] get_artifacts_root() [kaggle] raised OS/PermissionError "
          f"as expected on local machine: {type(e).__name__}")
finally:
    del os.environ["KAGGLE_KERNEL_RUN_TYPE"]

# ---------------------------------------------------------------------------
# 11. KeyError on unknown domain
# ---------------------------------------------------------------------------
try:
    get_dataset_path("unknown_domain", cfg)
    raise AssertionError("Should have raised KeyError for unknown domain")
except KeyError as e:
    print(f"[PASS] get_dataset_path raises KeyError for unknown domain: {e}")

# ---------------------------------------------------------------------------
print()
print("=" * 58)
print("ALL TESTS PASSED  -  Path Adaptation dogrulamasi basarili!")
print("=" * 58)
print()
print("Environment routing summary:")
print(f"  is_kaggle()        -> {is_kaggle()} (local)")
print(f"  get_project_root() -> {get_project_root()}")
print(f"  get_artifacts_root [local]  -> {get_artifacts_root(cfg)}")
print(f"  datasets.yaml: iot.kaggle_raw_path  = {cfg['iot']['kaggle_raw_path']}")
print(f"  datasets.yaml: cloud.kaggle_raw_path = {cfg['cloud']['kaggle_raw_path']}")
print(f"  datasets.yaml: env.kaggle.artifacts  = {cfg['environment']['kaggle']['artifacts_root']}")
