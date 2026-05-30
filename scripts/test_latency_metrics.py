"""Integration test for Adim 3: metrics.py latency functions."""

import sys
sys.path.insert(0, ".")

import numpy as np
import torch

import src.evaluation.metrics as M

# ---- Syntax / import check ----
print("[PASS] metrics.py imports cleanly")

# ---- Verify all expected symbols ----
expected = [
    "compute_classification_metrics",
    "measure_inference_latency",
    "measure_inference_latency_torch",
    "measure_inference_latency_auto",
    "_compute_latency_stats",
    "get_top_k_features_by_importance",
]
for fn_name in expected:
    assert hasattr(M, fn_name), f"Missing symbol: {fn_name}"
print("[PASS] All 6 symbols exported from metrics module")

# ---- measure_inference_latency_torch (CPU path) ----
from src.models.cloud_specialist import CloudSpecialist

model_cloud = CloudSpecialist(input_dim=76, num_classes=2)
model_cloud.eval()

results = M.measure_inference_latency_torch(
    model=model_cloud,
    input_dim=76,
    batch_sizes=(1, 32),
    n_warmup=3,
    n_runs=10,
)

assert set(results.keys()) == {1, 32}, f"Expected batch keys {{1,32}}, got {set(results.keys())}"
for bs, stats in results.items():
    for key in ["mean_ms", "std_ms", "min_ms", "max_ms", "p95_ms"]:
        assert key in stats, f"Missing key '{key}' in batch={bs} stats"
    assert stats["mean_ms"] > 0, "Latency must be positive"
    print(f"  batch={bs:2d}:  mean={stats['mean_ms']:.3f} ms   p95={stats['p95_ms']:.3f} ms")

print("[PASS] measure_inference_latency_torch (CPU, CloudSpecialist, batch=1 and 32)")

# ---- State restoration test ----
model_cloud.train()
assert model_cloud.training, "Pre-condition: model should be in training mode"

_ = M.measure_inference_latency_torch(
    model_cloud, input_dim=76, batch_sizes=(1,), n_warmup=1, n_runs=2
)
assert model_cloud.training, "Model must remain in training mode after latency measurement"
print("[PASS] model.training state correctly restored to True after measurement")

# ---- IoTSpecialist path ----
from src.models.iot_specialist import IoTSpecialist

model_iot = IoTSpecialist(input_dim=46, num_classes=2)
results_iot = M.measure_inference_latency_torch(
    model=model_iot,
    input_dim=46,
    batch_sizes=(1, 32),
    n_warmup=3,
    n_runs=10,
)
assert set(results_iot.keys()) == {1, 32}
print("[PASS] measure_inference_latency_torch (CPU, IoTSpecialist)")

# ---- Dispatcher: PyTorch branch ----
result_auto = M.measure_inference_latency_auto(
    model=model_cloud,
    input_dim_or_X=76,
    batch_sizes=(1,),
    n_warmup=1,
    n_runs=3,
)
assert isinstance(result_auto, dict) and 1 in result_auto
print("[PASS] measure_inference_latency_auto correctly dispatched to torch path")

# ---- Dispatcher: TypeError guard (wrong input type for PyTorch model) ----
try:
    M.measure_inference_latency_auto(model_cloud, input_dim_or_X=np.zeros((10, 76)))
    assert False, "Should have raised TypeError"
except TypeError as exc:
    print(f"[PASS] TypeError guard (PyTorch + ndarray input): {exc}")

# ---- _compute_latency_stats arithmetic ----
stats = M._compute_latency_stats([1.0, 2.0, 3.0, 4.0, 5.0])
assert abs(stats["mean_ms"] - 3.0) < 1e-9, f"mean wrong: {stats['mean_ms']}"
assert abs(stats["min_ms"]  - 1.0) < 1e-9, f"min wrong:  {stats['min_ms']}"
assert abs(stats["max_ms"]  - 5.0) < 1e-9, f"max wrong:  {stats['max_ms']}"
print("[PASS] _compute_latency_stats arithmetic correct")

# ---- Existing Keras function untouched (import + signature check) ----
import inspect
sig = inspect.signature(M.measure_inference_latency)
params = list(sig.parameters.keys())
assert params == ["model", "X", "n_warmup", "n_runs"], f"Keras fn signature changed: {params}"
print("[PASS] measure_inference_latency (Keras) signature unchanged")

print()
print("=" * 55)
print("ALL TESTS PASSED  -  Adim 3 dogrulamasi basarili!")
print("=" * 55)
