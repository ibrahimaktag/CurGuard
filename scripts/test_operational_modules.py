"""Quick unit test for the 3 operational modules in run_kaggle_full.py."""
import sys
sys.path.insert(0, ".")

import numpy as np

from scripts.run_kaggle_full import (
    _synthetic_anomaly_scores,
    _synthetic_routing,
    analyze_anomaly_override,
    analyze_router_cascade,
    compute_cost_sensitive_matrix,
)

rng = np.random.default_rng(42)
n = 5_000
y_true = (rng.random(n) > 0.4).astype(int)
y_pred = y_true.copy()
# Inject realistic noise: ~5% FN, ~3% FP
fn_idx = rng.choice(np.where(y_true == 1)[0], size=int(n * 0.05), replace=False)
fp_idx = rng.choice(np.where(y_true == 0)[0], size=int(n * 0.03), replace=False)
y_pred[fn_idx] = 0
y_pred[fp_idx] = 1

y_proba = np.column_stack([1 - y_pred.astype(float), y_pred.astype(float)])
y_proba += rng.normal(0, 0.05, y_proba.shape)
y_proba = np.clip(y_proba, 0.01, 0.99)
y_proba /= y_proba.sum(axis=1, keepdims=True)

# ---- 1. Router Cascade ----
routing = _synthetic_routing(n, "cloud")
r = analyze_router_cascade(routing, y_true, y_pred, "cloud")
assert "misrouting_rate_pct" in r
assert 0 <= r["misrouting_rate_pct"] <= 15
misrate = r["misrouting_rate_pct"]
f1delta = r["f1_delta_vs_all"]
print(f"[PASS] Router Cascade: misrouting={misrate}%  f1_delta={f1delta}")

# ---- 2. Anomaly Override ----
anomaly_scores = _synthetic_anomaly_scores(y_proba)
o = analyze_anomaly_override(y_true, y_pred, anomaly_scores, override_threshold=0.8)
assert "f1_lift" in o
assert "recall_lift" in o
assert o["n_overrides"] >= 0
n_ov = o["n_overrides"]
f1lift = o["f1_lift"]
rlift = o["recall_lift"]
print(f"[PASS] Anomaly Override: overrides={n_ov}  f1_lift={f1lift}  recall_lift={rlift}")

# ---- 3. Cost-Sensitive Threat Matrix ----
y_pred_ov = y_pred.copy()
y_pred_ov[(y_pred == 0) & (anomaly_scores >= 0.8)] = 1
c = compute_cost_sensitive_matrix(y_true, y_pred_ov, cost_fp=1.0, cost_fn=10.0)
assert c["cost_ratio_fn_vs_fp"] == 10.0
assert c["savings_pct"] > 0, "Model should save cost vs naive baseline"
fp_count = c["confusion_matrix"]["FP"]
fn_count = c["confusion_matrix"]["FN"]
total_cost = c["total_operational_cost"]
savings_pct = c["savings_pct"]
print(f"[PASS] Cost Matrix: FP={fp_count}  FN={fn_count}")
print(f"       total_cost={total_cost}  savings={savings_pct}%")

print()
print("=" * 52)
print("ALL 3 OPERATIONAL MODULES PASSED")
print("=" * 52)
