"""
Phase 4 End-to-End Runtime Verification Script.
Executes: Train -> GA -> GP -> Attach -> Save -> Reload -> Predict consistency.
"""
import time
import json
import numpy as np

print("=" * 65)
print("PHASE 4 — HYBRID PIPELINE RUNTIME VERIFICATION")
print("=" * 65)

# ── 1. Imports ────────────────────────────────────────────────────────────────
from models.deep_neuro_fuzzy_net import DeepNeuroFuzzyRiskModel
from models.train import (
    build_dataset, train_model, save_weights, load_weights,
    save_gp_expression, load_gp_expression,
    TRAINING_UNIVERSE, WEIGHTS_PATH, GP_EXPRESSION_PATH,
)
from models.genetic_optimizer import optimize_model
from models.genetic_programming import GP_CONFIG, run_gp, attach_gp_to_model
from models.hybrid_pipeline import load_full_model, HybridResult
from config.settings import GA_CONFIG

# ── 2. Build Dataset ──────────────────────────────────────────────────────────
print("\n[1/7] Building offline dataset …")
t0 = time.time()
dataset = build_dataset(
    tickers=["OFFLINE_A", "OFFLINE_B", "OFFLINE_C", "OFFLINE_D"],
    allow_network=False,
)
n = len(dataset["X_vec"])
print(f"      {n} samples  |  X_vec {dataset['X_vec'].shape}  |  elapsed {time.time()-t0:.1f}s")

# ── 3. Train ──────────────────────────────────────────────────────────────────
print("\n[2/7] Training BiLSTM + Neuro-Fuzzy model (5 epochs) …")
model = DeepNeuroFuzzyRiskModel()
t0 = time.time()
history = train_model(model, dataset, epochs=5, batch_size=64)
print(f"      train_mae={history['train_mae'][-1]:.3f}  "
      f"val_mae={history['val_mae'][-1]:.3f}  "
      f"val_acc={history['val_acc'][-1]:.1f}%  "
      f"elapsed {time.time()-t0:.1f}s")

# ── 4. Baseline prediction BEFORE evolution ──────────────────────────────────
x_probe = np.array([0.50, 0.20, 0.25, 0.20], dtype=np.float32)
score_before_ga, cat_before, mdd_before, _ = model.predict_risk(x_probe)
print(f"\n      Baseline (pre-evolution) score = {score_before_ga:.4f}  "
      f"cat={cat_before}  mdd={mdd_before:.4f}")
print(f"      GP expression on model        : {model.gp_expression_string}")

# ── 5. GA Optimisation ────────────────────────────────────────────────────────
print("\n[3/7] Running GA (pop=15, gen=10) …")
t0 = time.time()
ga_result = optimize_model(
    model, dataset,
    population_size=15, generations=10, seed=42,
)
meta = ga_result["metadata"]
chrom = ga_result["chromosome"]
print(f"      baseline_fitness={meta['baseline_fitness']:.6f}  "
      f"best_fitness={meta['best_fitness']:.6f}  "
      f"improved={meta['improved']}  "
      f"chromosome_len={len(chrom)}  "
      f"elapsed {time.time()-t0:.1f}s")

score_after_ga, _, _, _ = model.predict_risk(x_probe)
print(f"      Score after GA: {score_after_ga:.4f}  "
      f"(Δ={score_after_ga - score_before_ga:+.4f})")

# ── 6. GP Evolution ───────────────────────────────────────────────────────────
print("\n[4/7] Running GP (pop=40, gen=20) …")
gp_cfg = dict(GP_CONFIG)
gp_cfg.update({"population_size": 40, "generations": 20, "max_depth": 5, "max_nodes": 63})

t0 = time.time()
gp_result = run_gp(dataset, config=gp_cfg, seed=42)
print(f"      best_fitness={gp_result['fitness']:.6f}  "
      f"val_MSE={gp_result['val_mse']:.3f}  "
      f"val_R2={gp_result['val_r2']:.4f}  "
      f"nodes={gp_result['node_count']}  "
      f"elapsed {time.time()-t0:.1f}s")
print(f"      Expression: {gp_result['expression'][:120]}")

# ── 7. Attach GP to Model ─────────────────────────────────────────────────────
print("\n[5/7] Attaching GP expression to model …")
attach_gp_to_model(model, gp_result)
print(f"      model.gp_expression_string = {model.gp_expression_string[:100]}")

score_after_gp, cat_gp, mdd_gp, _ = model.predict_risk(x_probe)
print(f"      Score after GP attach: {score_after_gp:.4f}  "
      f"(Δ from pre-GA={score_after_gp - score_before_ga:+.4f})")
assert model.gp_expression_string != "(none)", "GP expression not attached!"

# ── 8. Save ───────────────────────────────────────────────────────────────────
print("\n[6/7] Saving model weights + GP expression …")
save_weights(model)
save_gp_expression(model)
print(f"      Weights  → {WEIGHTS_PATH}")
print(f"      GP tree  → {GP_EXPRESSION_PATH}")

# ── 9. Reload and verify consistency ─────────────────────────────────────────
print("\n[7/7] Reloading from disk in a fresh model instance …")
model2 = DeepNeuroFuzzyRiskModel()
weights_ok = load_weights(model2)
gp_ok      = load_gp_expression(model2)
print(f"      weights_loaded={weights_ok}  gp_loaded={gp_ok}")
print(f"      GP expr on reload: {model2.gp_expression_string[:100]}")

score_reload, cat_reload, mdd_reload, _ = model2.predict_risk(x_probe)
print(f"      Score after reload : {score_reload:.4f}")
print(f"      Δ original vs reload: {abs(score_after_gp - score_reload):.6f}")

TOLERANCE = 1e-4
assert abs(score_after_gp - score_reload) < TOLERANCE, (
    f"Prediction inconsistency after reload: {score_after_gp} vs {score_reload}"
)
print(f"      ✅ Reload consistency PASSED (tol={TOLERANCE})")

# ── 10. Tensor dimension check ────────────────────────────────────────────────
print("\n[BONUS] Tensor dimension verification …")
import torch
model2.eval()
B = 4
with torch.no_grad():
    vx = torch.randn(B, 4)
    sx = torch.randn(B, 20, 4)
    rs, cat_t, mdd_t, info = model2(vx, sx)
    gp_sig = info["gp_signal"]

print(f"      rs.shape={tuple(rs.shape)}  cat.shape={tuple(cat_t.shape)}  "
      f"mdd.shape={tuple(mdd_t.shape)}")
print(f"      gp_signal.shape={tuple(gp_sig.shape)}")
print(f"      gp_signal is non-zero: {bool((gp_sig != 0).any().item())}")
assert tuple(rs.shape)      == (B, 1), f"Bad risk shape: {rs.shape}"
assert tuple(gp_sig.shape)  == (B, 1), f"Bad GP signal shape: {gp_sig.shape}"
print("      ✅ Tensor dimensions PASSED")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ALL PHASE 4 CHECKS PASSED")
print(f"  Dataset    : {n} synthetic samples")
print(f"  Training   : 5 epochs, val_mae={history['val_mae'][-1]:.3f}")
print(f"  GA         : improved={meta['improved']}  Δfitness={meta['best_fitness']-meta['baseline_fitness']:+.6f}")
print(f"  GP         : nodes={gp_result['node_count']}  val_R2={gp_result['val_r2']:.4f}")
print(f"  Score path : {score_before_ga:.4f} → {score_after_ga:.4f} (GA) → {score_after_gp:.4f} (GP)")
print(f"  Reload Δ   : {abs(score_after_gp - score_reload):.2e}")
print("=" * 65)
