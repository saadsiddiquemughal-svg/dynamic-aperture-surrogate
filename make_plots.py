"""
Makes the two plots I actually care about:
  results/accuracy.png  - predicted vs true DA, both models
  results/da_map.png    - DA over the tune diagram, tracker vs surrogate
                          side by side, so you can see where the surrogate
                          gets the resonance structure right and where it
                          smooths things over
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from beamsim.henon import dynamic_aperture

# ---- accuracy scatter ----
d = np.load("results/test_predictions.npz")
metrics = json.load(open("results/metrics.json"))

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
for ax, key, label in [(axes[0], "pred_xgb", "XGBoost"),
                       (axes[1], "pred_mlp", "Neural network (MLP)")]:
    m = metrics["xgboost" if key == "pred_xgb" else "mlp"]
    ax.scatter(d["y_test"], d[key], s=6, alpha=0.4)
    lim = [0, 1.55]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True dynamic aperture (tracking)")
    ax.set_title(f"{label}\n$R^2$ = {m['r2']:.3f},  MAE = {m['mae']:.3f}")
axes[0].set_ylabel("Predicted dynamic aperture")
fig.suptitle("Surrogate accuracy on held-out lattice configurations")
fig.tight_layout()
fig.savefig("results/accuracy.png", dpi=150)
print("wrote results/accuracy.png")

# ---- DA map: tracker vs surrogate ----
N = 70
K2_FIXED = 1.0
q = np.linspace(0.05, 0.45, N)
QX, QY = np.meshgrid(q, q)

print("computing the ground-truth map (this is the slow part)...")
da_true = dynamic_aperture(QX.ravel(), QY.ravel(),
                           np.full(N * N, K2_FIXED),
                           n_turns=1000).reshape(N, N)

mlp = joblib.load("results/model_mlp.joblib")
Xg = np.column_stack([QX.ravel(), QY.ravel(), np.full(N * N, K2_FIXED)])
da_pred = mlp.predict(Xg).reshape(N, N)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
for ax, Z, title in [(axes[0], da_true, "Tracking (ground truth)"),
                     (axes[1], da_pred, "ML surrogate (MLP)")]:
    im = ax.pcolormesh(QX, QY, Z, cmap="viridis", vmin=0, vmax=da_true.max())
    ax.set_xlabel("Horizontal tune $Q_x$")
    ax.set_title(title)
axes[0].set_ylabel("Vertical tune $Q_y$")
fig.colorbar(im, ax=axes, label="Dynamic aperture", shrink=0.9)
fig.suptitle(f"Dynamic aperture across the tune diagram (k2 = {K2_FIXED}). "
             "Dark bands are resonances.")
fig.savefig("results/da_map.png", dpi=150, bbox_inches="tight")
print("wrote results/da_map.png")
