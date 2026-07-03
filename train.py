"""
Trains two models to predict DA straight from (qx, qy, k2), no tracking
needed at inference time. Also benchmarks how much faster that actually
is compared to calling dynamic_aperture() directly.
"""

import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

from beamsim.henon import dynamic_aperture

df = pd.read_csv("data/da_dataset.csv")
X = df[["qx", "qy", "k2"]].values
y = df["da"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

models = {
    "xgboost": XGBRegressor(
        n_estimators=600, max_depth=8, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=0,
    ),
    # scaling matters a lot here since qx/qy (0.05-0.45) and k2 (0.2-2.0)
    # are on pretty different scales
    "mlp": make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=2000,
                     early_stopping=True, random_state=0),
    ),
}

metrics = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics[name] = {
        "r2": float(r2_score(y_test, pred)),
        "mae": float(mean_absolute_error(y_test, pred)),
    }
    joblib.dump(model, f"results/model_{name}.joblib")
    print(f"{name:8s}  R^2 = {metrics[name]['r2']:.3f}   "
          f"MAE = {metrics[name]['mae']:.4f}")

# ---- speed comparison ----
# baseline: one config at a time through the actual tracker, like you'd
# do inside an optimisation loop
n_bench = 20
t0 = time.perf_counter()
for i in range(n_bench):
    dynamic_aperture(X_test[i, 0], X_test[i, 1], X_test[i, 2], n_turns=1000)
t_sim = (time.perf_counter() - t0) / n_bench

# surrogate: batched predict, which is how you'd actually use it
best = models["xgboost"]
t0 = time.perf_counter()
for _ in range(50):
    best.predict(X_test)
t_ml = (time.perf_counter() - t0) / (50 * len(X_test))

speedup = t_sim / t_ml
metrics["speedup"] = {
    "tracking_seconds_per_config": t_sim,
    "surrogate_seconds_per_config": t_ml,
    "speedup_factor": float(speedup),
}
print(f"\ntracking:  {t_sim*1e3:8.2f} ms / config")
print(f"surrogate: {t_ml*1e6:8.2f} us / config")
print(f"speedup:   {speedup:,.0f}x")

with open("results/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# save test predictions so make_plots.py doesn't need to reload models
np.savez("results/test_predictions.npz",
         y_test=y_test,
         pred_xgb=models["xgboost"].predict(X_test),
         pred_mlp=models["mlp"].predict(X_test))
