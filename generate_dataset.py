"""
Generate the training set: sample a bunch of (Qx, Qy, k2) combos and get
the ground-truth DA for each from the tracker.

Writes data/da_dataset.csv (columns: qx, qy, k2, da) and data/timing.txt
with the average time per DA evaluation - used later as the baseline
for the speedup comparison.
"""

import time
import numpy as np
import pandas as pd
from beamsim.henon import dynamic_aperture

RNG = np.random.default_rng(42)  # seeded so this is reproducible

N_SAMPLES = 4000
N_TURNS = 1000

# tunes: stay inside (0, 0.5), that's where the resonance structure lives.
# k2: weak enough to be near-linear at one end, strong enough to be nasty
# at the other.
qx = RNG.uniform(0.05, 0.45, N_SAMPLES)
qy = RNG.uniform(0.05, 0.45, N_SAMPLES)
k2 = RNG.uniform(0.2, 2.0, N_SAMPLES)

print(f"tracking {N_SAMPLES} configs, {N_TURNS} turns each...")
t0 = time.perf_counter()
da = dynamic_aperture(qx, qy, k2, n_turns=N_TURNS)
elapsed = time.perf_counter() - t0
per_eval = elapsed / N_SAMPLES

df = pd.DataFrame({"qx": qx, "qy": qy, "k2": k2, "da": da})
df.to_csv("data/da_dataset.csv", index=False)

with open("data/timing.txt", "w") as f:
    f.write(f"total_seconds={elapsed:.2f}\n")
    f.write(f"seconds_per_da_evaluation={per_eval:.4f}\n")

print(f"done in {elapsed:.1f}s -> {per_eval*1000:.1f} ms/config (batched)")
print(df.describe())
