#!/usr/bin/env python3
"""
Example: Using DP-RSS on general bounded data (Appendix A of the paper).

When data lies in [x_min, x_max] × [y_min, y_max] instead of [0,1]²,
simply pass the bounds via x_bounds and y_bounds. DP-RSS normalises
internally and returns parameters on the original scale.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dp_rss import dp_rss, l2_error_exact

# ── True model: y = 2.5x + 10  with x ∈ [20, 80], y ∈ [0, 250] ─────────────

TRUE_ALPHA = 2.5
TRUE_BETA  = 10.0
N          = 5000
SIGMA      = 5.0
X_MIN, X_MAX = 20.0, 80.0

np.random.seed(42)

# Generate data on the original scale
x = np.random.uniform(X_MIN, X_MAX, size=N)
y = TRUE_ALPHA * x + TRUE_BETA + np.random.normal(0, SIGMA, size=N)

# Compute observed y bounds (known or domain-specific)
Y_MIN, Y_MAX = 0.0, 250.0
y = np.clip(y, Y_MIN, Y_MAX)

print(f"True model:  y = {TRUE_ALPHA}x + {TRUE_BETA}")
print(f"Data range:  x ∈ [{X_MIN}, {X_MAX}],  y ∈ [{Y_MIN}, {Y_MAX}]")
print(f"Dataset size: {N},  noise σ = {SIGMA}")
print()

# ── Run DP-RSS with general bounds ───────────────────────────────────────────

for eps in [0.5, 1.0, 2.0, 5.0]:
    np.random.seed(0)
    result = dp_rss(
        x, y,
        epsilon=eps,
        x_bounds=(X_MIN, X_MAX),
        y_bounds=(Y_MIN, Y_MAX),
    )
    if result is None:
        print(f"ε={eps:<5}  → degenerate (returned None)")
    else:
        a_hat, b_hat = result
        mse = l2_error_exact(TRUE_ALPHA, TRUE_BETA, a_hat, b_hat, x_lower=X_MIN, x_upper=X_MAX)
        print(f"ε={eps:<5}  → α̂={a_hat:+.4f}  β̂={b_hat:+.4f}  MSE={mse:.4f}")
