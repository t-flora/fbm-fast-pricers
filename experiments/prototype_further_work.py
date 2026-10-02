"""
Prototypes for the two highest-priority items in TODO.md (not part of the pipeline).

  1. Variance-corrected low-rank sampler: add independent noise with variance
     diag(C - C_k) to the rank-k paths, restoring every marginal variance.
  2. Conditional geometric control variate: with rho = 0, log of the geometric
     average is Gaussian given the volatility path, so E[(G - K)^+ | sigma] is a
     Black-Scholes formula and Y = (G - K)^+ is a control variate with known mean.

Uses the exact eigendecomposition (the optimal rank-k truncation, which the rSVD
matches to ~2%) and one shared set of random numbers.  Runs in under a minute:

    uv run python experiments/prototype_further_work.py
"""

import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "data"))

from data.params import H, NU, MU0, S0, K, T  # noqa: E402
import validate_stability as vs  # noqa: E402

N, dt = 252, T / 252
C = vs.fbm_cov_matrix(N, H)
w, U = np.linalg.eigh(C); w, U = w[::-1], U[:, ::-1]

def asian_payoff(logvol, Z):
    sig = np.exp(MU0 + logvol)
    logS = np.log(S0) + np.cumsum(-0.5 * sig**2 * dt + sig * np.sqrt(dt) * Z, axis=1)
    return np.maximum(np.exp(logS).mean(1) - K, 0), sig

rng = np.random.default_rng(1); M = 200_000
z = rng.standard_normal((M, N)); Z = rng.standard_normal((M, N)); eps = rng.standard_normal((M, N))
L = np.linalg.cholesky(C)
exact, sig = asian_payoff(NU * z @ L.T, Z)
print(f"exact (Cholesky, CRN): {exact.mean():.4f} +- {exact.std()/np.sqrt(M):.4f}")

# Idea 1: low-rank + diagonal variance correction (same z columns, same Z)
for k in [8, 32, 128]:
    Lk = U[:, :k] * np.sqrt(w[:k])
    lowrank, _ = asian_payoff(NU * z[:, :k] @ Lk.T, Z)
    d = np.sqrt(np.maximum(np.diag(C) - (Lk**2).sum(1), 0))
    corrected, _ = asian_payoff(NU * (z[:, :k] @ Lk.T + eps * d), Z)
    se = np.sqrt(2) * exact.std() / np.sqrt(M)
    print(f"k={k:3d}: low-rank {100*(lowrank.mean()/exact.mean()-1):+.2f}%   low-rank+diag {100*(corrected.mean()/exact.mean()-1):+.2f}%   (SE of an unpaired diff {100*se/exact.mean():.2f}%; shared draws make it smaller)")

# Idea 2: geometric Asian control variate with closed-form conditional expectation given the vol path
# log G = log S0 + (1/N) sum_i log(S_i/S0): Gaussian given sigma, with
#   mean m = -(dt/2) sum_j sigma_j^2 (N-j+1)/N,  var v = dt sum_j sigma_j^2 ((N-j+1)/N)^2
wts = (N - np.arange(N)) / N
m = np.log(S0) - 0.5 * dt * (sig**2 * wts).sum(1)
v = dt * (sig**2 * wts**2).sum(1)
logS = np.log(S0) + np.cumsum(-0.5 * sig**2 * dt + sig * np.sqrt(dt) * Z, axis=1)
G = np.exp(logS.mean(1)); Y = np.maximum(G - K, 0)
d1 = (m - np.log(K) + v) / np.sqrt(v); EY = np.exp(m + v / 2) * norm.cdf(d1) - K * norm.cdf(d1 - np.sqrt(v))
beta = np.cov(exact, Y)[0, 1] / Y.var()
cv = exact - beta * (Y - EY)
print(f"corr(arith, geom) = {np.corrcoef(exact, Y)[0,1]:.4f}; check E[Y|sigma] vs Y mean: {EY.mean():.4f} vs {Y.mean():.4f}")
print(f"plain MC: {exact.mean():.4f} +- {exact.std()/np.sqrt(M):.4f};  control variate: {cv.mean():.4f} +- {cv.std()/np.sqrt(M):.4f};  variance reduction {exact.var()/cv.var():.0f}x")
