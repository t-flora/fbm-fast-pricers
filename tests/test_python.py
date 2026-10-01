"""
Correctness tests for the Python engine and analysis helpers.

Run with:  uv run pytest tests/

Each test targets a property that a past bug broke or would break:
  - gamma(0) = dt^{2H} and the closed-form minimum circulant eigenvalue
    (the stability script once halved gamma(0) and "found" negative eigenvalues)
  - sample covariance of simulated fBM paths matches nu^2 C
  - at nu = 0 the European pricer reproduces Black-Scholes
  - the variogram calibration recovers known (H, nu), with time in years
  - the Python rSVD is close to the optimal truncation
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

import calibrate  # noqa: E402
import validate_stability as stab  # noqa: E402
from rfsv_model import (  # noqa: E402
    bs_call_price,
    build_fgn_eigenvalues,
    price_european_call,
    simulate_log_vol_paths,
)


@pytest.mark.parametrize("H", [0.05, 0.10, 0.30, 0.50])
@pytest.mark.parametrize("N", [64, 252])
def test_circulant_min_eigenvalue_closed_form(N, H):
    dt = 1.0 / N
    assert stab.fgn_cov_row(N, H, dt)[0] == pytest.approx(dt ** (2 * H))
    lam_stab = stab.circulant_eigenvalues(N, H, dt)
    lam_engine = build_fgn_eigenvalues(N, H, dt)
    np.testing.assert_allclose(lam_stab, lam_engine, rtol=1e-10, atol=1e-14)
    closed = dt ** (2 * H) * (N ** (2 * H) - (N - 1) ** (2 * H))
    assert lam_engine.min() > 0
    assert lam_engine.min() == pytest.approx(closed, rel=1e-8)


def test_simulated_paths_have_fbm_covariance():
    N, M, H, nu = 16, 100_000, 0.10, 0.5
    paths = simulate_log_vol_paths(N, M, H, nu, dt=1.0 / N, seed=3)
    S = paths.T @ paths / M
    C = nu ** 2 * stab.fbm_cov_matrix(N, H)
    tol = 6 * np.sqrt(2 / M) * np.abs(C).max()  # ~6 standard errors per entry
    assert np.abs(S - C).max() < tol
    assert np.abs(paths.mean(axis=0)).max() < 6 * nu / np.sqrt(M)


@pytest.mark.parametrize("martingale_correct", [False, True])
def test_european_price_matches_black_scholes_at_zero_vol_of_vol(martingale_correct):
    S0, K, T, sigma, M = 100.0, 100.0, 0.25, 0.20, 200_000
    p = price_european_call(H=0.1, nu=0.0, K=K, T=T, S0=S0, r=0.0, N=20, M=M,
                            seed=5, mu0=np.log(sigma),
                            martingale_correct=martingale_correct)
    se = S0 * sigma * np.sqrt(T) * 0.6 / np.sqrt(M)  # payoff std < 0.6 S0 sigma sqrt(T)
    assert abs(p - bs_call_price(S0, K, T, 0.0, sigma)) < 4 * se


def test_variogram_recovers_known_parameters():
    H_true, nu_true, step = 0.10, 0.30, 1.0 / 252
    log_vol = simulate_log_vol_paths(20_000, 4, H_true, nu_true, dt=step, seed=9)
    fits = [calibrate.fit_variogram(path, step_years=step, lag_max=20) for path in log_vol]
    H_hat, nu_hat = np.mean(fits, axis=0)
    assert H_hat == pytest.approx(H_true, abs=0.02)
    assert nu_hat == pytest.approx(nu_true, rel=0.15)


@pytest.mark.parametrize("k", [4, 16, 64])
def test_rsvd_near_optimal(k):
    C = stab.fbm_cov_matrix(200, 0.10)
    w = np.linalg.eigvalsh(C)[::-1]
    U, S, _ = stab.rsvd(C, k)
    err = np.linalg.norm(C - (U * S) @ U.T)
    assert err / np.sqrt((w[k:] ** 2).sum()) < 1.05
