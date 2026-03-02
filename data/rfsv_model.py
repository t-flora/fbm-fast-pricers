"""
Vectorized numpy RFSV Monte Carlo engine.

Matches the C++ FFTW circulant-embedding convention exactly:
  - fGn autocovariance γ(k) embedded in 2N-circulant
  - sqrt_lam = sqrt(max(λ,0) / (2N))   [FFTW unnormalized backward convention]
  - increments = Re(IFFT(W) * 2N)[:N]  [multiply 2N to undo numpy's 1/(2N)]
  - log_vol = nu * cumsum(increments)   → fBM paths

Usage:
    from data.rfsv_model import price_asian_call, price_european_call, bs_implied_vol
"""

import numpy as np
from scipy.stats import norm as sp_norm
from scipy.optimize import brentq


# ── fGn covariance ───────────────────────────────────────────────────────────

def build_fgn_eigenvalues(N: int, H: float, dt: float) -> np.ndarray:
    """
    Build 2N circulant embedding of the fGn Toeplitz matrix.
    Returns the real eigenvalue array λ of length 2N (all ≥ 0 for H ≤ 0.5).
    """
    M = 2 * N
    c_emb = np.zeros(M, dtype=complex)
    ks = np.arange(N)
    h2 = 2.0 * H
    # Vectorized γ(k) for k = 0..N-1; protect (k-1) base for k=0,1
    safe_km1 = np.maximum(ks - 1, 0)                # avoid negative base
    km1 = np.where(ks <= 1, 0.0, safe_km1 ** h2)
    c_emb[:N] = 0.5 * dt ** h2 * ((ks + 1) ** h2 + km1 - 2.0 * ks ** h2)
    c_emb[0] = dt ** h2  # override k=0 term
    for j in range(1, N):
        c_emb[M - j] = c_emb[j]  # symmetric reflection; c_emb[N] stays 0
    lam = np.fft.fft(c_emb).real  # imaginary parts ≈ 0 for symmetric input
    return lam


# ── fBM path simulation (vectorized) ────────────────────────────────────────

def simulate_log_vol_paths(N: int, M: int, H: float, nu: float,
                           dt: float, seed: int = 42) -> np.ndarray:
    """
    Generate M fBM log-vol paths of length N via Davies-Harte circulant embedding.

    Returns array of shape (M, N) where paths[m, i] = nu * W^H(t_{i+1}).
    Normalization matches C++ FFTW backward (unnormalized) convention.
    """
    M_emb = 2 * N
    lam = build_fgn_eigenvalues(N, H, dt)
    sqrt_lam = np.sqrt(np.maximum(lam, 0.0) / M_emb)

    rng = np.random.default_rng(seed)
    a = rng.standard_normal((M, M_emb))
    b = rng.standard_normal((M, M_emb))
    W_freq = sqrt_lam[np.newaxis, :] * (a + 1j * b)
    # Multiply by M_emb to match FFTW's unnormalized IFFT (numpy ifft divides by M_emb)
    out = np.fft.ifft(W_freq, axis=1) * M_emb
    increments = out.real[:, :N]          # (M, N) fGn increments
    log_vol = nu * np.cumsum(increments, axis=1)  # (M, N) fBM paths
    return log_vol


# ── Price path simulation ────────────────────────────────────────────────────

def _simulate_price_paths(log_vol: np.ndarray, S0: float, r: float,
                          dt: float, seed: int = 99) -> np.ndarray:
    """
    Convert (M, N) log-vol paths → (M, N) GBM price paths.

    S_{t+dt} = S_t * exp((r - σ_t²/2)*dt + σ_t*sqrt(dt)*Z_t)
    where σ_t = exp(log_vol[t]).  Matches C++ log_vol_to_prices().
    """
    M, N = log_vol.shape
    sigma = np.exp(log_vol)                         # (M, N) instantaneous vol
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((M, N))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
    return np.exp(log_prices)                       # (M, N)


# ── Pricing functions ────────────────────────────────────────────────────────

def price_asian_call(H: float, nu: float, K: float, T: float = 1.0,
                     S0: float = 100.0, r: float = 0.0,
                     N: int = 252, M: int = 10000, seed: int = 42) -> float:
    """Price arithmetic Asian call under RFSV model via Monte Carlo."""
    dt = T / N
    log_vol = simulate_log_vol_paths(N, M, H, nu, dt, seed=seed)
    prices = _simulate_price_paths(log_vol, S0, r, dt, seed=seed + 1)
    A = np.mean(prices, axis=1)                     # arithmetic average per path
    payoff = np.maximum(A - K, 0.0)
    return float(np.exp(-r * T) * np.mean(payoff))


def price_european_call(H: float, nu: float, K: float, T: float = 1.0,
                        S0: float = 100.0, r: float = 0.0,
                        N: int = 252, M: int = 10000, seed: int = 42,
                        mu0: float = 0.0) -> float:
    """
    Price European call under RFSV model via Monte Carlo.

    mu0: log-vol drift (additive shift to log σ_t).
         log σ_t = mu0 + nu * W_t^H.
         Default mu0=0 → σ_0 = 1.0 (matches C++ params.hpp).
         Set mu0 = log(target_vol) to calibrate to market vol level.
    """
    dt = T / N
    log_vol = simulate_log_vol_paths(N, M, H, nu, dt, seed=seed) + mu0
    prices = _simulate_price_paths(log_vol, S0, r, dt, seed=seed + 1)
    S_T = prices[:, -1]                             # terminal price only
    payoff = np.maximum(S_T - K, 0.0)
    return float(np.exp(-r * T) * np.mean(payoff))


# ── Black-Scholes helpers ────────────────────────────────────────────────────

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0 or sigma <= 0:
        return max(S * np.exp(-0.0) - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d2)


def bs_implied_vol(price: float, S: float, K: float, T: float, r: float,
                   tol: float = 1e-6, vol_lo: float = 1e-4,
                   vol_hi: float = 20.0) -> float:
    """
    Invert Black-Scholes formula to get implied volatility.
    Returns NaN if the price is outside the no-arbitrage bounds or inversion fails.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + tol:
        return float("nan")
    try:
        return brentq(lambda v: bs_call_price(S, K, T, r, v) - price,
                      vol_lo, vol_hi, xtol=tol)
    except ValueError:
        return float("nan")


# ── Levy (1992) arithmetic Asian call approximation ─────────────────────────

def levy_asian_call(S0: float, K: float, T: float, r: float, sigma: float,
                    N: int = 252) -> float:
    """
    Levy (1992) lognormal moment-matching approximation for discrete arithmetic
    Asian call under constant-sigma GBM.

    Matches the RFSV model at nu → 0 (constant sigma_t = exp(0) = 1),
    providing a noise-free analytical benchmark.

    Reference: Levy (1992), "Pricing European average rate currency options",
    Journal of International Money and Finance.
    """
    dt = T / N
    t = np.arange(1, N + 1) * dt                   # payment times t_1, ..., t_N

    # First moment of arithmetic mean: E[A] = (1/N) sum_i E[S_{t_i}]
    mu_A = np.mean(S0 * np.exp(r * t))

    # Second moment: E[A^2] = (S0^2/N^2) sum_i sum_j exp(r(t_i+t_j) + sigma^2*min(t_i,t_j))
    a = S0 * np.exp(r * t)                          # (N,)
    outer_a = np.outer(a, a)                         # (N, N)
    min_t = np.minimum(t[:, None], t[None, :])       # (N, N) min(t_i, t_j)
    mu_A2 = np.mean(outer_a * np.exp(sigma ** 2 * min_t))

    # Lognormal fit: v^2 = log(E[A^2] / E[A]^2)
    ratio = mu_A2 / (mu_A ** 2)
    if ratio <= 1.0:
        # Degenerate: approximate with intrinsic
        return max(mu_A * np.exp(-r * T) - K * np.exp(-r * T), 0.0)
    v2 = np.log(ratio)
    v = np.sqrt(v2)

    # BS formula on the lognormal-approximated arithmetic mean
    d1 = (np.log(mu_A / K) + v2 / 2.0) / v
    d2 = d1 - v
    call = np.exp(-r * T) * (mu_A * sp_norm.cdf(d1) - K * sp_norm.cdf(d2))
    return float(call)
