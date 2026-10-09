"""
Vectorized numpy RFSV Monte Carlo engine.

Matches the C++ FFTW circulant-embedding convention exactly:
  - fGn autocovariance γ(k) embedded in 2N-circulant
  - sqrt_lam = sqrt(max(λ,0) / (2N))   [FFTW unnormalized backward convention]
  - increments = Re(IFFT(W) * 2N)[:N]  [multiply 2N to undo numpy's 1/(2N)]
  - log_vol = nu * cumsum(increments)   → fBM paths

Also: the approximate global low-rank (rSVD) sampler, optionally variance-corrected
(simulate_log_vol_paths_lowrank, mirroring src/rsvd/lowrank.hpp), and the conditional
geometric control variate (price_asian_call_cv), which accepts any path sampler.

Usage:
    from data.rfsv_model import price_asian_call, price_european_call, bs_implied_vol
"""

import numpy as np
from scipy.stats import norm as sp_norm
from scipy.optimize import brentq

try:
    from data.params import MU0
except ImportError:  # imported with data/ itself on sys.path (e.g. from tests/)
    from params import MU0


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


# ── fBM covariance and the low-rank (rSVD) sampler ──────────────────────────

def fbm_cov_matrix(N: int, H: float, T: float = 1.0) -> np.ndarray:
    """Dense fBM covariance C_ij = (t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H}) / 2 on t_j = j T / N."""
    dt = T / N
    t  = np.arange(1, N + 1) * dt
    ti, tj = t[:, None], t[None, :]
    return 0.5 * (ti ** (2 * H) + tj ** (2 * H) - np.abs(ti - tj) ** (2 * H))


def rsvd(A: np.ndarray, k: int, p: int = 5, q: int = 2,
         seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Halko-Martinsson-Tropp Algorithm 4.4 (randomized subspace iteration).

    Returns U (m x k), S (k,) such that A ≈ U * diag(S) * U^T for symmetric A.
    The third return value is None (Vt not needed here; symmetric => Vt = U^T).

    How it works:
      Stage A: random sketch Y = A @ Omega captures the k dominant column directions.
      Power iterations Y = (A A^T)^q @ Y amplify the signal-to-noise ratio by
      raising singular values to the power 2q+1, separating large from small.
      Re-orthonormalizing after every product keeps the small directions from
      being lost to round-off (sigma_1/sigma_k raised to 2q+1 can reach 1e15).
      Q is an orthonormal basis for the range of (A A^T)^q A.
      Stage B: project A onto Q to get a small (k+p) x n matrix B.
      SVD of B is cheap (O((k+p)^2 * n)); rotate back via Q to get U.
    """
    rng      = np.random.default_rng(seed)
    _m, n  = A.shape
    l     = k + p  # oversampled rank (p=5 reduces failure probability to near zero)
    Omega = rng.standard_normal((n, l))
    Q, _  = np.linalg.qr(A @ Omega)  # sketch: N x l  (captures top-l column directions)
    for _ in range(q):
        # Power iteration with QR after each product: range of (A A^T)^q A
        W, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ W)
    B    = Q.T @ A            # project: cheap (k+p) x n matrix
    U_b, S, _ = np.linalg.svd(B, full_matrices=False)
    U    = Q @ U_b            # rotate U back into the original N-dimensional space
    return U[:, :k], S[:k], None


def lowrank_factor(C: np.ndarray, rank_k: int, seed: int = 42) -> np.ndarray:
    """
    Approximate factor L_k = U diag(sqrt(max(S, 0)))  (N x k)  with L_k L_k^T ≈ C,
    from rSVD with oversampling 5 and 2 power iterations.  Mirrors lowrank_factor()
    in src/rsvd/lowrank.hpp.
    """
    k = min(rank_k, C.shape[0])
    U, S, _ = rsvd(C, k, p=5, q=2, seed=seed)
    return U * np.sqrt(np.maximum(S, 0.0))


def residual_sd(C: np.ndarray, Lk: np.ndarray) -> np.ndarray:
    """
    Variance correction: d_j = sqrt(max(C_jj - (L_k L_k^T)_jj, 0)), the per-step standard
    deviation of the variance the rank-k factor misses.  Adding independent N(0, d_j^2)
    noise at step j restores every marginal variance exactly; off-diagonal covariances
    stay those of L_k L_k^T.  Mirrors residual_sd() in src/rsvd/lowrank.hpp.
    """
    return np.sqrt(np.maximum(np.diag(C) - (Lk ** 2).sum(axis=1), 0.0))


def lowrank_sampler(N: int, H: float, nu: float, dt: float, rank_k: int,
                    corrected: bool = False, factor_seed: int = 42):
    """
    Build the rank-k factor once and return sampler(M, seed) -> (M, N) log-vol paths
    nu * (L_k z + d ⊙ eps), with d = 0 unless corrected.  The factor depends only on
    factor_seed, so runs with different path seeds share one sampler (as in C++, where
    the rSVD sketch has its own RNG).  The correction draws eps after z from the same
    stream default_rng(seed), so uncorrected paths are unchanged by the option.
    """
    C = fbm_cov_matrix(N, H, T=N * dt)          # grid {dt, 2dt, ..., N dt}
    Lk = lowrank_factor(C, rank_k, seed=factor_seed)
    d = residual_sd(C, Lk) if corrected else None

    def sample(M: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((M, Lk.shape[1]))
        x = z @ Lk.T                             # (M, N), O(N k) per path
        if d is not None:
            x += d * rng.standard_normal((M, N))
        return nu * x

    return sample


def simulate_log_vol_paths_lowrank(N: int, M: int, H: float, nu: float, dt: float,
                                   rank_k: int, seed: int = 42,
                                   corrected: bool = False) -> np.ndarray:
    """
    Approximate fBM log-vol paths from the global rank-k rSVD factor, shape (M, N) like
    simulate_log_vol_paths.  Plain paths have covariance nu^2 L_k L_k^T and lose the
    truncated variance; corrected=True restores the marginal variances (diag(nu^2 C)),
    the Python equivalent of lowrank::price_corrected / price_cv(..., corrected=true).
    """
    return lowrank_sampler(N, H, nu, dt, rank_k, corrected=corrected)(M, seed)


# ── Price path simulation ────────────────────────────────────────────────────

def _simulate_price_paths(log_vol: np.ndarray, S0: float, r: float,
                          dt: float, seed: int = 42) -> np.ndarray:
    """
    Convert (M, N) log-vol paths → (M, N) GBM price paths.

    S_{t+dt} = S_t * exp((r - σ_t²/2)*dt + σ_t*sqrt(dt)*Z_t)
    where σ_t = exp(log_vol[t]).  Matches C++ log_vol_to_prices().

    The shocks Z come from SeedSequence([seed, 1]), a stream independent of the
    log-vol stream default_rng(seed) and of every other integer seed.  (Using
    seed + 1 here would collide with the log-vol stream of the run seeded seed + 1.)
    """
    M, N = log_vol.shape
    sigma = np.exp(log_vol)                         # (M, N) instantaneous vol
    rng = np.random.default_rng([seed, 1])
    Z = rng.standard_normal((M, N))
    log_returns = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    log_prices = np.log(S0) + np.cumsum(log_returns, axis=1)
    return np.exp(log_prices)                       # (M, N)


# ── Pricing functions ────────────────────────────────────────────────────────

def price_asian_call(H: float, nu: float, K: float, T: float = 1.0,
                     S0: float = 100.0, r: float = 0.0,
                     N: int = 252, M: int = 10000, seed: int = 42,
                     mu0: float = MU0) -> float:
    """
    Price arithmetic Asian call under RFSV model via Monte Carlo.

    mu0: log-vol level, log sigma_t = mu0 + nu * W_t^H.  Defaults to
         log(SIGMA0) from data/params.py (sigma_0 = 0.2).
    """
    dt = T / N
    log_vol = simulate_log_vol_paths(N, M, H, nu, dt, seed=seed) + mu0
    prices = _simulate_price_paths(log_vol, S0, r, dt, seed=seed)
    A = np.mean(prices, axis=1)                     # arithmetic average per path
    payoff = np.maximum(A - K, 0.0)
    return float(np.exp(-r * T) * np.mean(payoff))


def asian_cv_terms(log_vol: np.ndarray, S0: float, r: float, dt: float, K: float,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-path terms of the conditional geometric control variate.

    log_vol: (M, N) total log-volatility paths (mu0 already added).  Returns
    (V, Y, EY): the arithmetic payoff, the geometric payoff, and its closed-form
    conditional expectation E[Y | sigma], all undiscounted.  The price shocks use the
    same stream as _simulate_price_paths(seed=seed), so V matches price_asian_call.
    K may be a scalar (each output has shape (M,)) or an array of strikes priced on the
    same paths (each output has shape (M,) + K.shape).

    With rho = 0, log G = log S0 + sum_j w_j [(r - sigma_j^2/2) dt + sigma_j sqrt(dt) Z_j],
    w_j = (N - j) / N, is Gaussian given sigma with mean m and variance v below, so
    E[(G - K)^+ | sigma] = e^{m + v/2} Phi(d1) - K Phi(d1 - sqrt v).
    """
    M, N = log_vol.shape
    K = np.asarray(K, dtype=float)
    col = (slice(None),) + (None,) * K.ndim      # per-path scalar -> broadcast over K
    sigma = np.exp(log_vol)
    Z = np.random.default_rng([seed, 1]).standard_normal((M, N))
    drift = (r - 0.5 * sigma ** 2) * dt
    log_S = np.log(S0) + np.cumsum(drift + sigma * np.sqrt(dt) * Z, axis=1)
    V = np.maximum(np.exp(log_S).mean(axis=1)[col] - K, 0.0)
    Y = np.maximum(np.exp(log_S.mean(axis=1))[col] - K, 0.0)
    w = (N - np.arange(N)) / N
    m = (np.log(S0) + (w * drift).sum(axis=1))[col]
    v = (dt * (w ** 2 * sigma ** 2).sum(axis=1))[col]
    sd = np.sqrt(v)
    d1 = (m - np.log(K) + v) / sd
    EY = np.exp(m + 0.5 * v) * sp_norm.cdf(d1) - K * sp_norm.cdf(d1 - sd)
    return V, Y, EY


def price_asian_call_cv(H: float, nu: float, K: float, T: float = 1.0,
                        S0: float = 100.0, r: float = 0.0,
                        N: int = 252, M: int = 10000, seed: int = 42,
                        mu0: float = MU0, pilot_M: int | None = None,
                        sampler=None, return_samples: bool = False) -> dict:
    """
    Arithmetic Asian call with the conditional geometric control variate.

    Estimator e^{-rT} mean(V - beta (Y - E[Y|sigma])), with beta = Cov(V, C) / Var(C)
    estimated on a separate pilot run (so the estimate is exactly unbiased).
    Returns dict(price, se, price_plain, se_plain, beta); price_plain uses the same
    paths as price_asian_call(..., seed=seed).

    K: a strike, or an array of strikes all priced from the same paths (beta is then
       estimated per strike and every entry of the result is an array over K).
    sampler: optional callable sampler(n_paths, seed) -> (n_paths, N) array of
       nu * W^H on the grid {dt, ..., N dt} (mu0 not added), e.g. lowrank_sampler(...).
       It replaces the exact FFT sampler; H and nu are then unused.
    return_samples: also return the per-path discounted estimates, "samples" (CV) and
       "samples_plain", shape (M,) + K.shape.  Two runs with the same seed share their
       random numbers path by path, so the SE of a price difference (a Greek, a premium)
       is the std of the per-path differences over sqrt(M).
    """
    dt = T / N
    disc = np.exp(-r * T)
    pilot_M = pilot_M or max(2000, M // 10)
    if sampler is None:
        def sampler(n, s):
            return simulate_log_vol_paths(N, n, H, nu, dt, seed=s)

    def terms(n, s):
        log_vol = sampler(n, s) + mu0
        V, Y, EY = asian_cv_terms(log_vol, S0, r, dt, K, seed=s)
        return V, Y - EY

    Vp, Cp = terms(pilot_M, seed + 1_000_003)
    if np.ndim(K) == 0:
        beta = np.cov(Vp, Cp)[0, 1] / Cp.var(ddof=1)
    else:  # one beta per strike (columnwise covariance)
        beta = ((Vp - Vp.mean(axis=0)) * (Cp - Cp.mean(axis=0))).sum(axis=0) / (pilot_M - 1)
        beta = beta / Cp.var(axis=0, ddof=1)
    V, C = terms(M, seed)
    est = V - beta * C

    def out(x):
        return float(x) if np.ndim(x) == 0 else x

    res = dict(price=out(disc * est.mean(axis=0)),
               se=out(disc * est.std(axis=0, ddof=1) / np.sqrt(M)),
               price_plain=out(disc * V.mean(axis=0)),
               se_plain=out(disc * V.std(axis=0, ddof=1) / np.sqrt(M)),
               beta=out(beta))
    if return_samples:
        res.update(samples=disc * est, samples_plain=disc * V)
    return res


def price_european_call(H: float, nu: float, K: float, T: float = 1.0,
                        S0: float = 100.0, r: float = 0.0,
                        N: int = 252, M: int = 10000, seed: int = 42,
                        mu0: float = MU0, martingale_correct: bool = False) -> float:
    """
    Price European call under RFSV model via Monte Carlo.

    mu0: log-vol drift (additive shift to log σ_t).
         log σ_t = mu0 + nu * W_t^H.
         Default log(SIGMA0) → σ_0 = 0.2 (matches C++ params.hpp).
         Set mu0 = log(target_vol) to calibrate to market vol level.
    martingale_correct: rescale S_T so its sample mean equals the forward
         S0 * exp(rT) exactly (moment matching).  With common random numbers
         across strikes, a sampling error in mean(S_T) acts like a shifted
         forward and tilts the whole implied-vol curve; this removes it.
    """
    dt = T / N
    log_vol = simulate_log_vol_paths(N, M, H, nu, dt, seed=seed) + mu0
    prices = _simulate_price_paths(log_vol, S0, r, dt, seed=seed)
    S_T = prices[:, -1]                             # terminal price only
    if martingale_correct:
        S_T = S_T * (S0 * np.exp(r * T) / S_T.mean())
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
