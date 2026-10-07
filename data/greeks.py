"""
Greeks of the ATM arithmetic Asian call: vega with respect to nu, and dV/dH.

Experiment design:
  Controlled:    H=0.10, nu=0.52, sigma_0=0.2, K=S0=100, T=1, r=0, N, M
  Independent:   bump size h of the central difference; estimator (finite
                 difference vs pathwise)
  Dependent:     dV/dnu and dV/dH with their standard errors

Two panels:
  (a) dV/dnu by central difference with common random numbers and the control
      variate, vs bump size h, against the pathwise estimator (plain and with
      a differentiated control variate)
  (b) dV/dH by central difference, vs bump size h

Usage:
    uv run python data/greeks.py [--M 20000] [--N 252]

Output:
    plots/figures/greeks.png

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

Bump-and-reprice with common random numbers:

  The central difference  [V(x+h) - V(x-h)] / (2h)  has bias O(h^2).  With
  independent random numbers for the two prices its variance is
  2 Var(V_hat) / (4h^2), which explodes as h -> 0.  With common random numbers
  (the same seed, hence the same Gaussian draws, for both bumps) each path's
  payoff moves continuously with x, so the per-path difference is O(h) and the
  variance of the estimator stays bounded as h -> 0.  Its standard error is the
  std of the per-path differences over sqrt(M), which is what is reported.

  Both bumps are priced with the control variate (price_asian_call_cv).  Each
  bump has its own beta from its own pilot, but the pilots also share their
  random numbers, so beta moves smoothly with x and the difference stays well
  behaved.  Each CV price is exactly unbiased, so the CV difference is too.

  Bumping H keeps the same Gaussian draws but changes the circulant eigenvalues,
  so every path changes smoothly with H as well.  There is no simple pathwise
  formula for dV/dH (it would need the derivative of the square-root spectrum),
  so only the finite difference is shown.

Pathwise vega:

  log sigma_t = mu0 + nu W_t^H, so d sigma_j / d nu = sigma_j W_j^H.  Differentiating
  the discrete scheme  log S_n = log S0 + sum_{j<=n} [(r - sigma_j^2/2) dt + sigma_j sqrt(dt) Z_j]
  path by path gives

      d log S_n / d nu = sum_{j<=n} sigma_j W_j^H (sqrt(dt) Z_j - sigma_j dt),
      d A / d nu       = (1/N) sum_n S_n d log S_n / d nu,
      d V / d nu       = 1{A > K} d A / d nu.

  The payoff (A - K)^+ is Lipschitz and A is differentiable in nu, so the
  derivative and the expectation can be exchanged: the pathwise estimator is
  unbiased (Glasserman 2004, Sec. 7.2).  It needs no bump at all.

  The control variate differentiates too.  E[Y - E[Y | sigma]] = 0 for every nu,
  so d/dnu (Y - E[Y | sigma]) also has mean zero and is a valid control for the
  pathwise vega.  dY/dnu = 1{G > K} G d log G / d nu, and the closed form
  E[Y | sigma] = e^{m + v/2} Phi(d1) - K Phi(d1 - sqrt v) differentiates through
  m and v:  dE/dm = e^{m+v/2} Phi(d1),  dE/dv = e^{m+v/2} [phi(d1)/(2 sqrt v) + Phi(d1)/2].
  Its beta comes from a separate pilot, as for prices.

Contested points:

  - Likelihood-ratio estimators are the usual alternative for discontinuous
    payoffs, but here the payoff is Lipschitz and the LR weight for nu would
    involve the inverse fBM covariance (an O(N^3) object); pathwise is simpler
    and has lower variance.
  - The finite difference at large h is biased (curvature of V in nu); at
    small h it is unbiased to O(h^2) and, with CRN, no noisier.  Agreement
    between FD at small h and pathwise is the check that both are right.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm as sp_norm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.rfsv_model import price_asian_call_cv, simulate_log_vol_paths
from data.params import H, NU, MU0, S0, K, T, R

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

NU_BUMPS = [0.005, 0.01, 0.02, 0.05, 0.1]
H_BUMPS = [0.0025, 0.005, 0.01, 0.02, 0.04]
SEED = 42


def fd_greek(param, h, N, M, seed=SEED, H=H, nu=NU, K=K):
    """
    Central difference in param ("nu" or "H") with common random numbers.
    Returns dict(cv=(est, se), plain=(est, se)) from the per-path differences.
    """
    out = {}
    runs = []
    for sign in (+1, -1):
        kw = dict(H=H, nu=nu)
        kw[param] += sign * h
        runs.append(price_asian_call_cv(K=K, T=T, S0=S0, r=R, N=N, M=M, seed=seed,
                                        return_samples=True, **kw))
    for name, key in [("cv", "samples"), ("plain", "samples_plain")]:
        d = (runs[0][key] - runs[1][key]) / (2 * h)
        out[name] = (float(d.mean()), float(d.std(ddof=1) / np.sqrt(M)))
    return out


def pathwise_vega_terms(X, nu, mu0, S0, r, dt, K, seed):
    """
    Per-path pathwise derivatives with respect to nu, undiscounted.

    X: (M, N) unit fBM paths W^H on the grid (log sigma = mu0 + nu X).  Uses the same
    price shocks as asian_cv_terms(seed=seed).  Returns (dV, dC): the derivative of the
    arithmetic payoff, and that of the control C = Y - E[Y | sigma] (mean zero).
    """
    M, N = X.shape
    sigma = np.exp(mu0 + nu * X)
    dsig = sigma * X                                  # d sigma / d nu
    Z = np.random.default_rng([seed, 1]).standard_normal((M, N))
    drift = (r - 0.5 * sigma ** 2) * dt
    ddrift = -sigma * dsig * dt
    log_S = np.log(S0) + np.cumsum(drift + sigma * np.sqrt(dt) * Z, axis=1)
    dlog_S = np.cumsum(ddrift + dsig * np.sqrt(dt) * Z, axis=1)
    S = np.exp(log_S)
    A = S.mean(axis=1)
    dV = (A > K) * (S * dlog_S).mean(axis=1)
    G = np.exp(log_S.mean(axis=1))
    dY = (G > K) * G * dlog_S.mean(axis=1)
    # Closed form E[Y | sigma] through its mean m and variance v (see asian_cv_terms)
    w = (N - np.arange(N)) / N
    m = np.log(S0) + (w * drift).sum(axis=1)
    v = dt * (w ** 2 * sigma ** 2).sum(axis=1)
    dm = (w * ddrift).sum(axis=1)
    dv = 2.0 * dt * (w ** 2 * sigma * dsig).sum(axis=1)
    sd = np.sqrt(v)
    d1 = (m - np.log(K) + v) / sd
    F = np.exp(m + 0.5 * v)
    dEY = F * sp_norm.cdf(d1) * dm + F * (sp_norm.pdf(d1) / (2 * sd) + 0.5 * sp_norm.cdf(d1)) * dv
    return dV, dY - dEY


def pathwise_vega(N, M, seed=SEED, H=H, nu=NU, mu0=MU0, K=K, pilot_M=None):
    """
    Pathwise dV/dnu: returns dict(cv=(est, se), plain=(est, se), beta).  The plain
    estimator is the mean of the per-path derivatives; the CV one subtracts
    beta * d/dnu (Y - E[Y | sigma]) with beta from a separate pilot run.
    """
    dt = T / N
    disc = np.exp(-R * T)
    pilot_M = pilot_M or max(2000, M // 10)

    def terms(n, s):
        X = simulate_log_vol_paths(N, n, H, 1.0, dt, seed=s)  # nu * X is the engine's path
        return pathwise_vega_terms(X, nu, mu0, S0, R, dt, K, seed=s)

    dVp, dCp = terms(pilot_M, seed + 1_000_003)
    beta = np.cov(dVp, dCp)[0, 1] / dCp.var(ddof=1)
    dV, dC = terms(M, seed)
    out = {"beta": float(beta)}
    for name, x in [("plain", dV), ("cv", dV - beta * dC)]:
        out[name] = (float(disc * x.mean()), float(disc * x.std(ddof=1) / np.sqrt(M)))
    return out


def plot_greeks(nu_fd, pw, h_fd, M, N, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle(f"Greeks of the ATM Asian call  (H={H}, $\\nu$={NU}, $\\sigma_0$=0.2, "
                 f"K=S0={S0:.0f}, T={T:.0f}, N={N}, M={M:,}; common random numbers)",
                 fontsize=11)

    # ── Panel (a): vega wrt nu, FD vs pathwise ──────────────────────────────
    ax = axes[0]
    hs = np.array(NU_BUMPS)
    est = np.array([nu_fd[h]["cv"][0] for h in NU_BUMPS])
    se = np.array([nu_fd[h]["cv"][1] for h in NU_BUMPS])
    pw_est, pw_se = pw["cv"]
    ax.axhspan(pw_est - pw_se, pw_est + pw_se, color="C2", alpha=0.25)
    ax.axhline(pw_est, color="C2", lw=1.5,
               label=rf"pathwise + CV: {pw_est:.3f} $\pm$ {pw_se:.3f}")
    ax.axhline(pw["plain"][0], color="gray", ls=":", lw=1.2,
               label=rf"pathwise, plain: {pw['plain'][0]:.3f} $\pm$ {pw['plain'][1]:.3f}")
    ax.errorbar(hs, est, yerr=se, fmt="o-", color="C0", capsize=4, ms=6, lw=1.5,
                label=r"central FD + CV, $\pm 1$ SE")
    ax.set_xscale("log")
    ax.set_xlabel(r"bump size $h$ in $\nu$")
    ax.set_ylabel(r"$\partial V / \partial \nu$")
    ax.set_title(r"(a) vega w.r.t. vol-of-vol $\nu$: finite difference vs pathwise")
    ax.legend(fontsize=9)

    # ── Panel (b): dV/dH by FD ──────────────────────────────────────────────
    ax2 = axes[1]
    hs = np.array(H_BUMPS)
    est = np.array([h_fd[h]["cv"][0] for h in H_BUMPS])
    se = np.array([h_fd[h]["cv"][1] for h in H_BUMPS])
    ax2.errorbar(hs, est, yerr=se, fmt="o-", color="C3", capsize=4, ms=6, lw=1.5,
                 label=r"central FD + CV, $\pm 1$ SE")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"bump size $h$ in $H$")
    ax2.set_ylabel(r"$\partial V / \partial H$")
    ax2.set_title(r"(b) sensitivity to the Hurst exponent $H$ (rougher = more expensive)")
    ax2.legend(fontsize=9)

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=20_000,
                        help="Monte Carlo paths per price (default: 20000)")
    parser.add_argument("--N", type=int, default=252,
                        help="Time steps per year (default: 252)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "greeks.png")
    print(f"Greeks of the ATM Asian call (M={args.M}, N={args.N}, H={H}, nu={NU}) ...")

    base = price_asian_call_cv(H, NU, K, T=T, S0=S0, r=R, N=args.N, M=args.M, seed=SEED)
    print(f"  price: {base['price']:.4f} ± {base['se']:.4f}")

    pw = pathwise_vega(args.N, args.M)
    print(f"\n  dV/dnu, pathwise:  plain {pw['plain'][0]:.4f} ± {pw['plain'][1]:.4f}"
          f"   CV {pw['cv'][0]:.4f} ± {pw['cv'][1]:.4f}"
          f"   (variance reduction {(pw['plain'][1] / pw['cv'][1]) ** 2:.1f}x)")

    # The z-score treats the two SEs as independent; they share paths, so it is conservative
    print(f"\n  {'h':>7} {'dV/dnu (FD + CV)':>20} {'FD plain':>18} {'FD - pathwise (CV)':>20}")
    nu_fd = {}
    for h in NU_BUMPS:
        nu_fd[h] = fd_greek("nu", h, args.N, args.M)
        (e, s), (ep, sp) = nu_fd[h]["cv"], nu_fd[h]["plain"]
        z = (e - pw["cv"][0]) / np.hypot(s, pw["cv"][1])
        print(f"  {h:7.4f} {e:12.4f} ± {s:.4f} {ep:10.4f} ± {sp:.4f} {z:+11.2f} SE")

    print(f"\n  {'h':>7} {'dV/dH (FD + CV)':>20} {'FD plain':>18}")
    h_fd = {}
    for h in H_BUMPS:
        h_fd[h] = fd_greek("H", h, args.N, args.M)
        (e, s), (ep, sp) = h_fd[h]["cv"], h_fd[h]["plain"]
        print(f"  {h:7.4f} {e:12.4f} ± {s:.4f} {ep:10.4f} ± {sp:.4f}")

    plot_greeks(nu_fd, pw, h_fd, args.M, args.N, out_path)


if __name__ == "__main__":
    main()
