"""
Phase 2: Asian Option Model Validation.

Compares RFSV Asian call prices with constant-volatility baselines across strikes,
and measures the roughness premium with and without variance normalization.

Outputs (plots/figures/validate_asian.png):
  (a) Price vs strike: Levy (sigma=SIGMA0, approximate), exact GBM by MC (nu=0), and
      RFSV at H in {0.05, 0.10, 0.30, 0.50} with nu = NU
  (b) Roughness premium p(H=0.10) - p(H=0.50) vs strike, two ways:
        fixed nu        -- both at nu = NU (mixes roughness with total variance)
        variance-matched -- nu at H=0.50 chosen so E[int_0^T sigma_t^2 dt] equals
                            its value at (H=0.10, nu=NU); isolates roughness

Every configuration reuses the same seeds (common random numbers), and each
simulated path is priced at every strike, so differences across H and K are
precise; premium error bars are the paired standard error across seeds.

Usage:
    uv run python data/validate_asian.py [--M 10000] [--N 252] [--n-seeds 10]

Output:
    plots/figures/validate_asian.png

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

What is the Levy (1992) approximation?

  Turnbull & Wakeman (1991) and Levy (1992) derived a semi-analytical formula
  for the arithmetic Asian call under *constant* sigma (standard GBM), by
  matching the first two moments of the arithmetic average to a lognormal
  distribution.  This gives a closed-form price in terms of Black-Scholes
  inputs.  It is accurate to < 1% at typical equity vols (sigma ~ 0.2), but
  NOT at the sigma_0 = 1.0 (100% vol) used here: at K = 100 it gives 23.72
  versus 22.29 +/- 0.12 for exact discrete GBM by MC (200k paths), i.e. it
  overprices by ~6%.  Treat the Levy curve as an approximate reference only;
  a nu = 0 MC run is the correct GBM baseline.

  We use it as a benchmark because:
  (1) At nu = 0, RFSV reduces to constant-sigma GBM (sigma = 1), which Levy
      approximates.
  (2) Levy is fast (no MC noise), giving a clean baseline.

Why do RFSV prices exceed Levy (even at H=0.5)?

  The RFSV model has sigma_t = exp(nu * W_t^H), which is stochastic.
  By Jensen's inequality, E[sigma^2] > E[sigma]^2 for any random sigma.
  This "volatility convexity" makes options more expensive than under constant
  sigma.  The effect grows with Var(log sigma_t) = nu^2 t^{2H}.  Note that for
  t < 1 year, t^{2H} is LARGER for small H, so lowering H at fixed nu also
  raises the total variance of log-vol, not just its roughness.

What is the roughness premium?

  RFSV(H=0.1) - RFSV(H=0.5).  At fixed nu this is NOT a pure roughness effect:
  Var(nu W_t^H) = nu^2 t^{2H}, so the two models also differ in how much log-vol
  varies on [0, T], and the expected integrated variance
  E[int_0^T sigma_t^2 dt] = int_0^T exp(2 nu^2 t^{2H}) dt is larger at H = 0.1.
  Panel (b) therefore also shows a variance-matched premium: nu at H = 0.5 is
  solved (brentq) so that E[int sigma^2 dt] equals its value at (H=0.1, nu=NU).
  What remains is the effect of path regularity alone.

Common random numbers and error bars:

  Every configuration (each H, nu) reuses seeds SEED_BASE + s, and each set of
  simulated paths is priced at every strike.  Differences between configurations
  are then far less noisy than independent runs would give.  The premium error
  bars are the standard error of the per-seed differences (paired), which is
  the honest uncertainty of a CRN difference.  Note the payoff is heavy-tailed at
  nu = 0.52 (per-path std ~ 90 at N = 252), so absolute prices carry an MC
  standard error of ~0.3 even with 10 seeds of 10,000 paths.

Contested point: Levy uses a lognormal approximation to the arithmetic average.
  At sigma = 1 it is biased (see above), so panel (a) also shows exact GBM by MC.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scipy.integrate import quad
from scipy.optimize import brentq
from data.rfsv_model import levy_asian_call, simulate_log_vol_paths, _simulate_price_paths
from data.params import NU, SIGMA0, MU0, S0, T, R

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# Baseline parameters matching src/common/params.hpp

# Strikes: ITM calls (K=80) through OTM calls (K=120)
STRIKES = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)

# H values to sweep
H_VALUES = {
    "H=0.05": 0.05,
    "H=0.10*": 0.10,
    "H=0.30": 0.30,
    "H=0.50": 0.50,
}

H_COLORS = {
    "H=0.05":  "#8e44ad",
    "H=0.10*": "#e74c3c",
    "H=0.30":  "#e67e22",
    "H=0.50":  "#27ae60",
}

# * = model value (H=0.10)

SEED_BASE = 42  # seed s of every configuration is SEED_BASE + s (common random numbers)


def integrated_variance(H, nu, T=T):
    """E[int_0^T sigma_t^2 dt] = int_0^T exp(2 nu^2 t^{2H}) dt for sigma_t = exp(nu W_t^H)."""
    return quad(lambda t: np.exp(2.0 * nu ** 2 * t ** (2.0 * H)), 0.0, T)[0]


def variance_matched_nu(H, H_ref=0.10, nu_ref=NU):
    """nu at Hurst H giving the same expected integrated variance as (H_ref, nu_ref)."""
    target = integrated_variance(H_ref, nu_ref)
    return brentq(lambda nu: integrated_variance(H, nu) - target, 1e-6, 5.0)


def asian_prices(H, nu, Ks, N, M, seed):
    """Asian call prices at every strike in Ks from ONE set of M simulated paths."""
    dt = T / N
    log_vol = simulate_log_vol_paths(N, M, H, nu, dt, seed=seed) + MU0
    A = _simulate_price_paths(log_vol, S0, R, dt, seed=seed).mean(axis=1)
    return np.exp(-R * T) * np.maximum(A[:, None] - Ks[None, :], 0.0).mean(axis=0)


def run_config(H, nu, Ks, N, M, n_seeds):
    """(n_seeds, len(Ks)) array of prices; seed s is shared across all configurations."""
    return np.array([asian_prices(H, nu, Ks, N, M, seed=SEED_BASE + s) for s in range(n_seeds)])


def run_levy_benchmark(Ks, sigma, N):
    """Levy analytical prices for arithmetic Asian call at constant sigma."""
    return np.array([levy_asian_call(S0, K, T, R, sigma, N) for K in Ks])


def paired_premium(a, b):
    """Mean and standard error of a - b across seeds (a, b share random numbers)."""
    d = a - b
    return d.mean(axis=0), d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])


def plot_validation(Ks, levy, gbm, rfsv, prem_fixed, prem_matched, nu_matched, n_seeds, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # ── Panel (a): Price vs Strike ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(Ks, levy, color="gray", linestyle="--", linewidth=1.8,
            label=rf"Levy approx. ($\sigma$={SIGMA0:.1f})")
    ax.plot(Ks, gbm.mean(axis=0), color="black", linestyle=":", linewidth=2,
            marker="s", markersize=4, label=rf"GBM by MC ($\nu$=0, $\sigma$={SIGMA0:.1f})")
    for label, arr in rfsv.items():
        ax.plot(Ks, arr.mean(axis=0), color=H_COLORS[label], linewidth=2,
                marker="o", markersize=5, label=f"RFSV {label}")
    ax.axvline(S0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Asian Call Price")
    ax.set_title("Asian call price vs strike\n"
                 rf"(S0=100, T=1, $\sigma_0$={SIGMA0:.1f}, $\nu$={NU:.2f}; mean of {n_seeds} seeds)")
    ax.legend(fontsize=8.5, loc="upper right")

    # ── Panel (b): Roughness premium, fixed vs variance-matched nu ───────────
    ax2 = axes[1]
    w = 2.2
    for (mean, se), dx, color, lab in [
        (prem_fixed, -w / 2, "#e74c3c", rf"fixed $\nu$={NU:.2f}"),
        (prem_matched, +w / 2, "#2980b9",
         rf"variance-matched ($\nu_{{0.5}}$={nu_matched:.3f})"),
    ]:
        ax2.bar(Ks + dx, mean, width=w, color=color, alpha=0.8, edgecolor="black",
                linewidth=0.5, yerr=se, capsize=2, label=lab)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel(r"p(H=0.10) $-$ p(H=0.50)")
    ax2.set_title("Roughness premium (error bars: $\\pm 1$ paired SE)")
    ax2.legend(fontsize=9)

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=10000,
                        help="Monte Carlo paths per seed (default: 10000)")
    parser.add_argument("--N", type=int, default=252,
                        help="Time steps per year (default: 252)")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Seeds per configuration, shared across configurations (default: 10)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "validate_asian.png")
    print(f"Asian option validation (M={args.M}, N={args.N}, seeds={args.n_seeds}) ...")

    levy = run_levy_benchmark(STRIKES, sigma=SIGMA0, N=args.N)
    gbm = run_config(0.10, 0.0, STRIKES, args.N, args.M, args.n_seeds)
    rfsv = {label: run_config(H, NU, STRIKES, args.N, args.M, args.n_seeds)
            for label, H in H_VALUES.items()}

    nu_matched = variance_matched_nu(0.50)
    print(f"\n  E[int sigma^2]: H=0.10, nu={NU:.2f} -> {integrated_variance(0.10, NU):.4f}; "
          f"H=0.50, nu={NU:.2f} -> {integrated_variance(0.50, NU):.4f}")
    print(f"  variance-matched nu at H=0.50: {nu_matched:.4f}")
    matched_h05 = run_config(0.50, nu_matched, STRIKES, args.N, args.M, args.n_seeds)

    prem_fixed = paired_premium(rfsv["H=0.10*"], rfsv["H=0.50"])
    prem_matched = paired_premium(rfsv["H=0.10*"], matched_h05)

    print(f"\n  {'K':>5} {'Levy':>7} {'GBM':>7} " + " ".join(f"{l:>8}" for l in rfsv)
          + f" {'prem_fix':>13} {'prem_match':>13}")
    for i, K in enumerate(STRIKES):
        row = " ".join(f"{arr[:, i].mean():8.3f}" for arr in rfsv.values())
        print(f"  {K:5.0f} {levy[i]:7.3f} {gbm[:, i].mean():7.3f} {row} "
              f"{prem_fixed[0][i]:+7.3f}±{prem_fixed[1][i]:.3f} "
              f"{prem_matched[0][i]:+7.3f}±{prem_matched[1][i]:.3f}")
    gbm_se = gbm[:, 4].std(ddof=1) / np.sqrt(args.n_seeds)
    print(f"\n  ATM: GBM {gbm[:, 4].mean():.3f} ± {gbm_se:.3f}  vs Levy {levy[4]:.3f}")

    plot_validation(STRIKES, levy, gbm, rfsv, prem_fixed, prem_matched, nu_matched,
                    args.n_seeds, out_path)


if __name__ == "__main__":
    main()
