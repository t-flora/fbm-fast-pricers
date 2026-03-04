"""
Phase 2: Asian Option Model Validation.

Compares RFSV Asian call prices against the Levy (1992) analytical approximation
for arithmetic Asian calls under constant-sigma GBM, across a range of strikes.

Two outputs:
  (a) Price vs Strike: Levy(sigma=1.0) + RFSV at H in {0.05, 0.10, 0.30, 0.50}
  (b) Roughness premium: RFSV(H=0.1) - RFSV(H=0.5) vs strike

Sanity checks:
  - At nu -> 0 (tiny), RFSV should converge to Levy with sigma=1.0
    (because sigma_t = exp(nu*W_t^H) -> 1 as nu -> 0)
  - RFSV(H=0.5) introduces stochastic vol (but smooth); prices > Levy due to
    Jensen's inequality / vol-of-vol convexity
  - RFSV(H=0.1) adds roughness; roughness premium concentrated near ATM

Usage:
    uv run python data/validate_asian.py [--M 5000] [--N 252]

Output:
    plots/validate_asian.png

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

What is the Levy (1992) approximation?

  Turnbull & Wakeman (1991) and Levy (1992) derived a semi-analytical formula
  for the arithmetic Asian call under *constant* sigma (standard GBM), by
  matching the first two moments of the arithmetic average to a lognormal
  distribution.  This gives a closed-form price in terms of Black-Scholes
  inputs.  It is not exact but is accurate to < 1% for typical parameters.

  We use it as a benchmark because:
  (1) At nu = 0, RFSV reduces to constant-sigma GBM, so RFSV should match Levy.
  (2) Levy is fast (no MC noise), giving a clean baseline.

Why do RFSV prices exceed Levy (even at H=0.5)?

  The RFSV model has sigma_t = exp(nu * W_t^H), which is stochastic.
  By Jensen's inequality, E[sigma^2] > E[sigma]^2 for any random sigma.
  This "volatility convexity" makes options more expensive than under constant
  sigma.  The effect is larger for larger nu and larger H (smoother => slower
  mean reversion back to sigma_0, so sigma_t wanders further from 1.0).

What is the roughness premium?

  RFSV(H=0.1) - RFSV(H=0.5) measures the additional price attributable
  purely to roughness, holding nu fixed.  Rougher vol (H=0.1) produces sharper,
  shorter-lived volatility spikes.  For Asian options (which average over time),
  these spikes partially cancel in the payoff, but their presence still raises
  option prices near ATM where the payoff is most sensitive to vol fluctuations.

Multiple seeds and error bars:

  We run n_seeds=3 independent Monte Carlo runs per (H, K) and display ±1sigma
  error bars.  This is important: without error bars it is impossible to tell
  whether differences between methods are real or just MC noise.  At M=5000,
  sigma_MC ~ 35/sqrt(5000) ~ 0.5, so two prices must differ by > 1.0 to be
  meaningfully separated.

Contested point: Levy uses a lognormal approximation to the arithmetic average.
  This approximation breaks down for deep OTM/ITM options or very high sigma.
  We show results across K in {70..130} where the approximation holds well.
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.rfsv_model import price_asian_call, levy_asian_call

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# Baseline parameters matching src/common/params.hpp
S0 = 100.0
T  = 1.0
R  = 0.0
NU = 0.30    # vol-of-vol (calibrated)

# Strikes: OTM puts through ITM calls
STRIKES = np.array([70, 80, 90, 95, 100, 105, 110, 120, 130], dtype=float)

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

# * = calibrated value (H=0.10)


def run_rfsv_sweep(Ks, H_values, nu, N, M, n_seeds=3):
    """
    Compute RFSV Asian call prices for each (H, K) combination.

    Runs n_seeds independent replications per (H, K) to estimate MC std-error.
    Returns (means_dict, stds_dict); stds_dict values are None if n_seeds==1.
    """
    means = {}
    stds  = {}
    for label, H in H_values.items():
        print(f"  {label} ...")
        prices_per_K = []
        for i, K in enumerate(Ks):
            seed_prices = [
                price_asian_call(H=H, nu=nu, K=K, T=T, S0=S0, r=R,
                                 N=N, M=M, seed=42 + i + 100 * s)
                for s in range(n_seeds)
            ]
            prices_per_K.append(seed_prices)
        arr = np.array(prices_per_K)          # (len(Ks), n_seeds)
        means[label] = arr.mean(axis=1)
        stds[label]  = arr.std(axis=1, ddof=1) if n_seeds > 1 else None
    return means, stds


def run_levy_benchmark(Ks, sigma, N):
    """Levy analytical prices for arithmetic Asian call at constant sigma."""
    return np.array([levy_asian_call(S0, K, T, R, sigma, N) for K in Ks])


def plot_validation(Ks, levy_prices, rfsv_means, rfsv_stds, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # ── Panel (a): Price vs Strike ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(Ks, levy_prices, color="black", linestyle="--", linewidth=2,
            label="Levy (σ=1.0, constant vol)")

    for label, prices in rfsv_means.items():
        yerr = rfsv_stds.get(label)
        ax.errorbar(Ks, prices, yerr=yerr, color=H_COLORS[label],
                    linewidth=2, marker="o", markersize=5,
                    capsize=3, elinewidth=1, label=f"RFSV {label}")

    ax.axvline(S0, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="ATM (K=S0)")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Asian Call Price")
    ax.set_title("Asian Call Price vs Strike\n(S0=100, T=1, ν=0.30)")
    ax.legend(fontsize=8.5, loc="upper right")
    ax.annotate("* calibrated  † error bars = ±1σ MC (3 seeds)",
                xy=(0.01, 0.01), xycoords="axes fraction", fontsize=7.5, alpha=0.7)

    # ── Panel (b): Roughness Premium ─────────────────────────────────────────
    ax2 = axes[1]
    label_h01 = "H=0.10*"
    label_h05 = "H=0.50"

    if label_h01 in rfsv_means and label_h05 in rfsv_means:
        premium = rfsv_means[label_h01] - rfsv_means[label_h05]
        ax2.bar(Ks, premium, color="#e74c3c", alpha=0.75, edgecolor="black",
                linewidth=0.6, width=2.5)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Strike K")
        ax2.set_ylabel("Roughness premium (H=0.10 − H=0.50)")
        ax2.set_title("Roughness Premium\nRFSV(H=0.10) − RFSV(H=0.50)")

        # Annotation offset as fraction of y-range to avoid clipping
        y_vals = premium[premium != 0]
        y_range = premium.max() - premium.min() if len(y_vals) > 0 else 1.0
        offset = 0.03 * abs(y_range)
        for x, y in zip(Ks, premium):
            sign = "+" if y >= 0 else ""
            va_dir = "bottom" if y >= 0 else "top"
            y_txt  = y + offset if y >= 0 else y - offset
            ax2.text(x, y_txt, f"{sign}{y:.2f}",
                     ha="center", va=va_dir, fontsize=8)

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def print_table(Ks, levy_prices, rfsv_means, rfsv_stds):
    print(f"\n  {'Strike':>8} {'Levy':>8}", end="")
    for label in rfsv_means:
        print(f" {label:>14}", end="")
    print()
    for i, K in enumerate(Ks):
        atm = " <ATM>" if K == S0 else ""
        print(f"  {K:>8.0f} {levy_prices[i]:>8.3f}", end="")
        for label, prices in rfsv_means.items():
            std = rfsv_stds.get(label)
            std_str = f"±{std[i]:.2f}" if std is not None else ""
            print(f" {prices[i]:>7.3f}{std_str:>6}", end="")
        print(atm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=5000,
                        help="Monte Carlo paths (default: 5000)")
    parser.add_argument("--N", type=int, default=252,
                        help="Time steps per year (default: 252)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
    out_path = os.path.join(out_dir, "validate_asian.png")

    print(f"Asian option validation (M={args.M}, N={args.N}) ...")
    print("\nComputing Levy benchmark (sigma=1.0) ...")
    levy_prices = run_levy_benchmark(STRIKES, sigma=1.0, N=args.N)

    print("\nComputing RFSV prices (3 seeds for ±1σ error bars) ...")
    rfsv_means, rfsv_stds = run_rfsv_sweep(STRIKES, H_VALUES, NU, args.N, args.M)

    print("\nPrice table:")
    print_table(STRIKES, levy_prices, rfsv_means, rfsv_stds)

    print("\nPlotting ...")
    plot_validation(STRIKES, levy_prices, rfsv_means, rfsv_stds, out_path)


if __name__ == "__main__":
    main()
