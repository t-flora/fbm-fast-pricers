"""
Phase 2: Asian Option Model Validation.

Compares RFSV Asian call prices against the Levy (1992) analytical approximation
for arithmetic Asian calls under constant-sigma GBM, across a range of strikes.

Two outputs:
  (a) Price vs Strike: Levy(sigma=1.0) + RFSV at H in {0.05, 0.10, 0.30, 0.50}
  (b) Roughness premium: RFSV(H=0.1) - RFSV(H=0.5) vs strike

Sanity checks:
  - At nu→0 (tiny), RFSV should converge to Levy with sigma=1.0
    (because sigma_t = exp(nu*W_t^H) → 1 as nu → 0)
  - RFSV(H=0.5) introduces stochastic vol (but smooth); prices > Levy due to
    Jensen's inequality / vol-of-vol convexity
  - RFSV(H=0.1) adds roughness; roughness premium concentrated near ATM

Usage:
    uv run python data/validate_asian.py [--M 5000] [--N 252]

Output:
    plots/validate_asian.png
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
    "H=0.05 (very rough)":  0.05,
    "H=0.10 (calibrated)":  0.10,
    "H=0.30 (moderately rough)": 0.30,
    "H=0.50 (smooth / Brownian)": 0.50,
}

H_COLORS = {
    "H=0.05 (very rough)":        "#8e44ad",
    "H=0.10 (calibrated)":        "#e74c3c",
    "H=0.30 (moderately rough)":  "#e67e22",
    "H=0.50 (smooth / Brownian)": "#27ae60",
}


def run_rfsv_sweep(Ks, H_values, nu, N, M):
    """Compute RFSV Asian call prices for each (H, K) combination."""
    results = {}
    for label, H in H_values.items():
        prices = []
        print(f"  {label} ...")
        for i, K in enumerate(Ks):
            p = price_asian_call(H=H, nu=nu, K=K, T=T, S0=S0, r=R,
                                 N=N, M=M, seed=42 + i)
            prices.append(p)
        results[label] = np.array(prices)
    return results


def run_levy_benchmark(Ks, sigma, N):
    """Levy analytical prices for arithmetic Asian call at constant sigma."""
    return np.array([levy_asian_call(S0, K, T, R, sigma, N) for K in Ks])


def plot_validation(Ks, levy_prices, rfsv_results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Panel (a): Price vs Strike ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(Ks, levy_prices, color="black", linestyle="--", linewidth=2,
            label="Levy approx. (sigma=1.0, constant vol)")

    for label, prices in rfsv_results.items():
        ax.plot(Ks, prices, color=H_COLORS[label], linewidth=2, marker="o",
                markersize=5, label=f"RFSV {label}")

    ax.axvline(S0, color="gray", linestyle=":", linewidth=1, alpha=0.6, label="ATM (K=S0)")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Asian Call Price")
    ax.set_title("Asian Call Price vs Strike\n(S0=100, T=1, nu=0.30)")
    ax.legend(fontsize=8, loc="upper right")

    # ── Panel (b): Roughness Premium ─────────────────────────────────────────
    ax2 = axes[1]
    label_h01 = "H=0.10 (calibrated)"
    label_h05 = "H=0.50 (smooth / Brownian)"

    if label_h01 in rfsv_results and label_h05 in rfsv_results:
        premium = rfsv_results[label_h01] - rfsv_results[label_h05]
        ax2.bar(Ks, premium, color="#e74c3c", alpha=0.75, edgecolor="black",
                linewidth=0.6, width=4.0)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Strike K")
        ax2.set_ylabel("Price difference")
        ax2.set_title("Roughness Premium\nRFSV(H=0.10) - RFSV(H=0.50)")
        for x, y in zip(Ks, premium):
            sign = "+" if y >= 0 else ""
            ax2.text(x, y + (0.02 if y >= 0 else -0.06), f"{sign}{y:.2f}",
                     ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def print_table(Ks, levy_prices, rfsv_results):
    print(f"\n  {'Strike':>8} {'Levy':>8}", end="")
    for label in rfsv_results:
        short = label.split("(")[0].strip()
        print(f" {short:>14}", end="")
    print()
    for i, K in enumerate(Ks):
        atm = " <ATM>" if K == S0 else ""
        print(f"  {K:>8.0f} {levy_prices[i]:>8.3f}", end="")
        for prices in rfsv_results.values():
            print(f" {prices[i]:>14.3f}", end="")
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

    print("\nComputing RFSV prices ...")
    rfsv_results = run_rfsv_sweep(STRIKES, H_VALUES, NU, args.N, args.M)

    print("\nPrice table:")
    print_table(STRIKES, levy_prices, rfsv_results)

    print("\nPlotting ...")
    plot_validation(STRIKES, levy_prices, rfsv_results, out_path)


if __name__ == "__main__":
    main()
