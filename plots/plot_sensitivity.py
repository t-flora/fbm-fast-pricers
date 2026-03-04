"""
Phase 3: Parameter Sensitivity Analysis.

Sweeps RFSV model parameters to show how Asian call prices depend on:
  - Hurst exponent H  (roughness of volatility)
  - vol-of-vol nu
  - Strike K (moneyness)

Figures produced:
  1. sensitivity_surface.png — 2-panel surface/heatmap: price vs (H, nu) at K=ATM
  2. sensitivity_strike.png  — price vs K for each H value

Usage:
    uv run python plots/plot_sensitivity.py [--M 3000] [--N 252]

Output:
    plots/sensitivity_surface.png
    plots/sensitivity_strike.png
"""

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.rfsv_model import price_asian_call

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

S0 = 100.0
T  = 1.0
R  = 0.0

H_GRID  = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.50])
NU_GRID = np.array([0.10, 0.20, 0.30, 0.40])
K_GRID  = np.array([80.0, 90.0, 100.0, 110.0, 120.0])


def run_grid(H_grid, nu_grid, K_atm, N, M):
    """Compute price grid over H × nu at fixed K."""
    grid = np.zeros((len(H_grid), len(nu_grid)))
    total = len(H_grid) * len(nu_grid)
    count = 0
    for i, H in enumerate(H_grid):
        for j, nu in enumerate(nu_grid):
            count += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid[i, j] = price_asian_call(H=H, nu=nu, K=K_atm, T=T, S0=S0, r=R,
                                              N=N, M=M, seed=100 * i + j)
            print(f"  [{count}/{total}] H={H:.2f}  nu={nu:.2f}  price={grid[i,j]:.3f}")
    return grid


def run_strike_sweep(H_grid, nu_fixed, K_grid, N, M):
    """Compute price vs K for each H value at fixed nu."""
    results = {}
    for i, H in enumerate(H_grid):
        prices = []
        for j, K in enumerate(K_grid):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = price_asian_call(H=H, nu=nu_fixed, K=K, T=T, S0=S0, r=R,
                                     N=N, M=M, seed=200 + 10 * i + j)
            prices.append(p)
        results[H] = np.array(prices)
    return results


def plot_surface(H_grid, nu_grid, price_grid, out_path, M):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.text(
        0.5, 1.01,
        f"ATM sensitivity  (K=S0=100, T=1yr, M={M:,} paths — MC std err "
        f"approx {35.0 / (M ** 0.5):.1f})",
        ha="center", fontsize=10, style="italic",
    )

    # ── Panel 1: Heatmap price vs (H, nu) via seaborn ────────────────────────
    ax = axes[0]
    # Build annotated DataFrame for seaborn.heatmap
    import pandas as pd
    df_heat = pd.DataFrame(
        price_grid,
        index=[f"{h:.2f}" for h in H_grid],
        columns=[f"{n:.2f}" for n in nu_grid],
    )
    sns.heatmap(df_heat, ax=ax, annot=True, fmt=".1f", cmap="viridis",
                cbar_kws={"label": "Asian call price"},
                linewidths=0.4, linecolor="white")
    ax.set_xlabel("Vol-of-vol  nu")
    ax.set_ylabel("Hurst exponent  H")
    ax.set_title("ATM Price vs (H, nu)\nLow H (rough) = higher price")
    # seaborn.heatmap inverts the y-axis by default; restore intuitive orientation
    ax.invert_yaxis()

    # ── Panel 2: Line plot price vs H for each nu ────────────────────────────
    ax2 = axes[1]
    cmap = plt.cm.plasma
    colors = [cmap(x) for x in np.linspace(0.2, 0.85, len(nu_grid))]
    for nu_val, color in zip(nu_grid, colors):
        nu_idx = list(nu_grid).index(nu_val)
        ax2.plot(H_grid, price_grid[:, nu_idx], marker="o", linewidth=2,
                 markersize=6, color=color, label=f"nu={nu_val:.2f}")

    ax2.set_xlabel("Hurst exponent H")
    ax2.set_ylabel("Asian call price")
    ax2.set_title("Price vs Roughness H\n(for each vol-of-vol nu)")
    ax2.legend(fontsize=9, title="nu")

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def plot_strike_sensitivity(H_grid, K_grid, results, nu_fixed, out_path):
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)

    # viridis: rough (low H) = dark purple, smooth (high H) = yellow — intuitive
    cmap = plt.cm.viridis
    colors = [cmap(x) for x in np.linspace(0.1, 0.9, len(H_grid))]

    all_prices = np.concatenate([results[H] for H in H_grid])
    y_lo = all_prices.min() * 0.97
    y_hi = all_prices.max() * 1.03

    for H, color in zip(H_grid, colors):
        label = f"H={H:.2f}"
        if H == 0.10:
            label += "*"
        elif H == 0.50:
            label += " (Brownian)"
        ax.plot(K_grid, results[H], marker="o", linewidth=2, markersize=6,
                color=color, label=label)

    ax.axvline(S0, color="gray", linestyle=":", linewidth=1.2, alpha=0.7,
               label=f"ATM  K={S0:.0f}")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Asian call price")
    ax.set_title(
        f"Asian Call Price vs Strike for Varying H\n"
        f"(nu={nu_fixed:.2f}, S0={S0:.0f}, T={T:.0f}yr;  * = calibrated)"
    )
    ax.set_ylim(y_lo, y_hi)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xticks(K_grid)

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=10_000,
                        help="Monte Carlo paths per grid point (default: 10000)")
    parser.add_argument("--N", type=int, default=252,
                        help="Time steps per year (default: 252)")
    args = parser.parse_args()

    out_dir = os.path.dirname(__file__)

    # ── Surface: H × nu at K=ATM ─────────────────────────────────────────────
    print(f"Computing H × nu price grid (K=ATM=100, M={args.M}, N={args.N}) ...")
    price_grid = run_grid(H_GRID, NU_GRID, K_atm=S0, N=args.N, M=args.M)

    print("\nPlotting surface ...")
    plot_surface(H_GRID, NU_GRID, price_grid,
                 os.path.join(out_dir, "sensitivity_surface.png"), M=args.M)

    # ── Strike sweep: K × H at nu=0.30 (calibrated) ─────────────────────────
    NU_FIXED = 0.30
    print(f"\nComputing K × H sweep (nu={NU_FIXED}, M={args.M}, N={args.N}) ...")
    strike_results = run_strike_sweep(H_GRID, NU_FIXED, K_GRID, args.N, args.M)

    print("\nPlotting strike sensitivity ...")
    plot_strike_sensitivity(H_GRID, K_GRID, strike_results, NU_FIXED,
                            os.path.join(out_dir, "sensitivity_strike.png"))

    # Print summary
    print("\nH × nu price grid (K=ATM):")
    print(f"  {'H\\nu':>6}", end="")
    for nu in NU_GRID:
        print(f"  {nu:>8.2f}", end="")
    print()
    for i, H in enumerate(H_GRID):
        print(f"  {H:>6.2f}", end="")
        for j in range(len(NU_GRID)):
            print(f"  {price_grid[i, j]:>8.3f}", end="")
        print()


if __name__ == "__main__":
    main()
