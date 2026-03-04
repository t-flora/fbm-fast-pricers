"""
Phase 3: Parameter Sensitivity Analysis.

Sweeps RFSV model parameters to show how Asian call prices depend on:
  - Hurst exponent H  (roughness of volatility)
  - vol-of-vol nu
  - Strike K (moneyness)

Figures produced:
  1. sensitivity_surface.png -- 2-panel: heatmap price vs (H, nu) at ATM, and
     line plot price vs H for each nu.
  2. sensitivity_strike.png  -- price vs K for each H value at fixed nu=0.30.

Usage:
    uv run python plots/plot_sensitivity.py [--M 10000] [--N 252]

Output:
    plots/sensitivity_surface.png
    plots/sensitivity_strike.png

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

Why does price increase as H decreases?

  This is the "roughness premium".  The RFSV model has sigma_t = exp(nu * W_t^H).
  Rougher processes (smaller H) produce more erratic volatility paths — lots of
  sharp spikes.  For Asian options, this matters because:
  (1) Jensen's inequality: E[f(sigma)] > f(E[sigma]) for convex f.
      Higher variability in sigma => higher expected payoff.
  (2) ATM options are most sensitive to vol changes (high gamma, high vega).
      The OTM/deep-ITM options are less affected (payoff is either always 0 or
      approximately linear in price, insensitive to vol).

Why does price increase with nu?

  nu = vol-of-vol.  Larger nu => sigma_t varies more around its mean of 1.0.
  Again by Jensen's inequality, more stochastic vol => higher option price.
  The effect is symmetric: nu enters as exp(nu * W), so doubling nu does not
  double the price, it amplifies in a nonlinear way.

The heatmap and ax.invert_yaxis():

  seaborn.heatmap() puts the first row of the DataFrame at the *top* of the y-axis
  (highest y position), while our H_grid goes from small to large (0.05 to 0.50).
  After seaborn draws the heatmap, the first row (H=0.05) appears at the top.
  We call ax.invert_yaxis() to flip it so H increases upward, matching the
  convention of H on the vertical axis increasing from bottom to top.

  Contested point: seaborn.heatmap's inversion behavior changed across versions.
  With annot=True, the annotations stay with their cells after inversion.
  A more robust alternative would be to build the DataFrame with reversed index.

The viridis colormap:

  We use viridis (dark purple = low price, bright yellow = high price) rather
  than coolwarm_r (which would make "rough" = red, "smooth" = blue — an
  arbitrary color assignment with confusing connotations of risk).  Viridis is
  perceptually uniform and colorblind-friendly.

Contested point: this script computes O(H * nu * K) = 6 * 4 + 6 * 5 = 54 MC runs
  at M=10000 paths each.  At ~0.2s per run (N=252, FFT Python engine), the total
  runtime is ~11 seconds.  This is manageable but limits the grid resolution.
  For a finer grid, the C++ benchmark should be used.
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
    # seaborn.heatmap places the first DataFrame row (H=0.05) at the TOP of the
    # y-axis, which reverses the natural ordering (H increases downward).
    # invert_yaxis() flips it so H increases upward, matching the line plot in
    # panel 2 and the reader's expectation ("rougher" = lower on the axis).
    # Note: the text annotations stay correctly aligned with their cells.
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
