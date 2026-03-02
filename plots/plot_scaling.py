"""
Block 5: Generate scaling plots from benchmark CSVs.

Figures produced:
  1. time_vs_N.png    — wall-clock time vs N (log-log) with fitted c·N^α constants
  2. error_vs_rank.png — Frobenius and price error vs rSVD rank k (dual y-axis)

Usage:
    uv run python plots/plot_scaling.py [--results-dir benchmarks/results]
"""

import argparse
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")

METHOD_STYLE = {
    "cholesky": {"label": "Dense Cholesky",        "marker": "o", "color": "#e74c3c"},
    "fft":      {"label": "Circulant+FFT",          "marker": "s", "color": "#2ecc71"},
    "hmatrix":  {"label": "H-matrix+rSVD (k=32)",  "marker": "^", "color": "#3498db"},
}

# Theoretical exponents for reference lines
THEORY_EXP = {
    "cholesky": 3.0,
    "fft":      1.0,   # O(N log N) ≈ O(N) for this range
    "hmatrix":  1.0,   # construction O(Nk²) + per-path O(Nk)
}

def fit_power_law(Ns, times):
    """Fit t = c * N^alpha via OLS on log-log scale. Returns (c, alpha, R^2)."""
    log_N = np.log(Ns)
    log_t = np.log(times)
    slope, intercept, r, _, _ = stats.linregress(log_N, log_t)
    return np.exp(intercept), slope, r**2


def plot_time_vs_N(df: pd.DataFrame, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]  # wall-clock time (log-log)
    Ns_all = np.sort(df["N"].unique()).astype(float)
    N_range = np.logspace(np.log10(Ns_all.min()), np.log10(Ns_all.max()), 200)

    fit_results = {}
    for method, style in METHOD_STYLE.items():
        sub = df[df["method"] == method].sort_values("N")
        if sub.empty:
            continue
        Ns = sub["N"].values.astype(float)
        times = sub["wall_time_s"].values
        c, alpha, r2 = fit_power_law(Ns, times)
        fit_results[method] = (c, alpha, r2)
        label = f"{style['label']}  [fit: {c:.2e}·N^{{{alpha:.2f}}}]"
        ax.loglog(Ns, times, marker=style["marker"], color=style["color"],
                  label=label, linewidth=2, markersize=8)
        ax.loglog(N_range, c * N_range**alpha, linestyle="--",
                  color=style["color"], alpha=0.5, linewidth=1)

    ax.set_xlabel("Path resolution N")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Scaling: Wall-clock Time vs N")
    ax.legend(fontsize=8, loc="upper left")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Second panel: fitted exponents with error bars and theoretical lines
    ax2 = axes[1]
    methods = list(fit_results.keys())
    alphas  = [fit_results[m][1] for m in methods]
    colors  = [METHOD_STYLE[m]["color"] for m in methods]
    theory  = [THEORY_EXP.get(m, None) for m in methods]
    r2s     = [fit_results[m][2] for m in methods]
    labels  = [METHOD_STYLE[m]["label"] for m in methods]

    x = np.arange(len(methods))
    bars = ax2.bar(x, alphas, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    for xi, (th, r2) in enumerate(zip(theory, r2s)):
        if th is not None:
            ax2.axhline(th, xmin=(xi - 0.4) / len(methods),
                        xmax=(xi + 0.4) / len(methods),
                        color="black", linestyle=":", linewidth=1.5, alpha=0.7)
        ax2.text(xi, alphas[xi] + 0.05, f"R²={r2:.3f}", ha="center", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel("Fitted exponent α  (t ≈ c·N^α)")
    ax2.set_title("Fitted Complexity Exponents\n(dashed = theoretical)")
    ax2.set_ylim(0, max(alphas) * 1.3)

    # Annotate constants
    for bar, m in zip(bars, methods):
        c = fit_results[m][0]
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                 f"c={c:.2e}", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()

    # Print fit summary to stdout
    print("\n  Complexity constant estimates (t = c · N^α):")
    print(f"  {'Method':<20} {'c':>12} {'α':>8} {'R²':>8}")
    for m in methods:
        c, a, r2 = fit_results[m]
        print(f"  {METHOD_STYLE[m]['label']:<20} {c:>12.3e} {a:>8.3f} {r2:>8.4f}")


def plot_error_vs_rank(df: pd.DataFrame, out_path: str):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ks = df["rank_k"].values

    # Frobenius error (left axis, primary — noise-free metric)
    if "frob_error" in df.columns:
        ax1.semilogy(ks, df["frob_error"] * 100, marker="D", color="#9b59b6",
                     linewidth=2, markersize=7, label="Frobenius error ||C-Ck||/||C|| (%)")
        ax1.set_ylabel("Frobenius norm error (%)", color="#9b59b6")
        ax1.tick_params(axis="y", labelcolor="#9b59b6")

    # Price error (right axis)
    if "abs_price_error" in df.columns:
        ax2.plot(ks, df["abs_price_error"], marker="o", color="#e74c3c",
                 linewidth=2, markersize=7, linestyle="--",
                 label="|price − reference|")
        ax2.set_ylabel("|Price error|  (absolute)", color="#e74c3c")
        ax2.tick_params(axis="y", labelcolor="#e74c3c")

    # MC noise floor
    ax2.axhline(0.4, color="#e74c3c", linestyle=":", alpha=0.5, linewidth=1,
                label="MC noise floor (≈1σ, M=10k)")

    ax1.set_xlabel("rSVD target rank k")
    ax1.set_title(f"H-matrix Approximation Quality vs Rank\n"
                  f"(N={df['N'].iloc[0]}, reference = avg of 500k Cholesky + FFT paths)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__))

    time_path  = os.path.join(args.results_dir, "time_vs_N.csv")
    error_path = os.path.join(args.results_dir, "error_vs_rank.csv")

    if os.path.exists(time_path):
        print("Plotting time_vs_N ...")
        df_time = pd.read_csv(time_path)
        plot_time_vs_N(df_time, os.path.join(out_dir, "time_vs_N.png"))
    else:
        print(f"WARNING: {time_path} not found — run ./build/benchmark first.")

    if os.path.exists(error_path):
        print("\nPlotting error_vs_rank ...")
        df_err = pd.read_csv(error_path)
        plot_error_vs_rank(df_err, os.path.join(out_dir, "error_vs_rank.png"))
    else:
        print(f"WARNING: {error_path} not found — run ./build/benchmark first.")


if __name__ == "__main__":
    main()
