"""
Plots for the two extensions (TODO.md items 1-2), from ./build/extensions output.

  variance_corrected_rank.png
    (a) price error vs rank: plain vs variance-corrected low-rank sampler (+-2 SE band)
    (b) increment (fGn) variance error vs rank
    (c) lag-1 increment covariance error vs rank
  control_variate.png
    (a) variance reduction of the conditional geometric control variate, per sampler
    (b) time to reach a 0.1% relative standard error, plain vs control variate (N=1000)

Usage:
    ./build/extensions
    uv run python plots/plot_extensions.py

Output:
    plots/figures/variance_corrected_rank.png, plots/figures/control_variate.png
"""

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.25)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "benchmarks", "results")
OUT = os.path.join(HERE, "figures")

# Fixed method order and colors (Cholesky/FFT/rSVD match plot_scaling.py)
METHODS = ["cholesky", "fft", "rsvd", "rsvd_corrected"]
LABELS = {"cholesky": "Cholesky", "fft": "FFT", "rsvd": "rSVD k=32",
          "rsvd_corrected": "rSVD k=32 + diag"}
COLORS = {"cholesky": "#e74c3c", "fft": "#2ecc71", "rsvd": "#3498db",
          "rsvd_corrected": "#8e44ad"}
PLAIN, CORR = COLORS["rsvd"], COLORS["rsvd_corrected"]

SIGMA_PAYOFF = 9.5  # payoff std at sigma0 = 0.2, nu = 0.52 (validate_convergence.py)


def plot_variance_correction(df: pd.DataFrame, out_path: str) -> None:
    ks = df["rank_k"].values
    M = int(df["M_paths"].iloc[0])
    se_pct = SIGMA_PAYOFF / np.sqrt(M) / df["reference_price"].iloc[0] * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    fig.suptitle(f"Variance-corrected low-rank sampler  (N={int(df['N'].iloc[0])}, "
                 f"{M:,} paths per rank)", fontsize=12)

    ax = axes[0]
    ax.axhspan(-2 * se_pct, 2 * se_pct, color="gray", alpha=0.15, label=r"$\pm 2$ SE")
    ax.axhline(0, color="black", lw=0.8)
    ax.plot(ks, df["plain_rel_error"] * 100, "o-", color=PLAIN, lw=2, ms=8, label="low-rank")
    ax.plot(ks, df["corrected_rel_error"] * 100, "s-", color=CORR, lw=2, ms=8,
            label="low-rank + diag")
    ax.set_title("(a) price error vs reference")
    ax.set_ylabel("relative price error (%)")

    for ax, col, title in [
        (axes[1], "incr_var_err", "(b) increment variance error"),
        (axes[2], "incr_lag1_err", "(c) lag-1 increment covariance error"),
    ]:
        ax.semilogy(ks, df[f"plain_{col}"] * 100, "o-", color=PLAIN, lw=2, ms=8, label="low-rank")
        ax.semilogy(ks, df[f"corrected_{col}"] * 100, "s-", color=CORR, lw=2, ms=8,
                    label="low-rank + diag")
        ax.set_title(title)
        ax.set_ylabel("mean relative error (%)")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("rank k")
        ax.legend(fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_control_variate(df: pd.DataFrame, out_path: str) -> None:
    Ns = sorted(df["N"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    fig.suptitle(f"Conditional geometric control variate  (M={int(df['M_paths'].iloc[0]):,} paths)",
                 fontsize=12)

    # (a) variance reduction, grouped by N
    ax = axes[0]
    width = 0.8 / len(METHODS)
    x = np.arange(len(Ns))
    for i, m in enumerate(METHODS):
        vals = [df[(df.method == m) & (df.N == n)]["variance_reduction"].iloc[0] for n in Ns]
        bars = ax.bar(x + (i - 1.5) * width, vals, width * 0.92, color=COLORS[m], label=LABELS[m])
        ax.bar_label(bars, fmt="%.0f×", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("variance reduction  (SE plain / SE cv)$^2$")
    ax.set_title("(a) variance reduction")
    ax.set_ylim(0, df["variance_reduction"].max() * 1.35)
    ax.legend(fontsize=9, loc="upper center", ncol=len(METHODS))

    # (b) time to 0.1% relative SE at the largest N, plain vs CV
    ax = axes[1]
    n = Ns[-1]
    sub = df[df.N == n].set_index("method").loc[METHODS]
    xm = np.arange(len(METHODS))
    for dx, col, alpha, lab in [(-0.2, "time_to_target_plain_s", 0.45, "plain MC"),
                               (0.2, "time_to_target_cv_s", 1.0, "control variate")]:
        bars = ax.bar(xm + dx, sub[col], 0.38, color=[COLORS[m] for m in METHODS],
                      alpha=alpha, edgecolor="black", linewidth=0.6, label=lab)
        ax.bar_label(bars, fmt="%.1f s", fontsize=8, padding=2)
    ax.set_yscale("log")
    ax.set_xticks(xm)
    ax.set_xticklabels([LABELS[m] for m in METHODS])
    ax.set_ylabel("time to 0.1% relative SE (s)")
    ax.set_title(f"(b) time to 0.1% relative standard error  (N={n})\n"
                 "plain rSVD k=32 is biased by ~4%: its SE says nothing about accuracy")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    # legend entries for plain vs CV, independent of method color
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="gray", alpha=0.45, edgecolor="black", label="plain MC"),
                       Patch(facecolor="gray", alpha=1.0, edgecolor="black", label="control variate")],
              fontsize=9)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    plot_variance_correction(pd.read_csv(os.path.join(RESULTS, "variance_corrected_rank.csv")),
                             os.path.join(OUT, "variance_corrected_rank.png"))
    plot_control_variate(pd.read_csv(os.path.join(RESULTS, "control_variate.csv")),
                         os.path.join(OUT, "control_variate.png"))


if __name__ == "__main__":
    main()
