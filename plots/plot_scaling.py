"""
Block 5: Generate scaling plots from benchmark CSVs.

Figures produced:
  1. time_vs_N.png    — wall-clock time vs N (log-log) with fitted c·N^α lines
  2. error_vs_rank.png — Frobenius and relative price error vs rSVD rank k
                         (shared log-scale y-axis, MC noise floor annotated)

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
    "rsvd":     {"label": "Low-rank rSVD (k=32)",   "marker": "^", "color": "#3498db"},
}

THEORY_EXP = {
    "cholesky": 2.0,   # O(M N^2) MC loop dominates the O(N^3) factorization at M = 10k
    "fft":      1.0,
    "rsvd":     1.0,
}


def fit_power_law(Ns, times):
    """Fit t = c * N^alpha via OLS on log-log scale. Returns (c, alpha, R^2)."""
    log_N = np.log(Ns)
    log_t = np.log(times)
    slope, intercept, r, _, _ = stats.linregress(log_N, log_t)
    return np.exp(intercept), slope, r ** 2


def plot_per_path_cost(df: pd.DataFrame, out_path: str):
    """
    Monte Carlo time per path, divided by N, vs N.  An O(N) method is flat, O(N log N)
    rises linearly on the log-x axis, and O(N^2) rises like N.  This separates the
    log N factor that a single power-law exponent hides.
    """
    if "mc_time_s" not in df.columns:
        print("  Skipping per-path cost plot (mc_time_s column missing).")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for method, style in METHOD_STYLE.items():
        sub = df[df["method"] == method].sort_values("N")
        if sub.empty:
            continue
        per_path_ns = sub["mc_time_s"].values / sub["M_paths"].values / sub["N"].values * 1e9
        ax.semilogx(sub["N"], per_path_ns, marker=style["marker"], color=style["color"],
                    linewidth=2, markersize=7, label=style["label"])
    ax.set_xlabel("Path resolution N")
    ax.set_ylabel("MC time per path per time step (ns)")
    ax.set_yscale("log")
    ax.set_title("Per-path cost normalized by N\n"
                 r"flat = $O(N)$,  rising linearly in $\log N$ = $O(N\log N)$,  slope 1 = $O(N^2)$")
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def plot_time_vs_N(df: pd.DataFrame, out_path: str):
    M_paths = int(df["M_paths"].iloc[0]) if "M_paths" in df.columns else 10_000
    Ns_all = np.sort(df["N"].unique()).astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.text(
        0.5, 1.01,
        f"Benchmark: M={M_paths:,} paths,  N in {{{', '.join(str(int(n)) for n in Ns_all)}}}",
        ha="center", fontsize=10, style="italic",
    )

    ax = axes[0]
    N_range = np.logspace(np.log10(Ns_all.min()), np.log10(Ns_all.max()), 200)

    fit_results = {}
    for method, style in METHOD_STYLE.items():
        sub = df[df["method"] == method].sort_values("N")
        if sub.empty:
            continue
        Ns    = sub["N"].values.astype(float)
        times = sub["wall_time_s"].values
        c, alpha, r2 = fit_power_law(Ns, times)
        fit_results[method] = (c, alpha, r2)

        # Data points
        ax.loglog(Ns, times, marker=style["marker"], color=style["color"],
                  label=style["label"], linewidth=2, markersize=8)
        # Fitted dashed line — fold R² into the legend label
        ax.loglog(N_range, c * N_range ** alpha, linestyle="--",
                  color=style["color"], alpha=0.65, linewidth=1.2,
                  label=f"  {c:.2e}$\\cdot N^{{{alpha:.2f}}}$  ($R^2$={r2:.3f})")

    ax.set_xlabel("Path resolution N")
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Scaling: wall-clock time vs N")
    ax.legend(fontsize=7.5, loc="upper left", ncol=1)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # ── Second panel: fitted exponents ────────────────────────────────────────
    ax2   = axes[1]
    methods = list(fit_results.keys())
    alphas  = [fit_results[m][1] for m in methods]
    colors  = [METHOD_STYLE[m]["color"] for m in methods]
    theory  = [THEORY_EXP.get(m, None) for m in methods]
    r2s     = [fit_results[m][2] for m in methods]
    labels  = [METHOD_STYLE[m]["label"] for m in methods]

    x    = np.arange(len(methods))
    bars = ax2.bar(x, alphas, color=colors, alpha=0.85, edgecolor="black", linewidth=0.8)
    for xi, (th, r2) in enumerate(zip(theory, r2s)):
        if th is not None:
            ax2.hlines(th, xi - 0.4, xi + 0.4,
                       color="black", linestyle=":", linewidth=1.5, alpha=0.7)
        ax2.text(xi, alphas[xi] + 0.05, f"$R^2$={r2:.3f}", ha="center", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right", fontsize=10)
    ax2.set_ylabel(r"Fitted exponent $\alpha$  ($t \approx c \cdot N^\alpha$)")
    ax2.set_title(
        "Fitted complexity exponents\n"
        r"(dotted = theoretical; note: Cholesky dominated by $O(M \cdot N^2)$ per-path cost)"
    )
    ax2.set_ylim(0, max(alphas + [t for t in theory if t is not None]) * 1.2)

    for bar, m in zip(bars, methods):
        c = fit_results[m][0]
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                 f"c={c:.2e}", ha="center", va="center",
                 color="white", fontsize=9, fontweight="bold")

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()

    print("\n  Complexity constant estimates (t = c · N^α):")
    print(f"  {'Method':<20} {'c':>12} {'α':>8} {'R²':>8}")
    for m in methods:
        c, a, r2 = fit_results[m]
        print(f"  {METHOD_STYLE[m]['label']:<20} {c:>12.3e} {a:>8.3f} {r2:>8.4f}")


def fbm_variance_lost(N: int, ks, H: float = 0.10, T: float = 1.0) -> np.ndarray:
    """
    Fraction of total path variance trace(C - C_k) / trace(C) discarded by the optimal
    rank-k truncation of the fBM covariance (exact eigendecomposition, noise-free).
    """
    t = np.arange(1, N + 1) * (T / N)
    C = 0.5 * (t[:, None] ** (2 * H) + t[None, :] ** (2 * H) - np.abs(t[:, None] - t[None, :]) ** (2 * H))
    w = np.linalg.eigvalsh(C)[::-1]
    return np.array([w[k:].sum() / w.sum() for k in ks])


# Paths per rank in the error experiment (M_ERR in benchmark.cpp) and the payoff std
# at sigma0 = 0.2, nu = 0.52, used for the price standard error
M_ERR = 100_000
SIGMA_PAYOFF = 9.5


def plot_error_vs_rank(df: pd.DataFrame, out_path: str):
    ref_price = df["reference_price"].iloc[0]
    N_val     = int(df["N"].iloc[0])
    ks        = df["rank_k"].values

    frob_pct  = df["frob_error"] * 100
    var_pct   = fbm_variance_lost(N_val, ks) * 100
    price_pct = (df["rsvd_price"] - ref_price) / ref_price * 100   # signed
    se_pct    = SIGMA_PAYOFF / np.sqrt(M_ERR) / ref_price * 100

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    fig.text(
        0.5, 1.01,
        f"N={N_val}, M={M_ERR:,} paths per rank,  reference = avg of 500k Cholesky + FFT paths",
        ha="center", fontsize=10, style="italic",
    )

    ax.semilogy(ks, frob_pct, marker="D", color="#9b59b6", linewidth=2, markersize=7,
                label=r"Frobenius  $\|C - C_k\|_F \,/\, \|C\|_F$")
    ax.semilogy(ks, var_pct, marker="s", color="#2980b9", linewidth=2, markersize=7,
                label=r"variance lost  ${\rm tr}(C - C_k) \,/\, {\rm tr}(C)$")
    ax.errorbar(ks, -price_pct, yerr=2 * se_pct, marker="o", color="#e74c3c",
                linewidth=2, markersize=7, linestyle="--", capsize=3,
                label=r"price underpricing  $(p_{\rm ref} - \hat{p}_k)/p_{\rm ref}$  ($\pm 2$ SE)")

    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("rSVD target rank k")
    ax.set_ylabel("Relative error (%)")
    ax.set_title(
        "Low-rank rSVD approximation quality vs rank\n"
        "price bias decays slowly, like the variance lost, not like the Frobenius error"
    )
    ax.legend(fontsize=8.5, loc="upper right")
    ax.grid(True, which="both", alpha=0.4)

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def plot_construction_breakdown(df: pd.DataFrame, out_path: str):
    """
    Stacked bar chart: construction (one-time setup) vs MC loop time.

    Separates the per-method one-time setup cost from the per-path Monte Carlo cost:
      - Cholesky:  covariance matrix build + LLT factorization (construction) vs M·N² MC loop
      - FFT:       fGn eigenvalue computation + FFTW plan creation  (negligible) vs M·N·logN MC
      - rSVD:      covariance build + rSVD (non-trivial)  vs  M·N·k MC loop

    Requires time_vs_N.csv to have construction_time_s and mc_time_s columns.
    """
    if "construction_time_s" not in df.columns or "mc_time_s" not in df.columns:
        print("  Skipping construction breakdown (CSV missing timing columns — rerun benchmark).")
        return

    M_paths = int(df["M_paths"].iloc[0]) if "M_paths" in df.columns else 10_000
    Ns      = np.sort(df["N"].unique()).astype(int)
    methods = [m for m in ["cholesky", "fft", "rsvd"] if m in df["method"].values]
    n_m     = len(methods)

    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    fig.text(
        0.5, 1.01,
        f"Construction vs MC time  (M={M_paths:,} paths)",
        ha="center", fontsize=10, style="italic",
    )

    bar_w = 0.25
    x     = np.arange(len(Ns))

    for mi, method in enumerate(methods):
        style = METHOD_STYLE[method]
        sub   = df[df["method"] == method].sort_values("N")
        tc    = sub["construction_time_s"].values
        tmc   = sub["mc_time_s"].values
        offset = (mi - n_m / 2.0 + 0.5) * bar_w

        # Construction (light)
        ax.bar(x + offset, tc, bar_w,
               color=style["color"], alpha=0.35, edgecolor="black", linewidth=0.5)
        # MC loop (dark)
        ax.bar(x + offset, tmc, bar_w, bottom=tc,
               color=style["color"], alpha=0.90, edgecolor="black", linewidth=0.5,
               label=style["label"])

        # Annotate construction % above each bar
        for i, (tc_i, tmc_i) in enumerate(zip(tc, tmc)):
            total  = tc_i + tmc_i
            pct    = tc_i / total * 100 if total > 0 else 0
            ax.text(x[i] + offset, total * 1.03, f"{pct:.1f}%",
                    ha="center", fontsize=7, color=style["color"],
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.85, edgecolor="none"))

    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in Ns])
    ax.set_ylabel("Wall-clock time (s)")
    ax.set_title(
        "One-time setup vs per-path Monte Carlo cost\n"
        "Light portion = construction;  dark = MC loop;  % = construction fraction",
        fontsize=10,
    )
    ax.legend(fontsize=9, title="Method")

    # Custom patch legend for light/dark
    from matplotlib.patches import Patch
    legend_patches = [
        Patch(facecolor="gray", alpha=0.35, edgecolor="black", label="construction (setup)"),
        Patch(facecolor="gray", alpha=0.90, edgecolor="black", label="MC loop (per-path)"),
    ]
    ax.legend(handles=legend_patches + [
        plt.Line2D([0], [0], color=METHOD_STYLE[m]["color"], lw=3, label=METHOD_STYLE[m]["label"])
        for m in methods
    ], fontsize=8, ncol=2)

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def plot_memory_vs_N(df: pd.DataFrame, out_path: str):
    """
    Two-panel memory plot from time_vs_N.csv:
      (a) theoretical peak bytes vs N (log-log) — Cholesky O(N^2), FFT O(N), rSVD variants
      (b) estimated memory bandwidth utilization for Cholesky (GB/s vs rated 100 GB/s)
    """
    L3_MB = 16.0  # Apple M2 P-cluster L2, the largest on-chip cache (M2 has no L3)
    BANDWIDTH_GBS = 100.0

    # Only rows with memory columns
    if "theoretical_peak_mb" not in df.columns:
        print("  Skipping memory plot (theoretical_peak_mb column missing).")
        return

    # Exclude rsvd_freed for the comparison panel (use theoretical_freed separately)
    methods_plot = ["cholesky", "fft", "rsvd", "rsvd_freed"]
    method_labels = {
        "cholesky":      r"Dense Cholesky  $O(N^2)$",
        "fft":           r"Circulant+FFT   $O(N)$",
        "rsvd":       r"rSVD (C held)   $O(N^2)$",
        "rsvd_freed": r"rSVD (C freed)  $O(Nk)$",
    }
    method_ls = {
        "cholesky":      "-",
        "fft":           "-",
        "rsvd":       "--",
        "rsvd_freed": ":",
    }

    Ns_all = np.sort(df["N"].unique()).astype(float)
    M_paths = int(df["M_paths"].iloc[0]) if "M_paths" in df.columns else 10_000

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.text(
        0.5, 1.01,
        f"Memory analysis  (M={M_paths:,} paths, largest cache={L3_MB:.0f} MB M2 L2, rated DRAM BW={BANDWIDTH_GBS:.0f} GB/s)",
        ha="center", fontsize=10, style="italic",
    )

    # ── Panel (a): theoretical peak MB vs N ──────────────────────────────────
    ax = axes[0]
    N_range = np.logspace(np.log10(Ns_all.min()), np.log10(Ns_all.max()), 200)

    for method in methods_plot:
        sub = df[df["method"] == method].sort_values("N")
        if sub.empty or "theoretical_peak_mb" not in sub.columns:
            continue
        color = METHOD_STYLE.get(method, {}).get("color", None)
        if color is None:
            color = "#95a5a6"  # gray for rsvd_freed
        ax.loglog(sub["N"], sub["theoretical_peak_mb"],
                  marker="o", markersize=7, linewidth=2,
                  linestyle=method_ls[method], color=color,
                  label=method_labels[method])
        if "measured_peak_mb" in sub.columns:
            # Lifetime peak measured in a forked child (includes allocator/GEMM workspace)
            meas = sub["measured_peak_mb"].clip(lower=1e-3)
            ax.loglog(sub["N"], meas, marker="D", markersize=6, linestyle="none",
                      markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.5)

    # L3 threshold line
    ax.axhline(L3_MB, color="#e74c3c", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"M2 L2 cache ({L3_MB:.0f} MB)")

    # O(N) and O(N^2) guide lines
    ax.loglog(N_range, 8e-6 * N_range, "lightgray", linewidth=1, linestyle="--")
    ax.loglog(N_range, 8e-6 * N_range ** 2, "lightgray", linewidth=1, linestyle=":")

    ax.set_xlabel("Path resolution N")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("Memory vs $N$: lines = dominant array, hollow = measured lifetime peak\n"
                 "(dashed = 16 MB cache; Cholesky and rSVD-held cross it at $N \\approx 1450$)")
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # ── Panel (b): Cholesky bandwidth utilization ─────────────────────────────
    ax2 = axes[1]
    chol = df[df["method"] == "cholesky"].sort_values("N")
    if not chol.empty and "est_bandwidth_GBs" in chol.columns:
        bw = chol["est_bandwidth_GBs"].values
        Ns = chol["N"].values
        ax2.bar(range(len(Ns)), bw, color=METHOD_STYLE["cholesky"]["color"],
                alpha=0.8, edgecolor="black", linewidth=0.8)
        ax2.axhline(BANDWIDTH_GBS, color="black", linestyle="--", linewidth=1.5,
                    label=f"Rated M2 bandwidth ({BANDWIDTH_GBS:.0f} GB/s)")
        for i, (n, b) in enumerate(zip(Ns, bw)):
            ax2.text(i, b + 1, f"{b:.0f}", ha="center", fontsize=10,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))
        ax2.set_xticks(range(len(Ns)))
        ax2.set_xticklabels([f"N={n}" for n in Ns])
        ax2.set_ylabel("Effective streaming rate of L (GB/s)")
        ax2.set_title(
            "Cholesky: effective rate of streaming $L$ (not measured DRAM traffic)\n"
            r"($4N(N{+}1) M$ bytes / wall time; $L$ fits in cache for $N \leq 1000$)"
        )
        ax2.legend(fontsize=9)
        ax2.set_ylim(0, BANDWIDTH_GBS * 1.15)
    else:
        ax2.text(0.5, 0.5, "est_bandwidth_GBs not available",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args()

    out_dir    = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(out_dir, exist_ok=True)
    time_path  = os.path.join(args.results_dir, "time_vs_N.csv")
    error_path = os.path.join(args.results_dir, "error_vs_rank.csv")

    if os.path.exists(time_path):
        print("Plotting time_vs_N ...")
        df_time = pd.read_csv(time_path)
        plot_time_vs_N(df_time, os.path.join(out_dir, "time_vs_N.png"))
        print("\nPlotting construction breakdown ...")
        plot_construction_breakdown(df_time, os.path.join(out_dir, "construction_breakdown.png"))
        print("\nPlotting per-path cost ...")
        plot_per_path_cost(df_time, os.path.join(out_dir, "per_path_cost.png"))
        if "theoretical_peak_mb" in df_time.columns:
            print("\nPlotting memory vs N ...")
            plot_memory_vs_N(df_time, os.path.join(out_dir, "memory_vs_N.png"))
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
