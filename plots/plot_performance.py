"""
Plots for the performance benchmark (./build/performance): batched sampling, the fast
Gaussian generator, and thread scaling (TODO.md items 4-6).

Figures produced (plots/figures/):
  perf_breakdown.png  — per-path cost split into random numbers, transform and price path,
                        for the original per-path loop vs batched B=64 with the fast generator
  perf_batched.png    — MC time per path vs N for the four configurations of each sampler,
                        and the local scaling exponent of Cholesky
  perf_batch_size.png — MC time per path vs block size B
  perf_threads.png    — speedup vs thread count, unbatched and batched, at two N

Usage:
    uv run python plots/plot_performance.py [--results-dir benchmarks/results]

GUIDE
-----
Configurations (perf_batched.csv): the generator and the batching are varied
independently, so the figure is a 2x2 design. Color encodes the generator (std =
mt19937 + normal_distribution, fast = xoshiro256++ + ziggurat), line style the batching
(dashed = one path, or one FFT pair, per block; solid = B = 64). "legacy" is the original
price_timed() loop: std generator, unbatched, with per-path allocations.

The local exponent d log t / d log N between neighbouring N separates the regimes that
a single power-law fit blends: Cholesky's per-path mat-vec is O(N^2) flops but streams
L from memory once per path, so its exponent rises above 2 once L leaves the 16 MB L2
(N ~ 1450). A blocked product reads L once per block, so the exponent should stay near 2.

Thread scaling: the M2 has 4 performance and 4 efficiency cores, so the speedup cannot
stay linear past 4 threads. Bandwidth-bound work (unbatched Cholesky at large N) should
flatten earlier, because extra threads compete for the same memory bus.
"""

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")

# Same method colors and markers as plot_scaling.py
METHOD_STYLE = {
    "cholesky": {"label": "Dense Cholesky", "marker": "o", "color": "#e74c3c"},
    "fft": {"label": "Circulant+FFT", "marker": "s", "color": "#2ecc71"},
    "rsvd": {"label": "Low-rank rSVD (k=32)", "marker": "^", "color": "#3498db"},
}
# Generator by color, batching by line style
RNG_COLOR = {"std": "#4a3aa7", "fast": "#eb6834"}
CONFIGS = {
    "legacy": {"label": "per-path loop (original), mt19937", "rng": "std", "ls": (0, (1, 1.5))},
    "unbatched_fast": {"label": "unbatched, ziggurat", "rng": "fast", "ls": "--"},
    "batched_std": {"label": "B = 64, mt19937", "rng": "std", "ls": "-"},
    "batched_fast": {"label": "B = 64, ziggurat", "rng": "fast", "ls": "-"},
}
STAGE_COLOR = {"rng": "#eda100", "transform": "#e87ba4", "payoff": "#008300"}


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_breakdown(df: pd.DataFrame):
    """Stacked per-path cost: original loop vs batched + fast generator, per N."""
    Ns = sorted(df["N"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.6 * len(Ns), 4.2), constrained_layout=True,
                             squeeze=False)
    print("\n  Per-path cost (us): original -> batched + fast generator")
    for ax, N in zip(axes[0], Ns):
        sub = df[df["N"] == N].set_index("method")
        rows, labels = [], []
        for method in METHOD_STYLE:
            if method not in sub.index:
                continue
            r = sub.loc[method]
            rows.append((r["rng_std_ns"], r["transform_unbatched_ns"], r["payoff_ns"]))
            rows.append((r["rng_fast_ns"], r["transform_batched_ns"], r["payoff_ns"]))
            labels += [f"{method}: per path", f"{method}: B=64, fast"]
            before, after = sum(rows[-2]), sum(rows[-1])
            print(f"    {method:<8} N={N:>4}: {before / 1e3:8.1f} -> {after / 1e3:7.1f}"
                  f"  ({before / after:.1f}x; rng {r['rng_std_ns'] / r['rng_fast_ns']:.1f}x,"
                  f" transform {r['transform_unbatched_ns'] / r['transform_batched_ns']:.1f}x)")
        rows = np.array(rows) / 1e3  # microseconds
        y = np.arange(len(rows))[::-1]
        left = np.zeros(len(rows))
        for j, (stage, name) in enumerate([("rng", "random numbers"), ("transform", "transform"),
                                           ("payoff", "price path + payoff")]):
            ax.barh(y, rows[:, j], left=left, height=0.7, color=STAGE_COLOR[stage],
                    edgecolor="white", linewidth=2, label=name)
            left += rows[:, j]
        for yi, total in zip(y, left):
            ax.text(total, yi, f" {total:.0f}", va="center", fontsize=9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, left.max() * 1.18)
        ax.set_xlabel("time per path (µs)")
        ax.set_title(f"N = {N}")
        ax.grid(axis="y", visible=False)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Per-path cost: original loop vs B = 64 with the ziggurat")
    save(fig, "perf_breakdown.png")


def plot_batched(df: pd.DataFrame):
    """MC time per path vs N for each configuration; Cholesky local exponents."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
    M = int(df["M_paths"].iloc[0])
    for ax, method in zip(axes.flat, METHOD_STYLE):
        sub = df[df["method"] == method]
        for cfg, style in CONFIGS.items():
            c = sub[sub["config"] == cfg].sort_values("N")
            if c.empty:
                continue
            per_path = c["mc_time_s"].values / c["M_paths"].values * 1e6
            yerr = [per_path - c["mc_time_q1_s"].values / c["M_paths"].values * 1e6,
                    c["mc_time_q3_s"].values / c["M_paths"].values * 1e6 - per_path]
            ax.errorbar(c["N"], per_path, yerr=yerr, color=RNG_COLOR[style["rng"]], ls=style["ls"],
                        marker=METHOD_STYLE[method]["marker"], markersize=6, linewidth=2,
                        capsize=2, label=style["label"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Path resolution N")
        ax.set_ylabel("MC time per path (µs)")
        ax.set_title(METHOD_STYLE[method]["label"])
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[0][0].legend(fontsize=8.5, loc="upper left")

    # Local exponent of the Cholesky MC time between neighbouring N
    ax = axes[1][1]
    sub = df[df["method"] == "cholesky"]
    print("\n  Cholesky local exponent d log t / d log N (last interval)")
    for cfg, style in CONFIGS.items():
        c = sub[sub["config"] == cfg].sort_values("N")
        if len(c) < 2:
            continue
        logN, logt = np.log(c["N"].values), np.log(c["mc_time_s"].values)
        slope = np.diff(logt) / np.diff(logN)
        mid = np.exp(0.5 * (logN[1:] + logN[:-1]))
        ax.plot(mid, slope, color=RNG_COLOR[style["rng"]], ls=style["ls"], marker="o",
                markersize=6, linewidth=2, label=style["label"])
        print(f"    {cfg:<15} {slope[-1]:.2f}  (N {int(c['N'].values[-2])} -> {int(c['N'].values[-1])})")
    ax.axhline(2.0, color="black", ls=":", linewidth=1.2)
    ax.text(0.02, 2.03, r"$O(N^2)$ flops", fontsize=9, transform=ax.get_yaxis_transform())
    N_l2 = np.sqrt(16 * 2**20 / 8)  # 8 N^2 bytes = 16 MB
    ax.axvline(N_l2, color="gray", ls="--", linewidth=1)
    ax.text(N_l2 * 1.04, 0.04, "L fills\n16 MB L2", fontsize=8, color="gray",
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("N (geometric midpoint of each interval)")
    ax.set_ylabel(r"local exponent  $d\log t / d\log N$")
    ax.set_title("Dense Cholesky: local scaling exponent")
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    fig.suptitle(f"Batching and the fast generator, one thread (M = {M:,}; bars = IQR of repeats)")
    save(fig, "perf_batched.png")

    print("\n  Speedup over the original loop (one thread)")
    for method in METHOD_STYLE:
        sub = df[df["method"] == method].pivot(index="N", columns="config", values="mc_time_s")
        for N, row in sub.iterrows():
            print(f"    {method:<8} N={N:>4}: unbatched+fast {row['legacy'] / row['unbatched_fast']:4.1f}x"
                  f"  B=64+std {row['legacy'] / row['batched_std']:4.1f}x"
                  f"  B=64+fast {row['legacy'] / row['batched_fast']:4.1f}x")


def plot_batch_size(df: pd.DataFrame):
    Ns = sorted(df["N"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(5.5 * len(Ns), 4.2), constrained_layout=True,
                             squeeze=False)
    for ax, N in zip(axes[0], Ns):
        for method, style in METHOD_STYLE.items():
            c = df[(df["N"] == N) & (df["method"] == method)].sort_values("batch")
            if c.empty:
                continue
            ax.plot(c["batch"], c["mc_time_s"] / c["M_paths"] * 1e6, marker=style["marker"],
                    color=style["color"], linewidth=2, markersize=7, label=style["label"])
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("block size B (paths per block)")
        ax.set_ylabel("MC time per path (µs)")
        ax.set_title(f"N = {N}")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    axes[0][0].legend(fontsize=9)
    fig.suptitle("MC time per path vs block size (fast generator, one thread)")
    save(fig, "perf_batch_size.png")


def plot_threads(df: pd.DataFrame):
    Ns = sorted(df["N"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(5.5 * len(Ns), 4.5), constrained_layout=True,
                             squeeze=False, sharey=True)
    max_t = int(df["threads"].max())
    print("\n  Speedup at the largest thread count / at 4 threads")
    for ax, N in zip(axes[0], Ns):
        ts = np.arange(1, max_t + 1)
        ax.plot(ts, ts, color="black", ls=":", linewidth=1.2, label="linear")
        if max_t > 4:
            ax.axvspan(4.5, max_t + 0.5, color="gray", alpha=0.12, linewidth=0)
            ax.text(4.6, 0.6, "efficiency cores", fontsize=8, color="dimgray")
        for method, style in METHOD_STYLE.items():
            sub = df[(df["N"] == N) & (df["method"] == method)]
            for B in sorted(sub["batch"].unique()):
                c = sub[sub["batch"] == B].sort_values("threads")
                t1 = c[c["threads"] == 1]["mc_time_s"].values[0]
                speedup = t1 / c["mc_time_s"].values
                batched = B > 2
                ax.plot(c["threads"], speedup, color=style["color"], marker=style["marker"],
                        ls="-" if batched else "--", linewidth=2, markersize=6,
                        label=f"{method}, {'B = 64' if batched else 'unbatched'}")
                at4 = speedup[c["threads"].values == 4]
                print(f"    {method:<8} N={N:>4} B={B:>2}: {speedup[-1]:.2f}x at {int(c['threads'].max())}"
                      + (f", {at4[0]:.2f}x at 4" if len(at4) else ""))
        ax.set_xlabel("threads")
        ax.set_title(f"N = {N}")
        ax.set_xticks(np.arange(1, max_t + 1))
        ax.set_ylim(0, max_t + 0.5)
    axes[0][0].set_ylabel("speedup over one thread")
    axes[0][-1].legend(fontsize=8, loc="upper left")
    fig.suptitle("Thread scaling (fast generator; M2: 4 performance + 4 efficiency cores)")
    save(fig, "perf_threads.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=RESULTS_DIR)
    args = ap.parse_args()
    for name, fn in [("perf_breakdown.csv", plot_breakdown), ("perf_batched.csv", plot_batched),
                     ("perf_batch_size.csv", plot_batch_size), ("perf_threads.csv", plot_threads)]:
        path = os.path.join(args.results_dir, name)
        if not os.path.exists(path):
            print(f"  Skipping {name} (not found; run ./build/performance)")
            continue
        fn(pd.read_csv(path))


if __name__ == "__main__":
    main()
