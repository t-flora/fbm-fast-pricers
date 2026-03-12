"""
Memory profiling for the Python RFSV engine.

Measures heap allocation (tracemalloc) in simulate_log_vol_paths across
a grid of (N, M) values. Compares measured bytes to the theoretical
O(M * N * 8) allocation for the full paths array.

Usage:
    uv run python data/profile_memory.py [--max-N 512] [--max-M 10000]

Output:
    plots/memory_profile.png
"""

import argparse
import os
import sys
import tracemalloc
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.rfsv_model import simulate_log_vol_paths

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

H    = 0.10
NU   = 0.30
T    = 1.0

# Hardware reference (Apple M2)
L3_MB = 16.0
L3_BYTES = L3_MB * 1024 * 1024


def measure_peak_bytes(N: int, M: int, seed: int = 42) -> tuple[float, float]:
    """
    Returns (peak_bytes, wall_time_s) for one call to simulate_log_vol_paths.
    Uses tracemalloc to capture the peak heap allocation inside the call.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    paths = simulate_log_vol_paths(N=N, M=M, H=H, nu=NU, dt=T/N, seed=seed)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del paths
    return float(peak), wall


def theoretical_bytes(N: int, M: int) -> float:
    """
    Dominant allocation: M paths × N time steps × 8 bytes (float64).
    The engine actually creates several such arrays (vol, prices, increments),
    so the true peak is ~ 3–5 × M × N × 8 bytes.
    """
    return float(M) * N * 8.0


def plot_memory_profile(results: dict, out_path: str):
    """
    Two-panel figure:
      (a) Peak heap bytes vs N (log-log) for fixed M, with O(N) / O(N^2) refs
      (b) Peak heap bytes vs M*N (total elements) — should be linear
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.text(
        0.5, 1.01,
        f"Python RFSV engine — tracemalloc peak heap  (H={H}, nu={NU}, T={T}yr)",
        ha="center", fontsize=10, style="italic",
    )

    cmap = plt.cm.plasma
    M_values = sorted({m for (_, m) in results})
    colors = {m: cmap(x) for m, x in zip(M_values,
              np.linspace(0.15, 0.85, len(M_values)))}

    # ── Panel (a): bytes vs N for each M ────────────────────────────────────
    ax = axes[0]
    N_values = sorted({n for (n, _) in results})

    for M in M_values:
        pts = [(n, results[(n, M)][0]) for n in N_values if (n, M) in results]
        if not pts:
            continue
        ns, bs = zip(*pts)
        ax.loglog(ns, bs, marker="o", linewidth=2, markersize=6,
                  color=colors[M], label=f"M={M:,}")

    # Reference lines
    n_ref = np.array(N_values, dtype=float)
    m_mid = M_values[len(M_values) // 2]
    ref_base = theoretical_bytes(N_values[0], m_mid)

    ax.loglog(n_ref, ref_base * (n_ref / N_values[0]),
              "k--", linewidth=1, alpha=0.5, label="O(N) ref")
    ax.loglog(n_ref, ref_base * (n_ref / N_values[0]) ** 2,
              "k:", linewidth=1, alpha=0.5, label="O(N²) ref")

    # Mark L3 spill threshold
    ax.axhline(L3_BYTES, color="#e74c3c", linestyle="--", linewidth=1.2,
               alpha=0.7, label=f"L3 cache ({L3_MB:.0f} MB)")

    ax.set_xlabel("Time steps N")
    ax.set_ylabel("Peak heap allocation (bytes)")
    ax.set_title("Memory vs N\n(log-log; dashed = L3 spill threshold)")
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f} MB" if x >= 1e6 else f"{x/1e3:.0f} KB"
    ))

    # ── Panel (b): bytes vs M*N (total elements) ─────────────────────────────
    ax2 = axes[1]
    xs, ys, cs = [], [], []
    for (N, M), (peak, _) in results.items():
        xs.append(float(N) * M)
        ys.append(peak)
        cs.append(colors[M])

    ax2.scatter(xs, ys, c=cs, s=50, zorder=3, edgecolors="none")
    # Fit line through origin
    mn = np.array(xs)
    pk = np.array(ys)
    slope = np.dot(mn, pk) / np.dot(mn, mn)  # OLS through origin
    x_line = np.linspace(0, max(xs), 100)
    ax2.plot(x_line, slope * x_line, "k--", linewidth=1.5, alpha=0.7,
             label=f"{slope:.1f} bytes/element")

    # Theoretical (1×)
    ax2.plot(x_line, 8.0 * x_line, "gray", linewidth=1, linestyle=":",
             label="8 bytes/element (1 array)")
    ax2.axhline(L3_BYTES, color="#e74c3c", linestyle="--", linewidth=1.2,
                alpha=0.7, label=f"L3 ({L3_MB:.0f} MB)")

    ax2.set_xlabel("Total elements  M × N")
    ax2.set_ylabel("Peak heap allocation (bytes)")
    ax2.set_title(
        "Memory vs total elements\n"
        f"Fitted slope = {slope:.1f} bytes/elem  "
        f"({slope/8:.1f}x float64 arrays)"
    )
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{x/1e6:.1f} MB" if x >= 1e6 else f"{x/1e3:.0f} KB"
    ))

    # Colorbar for M legend in scatter plot
    for M, c in colors.items():
        ax2.scatter([], [], c=[c], s=40, label=f"M={M:,}", edgecolors="none")
    ax2.legend(fontsize=7, ncol=2, loc="upper left")

    fig.savefig(out_path, dpi=150)
    print(f"\n  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-N", type=int, default=512,
                        help="Maximum N to sweep (default: 512)")
    parser.add_argument("--max-M", type=int, default=10_000,
                        help="Maximum M to sweep (default: 10000)")
    args = parser.parse_args()

    # Sweep grids — keep manageable (tracemalloc adds overhead)
    N_values = [n for n in [64, 128, 252, 512, 1024] if n <= args.max_N]
    M_values = [m for m in [100, 1_000, 5_000, 10_000] if m <= args.max_M]

    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "memory_profile.png")

    print(f"Profiling Python RFSV engine memory (H={H}, nu={NU})")
    print(f"N in {N_values}")
    print(f"M in {M_values}")
    print(f"L3 cache: {L3_MB:.0f} MB = {L3_BYTES:.0f} bytes\n")

    print(f"{'N':>6} {'M':>8} {'peak_MB':>10} {'theory_MB':>12} "
          f"{'ratio':>7} {'wall_s':>8} {'cache_pressure':>15}")
    print("-" * 72)

    results = {}
    for N in N_values:
        for M in M_values:
            peak, wall = measure_peak_bytes(N, M)
            theory = theoretical_bytes(N, M)
            ratio = peak / theory if theory > 0 else float("nan")
            pressure = peak / L3_BYTES
            results[(N, M)] = (peak, wall)
            print(f"{N:>6} {M:>8,} {peak/1e6:>10.2f} {theory/1e6:>12.2f} "
                  f"{ratio:>7.1f} {wall:>8.3f} {pressure:>15.2f}")

    print("\nPlotting ...")
    plot_memory_profile(results, out_path)

    # Summary
    print("\nKey insights:")
    for N in N_values:
        for M in M_values:
            peak, _ = results[(N, M)]
            theory = theoretical_bytes(N, M)
            n_arrays = peak / theory
            pressure = peak / L3_BYTES
            if pressure > 1.0:
                note = f"  *** L3 SPILL (pressure={pressure:.1f}x)"
            elif pressure > 0.5:
                note = f"  (half L3: {pressure:.2f}x)"
            else:
                note = ""
            print(f"  N={N:4d} M={M:6,d}: {peak/1e6:6.2f} MB peak "
                  f"= {n_arrays:.1f}x float64 arrays{note}")


if __name__ == "__main__":
    main()
