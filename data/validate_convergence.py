"""
MC Convergence Study: price vs number of paths M.

Answers: "How many paths are needed for the MC price to converge?"

Experiment design:
  Controlled:    N=252, H=0.10, nu=0.30, K=100, T=1, S0=100, r=0
  Independent:   M in {100, 250, 500, 1k, 2.5k, 5k, 10k, 25k}
  Dependent:     mean price and MC std-error across 5 independent seeds

Two panels:
  (a) Price +/- 1sigma vs M on log x-axis
      Overlay: reference price (horizontal line) and +/-2*sigma_payoff/sqrt(M) band
  (b) Log-log: MC std-error vs M; fitted slope (should be approx -0.5)

Output:
    plots/convergence.png

Usage:
    uv run python data/validate_convergence.py [--n-seeds 5] [--max-M 25000]

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

The Central Limit Theorem (CLT) guarantees that the Monte Carlo estimator

    p_hat = (1/M) * sum_{i=1}^{M} payoff_i

has standard error  sigma_payoff / sqrt(M),  regardless of dimension.
This "1/sqrt(M)" rate is the fundamental MC convergence law.

Why is sigma_payoff so large (~35)?
  Our RFSV model uses sigma_0 = exp(nu * W_0^H) = 1.0 (100% annualised vol).
  At 100% vol the stock path swings wildly, producing hugely variable payoffs.
  The mean still converges correctly — it just takes more paths to pin it down.

Estimating sigma_payoff:
  We don't know sigma_payoff analytically, so we back it out from the largest-M
  run using the CLT relation:
    std(seed_prices at M_max) = sigma_payoff / sqrt(M_max)
  => sigma_payoff_hat = std(seed_prices) * sqrt(M_max)
  This is then used to draw the theoretical confidence bands in panel (a).

Panel (b): why the fitted slope may differ from -0.5:
  With only 5 seeds, the standard deviation estimate itself has high variance,
  especially at small M (few samples from a heavy-tailed payoff distribution).
  This is expected and does not indicate a bug — it is a consequence of
  estimating a variance with 5 observations.  More seeds would tighten the fit.

Contested point: the reference price (p_ref = 23.58) comes from 500k-path C++
  Cholesky + FFT runs.  If the Python FFT engine had any normalization mismatch,
  the comparison would be unfair.  The close agreement validates both engines.
"""

import os
import sys
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.rfsv_model import price_asian_call


# ── Parameters ────────────────────────────────────────────────────────────────
H   = 0.10
NU  = 0.30
K   = 100.0
T   = 1.0
S0  = 100.0
R   = 0.0
N   = 252

M_VALUES = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000]
BASE_SEEDS = [42, 142, 242, 342, 442]


def _load_reference_price() -> float:
    """Read reference price from benchmarks/results/reference_price.txt, or use known value."""
    ref_path = os.path.join(os.path.dirname(__file__), "..",
                            "benchmarks", "results", "reference_price.txt")
    try:
        with open(ref_path) as f:
            for line in f:
                if line.startswith("reference_price="):
                    return float(line.split("=")[1])
    except FileNotFoundError:
        pass
    return 23.58   # known value: avg of 500k-path Cholesky + FFT runs


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Independent replications per M (default 5)")
    parser.add_argument("--max-M", type=int, default=25_000,
                        help="Maximum M to include (default 25000)")
    args = parser.parse_args()

    seeds = BASE_SEEDS[:args.n_seeds]
    m_values = [m for m in M_VALUES if m <= args.max_M]

    ref_price = _load_reference_price()
    print(f"Reference price: {ref_price:.4f}")
    print(f"Running {len(m_values)} M-values × {len(seeds)} seeds …\n")

    # ── Collect results ───────────────────────────────────────────────────────
    mean_prices = []
    std_prices  = []
    mean_times  = []

    for M in m_values:
        seed_prices = []
        seed_times  = []
        for s in seeds:
            t0 = time.perf_counter()
            p  = price_asian_call(H=H, nu=NU, K=K, T=T, S0=S0, r=R, N=N, M=M, seed=s)
            seed_times.append(time.perf_counter() - t0)
            seed_prices.append(p)

        mp  = np.mean(seed_prices)
        sp  = np.std(seed_prices, ddof=1)
        mt  = np.mean(seed_times)
        mean_prices.append(mp)
        std_prices.append(sp)
        mean_times.append(mt)
        print(f"  M={M:>6,d}  price={mp:.4f} ± {sp:.4f}  ({mt:.2f}s/run)")

    mean_prices = np.array(mean_prices)
    std_prices  = np.array(std_prices)
    mean_times  = np.array(mean_times)
    m_arr       = np.array(m_values, dtype=float)

    # Estimate sigma_payoff from the largest-M run.
    # CLT: std(seed_prices at M) = sigma_payoff / sqrt(M)
    # => sigma_payoff = std(seed_prices) * sqrt(M)
    # We use the largest M because it has the best-estimated std (most samples).
    sigma_payoff = std_prices[-1] * np.sqrt(m_arr[-1])
    print(f"\nEstimated sigma_payoff = {sigma_payoff:.4f}  (from M={m_values[-1]:,} run)")

    # Log-log fit: log(std) = intercept + slope * log(M)
    # Under CLT, slope = -0.5 exactly.  Deviations are due to noisy std estimates
    # (especially at small M where we have only n_seeds observations of the variance).
    log_m   = np.log10(m_arr)
    log_std = np.log10(np.maximum(std_prices, 1e-8))
    slope, intercept, r2_val, _, _ = linregress(log_m, log_std)
    r2 = r2_val ** 2  # linregress returns Pearson r, not R^2 -- must square it
    print(f"Log-log fit: slope = {slope:.3f}  (expected approx -0.50),  R^2 = {r2:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        f"MC Convergence: Asian Call Price vs Paths  "
        f"(N={N}, H={H}, $\\nu$={NU}, K={K}, T={T})",
        fontsize=11,
    )

    # ── Panel (a): price ± 1σ vs M ────────────────────────────────────────────
    band_m  = np.logspace(np.log10(m_arr[0]) - 0.1, np.log10(m_arr[-1]) + 0.1, 200)
    band_hi = ref_price + 2 * sigma_payoff / np.sqrt(band_m)
    band_lo = ref_price - 2 * sigma_payoff / np.sqrt(band_m)

    ax1.fill_between(band_m, band_lo, band_hi, alpha=0.15, color="C0",
                     label=r"$\pm 2\,\hat{\sigma}_{\rm payoff}/\sqrt{M}$ band")
    ax1.axhline(ref_price, color="k", ls="--", lw=1.2, label=f"Reference  {ref_price:.4f}")
    ax1.errorbar(m_arr, mean_prices, yerr=std_prices,
                 fmt="o-", color="C1", ms=5, capsize=4, lw=1.5,
                 label=r"RFSV price $\pm 1\sigma$ (5 seeds)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Paths $M$")
    ax1.set_ylabel("Asian call price")
    ax1.set_title("(a) Price converges to reference as $M$ grows")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    # ── Panel (b): log-log std vs M ───────────────────────────────────────────
    fit_line = 10 ** (intercept + slope * log_m)

    ax2.loglog(m_arr, std_prices, "o", color="C1", ms=6, label=r"empirical $\sigma(M)$")
    ax2.loglog(m_arr, fit_line, "--", color="C0", lw=1.5,
               label=f"fit: slope = {slope:.3f}  ($R^2={r2:.3f}$)")

    # Theoretical -0.5 reference line passing through last data point
    theory_line = std_prices[-1] * (m_arr[-1] / m_arr) ** 0.5
    ax2.loglog(m_arr, theory_line, ":", color="gray", lw=1.2,
               label=r"theory: slope $= -0.50$")

    ax2.set_xlabel("Paths $M$")
    ax2.set_ylabel(r"MC std-error  $\sigma(\hat{p}_M)$")
    ax2.set_title(
        f"(b) Log-log: $\\sigma \\propto M^{{-0.5}}$\n"
        f"Fitted slope = {slope:.3f},  $R^2$ = {r2:.3f}",
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "..", "plots", "convergence.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved  {os.path.normpath(out)}")
    plt.show()


if __name__ == "__main__":
    main()
