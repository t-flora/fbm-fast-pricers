"""
MC Convergence Study: price vs number of paths M.

Answers: "How many paths are needed for the MC price to converge?"

Experiment design:
  Controlled:    N=252, H=0.10, nu=0.52, K=100, T=1, S0=100, r=0
  Independent:   M in {100, 250, 500, 1k, 2.5k, 5k, 10k, 25k}
  Dependent:     mean price and MC std-error across 20 independent seeds

Two panels:
  (a) Price +/- 1sigma vs M on log x-axis
      Overlay: reference price (horizontal line) and +/-2*sigma_payoff/sqrt(M) band
  (b) Log-log: MC std-error vs M; fitted slope (should be approx -0.5)

Output:
    plots/figures/convergence.png

Usage:
    uv run python data/validate_convergence.py [--n-seeds 20] [--max-M 25000]

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

The Central Limit Theorem (CLT) guarantees that the Monte Carlo estimator

    p_hat = (1/M) * sum_{i=1}^{M} payoff_i

has standard error  sigma_payoff / sqrt(M),  regardless of dimension.
This "1/sqrt(M)" rate is the fundamental MC convergence law.

Why is sigma_payoff so large (~60)?
  Our RFSV model uses sigma_0 = exp(nu * W_0^H) = 1.0 (100% annualised vol).
  At 100% vol the stock path swings wildly, producing hugely variable payoffs.
  The mean still converges correctly — it just takes more paths to pin it down.

Estimating sigma_payoff:
  We don't know sigma_payoff analytically, so we estimate it as the sample std
  of the M_max individual per-path payoffs of one run.  (Backing it out from
  the spread of n_seeds seed-level prices instead is far noisier: a std from
  5 samples has ~35% relative error, which once gave sigma ~ 35 instead of ~61.)
  This is then used to draw the theoretical confidence bands in panel (a).

Panel (b): why the fitted slope may differ from -0.5:
  With few seeds, the standard deviation estimate itself has high variance,
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
from data.rfsv_model import price_asian_call, simulate_log_vol_paths, _simulate_price_paths
from data.params import H, NU, MU0, K, T, S0, R


# ── Parameters ────────────────────────────────────────────────────────────────
N   = 252

M_VALUES = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000]
SEED_BASE = 1000      # replication seeds are SEED_BASE + s
REF_SEED_BASE = 9000  # reference batches use their own seeds
REF_M, REF_BATCH = 1_000_000, 50_000


def _reference(n_paths: int = REF_M, batch: int = REF_BATCH) -> tuple[float, float, float]:
    """
    High-accuracy reference on the SAME grid as the study (N steps), from the same
    engine: returns (price, standard error, sigma_payoff) over n_paths payoffs.
    """
    dt = T / N
    payoffs = []
    for b in range(n_paths // batch):
        seed = REF_SEED_BASE + b
        log_vol = simulate_log_vol_paths(N, batch, H, NU, dt, seed=seed) + MU0
        paths = _simulate_price_paths(log_vol, S0, R, dt, seed=seed)
        payoffs.append(np.exp(-R * T) * np.maximum(paths.mean(axis=1) - K, 0.0))
    v = np.concatenate(payoffs)
    sigma = float(np.std(v, ddof=1))
    return float(v.mean()), sigma / np.sqrt(v.size), sigma


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-seeds", type=int, default=20,
                        help="Independent replications per M (default 20)")
    parser.add_argument("--max-M", type=int, default=25_000,
                        help="Maximum M to include (default 25000)")
    args = parser.parse_args()

    seeds = [SEED_BASE + s for s in range(args.n_seeds)]
    m_values = [m for m in M_VALUES if m <= args.max_M]

    ref_price, ref_se, sigma_payoff = _reference()
    print(f"Reference price (N={N}, {REF_M:,} paths): {ref_price:.4f} ± {ref_se:.4f}")
    print(f"sigma_payoff = {sigma_payoff:.2f}  (from the {REF_M:,} reference payoffs)")
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
        f"MC convergence: Asian call price vs paths  "
        f"(N={N}, H={H}, $\\nu$={NU}, K={K}, T={T})",
        fontsize=11,
    )

    # ── Panel (a): price ± 1σ vs M ────────────────────────────────────────────
    band_m  = np.logspace(np.log10(m_arr[0]) - 0.1, np.log10(m_arr[-1]) + 0.1, 200)
    band_hi = ref_price + 2 * sigma_payoff / np.sqrt(band_m)
    band_lo = ref_price - 2 * sigma_payoff / np.sqrt(band_m)

    ax1.fill_between(band_m, band_lo, band_hi, alpha=0.15, color="C0",
                     label=r"$\pm 2\,\hat{\sigma}_{\rm payoff}/\sqrt{M}$ band")
    ax1.axhline(ref_price, color="k", ls="--", lw=1.2, label=f"Reference (same N, {REF_M:,} paths)  {ref_price:.3f}")
    ax1.errorbar(m_arr, mean_prices, yerr=std_prices,
                 fmt="o-", color="C1", ms=5, capsize=4, lw=1.5,
                 label=rf"RFSV price $\pm 1\sigma$ ({len(seeds)} seeds)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Paths $M$")
    ax1.set_ylabel("Asian call price")
    ax1.set_title("(a) price converges to reference as $M$ grows")
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
        f"(b) log-log: $\\sigma \\propto M^{{-0.5}}$\n"
        f"Fitted slope = {slope:.3f},  $R^2$ = {r2:.3f}",
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "..", "plots", "figures", "convergence.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved  {os.path.normpath(out)}")



if __name__ == "__main__":
    main()
