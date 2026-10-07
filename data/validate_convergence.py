"""
MC Convergence Study: price vs number of paths M, plain MC and control variate.

Answers: "How many paths are needed for the MC price to converge, and how many
fewer does the control variate need?"

Experiment design:
  Controlled:    N=252, H=0.10, nu=0.52, K=100, T=1, S0=100, r=0
  Independent:   M in {100, 250, 500, 1k, 2.5k, 5k, 10k, 25k}; estimator (plain MC
                 or the conditional geometric control variate)
  Dependent:     mean price and MC std-error across 20 independent seeds

Both estimators are computed from the SAME paths in every run
(price_asian_call_cv returns the plain estimate on its paths as price_plain), so
the comparison isolates the estimator.

Two panels:
  (a) Price +/- 1sigma vs M on log x-axis, plain and CV side by side
      Overlay: reference price (horizontal line) and +/-2*sigma/sqrt(M) bands
  (b) Log-log: MC std-error vs M for both; fitted slopes (should be approx -0.5)

Output:
    plots/figures/convergence.png

Usage:
    uv run python data/validate_convergence.py [--n-seeds 20] [--max-M 25000]

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

The Central Limit Theorem (CLT) guarantees that the Monte Carlo estimator

    p_hat = (1/M) * sum_{m=1}^{M} payoff_m

has standard error  sigma_payoff / sqrt(M),  regardless of dimension.
This "1/sqrt(M)" rate is the fundamental MC convergence law.

The control variate does not change the rate, only the constant:

    p_hat_CV = (1/M) * sum_m [V_m - beta (Y_m - E[Y | sigma]_m)]

where V is the arithmetic payoff, Y the geometric-average payoff and E[Y | sigma]
its closed form given the volatility path.  Its standard error is
sigma_CV / sqrt(M) with sigma_CV = sigma_payoff sqrt(1 - rho^2), rho = corr(V, C).
Both curves in panel (b) should have slope -0.5; the vertical gap between them is
log10 of the standard-error ratio (about 5, i.e. a variance reduction of ~25).

Estimating sigma_payoff and sigma_CV:
  We don't know them analytically, so we estimate each as the sample std of the
  10^6 per-path estimates of the reference run.  (Backing them out from the
  spread of n_seeds seed-level prices instead is far noisier: a std from 5
  samples has ~35% relative error.)  They draw the theoretical confidence bands
  in panel (a).

Why the pilot does not bias the CV:
  beta = Cov(V, C) / Var(C) is estimated on a separate pilot run (10% of M, at
  least 2,000 paths), independent of the M paths it is applied to, so
  E[p_hat_CV] = E[V] exactly at every M.  At small M the pilot is larger than the
  run itself; it is excluded from M, so the CV curve at M = 100 is slightly
  flattered relative to its cost.

Panel (b): why the fitted slope may differ from -0.5:
  With few seeds, the standard deviation estimate itself has high variance,
  especially at small M (few samples from a heavy-tailed payoff distribution).
  This is expected and does not indicate a bug — it is a consequence of
  estimating a variance from n_seeds observations.  More seeds would tighten the fit.

Contested point: the reference price comes from the same Python FFT engine on
  the same grid (N=252), with the control variate over 10^6 paths, so the
  comparison is free of discretization bias; agreement with the C++ reference
  (computed at N=500) is a separate cross-engine check.
"""

import os
import sys
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data.rfsv_model import price_asian_call_cv
from data.params import H, NU, K, T, S0, R


# ── Parameters ────────────────────────────────────────────────────────────────
N   = 252

M_VALUES = [100, 250, 500, 1_000, 2_500, 5_000, 10_000, 25_000]
SEED_BASE = 1000      # replication seeds are SEED_BASE + s
REF_SEED_BASE = 9000  # reference batches use their own seeds
REF_M, REF_BATCH = 1_000_000, 50_000


def _reference(n_paths: int = REF_M, batch: int = REF_BATCH) -> dict:
    """
    High-accuracy reference on the SAME grid as the study (N steps), from the same
    engine.  Returns, for the plain and the CV estimator over n_paths paths, the
    price, its standard error and the per-path std (sigma_payoff, sigma_cv).
    """
    plain, cv = [], []
    for b in range(n_paths // batch):
        res = price_asian_call_cv(H=H, nu=NU, K=K, T=T, S0=S0, r=R, N=N, M=batch,
                                  seed=REF_SEED_BASE + b, return_samples=True)
        plain.append(res["samples_plain"])
        cv.append(res["samples"])
    out = {}
    for name, v in [("plain", np.concatenate(plain)), ("cv", np.concatenate(cv))]:
        sigma = float(np.std(v, ddof=1))
        out[name] = dict(price=float(v.mean()), se=sigma / np.sqrt(v.size), sigma=sigma)
    return out


def _loglog_fit(m_arr, std):
    """Slope, intercept and R^2 of log10(std) vs log10(M)."""
    slope, intercept, r_val, _, _ = linregress(np.log10(m_arr), np.log10(np.maximum(std, 1e-12)))
    return slope, intercept, r_val ** 2  # linregress returns Pearson r, not R^2 -- square it


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

    ref = _reference()
    ref_price = ref["cv"]["price"]   # the more precise of the two
    print(f"Reference (N={N}, {REF_M:,} paths): CV {ref['cv']['price']:.4f} ± {ref['cv']['se']:.4f}"
          f"   plain {ref['plain']['price']:.4f} ± {ref['plain']['se']:.4f}")
    print(f"sigma_payoff = {ref['plain']['sigma']:.3f}, sigma_CV = {ref['cv']['sigma']:.3f}"
          f"  -> variance reduction {(ref['plain']['sigma'] / ref['cv']['sigma']) ** 2:.1f}x")
    print(f"Running {len(m_values)} M-values × {len(seeds)} seeds …\n")

    # ── Collect results ───────────────────────────────────────────────────────
    est = {"plain": dict(mean=[], std=[]), "cv": dict(mean=[], std=[])}
    mean_times = []

    for M in m_values:
        prices = {"plain": [], "cv": []}
        seed_times = []
        for s in seeds:
            t0 = time.perf_counter()
            res = price_asian_call_cv(H=H, nu=NU, K=K, T=T, S0=S0, r=R, N=N, M=M, seed=s)
            seed_times.append(time.perf_counter() - t0)
            prices["plain"].append(res["price_plain"])  # = price_asian_call(..., seed=s)
            prices["cv"].append(res["price"])
        for name, p in prices.items():
            est[name]["mean"].append(np.mean(p))
            est[name]["std"].append(np.std(p, ddof=1))
        mean_times.append(np.mean(seed_times))
        print(f"  M={M:>6,d}  plain {est['plain']['mean'][-1]:.4f} ± {est['plain']['std'][-1]:.4f}"
              f"   CV {est['cv']['mean'][-1]:.4f} ± {est['cv']['std'][-1]:.4f}"
              f"   ({mean_times[-1]:.2f}s/run incl. pilot)")

    m_arr = np.array(m_values, dtype=float)
    for name in est:
        est[name] = {k: np.array(v) for k, v in est[name].items()}
        # Log-log fit: log(std) = intercept + slope * log(M).  Under the CLT the slope is
        # -0.5 exactly; deviations come from noisy std estimates (n_seeds samples each).
        est[name]["fit"] = _loglog_fit(m_arr, est[name]["std"])
        slope, _, r2 = est[name]["fit"]
        print(f"Log-log fit ({name:>5}): slope = {slope:.3f}  (expected approx -0.50),  R^2 = {r2:.4f}")
    vr = (est["plain"]["std"] / est["cv"]["std"]) ** 2
    vr_ref = (ref["plain"]["sigma"] / ref["cv"]["sigma"]) ** 2
    print(f"Variance reduction from the seed spreads: {vr.min():.1f}x to {vr.max():.1f}x "
          f"(median {np.median(vr):.1f}x; per path, from the reference: {vr_ref:.1f}x)")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        f"MC convergence: Asian call price vs paths, plain MC and control variate  "
        f"(N={N}, H={H}, $\\nu$={NU}, K={K}, T={T})",
        fontsize=11,
    )
    style = {"plain": dict(color="C1", label="plain MC", dx=0.94, fmt="o-"),
             "cv": dict(color="C2", label="control variate", dx=1.06, fmt="s-")}

    # ── Panel (a): price ± 1σ vs M ────────────────────────────────────────────
    band_m = np.logspace(np.log10(m_arr[0]) - 0.1, np.log10(m_arr[-1]) + 0.1, 200)
    for name in ("plain", "cv"):
        st, sig = style[name], ref[name]["sigma"]
        ax1.fill_between(band_m, ref_price - 2 * sig / np.sqrt(band_m),
                         ref_price + 2 * sig / np.sqrt(band_m), alpha=0.13, color=st["color"],
                         label=rf"{st['label']}: $\pm 2\,\hat{{\sigma}}/\sqrt{{M}}$ "
                               rf"($\hat{{\sigma}}$={sig:.2f})")
    ax1.axhline(ref_price, color="k", ls="--", lw=1.2,
                label=f"Reference (same N, {REF_M:,} CV paths)  {ref_price:.3f}")
    for name in ("plain", "cv"):
        st = style[name]
        ax1.errorbar(m_arr * st["dx"], est[name]["mean"], yerr=est[name]["std"],
                     fmt=st["fmt"], color=st["color"], ms=5, capsize=4, lw=1.5,
                     label=rf"{st['label']} $\pm 1\sigma$ ({len(seeds)} seeds)")

    ax1.set_xscale("log")
    ax1.set_xlabel("Paths $M$")
    ax1.set_ylabel("Asian call price")
    ax1.set_title("(a) price converges to reference as $M$ grows")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.3)

    # ── Panel (b): log-log std vs M ───────────────────────────────────────────
    log_m = np.log10(m_arr)
    for name in ("plain", "cv"):
        st = style[name]
        slope, intercept, r2 = est[name]["fit"]
        ax2.loglog(m_arr, est[name]["std"], st["fmt"][0], color=st["color"], ms=6,
                   label=rf"{st['label']}: empirical $\sigma(M)$")
        ax2.loglog(m_arr, 10 ** (intercept + slope * log_m), "--", color=st["color"], lw=1.5,
                   label=f"fit: slope = {slope:.3f}  ($R^2={r2:.3f}$)")
        # Theoretical rate from the reference per-path std
        ax2.loglog(m_arr, ref[name]["sigma"] / np.sqrt(m_arr), ":", color="gray", lw=1.2,
                   label=r"theory $\hat{\sigma}/\sqrt{M}$" if name == "plain" else None)

    ax2.set_xlabel("Paths $M$")
    ax2.set_ylabel(r"MC std-error  $\sigma(\hat{p}_M)$")
    ax2.set_title(
        f"(b) log-log: $\\sigma \\propto M^{{-0.5}}$ for both\n"
        f"slopes {est['plain']['fit'][0]:.3f} (plain), {est['cv']['fit'][0]:.3f} (CV); "
        f"variance reduction {vr_ref:.1f}x (per path)",
    )
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)

    out = os.path.join(os.path.dirname(__file__), "..", "plots", "figures", "convergence.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nSaved  {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
