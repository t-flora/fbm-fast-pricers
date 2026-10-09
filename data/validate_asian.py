"""
Phase 2: Asian Option Model Validation.

Compares RFSV Asian call prices with constant-volatility baselines across strikes,
and measures the roughness premium with and without variance normalization.

Outputs (plots/figures/validate_asian.png, plots/figures/roughness_share.csv):
  (a) Price vs strike: Levy (sigma=SIGMA0, approximate), exact GBM by MC (nu=0), and
      RFSV at H in {0.05, 0.10, 0.30, 0.50} with nu = NU
  (b) Roughness premium p(H=0.10) - p(H=0.50) vs strike, four ways:
        fixed nu         -- both at nu = NU (mixes roughness with total variance)
        E[int sigma^2]   -- nu at H=0.50 chosen so E[int_0^T sigma_t^2 dt] equals
                            its value at (H=0.10, nu=NU)
        int Var(log sig) -- nu at H=0.50 matching int_0^T Var(log sigma_t) dt
                            (closed form)
        ATM implied vol  -- nu at H=0.50 matching the European ATM price (hence the
                            ATM implied volatility) at maturity T
      The legend gives each normalization's roughness share: the fraction of the
      fixed-nu ATM premium that survives the normalization.

Every price uses the conditional geometric control variate (price_asian_call_cv),
every configuration reuses the same seeds (common random numbers), and each
simulated path is priced at every strike, so differences across H and K are
precise; premium error bars are the paired standard error across seeds.

Usage:
    uv run python data/validate_asian.py [--M 10000] [--N 252] [--n-seeds 10]

Output:
    plots/figures/validate_asian.png
    plots/figures/roughness_share.csv   (one row per normalization)

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

What is the Levy (1992) approximation?

  Turnbull & Wakeman (1991) and Levy (1992) derived a semi-analytical formula
  for the arithmetic Asian call under *constant* sigma (standard GBM), by
  matching the first two moments of the arithmetic average to a lognormal
  distribution.  This gives a closed-form price in terms of Black-Scholes
  inputs.  It is accurate to < 1% at typical equity vols (sigma ~ 0.2, the
  sigma_0 used here), but not at high vol: at sigma = 1 it overprices exact
  discrete GBM by ~6%.  A nu = 0 MC run is therefore also shown as the exact
  GBM baseline.

  We use it as a benchmark because:
  (1) At nu = 0, RFSV reduces to constant-sigma GBM (sigma = SIGMA0), which Levy
      approximates.
  (2) Levy is fast (no MC noise), giving a clean baseline.

Why do RFSV prices exceed Levy (even at H=0.5)?

  The RFSV model has sigma_t = sigma_0 exp(nu * W_t^H), which is stochastic.
  By Jensen's inequality, E[sigma^2] > E[sigma]^2 for any random sigma.
  This "volatility convexity" makes options more expensive than under constant
  sigma.  The effect grows with Var(log sigma_t) = nu^2 t^{2H}.  Note that for
  t < 1 year, t^{2H} is LARGER for small H, so lowering H at fixed nu also
  raises the total variance of log-vol, not just its roughness.

What is the roughness premium, and why normalize?

  RFSV(H=0.1) - RFSV(H=0.5).  At fixed nu this is NOT a pure roughness effect:
  Var(nu W_t^H) = nu^2 t^{2H}, so the two models also differ in how much log-vol
  varies on [0, T].  A normalization raises nu at H = 0.5 until some measure of
  "how much volatility" agrees, and what remains of the premium is attributed to
  path regularity.  The roughness share is

      share = premium(normalized) / premium(fixed nu)    at K = S0.

  Which measure to hold fixed is a modelling choice, so three are reported:

  (1) E[int_0^T sigma_t^2 dt] = sigma_0^2 int_0^T exp(2 nu^2 t^{2H}) dt: the expected
      total variance of returns (solved for nu by brentq on the 1-D integral).
  (2) int_0^T Var(log sigma_t) dt = nu^2 T^{2H+1} / (2H+1): the integrated
      log-vol variance, which gives nu(0.5) = nu sqrt((2*0.5+1)/(2H+1) T^{2H-1})
      in closed form (0.671 from (0.10, 0.52) at T = 1).  It ignores the lognormal
      convexity, so it asks for slightly more vol-of-vol than (1).
  (3) The European ATM implied volatility at maturity T, the quantity a trader
      would hold fixed.  There is no closed form, so it is matched by MC root-
      finding, made precise by two choices:
        - conditional (Hull-White mixing) MC: with rho = 0, log S_T given the vol
          path is Gaussian with total variance sum_j sigma_j^2 dt, so
          E[(S_T - K)^+ | sigma] is the Black-Scholes price at the path's RMS vol,
          exact for the discrete scheme.  This removes all price-shock noise.
        - common random numbers: the vol paths at H = 0.5 are generated once per
          seed and only rescaled by nu, so the MC price is a smooth, increasing
          function of nu and brentq converges cleanly.
      The calibration uses its own seeds (not the pricing seeds), so the matched nu
      is a constant independent of the paths that price the premium.  Its residual
      MC error (delta method on the paired per-path differences) is printed; it is
      ~1e-3, negligible next to the spread between normalizations.

Common random numbers and error bars:

  Every configuration (each H, nu) reuses seeds SEED_BASE + s (and the matching
  pilot seeds of the control variate), and each set of simulated paths is priced
  at every strike.  Differences between configurations are then far less noisy
  than independent runs would give.  The premium error bars are the standard
  error of the per-seed differences of the CV estimates (paired), which is the
  honest uncertainty of a CRN difference; each CV estimate is exactly unbiased
  because its beta comes from a separate pilot run.  The share's SE uses the
  delta method on the per-seed (normalized, fixed) premium pairs.

The control variate:

  price_asian_call_cv subtracts beta (Y - E[Y | sigma]), with Y the geometric-average
  payoff and E[Y | sigma] its closed form given the vol path.  It cancels the
  price-shock noise; the variance reduction printed at the end (about 25x at the
  money) is per path, so 10 seeds of 10,000 CV paths are worth ~2.5M plain paths.

Contested point: Levy uses a lognormal approximation to the arithmetic average.
  It is accurate at sigma = 0.2 but not at high vol, so panel (a) also shows exact
  GBM by MC.
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm as sp_norm
from data.rfsv_model import (levy_asian_call, simulate_log_vol_paths, price_asian_call_cv,
                             bs_implied_vol)
from data.params import NU, SIGMA0, MU0, S0, T, R

sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)

# Baseline parameters matching src/common/params.hpp

# Strikes: ITM calls (K=80) through OTM calls (K=120)
STRIKES = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120], dtype=float)
ATM = int(np.argmin(np.abs(STRIKES - S0)))

# H values to sweep
H_VALUES = {
    "H=0.05": 0.05,
    "H=0.10*": 0.10,
    "H=0.30": 0.30,
    "H=0.50": 0.50,
}

H_COLORS = {
    "H=0.05":  "#8e44ad",
    "H=0.10*": "#e74c3c",
    "H=0.30":  "#e67e22",
    "H=0.50":  "#27ae60",
}

# * = model value (H=0.10)

SEED_BASE = 42       # seed s of every configuration is SEED_BASE + s (common random numbers)
CALIB_SEED_BASE = 7000  # separate seeds for the ATM-implied-vol calibration of nu


def integrated_variance(H, nu, T=T):
    """E[int_0^T sigma_t^2 dt] / sigma_0^2 = int_0^T exp(2 nu^2 t^{2H}) dt."""
    return quad(lambda t: np.exp(2.0 * nu ** 2 * t ** (2.0 * H)), 0.0, T)[0]


def variance_matched_nu(H, H_ref=0.10, nu_ref=NU):
    """nu at Hurst H giving the same expected integrated variance as (H_ref, nu_ref)."""
    target = integrated_variance(H_ref, nu_ref)
    return brentq(lambda nu: integrated_variance(H, nu) - target, 1e-6, 5.0)


def logvar_matched_nu(H, H_ref=0.10, nu_ref=NU, T=T):
    """nu at Hurst H with the same int_0^T Var(log sigma_t) dt = nu^2 T^{2H+1}/(2H+1)."""
    return nu_ref * np.sqrt((2 * H + 1) / (2 * H_ref + 1) * T ** (2 * (H_ref - H)))


def _bs_call(S, K, T, r, sigma):
    """Black-Scholes call, vectorized over sigma > 0."""
    sd = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + r * T) / sd + 0.5 * sd
    return S * sp_norm.cdf(d1) - K * np.exp(-r * T) * sp_norm.cdf(d1 - sd)


def european_atm_paths(H, nu, N, M, n_seeds, seed_base=CALIB_SEED_BASE):
    """
    Per-path conditional ATM European prices E[e^{-rT}(S_T - S0)^+ | sigma], pooled over
    n_seeds seeds: Black-Scholes at the path's RMS vol sqrt(mean_j sigma_j^2), which is
    exact for the discrete scheme (rho = 0).  No price shocks are drawn.
    """
    dt = T / N
    out = []
    for s in range(n_seeds):
        log_vol = simulate_log_vol_paths(N, M, H, nu, dt, seed=seed_base + s) + MU0
        out.append(_bs_call(S0, S0, T, R, np.sqrt(np.exp(2.0 * log_vol).mean(axis=1))))
    return np.concatenate(out)


def atm_iv_matched_nu(N, M, n_seeds, H=0.50, H_ref=0.10, nu_ref=NU):
    """
    nu at Hurst H whose European ATM price (so ATM implied vol) equals that of
    (H_ref, nu_ref), by brentq on conditional-MC prices with common random numbers.
    Returns (nu, se_nu, atm_iv_ref).
    """
    ref = european_atm_paths(H_ref, nu_ref, N, M, n_seeds)

    def gap(nu):
        return european_atm_paths(H, nu, N, M, n_seeds).mean() - ref.mean()

    nu_star = brentq(gap, 1e-3, 2.0, xtol=1e-4)
    # Delta method: SE of the paired price difference divided by the slope d(price)/d(nu)
    diff = european_atm_paths(H, nu_star, N, M, n_seeds) - ref
    h = 0.01
    slope = (gap(nu_star + h) - gap(nu_star - h)) / (2 * h)
    se_nu = diff.std(ddof=1) / np.sqrt(diff.size) / slope
    return nu_star, se_nu, bs_implied_vol(ref.mean(), S0, S0, T, R)


def run_config(H, nu, Ks, N, M, n_seeds):
    """
    CV prices at every strike in Ks, one path set per seed; seed s is shared across all
    configurations.  Returns dict of (n_seeds, len(Ks)) arrays: price, se, price_plain,
    se_plain (the last two are plain MC on the same paths).
    """
    runs = [price_asian_call_cv(H, nu, Ks, T=T, S0=S0, r=R, N=N, M=M, seed=SEED_BASE + s)
            for s in range(n_seeds)]
    return {key: np.array([res[key] for res in runs])
            for key in ("price", "se", "price_plain", "se_plain")}


def run_levy_benchmark(Ks, sigma, N):
    """Levy analytical prices for arithmetic Asian call at constant sigma."""
    return np.array([levy_asian_call(S0, K, T, R, sigma, N) for K in Ks])


def paired_premium(a, b):
    """Mean and standard error of a - b across seeds (a, b share random numbers)."""
    d = a - b
    return d.mean(axis=0), d.std(axis=0, ddof=1) / np.sqrt(d.shape[0])


def roughness_share(rough, smooth_norm, smooth_fixed, k=ATM):
    """
    share = mean(d_norm) / mean(d_fixed) at strike index k, with d the per-seed premia;
    SE by the delta method, Var(share) ~ Var(d_norm - share d_fixed) / (n mean(d_fixed)^2).
    """
    dn = rough[:, k] - smooth_norm[:, k]
    df = rough[:, k] - smooth_fixed[:, k]
    share = dn.mean() / df.mean()
    se = (dn - share * df).std(ddof=1) / np.sqrt(dn.size) / abs(df.mean())
    return share, se


def plot_validation(Ks, levy, gbm, rfsv, premia, n_seeds, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    # ── Panel (a): Price vs Strike ───────────────────────────────────────────
    ax = axes[0]
    ax.plot(Ks, levy, color="gray", linestyle="--", linewidth=1.8,
            label=rf"Levy approx. ($\sigma$={SIGMA0:.1f})")
    ax.plot(Ks, gbm.mean(axis=0), color="black", linestyle=":", linewidth=2,
            marker="s", markersize=4, label=rf"GBM by MC ($\nu$=0, $\sigma$={SIGMA0:.1f})")
    for label, arr in rfsv.items():
        ax.plot(Ks, arr.mean(axis=0), color=H_COLORS[label], linewidth=2,
                marker="o", markersize=5, label=f"RFSV {label}")
    ax.axvline(S0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Asian Call Price")
    ax.set_title("Asian call price vs strike\n"
                 rf"(S0=100, T=1, $\sigma_0$={SIGMA0:.1f}, $\nu$={NU:.2f}; "
                 f"control variate, mean of {n_seeds} seeds)")
    ax.legend(fontsize=8.5, loc="upper right")

    # ── Panel (b): Roughness premium under each normalization ────────────────
    ax2 = axes[1]
    colors = ["#e74c3c", "#2980b9", "#16a085", "#8e44ad"]
    w = 1.1
    for j, (p, color) in enumerate(zip(premia, colors)):
        dx = (j - (len(premia) - 1) / 2) * w
        lab = rf"{p['label']} ($\nu_{{0.5}}$={p['nu']:.3f})"
        if j > 0:
            lab += f": share {100 * p['share']:.0f}%"
        ax2.bar(Ks + dx, p["mean"], width=w, color=color, alpha=0.85, edgecolor="black",
                linewidth=0.4, yerr=p["se"], capsize=1.5, label=lab)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_ylim(min(0.0, min((p["mean"] - p["se"]).min() for p in premia)),
                 1.45 * max((p["mean"] + p["se"]).max() for p in premia))  # room for legend
    ax2.set_xlabel("Strike K")
    ax2.set_ylabel(r"p(H=0.10) $-$ p(H=0.50)")
    ax2.set_title("Roughness premium by normalization of $\\nu$ at H=0.5\n"
                  "(share = normalized / fixed-$\\nu$ premium at K=100; "
                  "error bars: $\\pm 1$ paired SE)")
    ax2.legend(fontsize=8.5, loc="upper right")

    fig.savefig(out_path, dpi=150)
    print(f"  Saved: {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=10000,
                        help="Monte Carlo paths per seed (default: 10000)")
    parser.add_argument("--N", type=int, default=252,
                        help="Time steps per year (default: 252)")
    parser.add_argument("--n-seeds", type=int, default=10,
                        help="Seeds per configuration, shared across configurations (default: 10)")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "..", "plots", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "validate_asian.png")
    csv_path = os.path.join(out_dir, "roughness_share.csv")
    print(f"Asian option validation (M={args.M}, N={args.N}, seeds={args.n_seeds}) ...")

    levy = run_levy_benchmark(STRIKES, sigma=SIGMA0, N=args.N)
    gbm = run_config(0.10, 0.0, STRIKES, args.N, args.M, args.n_seeds)
    runs = {label: run_config(H, NU, STRIKES, args.N, args.M, args.n_seeds)
            for label, H in H_VALUES.items()}
    rfsv = {label: r["price"] for label, r in runs.items()}

    # ── nu at H = 0.5 under each normalization ───────────────────────────────
    nu_intvar = variance_matched_nu(0.50)
    nu_logvar = logvar_matched_nu(0.50)
    nu_iv, nu_iv_se, iv_ref = atm_iv_matched_nu(args.N, args.M, args.n_seeds)
    print(f"\n  E[int sigma^2]/sigma0^2: H=0.10, nu={NU:.2f} -> {integrated_variance(0.10, NU):.4f}; "
          f"H=0.50, nu={NU:.2f} -> {integrated_variance(0.50, NU):.4f}")
    print(f"  matched nu at H=0.50:  E[int sigma^2] {nu_intvar:.4f}   "
          f"int Var(log sigma) {nu_logvar:.4f}   ATM IV {nu_iv:.4f} ± {nu_iv_se:.4f}")

    rough = rfsv["H=0.10*"]
    normalizations = [("fixed $\\nu$", "fixed nu", NU, rfsv["H=0.50"]),
                      ("E[$\\int\\sigma^2$] matched", "E[int sigma^2 dt]", nu_intvar, None),
                      ("$\\int$Var(log $\\sigma$) matched", "int Var(log sigma) dt", nu_logvar, None),
                      ("ATM IV matched", "ATM implied vol", nu_iv, None)]
    premia = []
    for label, name, nu05, smooth in normalizations:
        if smooth is None:
            smooth = run_config(0.50, nu05, STRIKES, args.N, args.M, args.n_seeds)["price"]
        mean, se = paired_premium(rough, smooth)
        share, share_se = roughness_share(rough, smooth, rfsv["H=0.50"])
        iv05 = bs_implied_vol(european_atm_paths(0.50, nu05, args.N, args.M, args.n_seeds).mean(),
                              S0, S0, T, R)
        premia.append(dict(label=label, name=name, nu=nu05, mean=mean, se=se,
                           share=share, share_se=share_se, iv=iv05))

    print(f"\n  {'K':>5} {'Levy':>7} {'GBM':>7} " + " ".join(f"{l:>8}" for l in rfsv)
          + "".join(f" {'prem_' + str(j):>13}" for j in range(len(premia))))
    for i, K in enumerate(STRIKES):
        row = " ".join(f"{arr[:, i].mean():8.3f}" for arr in rfsv.values())
        prem = "".join(f" {p['mean'][i]:+7.3f}±{p['se'][i]:.3f}" for p in premia)
        print(f"  {K:5.0f} {levy[i]:7.3f} {gbm['price'][:, i].mean():7.3f} {row}{prem}")
    print("  (prem_0..3 = fixed nu, E[int sigma^2], int Var(log sigma), ATM IV)")

    gbm_atm = gbm["price"][:, ATM]
    print(f"\n  ATM: GBM {gbm_atm.mean():.4f} ± {gbm_atm.std(ddof=1) / np.sqrt(args.n_seeds):.4f}"
          f"  vs Levy {levy[ATM]:.4f}")
    print(f"  ATM European implied vol of the rough model (H=0.10, nu={NU:.2f}): {100 * iv_ref:.2f}%")

    print(f"\n  Roughness share of the ATM premium (fixed-nu premium "
          f"{premia[0]['mean'][ATM]:.4f} ± {premia[0]['se'][ATM]:.4f}):")
    print(f"  {'normalization':<24} {'nu(H=0.5)':>9} {'ATM IV':>7} {'premium':>16} {'share':>14}")
    for p in premia:
        print(f"  {p['name']:<24} {p['nu']:9.4f} {100 * p['iv']:6.2f}% "
              f"{p['mean'][ATM]:+8.4f} ± {p['se'][ATM]:.4f} "
              f"{100 * p['share']:6.1f} ± {100 * p['share_se']:.1f}%")

    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["normalization", "nu_H050", "atm_iv_H050", "premium_atm", "premium_atm_se",
                     "roughness_share", "roughness_share_se", "N", "M", "n_seeds"])
        for p in premia:
            wr.writerow([p["name"], f"{p['nu']:.5f}", f"{p['iv']:.5f}", f"{p['mean'][ATM]:.5f}",
                         f"{p['se'][ATM]:.5f}", f"{p['share']:.5f}", f"{p['share_se']:.5f}",
                         args.N, args.M, args.n_seeds])
    print(f"  Saved: {csv_path}")

    # Variance reduction of the control variate, per path, at the money
    print("\n  Control variate at K=100: CV SE vs plain SE per seed (same paths)")
    for label, r in [("GBM", gbm)] + list(runs.items()):
        vr = (r["se_plain"][:, ATM] ** 2).mean() / (r["se"][:, ATM] ** 2).mean()
        print(f"    {label:>8}: SE {r['se'][:, ATM].mean():.4f} vs {r['se_plain'][:, ATM].mean():.4f}"
              f"  -> variance reduction {vr:6.1f}x")

    plot_validation(STRIKES, levy, gbm["price"], rfsv, premia, args.n_seeds, out_path)


if __name__ == "__main__":
    main()
