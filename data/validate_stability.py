"""
Numerical stability analysis — three sub-questions.

Panel (a) FFT circulant-embedding eigenvalues vs H
  - Sweep H in {0.05, 0.10, ..., 0.501}
  - Report: min(lambda), fraction negative (would need clipping), relative energy lost
  - Verifies the embedding is PSD, i.e. the FFT sampler is exact on the grid

Panel (b) rSVD condition number vs rank
  - kappa(L_k) = max(sqrt(S)) / min(sqrt(S)) for rank k in {2..128}
  - Large kappa => near-singular factor => poor path approximation

Panel (c) Cholesky covariance conditioning vs N
  - kappa(C) = lambda_max / lambda_min for N in {32..1000}
  - Explains why Cholesky may need regularization at large N
  - Also shows why small singular values dominate rSVD truncation error

Usage:
    uv run python data/validate_stability.py [--N 252] [--rank-max 128]

Output:
    plots/figures/stability_report.png

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

Panel (a) — FFT eigenvalue clipping

  The circulant embedding works by embedding the fGn (increment) covariance
  into a larger 2N-periodic matrix whose eigenvalues are the FFT of its first
  row.  When all eigenvalues are non-negative, we can use IFFT to sample exact
  fBM paths in O(N log N) — the Davies-Harte / Wood-Chan method.

  If any eigenvalue were negative it would have to be clipped to zero, and the
  sampled increments would no longer have exactly the fGn covariance.

  Key result: for every H tested (0.05 to 0.501) and every N, all 2N
  eigenvalues are strictly positive, so no clipping happens and the FFT
  sampler is exact on the grid.  The margin is thin for rough H, though:
  at H = 0.1 the smallest eigenvalue is 0.24% of gamma(0) at N = 252 and
  0.08% at N = 1000, shrinking like N^{-(1-2H)} because the fGn spectral
  density vanishes at frequency zero for H < 1/2.  For
  H <= 0.5 the fGn autocovariance is non-positive at every nonzero lag, the
  case covered by Craigmile (2003).  The C++ pricer asserts this at runtime
  (fft.hpp throws if any eigenvalue < -1e-8).

  Pitfall: gamma(0) must equal dt^{2H}.  Dropping the |k-1|^{2H} term at k = 0
  halves gamma(0) and spuriously produces ~28% negative eigenvalues at H = 0.1
  (an earlier version of this script had exactly that bug).

Panel (b) — rSVD condition number

  L_k = U * diag(sqrt(S)) is our approximate Cholesky factor (N x k).
  Its condition number kappa(L_k) = sqrt(S_max) / sqrt(S_min) measures how
  much the singular values of L_k vary.

  Large kappa means L_k is nearly singular: a tiny perturbation in the random
  draw z can produce a wildly different path.  For k = 32 at H = 0.1 we find
  kappa ~ 18 — manageable.  At larger k we include smaller singular values
  (denominator shrinks), so kappa grows.

  Contested point: is a condition number of 18 "large"?  For floating-point
  arithmetic with 64-bit doubles, we have ~15 significant digits, so kappa = 18
  loses at most 1.3 digits — far from catastrophic.  The issue is more subtle:
  paths associated with small singular values have the wrong scale, inflating
  variance in the MC estimate.

Panel (c) — Cholesky covariance conditioning

  kappa(C) = lambda_max / lambda_min measures how "stretched" the distribution
  is.  For C to be invertible (required for Cholesky), all eigenvalues must be
  strictly positive.  As N grows, lambda_min shrinks (more correlated time steps
  → near-singular matrix), and kappa grows as N^alpha.

  At N = 252, kappa ~ 789.  At N = 1000, kappa ~ 4e3 (fit: kappa ~ 1.08 N^1.19),
  so float64 Cholesky stays safe until N ~ 1e12.  The growth still explains why the approximation
  quality of the rank-k rSVD degrades: the spectrum spans many decades, so
  truncating at rank k discards non-negligible low-frequency structure.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

from data.params import H as H_DEFAULT, NU as NU_DEFAULT, T  # noqa: E402
# fbm_cov_matrix and rsvd live in the engine (shared with the low-rank sampler);
# re-exported here for the tests and experiments that use them as stab.*
from data.rfsv_model import fbm_cov_matrix, rsvd  # noqa: E402,F401


# ── Covariance helpers ────────────────────────────────────────────────────────

def fgn_cov_row(N: int, H: float, dt: float) -> np.ndarray:
    """First row of the Toeplitz fGn covariance matrix."""
    k = np.arange(N, dtype=float)
    h2 = 2.0 * H
    # |k-1|^{2H} is 1 at k = 0 (not 0), giving gamma(0) = dt^{2H} = Var of one increment
    km1 = np.abs(k - 1) ** h2
    return 0.5 * dt ** h2 * (np.abs(k + 1) ** h2 + km1 - 2.0 * k ** h2)


def circulant_eigenvalues(N: int, H: float, dt: float) -> np.ndarray:
    """
    Eigenvalues of the 2N-circulant embedding of fGn covariance.

    The fGn Toeplitz matrix T (N x N) is embedded into a circulant C (2N x 2N)
    whose first row is:
        c = [gamma(0), ..., gamma(N-1), 0, gamma(N-1), ..., gamma(1)]
    The symmetric reflection ensures C is real-symmetric so its eigenvalues
    are real.  Fact: the eigenvalues of a circulant are the DFT of its first row.
    """
    M    = 2 * N
    c    = np.zeros(M)
    row  = fgn_cov_row(N, H, dt)
    c[:N]   = row          # forward half: gamma(0), ..., gamma(N-1)
    c[N+1:] = row[1:][::-1]  # reflected half: gamma(N-1), ..., gamma(1); c[N]=0
    # FFT gives complex output; imaginary parts should be ~0 (real symmetric input)
    lam = np.fft.fft(c).real
    return lam


# ── Panel (a): FFT circulant eigenvalues ─────────────────────────────────────

def panel_fft_clipping(ax, N: int = 252):
    H_values = [0.05, 0.10, 0.20, 0.30, 0.40, 0.45, 0.49, 0.499, 0.50, 0.501]

    # min(lambda) scales like dt^{2H}, so normalize by gamma(0) = mean(lambda)
    # to compare across H.  Two grids show how the margin shrinks with N.
    for n, style in [(N, "o-"), (4 * N, "s--")]:
        rel_min, n_neg = [], 0
        for H in H_values:
            lam = circulant_eigenvalues(n, H, T / n)
            rel_min.append(lam.min() / lam.mean())
            n_neg += int((lam < 0).sum())
        ax.semilogy(H_values, np.maximum(rel_min, 1e-16), style, linewidth=2,
                    markersize=6, label=f"N={n}  ({n_neg} negative eigenvalues)")

    ax.axvline(0.5, color="#e74c3c", linestyle="--", linewidth=1.2,
               alpha=0.7, label="H = 0.5 (white noise)")
    ax.set_xlabel("Hurst exponent H")
    ax.set_ylabel(r"$\min(\lambda) \,/\, \gamma(0)$")
    ax.set_title(
        "FFT: smallest circulant eigenvalue vs H\n"
        r"all $\lambda > 0$ $\rightarrow$ no clipping $\rightarrow$ exact fGn sampling"
    )
    ax.legend(fontsize=8)


# ── Panel (b): rSVD condition number vs rank ──────────────────────────────────

def panel_rsvd_conditioning(ax, N: int = 252):
    C     = fbm_cov_matrix(N, H_DEFAULT, T)
    ranks = [2, 4, 8, 16, 32, 64, 96, 128]
    ranks = [k for k in ranks if k <= N]

    kappas, frob_errs = [], []
    frob_C = np.linalg.norm(C, "fro")

    for k in ranks:
        U, S, _ = rsvd(C, k)
        kappas.append(np.sqrt(S[0]) / np.sqrt(max(S[-1], 1e-15)))
        Ck = (U * S) @ U.T
        frob_errs.append(np.linalg.norm(C - Ck, "fro") / frob_C)

    ranks_arr  = np.array(ranks)
    kappas_arr = np.array(kappas)
    frob_arr   = np.array(frob_errs)

    color_k = "#8e44ad"
    ax.semilogy(ranks_arr, kappas_arr, "o-", color=color_k,
                linewidth=2, markersize=7,
                label=r"$\kappa(L_k) = \sqrt{S_{\max}} / \sqrt{S_{\min}}$")
    ax.set_xlabel("rSVD rank k")
    ax.set_ylabel(r"Condition number $\kappa(L_k)$", color=color_k)
    ax.tick_params(axis="y", labelcolor=color_k)
    ax.set_title(
        f"rSVD: condition number vs rank  (N={N}, H={H_DEFAULT})\n"
        r"$\kappa(L_k) < 40$ for all $k$ $\rightarrow$ well-conditioned factor"
    )

    ax2 = ax.twinx()
    ax2.semilogy(ranks_arr, frob_arr * 100, "s--", color="#27ae60",
                 linewidth=2, markersize=6, label="Frobenius error %")
    ax2.set_ylabel("Frobenius error (%)", color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # Annotate the MC noise floor
    ax2.axhline(100 * (61.0 / np.sqrt(10_000)) / 23.58,  # sigma_V / sqrt(M) at M = 10k, as % of price
                color="#27ae60", linestyle=":",
                linewidth=1, alpha=0.6, label="MC noise floor")


# ── Panel (c): Cholesky covariance conditioning vs N ─────────────────────────

def panel_cholesky_conditioning(ax):
    Ns     = [32, 64, 128, 252, 500, 750, 1000]
    kappas = []
    for N in Ns:
        C  = fbm_cov_matrix(N, H_DEFAULT, T)
        sv = np.linalg.svd(C, compute_uv=False)
        kappas.append(sv[0] / max(sv[-1], 1e-15))

    Ns_arr = np.array(Ns)
    k_arr  = np.array(kappas)

    ax.loglog(Ns_arr, k_arr, "o-", color="#e74c3c",
              linewidth=2.5, markersize=8,
              label=r"$\kappa(C) = \lambda_{\max} / \lambda_{\min}$")

    # Fit power law
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        log_N = np.log(Ns_arr)
        log_k = np.log(k_arr)
        alpha, log_c = np.polyfit(log_N, log_k, 1)
    c_fit = np.exp(log_c)
    N_range = np.logspace(np.log10(Ns_arr[0]), np.log10(Ns_arr[-1]), 200)
    ax.loglog(N_range, c_fit * N_range ** alpha, "--", color="#e74c3c",
              alpha=0.5, linewidth=1.5,
              label=f"fit: $\\kappa \\approx {c_fit:.2f} \\cdot N^{{{alpha:.2f}}}$")

    # Machine epsilon thresholds
    eps = np.finfo(float).eps
    ax.axhline(1.0 / eps, color="gray", linestyle=":", linewidth=1.2,
               label=r"$1/\varepsilon$ (float64 limit)")

    ax.set_xlabel("Path resolution N")
    ax.set_ylabel("Condition number κ(C)")
    ax.set_title(
        f"Cholesky: κ(C) vs N  (H={H_DEFAULT})\n"
        "polynomial growth, far below the float64 limit"
    )
    ax.legend(fontsize=9)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Annotate actual N values
    for n, k in zip(Ns_arr, k_arr):
        if n in (252, 1000):
            ax.annotate(f"N={n}\nκ={k:.0e}", xy=(n, k),
                        xytext=(n * 1.05, k * 0.6),
                        fontsize=7, arrowprops=dict(arrowstyle="->", color="gray"),
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="lightgray"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=252,
                        help="N for panels (a) and (b) (default: 252)")
    parser.add_argument("--rank-max", type=int, default=128)
    args = parser.parse_args()

    out_dir  = os.path.join(os.path.dirname(__file__), "..", "plots", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stability_report.png")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    fig.text(
        0.5, 1.01,
        f"Numerical stability analysis  (H={H_DEFAULT}, $\\nu$={NU_DEFAULT}, T={T}yr, N={args.N} for panels a\u2013b)",
        ha="center", fontsize=10, style="italic",
    )

    print("Panel (a): FFT eigenvalue clipping ...")
    panel_fft_clipping(axes[0], N=args.N)

    print("Panel (b): rSVD conditioning ...")
    panel_rsvd_conditioning(axes[1], N=args.N)

    print("Panel (c): Cholesky conditioning vs N ...")
    panel_cholesky_conditioning(axes[2])

    for ax, label in zip(axes, ["(a)", "(b)", "(c)"]):
        ax.text(0.02, 0.98, label, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # ── Summary statistics ─────────────────────────────────────────────────
    print(f"\nStability summary (N={args.N}, H={H_DEFAULT}):")
    dt = T / args.N

    def clipping_stats(lam):
        neg   = lam < 0
        frac  = neg.mean()
        total = lam.sum()
        lost  = (-lam[neg]).sum() / max(total, 1e-12) if neg.any() else 0.0
        return lam.min(), frac, lost

    lam_default = circulant_eigenvalues(args.N, H_DEFAULT, dt)
    mn, fr, el = clipping_stats(lam_default)
    print(f"  FFT at H={H_DEFAULT}: min(λ)={mn:.4f}, "
          f"frac_clipped={fr*100:.2f}%, energy_lost={el*100:.4f}%")

    for H_test in [0.499, 0.50, 0.501]:
        lam = circulant_eigenvalues(args.N, H_test, dt)
        mn, fr, el = clipping_stats(lam)
        print(f"  FFT at H={H_test}: min(λ)={mn:.6f}, "
              f"frac_clipped={fr*100:.3f}%, energy_lost={el*100:.4f}%")

    C = fbm_cov_matrix(args.N, H_DEFAULT, T)
    sv = np.linalg.svd(C, compute_uv=False)
    print(f"\n  Cholesky κ(C) at N={args.N}: {sv[0]/max(sv[-1],1e-15):.2e}")
    print(f"  (λ_max={sv[0]:.4f}, λ_min={sv[-1]:.2e})")

    for k in [8, 32, 64]:
        if k > args.N:
            continue
        _, S, _vt = rsvd(C, k)  # noqa: F841
        kappa = np.sqrt(S[0]) / np.sqrt(max(S[-1], 1e-15))
        print(f"  rSVD κ(L_k) at k={k:3d}: {kappa:.2e}")


if __name__ == "__main__":
    main()
