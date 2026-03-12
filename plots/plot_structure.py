"""
Structural Analysis: Why each fBM simulation algorithm works.

Two key insights visualised:

  1. Toeplitz structure of fGn increments (motivates FFT circulant embedding)
       - fBM covariance C(t_i, t_j) is NOT translation-invariant (non-stationary).
       - fGn increment covariance gamma(|i-j|) IS translation-invariant => Toeplitz.
       - A Toeplitz matrix embeds in a PSD circulant => exact O(N log N) sampling.

  2. Low-rank off-diagonal structure of fBM covariance (motivates H-matrix + rSVD)
       - The kernel C(s,t) is smooth away from the diagonal s approx t.
       - Smooth kernels on well-separated blocks have low numerical rank
         (Candes, Demanet & Ying 2008).
       - Rougher H => slower singular-value decay => higher rank needed.

Output:
    plots/structure_analysis.png

Usage:
    uv run python plots/plot_structure.py [--N-small 64] [--N-large 128]

──────────────────────────────────────────────────────────────────────────
BEGINNER'S GUIDE
──────────────────────────────────────────────────────────────────────────

Panel (a) vs (b): fBM vs fGn covariance heatmaps

  fBM covariance: C(t_i, t_j) = 0.5 * (t_i^{2H} + t_j^{2H} - |t_i-t_j|^{2H})
  The value at position (i,j) depends on both i AND j — each diagonal has a
  different color, confirming non-stationarity.

  fGn covariance: gamma(|i-j|) — depends only on the lag |i-j|, not on i or j
  separately.  Each anti-diagonal has the same color: this is Toeplitz.

  A Toeplitz matrix T has T[i,j] = f(|i-j|).  Any Toeplitz T can be embedded
  into a 2N x 2N circulant C (by mirroring its first row), and a circulant
  is diagonalized by the DFT: C = F * diag(lambda) * F*.  This means we can
  sample from N(0, T) exactly using O(N log N) FFTs, not O(N^3) Cholesky.

Panel (c): diagonal slice profiles

  For a Toeplitz matrix, C[i, i+k] = gamma(k) for ALL i (flat in i).
  For fBM, C[i, i+k] grows with i because later time points t_i = i*dt are
  larger, inflating the t_i^{2H} terms.

  The plot overlays two rows (i=0 and i=N/4) for both fBM and fGn.
  If the script is working correctly, the two fGn curves must overlap exactly
  (verified numerically: max deviation = 0.00e+00).

Panel (d): off-diagonal singular value decay

  The "off-diagonal block" is the top-right N/2 x N/2 quadrant of C, which
  represents covariances between the first and second halves of the path — a
  well-separated pair of time intervals.

  Singular values are normalised by sigma_1 (the largest), so we compare
  the *relative* decay rates.  For H=0.5 (standard BM), the kernel is smoother,
  the block has fast-decaying singular values, and a small rank k suffices.
  For H=0.1 (rough fBM), the kernel has a singularity near zero, the block
  decays slowly, and we need a larger rank to capture the same fraction of energy.

  Contested point: the "off-diagonal block" is just N/2 x N/2.  A full H-matrix
  would recursively partition the matrix and compress each off-diagonal sub-block
  separately.  Our implementation uses a *global* rank-k approximation, which
  is simpler but less space-efficient than a true hierarchical H-matrix.
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import toeplitz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Covariance builders ───────────────────────────────────────────────────────

def fbm_cov(N: int, H: float, T: float = 1.0) -> np.ndarray:
    """N×N fBM covariance  C_{ij} = ½(t_i^{2H} + t_j^{2H} − |t_i−t_j|^{2H})."""
    dt = T / N
    t = np.arange(1, N + 1) * dt
    ti, tj = t[:, None], t[None, :]
    return 0.5 * (ti ** (2 * H) + tj ** (2 * H) - np.abs(ti - tj) ** (2 * H))


def fgn_cov(N: int, H: float, T: float = 1.0) -> np.ndarray:
    """
    N×N fGn increment covariance — Toeplitz with first row γ(0), γ(1), …, γ(N-1).

    γ(k) = dt^{2H}/2 · (|k+1|^{2H} + |k−1|^{2H} − 2k^{2H})

    Matches the autocovariance used in data/rfsv_model.py and the C++ FFT pricer.
    """
    dt = T / N
    ks = np.arange(N, dtype=float)
    h2 = 2.0 * H
    safe_km1 = np.maximum(ks - 1, 0)
    km1 = np.where(ks <= 1, 0.0, safe_km1 ** h2)
    gamma = 0.5 * dt ** h2 * ((ks + 1) ** h2 + km1 - 2.0 * ks ** h2)
    gamma[0] = dt ** h2          # γ(0) = Var[ΔW^H] = dt^{2H}
    return toeplitz(gamma)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--N-small", type=int, default=64,
                        help="Matrix size for heatmap panels (default 64)")
    parser.add_argument("--N-large", type=int, default=128,
                        help="Matrix size for SVD analysis (default 128)")
    args = parser.parse_args()

    N_sm = args.N_small
    N_lg = args.N_large
    H_rough  = 0.10
    H_smooth = 0.50
    T = 1.0

    print(f"Building covariance matrices  (N_small={N_sm}, N_large={N_lg}) …")

    C_fbm_sm  = fbm_cov(N_sm, H_rough,  T)
    C_fgn_sm  = fgn_cov(N_sm, H_rough,  T)
    C_fbm_lg  = fbm_cov(N_lg, H_rough,  T)
    C_fbm_s05 = fbm_cov(N_lg, H_smooth, T)

    # Off-diagonal block: top-right quadrant  [0:half, half:N]
    half = N_lg // 2
    block_rough  = C_fbm_lg [:half, half:]
    block_smooth = C_fbm_s05[:half, half:]

    print("Computing SVD of off-diagonal blocks …")
    sv_rough  = np.linalg.svd(block_rough,  compute_uv=False)
    sv_smooth = np.linalg.svd(block_smooth, compute_uv=False)
    sv_rough  /= sv_rough[0]
    sv_smooth /= sv_smooth[0]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_fbm  = fig.add_subplot(gs[0, 0])
    ax_fgn  = fig.add_subplot(gs[0, 1])
    ax_diag = fig.add_subplot(gs[1, 0])
    ax_svd  = fig.add_subplot(gs[1, 1])

    fig.suptitle(
        f"Structural analysis: why FFT and low-rank rSVD work\n"
        f"H = {H_rough},  N = {N_sm} (heatmaps),  N = {N_lg} (SVD)",
        fontsize=11, y=1.01,
    )

    # ── (a) fBM covariance heatmap ────────────────────────────────────────────
    vmax_a = C_fbm_sm.max()
    im_a = ax_fbm.imshow(C_fbm_sm, aspect="auto", cmap="plasma",
                         origin="lower", vmin=0, vmax=vmax_a)
    plt.colorbar(im_a, ax=ax_fbm, fraction=0.046, pad=0.04)
    ax_fbm.set_title(
        "(a) fBM covariance  $C(t_i,t_j)$\n"
        "non-stationary — diagonals are not constant",
        fontsize=9,
    )
    ax_fbm.set_xlabel("time step $j$")
    ax_fbm.set_ylabel("time step $i$")

    # ── (b) fGn covariance heatmap ────────────────────────────────────────────
    vmax_b = C_fgn_sm.max()
    im_b = ax_fgn.imshow(C_fgn_sm, aspect="auto", cmap="plasma",
                         origin="lower", vmin=0, vmax=vmax_b)
    plt.colorbar(im_b, ax=ax_fgn, fraction=0.046, pad=0.04)
    ax_fgn.set_title(
        "(b) fGn covariance  $\\gamma(|i-j|)$\n"
        r"Toeplitz — constant diagonals $\rightarrow$ circulant embedding",
        fontsize=9,
    )
    ax_fgn.set_xlabel("time step $j$")
    ax_fgn.set_ylabel("time step $i$")

    # ── (c) Diagonal slice profiles ───────────────────────────────────────────
    # For a Toeplitz matrix:  C[i, i+k] = γ(k) for all i  (flat in i)
    # For fBM:                C[i, i+k] varies with i       (non-stationary)
    i0 = 0
    i1 = N_sm // 4
    max_lag = N_sm - i1 - 1     # so both rows can reach the same lags
    ks = np.arange(max_lag)

    fgn_row0 = np.array([C_fgn_sm[i0, i0 + k] for k in ks])
    fgn_row1 = np.array([C_fgn_sm[i1, i1 + k] for k in ks])
    fbm_row0 = np.array([C_fbm_sm[i0, i0 + k] for k in ks])
    fbm_row1 = np.array([C_fbm_sm[i1, i1 + k] for k in ks])

    ax_diag.plot(ks, fgn_row0, "-",  color="C0", lw=1.8, label=f"fGn  row $i={i0}$")
    ax_diag.plot(ks, fgn_row1, "--", color="C0", lw=1.8, label=f"fGn  row $i={i1}$  (overlaps)")
    ax_diag.plot(ks, fbm_row0, "-",  color="C1", lw=1.8, label=f"fBM  row $i={i0}$")
    ax_diag.plot(ks, fbm_row1, "--", color="C1", lw=1.8, label=f"fBM  row $i={i1}$")

    ax_diag.set_xlabel("lag $k$")
    ax_diag.set_ylabel("covariance  $C[i,\\, i+k]$")
    ax_diag.set_title(
        "(c) diagonal slice  $C[i,\\,i+k]$ vs lag $k$\n"
        "fGn rows overlap (Toeplitz); fBM rows diverge (non-stationary)",
        fontsize=9,
    )
    ax_diag.legend(fontsize=8)
    ax_diag.grid(True, alpha=0.3)

    # ── (d) Off-diagonal singular value decay ─────────────────────────────────
    n_sv = min(half, 50)
    ranks = np.arange(1, n_sv + 1)

    ax_svd.semilogy(ranks, sv_rough [:n_sv], "o-", ms=4, lw=1.5,
                    label=f"H = {H_rough:.2f}  (rough)")
    ax_svd.semilogy(ranks, sv_smooth[:n_sv], "s-", ms=4, lw=1.5,
                    label=f"H = {H_smooth:.2f}  (smooth / Brownian)")

    threshold = 0.01
    ax_svd.axhline(threshold, color="gray", ls=":", lw=1.2, label=f"{threshold*100:.0f}% threshold")

    # Annotate where each curve crosses the threshold
    def crossing_rank(sv: np.ndarray, thresh: float) -> int:
        idx = np.argmax(sv < thresh)
        return int(idx) + 1 if sv[idx] < thresh else len(sv)

    r_rough  = crossing_rank(sv_rough,  threshold)
    r_smooth = crossing_rank(sv_smooth, threshold)
    for r, col, lbl in [(r_rough, "C0", f"k={r_rough}"), (r_smooth, "C1", f"k={r_smooth}")]:
        ax_svd.axvline(r, color=col, ls=":", lw=1.0, alpha=0.7)
        ax_svd.text(r + 0.4, threshold * 1.8, lbl, color=col, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="none"))

    ax_svd.set_xlabel("rank $k$")
    ax_svd.set_ylabel("normalised singular value  $\\sigma_k / \\sigma_1$")
    ax_svd.set_title(
        f"(d) off-diagonal block singular value decay  (N={N_lg})\n"
        r"Rougher $H$ $\rightarrow$ slower decay $\rightarrow$ higher rank needed for accuracy",
        fontsize=9,
    )
    ax_svd.legend(fontsize=9)
    ax_svd.grid(True, which="both", alpha=0.3)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = os.path.join(os.path.dirname(__file__), "figures", "structure_analysis.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved  {out}")

    # ── Console summary ───────────────────────────────────────────────────────
    print(f"\nToeplitz check (max row-to-row variation in fGn diagonal slices):")
    max_var = np.max(np.abs(fgn_row0 - fgn_row1))
    print(f"  max |fGn[{i0},{i0}+k] − fGn[{i1},{i1}+k]| = {max_var:.2e}  (should be ~0)")

    print(f"\nOff-diagonal SVD at 1% threshold:")
    print(f"  H={H_rough}  (rough):  rank ≥ {r_rough}")
    print(f"  H={H_smooth} (smooth): rank ≥ {r_smooth}")
    print(f"  Rough requires {r_rough / max(r_smooth, 1):.1f}× higher rank for same accuracy.")


if __name__ == "__main__":
    main()
