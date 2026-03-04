# Final Project TODO

Tasks to complete before submitting as a final project for a fast analysis-based algorithms class.
Guided by `include.md`: memory use, runtime efficiency, accuracy, stability, proper experiment design, structural analysis.

---

## ✅ Completed

- **#15** `plots/plot_structure.py` — Toeplitz heatmaps + off-diagonal SVD decay → `plots/structure_analysis.png`
- **#12** `data/validate_convergence.py` — MC convergence study → `plots/convergence.png`
- **#16** C++ `price_timed()` in all three .hpp files; construction/MC split in `time_vs_N.csv`; stacked bar chart → `plots/construction_breakdown.png`
- **#13** C++ memory metrics (RSS, theoretical_peak_mb, cache_pressure, est_bandwidth_GBs) in `time_vs_N.csv`; `hmatrix_freed` variant; `data/profile_memory.py` (tracemalloc); `plots/memory_vs_N.png`, `plots/memory_profile.png`
- **#8** `plot_scaling.py` — fit lines in legend, R² annotations, constrained_layout, shared y-axis for error_vs_rank, proper MC noise floor
- **#9** `validate_asian.py` — shortened labels, bar width 2.5, annotation offset fix, ±1σ error bars (3 seeds), y-axis rename
- **#10** `plot_sensitivity.py` — seaborn.heatmap, viridis colormap, ylim fix, M default 10000, MC std err in subtitle
- **#11** `validate_iv.py` — σ_eff in title, intrinsic region shading, round moneyness ticks, integer % y-axis, mu0s passed to plot

---

## Plot Fixes

### 8 · Fix `plot_scaling.py` — `time_vs_N` and `error_vs_rank`
- `time_vs_N`: fitted dashed lines have no legend entry; add them as "c·N^α fit". Annotate R² directly on the log-log panel. Add a note that Cholesky's fitted α=1.54 reflects O(M·N²) per-path cost dominating O(N³) factorization at M=10k.
- `error_vs_rank`: normalize price error by reference price (put both metrics on a shared log-scale y-axis instead of dual y-axis). Compute the MC noise floor properly as σ_payoff/√M rather than hardcoding 0.4.
- Both: use `constrained_layout` instead of `tight_layout`; add `fig.text` subtitle with experimental parameters.

### 9 · Fix `validate_asian.py` — roughness premium panel and legend
- Bar `width=4.0` causes overlap at K=95/100/105 — reduce to 2.5 or use step spacing.
- Text annotation offset `y + 0.02` clips into bars; compute offset as fraction of y-range.
- Shorten legend labels to "H=0.05", "H=0.10*", etc. with footnote; current strings overflow.
- Rename panel (b) y-axis from "Price difference" → "Roughness premium (H=0.10 − H=0.50)".
- Add ±1σ MC error bars to all price lines in panel (a): σ_MC = std(payoffs)/√M.

### 10 · Fix `plot_sensitivity.py` — heatmap alignment and colormap
- `imshow` extent misaligns tick labels vs cells — switch to `seaborn.heatmap(annot=True)` or `pcolormesh` with explicit edges.
- Add subtitle noting MC std err ≈ 0.9 at M=3000; default M should be 10000 (noise → 0.3).
- `coolwarm_r` maps rough (low H) = red, smooth (high H) = blue — counterintuitive. Switch to `viridis` or `plasma`.
- Add axis ticks at every H and nu grid point; they're missing on the heatmap x-axis.
- Set `ylim` in `sensitivity_strike` to [min·0.97, max·1.03] — curves span ~2 units but axis starts near 0.

### 11 · Fix `validate_iv.py` — IV smile plot
- NaN RFSV IV for deep-ITM calls creates an abrupt line start; add shaded "intrinsic region" annotation.
- Moneyness x-axis ticks should be at round values (0.80, 0.90, 1.00, 1.10).
- Format IV y-axis as integer percentages with gridlines at 10%, 15%, 20%, 25%.
- Include calibrated σ_eff in plot title, e.g. "RFSV scaled to σ_eff=0.159".
- `calibrate_mu0` uses M=500 (noisy); document ±uncertainty or increase to M=1000.

---

## New Experiments

### 12 · MC convergence: price vs M (paths)  ← *"evaluate accuracy"*
New script `data/validate_convergence.py`:
- **Controlled**: N=252, H=0.10, nu=0.30, K=100, T=1 (all fixed).
- **Independent**: M ∈ {100, 250, 500, 1k, 2.5k, 5k, 10k, 25k}.
- **Dependent**: mean price and std across 5 independent seeds per M.
- Plot: price ± 1σ vs M on log x-axis; overlay reference price and 1/√M confidence bands.
- Confirm σ ∝ 1/√M empirically (log-log slope ≈ −0.5); report R².
- Run for all 3 C++ methods + Python engine; also report wall time vs M (should be linear).
- Output: `plots/convergence.png`

### 13 · Memory profiling  ← *"evaluate memory use"*
**C++** (`benchmarks/benchmark.cpp` or separate `benchmarks/memory_benchmark.cpp`):
- Use `getrusage(RUSAGE_SELF).ru_maxrss` on macOS to record peak RSS after each method.
- Theoretical: Cholesky O(N²)·8 bytes (N=1000 → 8 MB), FFT O(N), H-matrix O(N·k) runtime / O(N²) construction.
- Add `peak_rss_mb` column to `time_vs_N.csv`; add a memory-vs-N panel to `plot_scaling.py`.

**Python** (`data/profile_memory.py`):
- Use `tracemalloc` to measure peak allocation in `simulate_log_vol_paths` per (N, M).
- Output: `plots/memory_vs_N.png` (or fold into `plot_scaling.py` as a third panel).

### 14 · Numerical stability analysis  ← *"evaluate stability"*
New script `data/validate_stability.py`, three sub-questions:

1. **FFT eigenvalue clipping** (H → 0.5 boundary): sweep H ∈ {0.40, 0.45, 0.49, 0.499, 0.50, 0.501}. Report min(λ), fraction of λ clipped to 0, relative energy lost = Σclip(λ,0)/Σλ. At H=0.5 with nu→0, RFSV should match GBM price exactly.
2. **rSVD conditioning**: report condition number of L_k = max(√S)/min(√S) at each rank k. Large κ → near-singular approximate paths.
3. **Cholesky ill-conditioning**: report κ(C) = λ_max/λ_min vs N. Explains why Cholesky may need regularization at large N.
- Output: `plots/stability_report.png` (3 sub-panels)

### 15 · Structural analysis: why each method works  ← *"translational invariance / symmetry"*
New script `plots/plot_structure.py`, two key insights:

1. **Toeplitz structure of fGn** (motivates FFT):
   - Plot fGn covariance matrix as heatmap → should show constant diagonals.
   - Contrast with fBM covariance matrix → not constant diagonals (non-stationary).
   - This translational invariance of increments is exactly why the circulant embedding works.

2. **Low-rank off-diagonal structure** (motivates H-matrix/rSVD):
   - Partition C into 4 quadrants; plot singular value decay of the off-diagonal block.
   - Compare decay rate for H=0.1 vs H=0.5 — rough H has slower decay (higher Frobenius error at given k).
   - Connect to Candès-Demanet-Ying (2008): smoothness of C(s,t) away from diagonal ⟹ low-rank off-diagonal blocks.
- Output: `plots/structure_analysis.png`

### 16 · Construction vs per-path MC time breakdown  ← *"proper experiment design"*
Separate one-time setup from per-path cost for all 3 methods:
- **Cholesky**: time the `llt.compute()` (O(N³)) separately from the MC loop (O(M·N²)).
- **FFT**: time the `fftw_plan` creation + eigenvalue computation vs per-path `fftw_execute`.
- **H-matrix**: already tracked as `construction_time_s` / `mc_time_s` in CSV.
Add stacked bar chart to `plot_scaling.py` showing construction vs MC time per method at N=252/500/1000.
Expected: FFT construction negligible; Cholesky factorization visible at N=1000; rSVD construction non-trivial.

---

## Final Polish

### 17 · Regenerate all plots at production quality
- `validate_asian.py`: M=10000, N=252, 5 seeds, mean ± std error bars.
- `validate_iv.py`: M=3000, N=63; calibration step M=1000.
- `plot_sensitivity.py`: M=10000, N=252 (noise floor 0.3).
- `validate_convergence.py`: M up to 25000.
- Document expected runtimes in CLAUDE.md.

### 18 · Write final report (`report.md` or README overhaul)
Sections:
1. Abstract (4–5 sentences)
2. Introduction: why Asian options? why rough vol? why fast algorithms?
3. Mathematical background (existing LaTeX, already good)
4. Algorithms: pseudocode + complexity table for all 3 methods
5. Experiments (each with a controlled-variable table):
   - Timing + memory vs N
   - Accuracy vs rank (Frobenius + price error)
   - MC convergence (price vs M)
6. Model validation: IV smile, Asian benchmark, sensitivity surface
7. Structural analysis: Toeplitz / low-rank geometry of the covariance
8. Stability: conditioning, eigenvalue regime boundaries
9. Discussion: limitations, future work (hybrid scheme, OpenMP, mean reversion)
10. Conclusion + results summary table

---

## Priority order
1. **#15** (structural analysis) — core theoretical insight the class cares about most
2. **#12** (MC convergence) — easiest new experiment, directly answers "evaluate accuracy"
3. **#16** (time breakdown) — needed to properly interpret the scaling plots
4. **#8–11** (plot fixes) — quick cleanup, high visibility impact
5. **#13** (memory) — straightforward C++ addition
6. **#14** (stability) — most open-ended, do last
7. **#17** (production runs) — do immediately before #18
8. **#18** (report) — final step
