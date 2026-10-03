# Changelog

## Extensions and CI, October 2026 (branch `feat/variance-correction-and-cv`)

Implements items 1–3 of the roadmap in `TODO.md` as *additional* methods. The existing pricers, the main benchmark CSVs, and the report's numbers are unchanged; the new methods are opt-in and measured by a separate program.

### Added

- **Variance-corrected low-rank sampler** (`src/rsvd/lowrank.hpp`: `residual_sd`, `price_corrected`, `price_corrected_timed`, and `price_cv(..., corrected=true)`). It adds independent noise with variance $\text{diag}(C - L_k L_k^\top)$ at each step, which restores every marginal variance. At $N = 500$ the price bias falls from between $-3.7\%$ and $-8.1\%$ to within one standard error ($\approx 0.56\%$) at every rank, for about 40% more MC time at $k = 32$.
- **Conditional geometric control variate**:
  - shared driver `src/common/control_variate.hpp` (`mc_control_variate`, `CVResult`);
  - per-path terms `asian_sample()` in `src/common/asian_payoff.hpp`;
  - `price_cv()` in all three samplers, and `asian_cv_terms()` / `price_asian_call_cv()` in `data/rfsv_model.py`.
  
  It estimates $\beta$ on a separate 10% pilot run, so the estimate is exactly unbiased, and reports the standard error with every price. Variance reduction is 26–29 for every sampler at $N = 252$ and $1000$; at $N = 1000$ the time to a 0.1% relative standard error falls from 355 s to 15 s (Cholesky) and from 218 s to 9.6 s (FFT).
- **Extensions benchmark** `benchmarks/extensions.cpp`, writing `variance_corrected_rank.csv` and `control_variate.csv`, and `plots/plot_extensions.py`, producing `variance_corrected_rank.png` and `control_variate.png`.
- **Tests**:
  - C++, 9 new checks (30 in total): the arithmetic payoff of `asian_sample()` equals the plain payoff to $10^{-15}$; for a fixed volatility path the closed-form conditional expectation matches a 400,000-draw Monte Carlo average; the Cholesky and FFT control-variate prices agree; the variance reduction exceeds 10; the corrected sampler's marginal variances equal $\text{diag}(C)$ and its price matches the exact one; and, as a negative control, the uncorrected sampler's bias is detected.
  - Python, 2 new tests: the closed form against Monte Carlo; the control-variate estimator's plain part reuses exactly the paths of `price_asian_call`, it agrees across seeds, and its variance reduction exceeds 10.
- **Continuous integration**: `.github/workflows/ci.yml` builds and runs both test suites on every push and pull request, on Ubuntu (GCC) and macOS (Apple Clang). Verified locally beforehand with a GCC 16 build (clean under `-Wall -Wextra`, all tests passing).

### Findings

- **The variance correction fixes the price, not the roughness.** Plain low-rank paths miss 93–100% of the increment (fGn) variance. The correction's per-step noise is white in levels, which makes increments too rough instead (increment-variance error 119% at $k = 2$, 1.8% at $k = 64$, 16% at $k = 128$). The Asian price depends on volatility levels, so it is unaffected; roughness-sensitive payoffs would need a banded correction.
- **The control variate's limit is volatility-path noise.** The reduction is set by $\text{corr}(V, C) = 0.981$, not by the raw arithmetic–geometric correlation of 0.998. Because $C$ has mean zero given the volatility path, it cannot remove the path-driven variance. That part is most of what remains: $\mathbb{E}[(G - K)^+ \mid \sigma]$ alone carries 2.7% of $\text{Var}(V)$, against about 4% left after the control.
- **A smaller standard error does not fix bias.** With the control variate, the plain rSVD sampler converges $27\times$ faster to a price that is still about 4% too low.

### Fixed

- `benchmarks/extensions.cpp` includes `<cstdlib>` for `std::exit` (needed by GCC).


## Review, October 2026 (branch `review-fixes`, merged in `0f0744d`)

A correctness review of the whole project, followed by fixes, new tests, re-run experiments, and a revised README and report. Several results in the previous version were caused by bugs or by experiments with too little statistical power. They are listed under [Retracted results](#retracted-results) so the old report text can be checked against the new one.

All numbers below come from the final benchmark run ($M = 10{,}000$ paths unless noted, Apple M2, single-threaded).

### Summary

- **The FFT sampler is exact, not "asymptotically exact".** Every circulant eigenvalue is positive for $H \leq \tfrac{1}{2}$, with a closed-form minimum. The earlier "28% negative eigenvalues" result came from a bug.
- **The model parameters changed.** $\nu = 0.52$ (the literature value converted to the engine's year units) and a new base volatility $\sigma_0 = 0.2$. Every price in the report changed as a result.
- **Benchmarks are fairer and wider.** Cholesky no longer multiplies by zeros, FFT draws two paths per transform, and $N$ now spans 64 to 4000. FFT ends up $7.6\times$ faster than Cholesky at $N = 4000$.
- **The low-rank sampler is biased, and the Frobenius error hid it.** It underprices by 4–8% at every rank because it discards 13–38% of the path variance.
- **New experiment:** a variance-matched roughness premium. About 70% of the premium is due to roughness itself.
- **Tests:** C++ and Python suites now run in the pipeline. Both catch the original bug.

---

### Correctness fixes

| Area | Problem | Fix | Effect on results |
|---|---|---|---|
| `data/validate_stability.py` | $\gamma(0)$ was computed as $\tfrac{1}{2}\Delta t^{2H}$ instead of $\Delta t^{2H}$ (the $\lvert k-1\rvert^{2H}$ term was dropped at $k = 0$) | Use the full formula | The "28% negative eigenvalues, 15% energy clipped" result disappears: no eigenvalue is ever negative |
| `src/rsvd/rsvd.hpp` | Implemented Halko et al. Algorithm 4.3 (no re-orthonormalization) while claiming 4.4. At $k = 128$, $(\sigma_1/\sigma_{k+p})^5 \approx 2 \times 10^{15}$, at the limit of double precision | QR after every product (true Algorithm 4.4); sketch width clamped to the matrix size | Frobenius errors changed by under 0.02%; removes a failure mode at larger $k$ or $N$ |
| `src/cholesky/cholesky.hpp` | `L * z` multiplied a dense copy of $L$, zeros included ($2N^2$ flops); three $N \times N$ matrices were alive | In-place `LLT` and a triangular mat-vec ($N^2$ flops, one matrix) | About 30% faster at $N = 1000$; prices unchanged |
| `data/calibrate.py` | `--ticker` was parsed but never applied to Oxford-Man data (which stacks about 30 indices); $\nu$ was a raw standard deviation, ignoring $\Delta^{H}$ scaling and time units | Filter by `Symbol`; fit $\nu$ from the variogram intercept with lags in years | yfinance proxy now reports $\nu$ per year and per day |
| `data/validate_convergence.py` | $\sigma_\text{payoff}$ was estimated from the spread of only five seed means; the reference price was on a different grid ($N = 500$ vs 252) | Per-path standard deviation; a $10^6$-path reference on the same grid; 20 seeds | Fitted slope is now $-0.488$ (theory $-0.5$); it was $-0.71$ |
| `plots/plot_sensitivity.py` | Every cell used an independent seed, so cell-to-cell differences were Monte Carlo noise | Common random numbers across all cells | Trends in $H$ and $\nu$ are now visible and smooth |
| `data/validate_iv.py` | A seed per strike; then, with shared seeds, a 0.05% sampling error in the forward tilted the IV curve into a fake skew | Common random numbers plus a martingale (moment-matching) correction of $S_T$; $M = 20{,}000$; capped y-axis | The model smile is a symmetric U, as $\rho = 0$ requires |
| `data/rfsv_model.py` | Price shocks used `default_rng(seed + 1)`, which collides with another run's log-vol stream | `default_rng([seed, 1])` | Removes correlated errors across runs |
| `plots/plot_scaling.py` | Theoretical Cholesky exponent drawn as 3 (the $O(MN^2)$ loop dominates, so it is 2); reference lines placed with data coordinates as axis fractions | Correct exponent and `hlines` | Figure only |
| `benchmarks/benchmark.cpp` | `peak_rss_mb` was the RSS change after the call returned (it could read 0); "L3 = 16 MB" is really the M2's L2; macOS-only `mach` API | Fork a child per method and read its peak RSS via `wait4`; POSIX only | New `measured_peak_mb` column; builds on Linux |

### Model and parameter changes

Both changes were decided by the user and affect every price in the report.

- **$\nu = 0.52$ instead of $0.30$.** Gatheral et al. quote $\nu \approx 0.3$ with time in days. The engine measures time in years ($T = 1$), where the same fit is $0.3 \cdot 252^{0.1} \approx 0.52$.
- **Base volatility $\sigma_0 = 0.2$:** $\log \sigma_t = \log \sigma_0 + \nu W_t^H$. Previously $\sigma_0 = 1$ (100% volatility). At $\nu = 0.52$ that payoff was so heavy-tailed that one path in $10^6$ carried 34% of the sample variance, and the sample standard deviation kept growing with $M$. At $\sigma_0 = 0.2$ the largest path carries 0.11%, and batch standard deviations range only from 9.3 to 9.6.
- **One source of truth for parameters.** `src/common/params.hpp` and the new `data/params.py` hold the parameters, and a test fails if they disagree. Python scripts no longer hardcode $H$ or $\nu$.
- The at-the-money price is now $p_\text{ref} = 5.3144$ (it was 23.58 at $\sigma_0 = 1$).

### Performance

- **Two fGn paths per inverse FFT.** The real and imaginary parts of one transform are independent fGn samples (their cross-covariance vanishes because $\lambda_j = \lambda_{2N-j}$). `FbmSampler::sample_pair()` uses both, halving the Gaussian draws that dominated the FFT per-path cost. $N = 1000$: 1.04 s to 0.71 s.
- **Fair Cholesky baseline** (see Correctness fixes).
- **Refactor.** Each header exposes its building block (`fbm_cholesky_factor`, `circulant_eigenvalues` and the `FbmSampler` class, `lowrank_factor`), and `price()` wraps `price_timed()` instead of duplicating it. Prices are unchanged to the last printed digit.

### Benchmark and experiment changes

- **Wider benchmark:** $N \in \{64, 128, 252, 500, 1000, 2000, 4000\}$, each timing the median of 3 repeats. The rSVD price-error experiment uses 100,000 paths per rank (was 10,000).
- **New plots:** per-path cost normalized by $N$ (`per_path_cost.png`), measured peaks on the memory plot, and variance lost $\text{tr}(C - C_k)/\text{tr}(C)$ on the error-vs-rank plot.
- **`validate_asian.py` rewritten:**
  - common random numbers across $H$ and $K$, with each simulated path priced at every strike;
  - an exact-GBM Monte Carlo baseline alongside Lévy;
  - 10 seeds with paired standard errors;
  - a variance-matched premium, with $\nu$ at $H = 0.5$ solved so that $\mathbb{E}\int_0^T \sigma_t^2\,dt$ matches $(H, \nu) = (0.10, 0.52)$.
- **`plot_structure.py`:** panel (d) now contrasts the off-diagonal block spectrum with the full matrix spectrum.
- **IV smile re-pulled** for a snapshot taken October 2, 2026 (SPY at \$771.09).

### Tests and tooling

- **`tests/test_samplers.cpp`** (21 checks, run with `ctest`):
  - $LL^\top = C$;
  - FFT eigenvalue minimum matches the closed form;
  - FFT path covariance matches $C$ for both the real- and imaginary-part paths, with no correlation between them;
  - rSVD within 5% of the Eckart–Young optimum;
  - Cholesky and FFT prices agree.
- **`tests/test_python.py`** (16 tests, run with pytest):
  - the same eigenvalue and covariance checks for the numpy engine;
  - the Black–Scholes limit at $\nu = 0$;
  - variogram recovery of known $(H, \nu)$;
  - the C++/Python parameter files agree.
- **Mutation check:** re-introducing the halved $\gamma(0)$ makes 4 C++ checks and 8 Python tests fail.
- **Pipeline:** `run_pipeline.sh` runs both suites before benchmarking and uses the new production parameters.
- **Build:** CMake exports `compile_commands.json`, so clangd finds the Eigen and FFTW headers. `pytest` was added as a uv dev dependency.

### Documentation and report

- **README** rewritten: headline results, quick start, model, the three samplers (including the positivity proof), results, validation, calibration, future work.
- **ALGORITHMS.md, CLAUDE.md, TODO.md:** corrected to match the code, with stale `src/hmatrix` paths fixed. The FFT walkthrough now matches the refactored code and has a direct covariance derivation.
- **Report** (`report-files/`):
  - new abstract;
  - proof in Section 2.3 that the embedding is PSD for $H \leq \tfrac{1}{2}$, with $\lambda_{\min} = \Delta t^{2H}(N^{2H} - (N-1)^{2H})$;
  - new subsections on the in-place Cholesky, two-path FFT, and testing;
  - a local-exponent table and per-path cost figure;
  - corrected memory and bandwidth discussion, plus a trace-error analysis of the low-rank sampler;
  - the variance-matched premium and new IV discussion;
  - Craigmile (2003) added to the bibliography, and the Wood–Chan note corrected (`plainnat` prints notes).
- **Housekeeping:**
  - superseded drafts and the uv `main.py` stub moved to `obsolete/`;
  - LaTeX build artifacts and `report-files/versions/` added to `.gitignore`;
  - `dataflow.d2` / `dataflow.svg` regenerated.

---

### Retracted results

Claims in the previous version that are wrong, and what replaces them.

| Previous claim | Status | Correct result |
|---|---|---|
| FFT embedding has 28% negative eigenvalues and 15% clipped energy at $H = 0.1$; the method is "asymptotically exact" | Bug in the stability script | All eigenvalues are positive for $H \leq \tfrac{1}{2}$ (Craigmile 2003); $\min \lambda / \gamma(0) = 0.24\%$ at $N = 252$; exact on the grid |
| $\sigma_\text{payoff} \approx 35$, SE $\approx 0.35$ at $M = 10{,}000$ | Estimated from five samples | At the old $\sigma_0 = 1$, $\nu = 0.3$ it was $\approx 61$; at the current parameters it is $9.46$ (SE $\approx 0.095$) |
| rSVD needs $k \geq 64$ for under 1% price error | Monte Carlo noise | Underprices by 3.7–8.1% at every rank tested ($k \leq 128$), resolved at 100,000 paths |
| Frobenius error is the meaningful accuracy metric for the low-rank sampler | Misleading | Price bias follows the variance lost ($13.2\%$ at $k = 128$), not the Frobenius error ($1.2\%$) |
| Old ALGORITHMS.md: the price error rises with rank through "systematic accidental cancellation" | Explanation fitted to noise | Errors were within about 1 SE of zero at 10,000 paths |
| Cholesky is DRAM-bandwidth bound at $N \leq 1000$ (47 GB/s "measured") | Not supported | $L$ fits in the 16 MB L2 at $N \leq 1000$. The effective streaming rate peaks at 35 GB/s and falls once $L$ spills (31 and 29 GB/s at $N = 2000$, 4000) |
| RFSV lies above the Lévy curve (sanity check) | Contradicted by the figure | At $\sigma = 1$ Lévy overpriced exact GBM by 6%. At $\sigma_0 = 0.2$ Lévy is accurate (4.625 vs $4.606 \pm 0.027$) and RFSV lies above it |
| Roughness premium 2.1 (9%); price decreases with $H$ and increases with $\nu$ in the sensitivity grid | Independent seeds per cell, so pure noise | With common random numbers the ATM premium is $0.468 \pm 0.008$ at $\nu = 0.52$, and $0.330 \pm 0.009$ variance-matched |
| RFSV reproduces SPY's steep ATM skew | Contradicted by the figure | With $\rho = 0$ the model smile is symmetric: curvature, no skew |
| The integrated-variance confound is "modest" (6%) | Understated | At $\nu = 0.52$ low $H$ raises $\mathbb{E}\int \sigma_t^2\,dt$ by 18.6%; about 30% of the premium is variance, 70% roughness |
| $\kappa(C) \approx 10^4$ at $N = 1000$, growing as $N^{1.5}$ | Wrong numbers | $\kappa(C) \approx 4 \times 10^3$, fitting $1.08\,N^{1.19}$ |
| FFT is $1.5\times$ faster than Cholesky at $N = 1000$ | Unfair baseline | Against a triangular Cholesky and one path per transform it was 9%; with two paths per transform it is $1.6\times$ ($7.6\times$ at $N = 4000$) |
| The Hybrid Scheme would improve this model's accuracy (TODO.md) | Wrong | Circulant embedding already samples fGn exactly; the Hybrid Scheme applies to Volterra models such as rough Bergomi |

### Commits

| Commit | Change |
|---|---|
| `80a310b` | Correct the rSVD algorithm, halve Cholesky work, fix benchmark docs |
| `53155ef` | Correct the analysis-script bugs behind several reported results |
| `bea809f` | Regenerate figures and dataflow diagram |
| `eb1ac0e` | Rewrite README; correct ALGORITHMS, CLAUDE, TODO |
| `4d64d6e` | Move superseded drafts to `obsolete/`, ignore LaTeX artifacts |
| `873a73b` | Add the LaTeX report |
| `62c7f92` | Expose sampler building blocks, deduplicate price functions |
| `b308395` | Measure true peak memory in a forked child, drop the mach API |
| `9688108` | Add C++ and Python test suites, run them in the pipeline |
| `70879a6` | Document tests, measured peak memory, refactored FFT sampler |
| `e3fe9ca` | $\nu = 0.52$, $\sigma_0 = 0.2$, two FFT paths per inverse transform |
| `9f95112` | Benchmark $N$ from 64 to 4000, median of 3 timings, 100k paths per rank |
| `c0d7631` | Variance-matched roughness premium, same-grid convergence reference |
| `5adab17` | Regenerate figures for the final parameters and wider benchmark |
| `ae7ba25` | Update README, guides and report for the final parameters and results |
