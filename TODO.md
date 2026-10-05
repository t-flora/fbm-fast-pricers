# Further Work

A prioritized roadmap written after the October 2026 review (merged in `0f0744d`; see `CHANGELOG.md`). Each item gives the evidence for it from this repo, a concrete plan, and a criterion for when it is done. Effort is a rough guess: **S** is about a day, **M** a few days, **L** a week or more.

**Status.** Items 1–3 are done (see [Done](#done) at the end; details in the README's Extensions section and `CHANGELOG.md`). The remaining items keep their numbers.

## Priorities at a glance

| # | Item | Why it matters | Effort |
|---|---|---|---|
| 1 | ~~Variance-corrected low-rank sampler~~ | **Done:** bias within one SE at every rank | S |
| 2 | ~~Conditional geometric control variate~~ | **Done:** variance reduction 26–29 for every sampler | S–M |
| 3 | ~~Continuous integration~~ | **Done:** Ubuntu + macOS on every push | S |
| 4 | Batched sampling (GEMM, batched FFT, GPU) | Cholesky is bandwidth-bound at large $N$; a matrix-matrix product is not | S–M |
| 5 | Faster Gaussian generator | Random numbers dominate the FFT and rSVD per-path cost | S |
| 6 | Multithreading | Paths are independent; thread scaling also shows which methods are bandwidth-bound | M |
| 7 | Hierarchical (HODLR) sampler | Far-field blocks are rank $\leq 3$; works where the FFT cannot | L |
| 8 | Spot-vol correlation $\rho < 0$ | Needed for a real skew comparison against SPY | L |
| 9 | Randomized quasi-Monte Carlo | Faster than $M^{-1/2}$ convergence | M |
| 10 | Model extensions | Mean reversion (fOU), long memory ($H > \tfrac12$), surface calibration | M each |
| 11 | Smaller items | Normalizations, Greeks, hardware counters, timing, report polish | S each |

Suggested order for what remains: 4–6 (performance), then 7–10.

---

## 4. Batched sampling as matrix-matrix products (S–M)

**Evidence.** At $N = 4000$ the Cholesky loop streams the 122 MB factor once per path, and its effective rate falls from 35 GB/s ($N = 1000$) to 29 GB/s once $L$ leaves the 16 MB L2. It is bandwidth-bound: one mat-vec does 2 flops per 8-byte element.

**Plan.**
- Generate paths in blocks of $B$ (for example 64): compute $LZ$ for an $N \times B$ matrix $Z$ with a triangular matrix-matrix product, which reuses each element of $L$ $B$ times.
- Do the same for $L_k Z$ in the low-rank sampler, and use `fftw_plan_many_dft` for batched FFTs.
- Re-run the benchmark and compare the speedups with the bandwidth-bound regime.
- The same batching is what a GPU port needs: cuBLAS for the Cholesky and low-rank products, batched cuFFT for the circulant sampler. Do the CPU version first; it shows how much of the gap is memory traffic.

**Done when:** the Cholesky local exponent at large $N$ drops back to about 2 (no cache penalty), and all three methods are benchmarked batched and unbatched.

## 5. Faster Gaussian generator (S)

**Evidence.** At $N = 1000$, one FFT transform's $4N$ Gaussian draws take 60–70 µs with `std::mt19937` and `std::normal_distribution`, against about 8 µs for the inverse FFT. The FFT per-path cost is flat at 67–73 ns per time step, which is mostly random number generation.

**Plan.**
- Replace the generator with a fast one (xoshiro256++ or a counter-based Philox) and a ziggurat or vectorized Box–Muller normal sampler, behind the existing `make_rng` and `randn` interface.
- Keep a fixed-seed regression test, since prices will change once more.

**Done when:** the per-path breakdown (random numbers vs transform vs price path) is measured and reported, and the FFT per-path cost falls measurably.

## 6. Multithreading (M)

**Plan.**
- Parallelize the path loop with OpenMP and reduce the payoff sum.
- Give each thread an independent, reproducible stream: a counter-based generator (item 5) or one `std::seed_seq` per thread.
- Report speedup against thread count for each method on the M2's four performance cores.

**Why it is interesting.** The FFT and low-rank samplers should scale nearly linearly. Cholesky at large $N$ should flatten, because extra threads compete for the same memory bandwidth. That gives a direct test of the bandwidth argument in Section 5.2 of the report.

**Done when:** thread-scaling curves for all methods at two $N$ values are in the benchmark output.

## 7. Hierarchical (HODLR) sampler (L)

**Evidence.** `plots/plot_structure.py` shows that well-separated off-diagonal blocks of $C$ fall below 1% of their top singular value by rank 3 at $H = 0.1$. Only the full matrix has the slowly decaying spectrum that defeats a global rank-$k$ approximation. A hierarchical off-diagonal low-rank (HODLR) matrix keeps near-diagonal blocks dense and compresses only the far field.

**Plan.**
- Build a HODLR approximation of $C$ by recursive bisection, compressing off-diagonal blocks with the existing rSVD to a tolerance.
- Compute a symmetric factorization $C \approx WW^\top$ (Ambikasaran, O'Neil and Singh describe one for HODLR matrices). That gives about $O(N \log^2 N)$ setup and an $O(N \log N)$ per-path product.
- Benchmark against FFT and the corrected low-rank sampler.
- For background, the general $\mathcal{H}$-matrix theory (admissibility conditions, $\mathcal{H}$-Cholesky) is in Hackbusch (2015).

**Why bother when FFT is exact.** The circulant embedding needs a uniform grid and stationary increments. A hierarchical sampler does not, so it would handle a trading-calendar grid, a time-varying $H$, or a different kernel. Testing it on a non-uniform grid is the natural showcase.

**Done when:** the sampler matches $C$ to a user tolerance, passes the covariance test, and is benchmarked on both a uniform and a non-uniform grid.

## 8. Spot-vol correlation $\rho < 0$ (L)

**Evidence.** The IV comparison (Section 6.1 of the report) can only test smile curvature: with $\rho = 0$ the model smile is symmetric, while SPY's is strongly skewed.

**Plan.**
- Correlation needs the Brownian motion that drives the volatility. The cleanest route is the Volterra (Riemann–Liouville) representation $\int_0^t (t-s)^{H - 1/2}\,dB_s$ used by rough Bergomi, where the joint covariance of the volatility driver and $B$ is explicit.
- Sample the joint Gaussian vector by Cholesky; the Hybrid Scheme (Bennedsen, Lunde & Pakkanen 2017) then becomes the relevant fast method. It does not apply to the current model, where circulant embedding is already exact.
- Calibrate $\rho$ to the ATM skew and redo the IV comparison.

**Done when:** a calibrated $\rho$ reproduces the sign and rough size of SPY's short-dated ATM skew, and the report states what changes in the algorithms (circulant embedding no longer applies directly).

## 9. Randomized quasi-Monte Carlo (M)

**Plan.**
- Replace the Gaussian draws with scrambled Sobol points (`scipy.stats.qmc` in Python).
- Order dimensions by importance. For Cholesky and low-rank this means principal components, i.e. the eigenvectors of $C$ in decreasing eigenvalue order. For FFT it means frequency components in decreasing $\lambda_j$.
- Measure RMSE against $M$ across independent scramblings. The Asian payoff's kink limits the gain, so measure rather than assume a rate.

**Done when:** the RMSE slope is measured for each sampler and compared with $-0.5$, alone and combined with the control variate from item 2.

## 10. Model extensions (M each)

- **Mean reversion.** Gatheral et al.'s full RFSV model drives log-volatility with a fractional Ornstein–Uhlenbeck (fOU) process rather than plain fBM, with a very slow reversion rate. The fOU process is itself stationary, so its covariance is Toeplitz and circulant embedding applies to the process directly. Positivity of the embedding has to be re-checked, since the argument in Section 2.3 is specific to fGn. This matters for long maturities, where plain fBM lets volatility drift without bound.
- **Long-memory regime ($H > \tfrac{1}{2}$).** For $H > \tfrac{1}{2}$ the fGn autocovariance is positive, so the positivity proof in Section 2.3 does not apply. The test suite already checks eigenvalues numerically up to $H = 0.501$. Extend the stability sweep and tests to $H \in [0.6, 0.9]$, and measure how the rSVD rank requirements change, since the spectrum decays faster when $H$ is large.
- **Calibration to the implied-volatility surface.** $H$ and $\nu$ come from realized-variance time series. Once $\rho$ exists (item 8), fit $(H, \nu, \rho, \sigma_0)$ to the SPY surface across strikes and maturities, minimizing $\sum_{K, T} (\sigma^{\text{RFSV}}_{K,T} - \sigma^{\text{market}}_{K,T})^2$. The control variate (item 2) and common random numbers make the repeated pricing affordable.

## 11. Smaller items (S each)

- **Other variance normalizations** for the roughness premium: match $\int_0^T \text{Var}(\log\sigma_t)\,dt$, or the ATM implied volatility, and report how the roughness share (70% today) moves.
- **Greeks.** Bump-and-reprice with common random numbers is now cheap and precise. Add vega with respect to $\nu$ and sensitivity to $H$, and compare with pathwise or likelihood-ratio estimators.
- **Hardware counters.** On Linux, measure actual DRAM traffic with `perf` (uncore counters) to replace the effective-bandwidth estimate.
- **Timing robustness.** Repeats still vary by up to 15%. Use more repeats and report the spread (interquartile range) in the CSV and plots.
- **Calibration data.** If an Oxford-Man (or other 5-minute realized variance) dataset becomes available, re-run `data/calibrate.py` and compare with the yfinance proxy.
- **Report polish.** Five small overfull boxes remain (Sections 4, 5, 7 and 8; all under 32pt).
- **A real fast mode for the pipeline.** `run_pipeline.sh --fast` shrinks the Python experiments but still runs the full benchmark (about 8 minutes) and the extensions (about 45 s). Add a quick mode to `benchmark` and `extensions` (for example $N \leq 500$, fewer paths, no 500k-path reference), so the whole pipeline can run in a few minutes.
- **CI smoke test of the analysis scripts.** CI runs only the unit tests; the plotting and validation scripts are never exercised automatically. Once the quick mode exists, add a CI job running `./run_pipeline.sh --fast --no-iv` and checking that every figure is produced.
- **Test the Oxford-Man loader.** `calibrate.load_oxford_man` (the `Symbol` filter and header handling) has no test because the data is not in the repo. A tiny synthetic CSV in the published long format would cover it.
- **Use the control variate in the validation experiments.** `validate_asian.py`, `plot_sensitivity.py` and `validate_convergence.py` still use plain Monte Carlo. Switching to `price_asian_call_cv` would cut their noise by a factor of about 27 at no extra cost; the convergence study could show plain and control-variate curves side by side.
- **Variance correction in the Python engine.** The corrected low-rank sampler exists only in C++.
- **Watch the Ubuntu 26 runner migration.** GitHub moves `ubuntu-latest` to Ubuntu 26 from October 19, 2026, with a newer GCC and Eigen. If CI breaks, fix it or pin `ubuntu-24.04`.

---

## Done

### 1. Variance-corrected low-rank sampler

`lowrank::price_corrected` and `lowrank::price_cv(..., corrected=true)` add independent noise with variance $\text{diag}(C - L_k L_k^\top)$ at each step. Measured by `./build/extensions` ($N = 500$, 100,000 paths per rank): the price error, between $-3.7\%$ and $-8.1\%$ for the plain sampler, falls to within one standard error ($\approx 0.56\%$) at every rank from 2 to 128, for about 40% more MC time at $k = 32$.

**Caveat found while testing.** The plain low-rank paths miss 93–100% of the increment (fGn) variance: they are far too smooth. The correction's noise is white in levels, so the increments become too rough instead (increment-variance error 119% at $k = 2$, 1.8% at $k = 64$, 16% at $k = 128$). The Asian price depends on volatility levels, so the fix works here. A payoff sensitive to path roughness would need the **banded correction** (a banded Cholesky of the residual), which remains open.

### 2. Conditional geometric control variate

`price_cv()` in all three samplers (shared driver `src/common/control_variate.hpp`) and `price_asian_call_cv()` in `data/rfsv_model.py`. $\beta$ is estimated on a separate pilot run of 10% of the paths, so the estimate is exactly unbiased. The variance reduction is 26–29 for every sampler at $N = 252$ and $1000$. Time to a 0.1% relative standard error at $N = 1000$ falls from 355 s to 15 s (Cholesky) and from 218 s to 9.6 s (FFT).

The reduction is set by $\text{corr}(V, C) = 0.981$, not by the raw payoff correlation (0.998). The control cancels price-shock noise but not volatility-path noise, which is most of the remaining 4% of the variance. **Open follow-up:** a second control on the volatility path (or quasi-Monte Carlo, item 9) to remove that part.

### 3. Continuous integration

`.github/workflows/ci.yml` builds the C++ code and runs `ctest` and `pytest` on every push and pull request, on Ubuntu (GCC, apt packages) and macOS (Apple Clang, Homebrew). The full benchmark and the live-data IV script are not run in CI.
