# Further Work

A prioritized roadmap written after the October 2026 review (merged in `0f0744d`; see `CHANGELOG.md`). Each item gives the evidence for it from this repo, a concrete plan, and a criterion for when it is done. Effort is a rough guess: **S** is about a day, **M** a few days, **L** a week or more.

**Status.** Items 1–6 are done, and most of item 11 (see [Done](#done) at the end; details in the README's Extensions and Performance sections and `CHANGELOG.md`). The remaining items keep their numbers.

## Priorities at a glance

| # | Item | Why it matters | Effort |
|---|---|---|---|
| 1 | ~~Variance-corrected low-rank sampler~~ | **Done:** bias within one SE at every rank | S |
| 2 | ~~Conditional geometric control variate~~ | **Done:** variance reduction 26–29 for every sampler | S–M |
| 3 | ~~Continuous integration~~ | **Done:** Ubuntu (gating) + macOS (informational) | S |
| 4 | ~~Batched sampling (GEMM, batched FFT)~~ | **Done:** Cholesky $4.0\times$ faster at $N = 4000$; GPU port still open | S–M |
| 5 | ~~Faster Gaussian generator~~ | **Done:** $4.0\times$ per draw; FFT path $2.1\times$ | S |
| 6 | ~~Multithreading~~ | **Done:** per-path Cholesky stalls at $1.8\times$, batched reaches $4.1\times$ | M |
| 7 | Hierarchical (HODLR) sampler | Far-field blocks are rank $\leq 3$; works where the FFT cannot | L |
| 8 | Spot-vol correlation $\rho < 0$ | Needed for a real skew comparison against SPY | L |
| 9 | Randomized quasi-Monte Carlo | Faster than $M^{-1/2}$ convergence | M |
| 10 | Model extensions | Mean reversion (fOU), long memory ($H > \tfrac12$), surface calibration | M each |
| 11 | Smaller items | Most done; hardware counters, calibration data, banded correction, GPU remain | S each |

Suggested order for what remains: 7–10, with the open parts of 11 as filler.

---

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

Done items are listed under [Done](#done). Still open:

- **GPU port of the batched pricers.** The batched code is already in GPU form: cuBLAS `trmm`/`gemm` for Cholesky and low-rank, batched cuFFT for the circulant sampler, a counter-based generator (Philox) per block. Compare against the batched, multithreaded CPU code, not the per-path loop. Once the transform and generator are fast, the price path's $2N$ exponentials are the largest stage of a low-rank path and a third of an FFT path, so that loop needs vectorizing too (for example `-fveclib` or a SIMD `exp`).
- **Vendor BLAS.** A quick trial linking Eigen to Apple Accelerate (`EIGEN_USE_BLAS`) was not faster than Eigen's own product at $B = 64$ and slower unbatched. It was measured on a loaded machine, so it is not conclusive.
- **Hardware counters.** On Linux, measure actual DRAM traffic with `perf` (uncore counters) to replace the effective-bandwidth estimate. Not possible on the macOS development machine.
- **Calibration data.** If an Oxford-Man (or other 5-minute realized variance) dataset becomes available, re-run `data/calibrate.py` and compare with the yfinance proxy. The loader is now tested on a synthetic file.
- **Banded variance correction** (from item 1): replace the diagonal noise by a banded factorization of $C - L_k L_k^\top$ so the increments are right too.
- **Second control on the volatility path** (from item 2), e.g. the integrated variance $\int_0^T \sigma_t^2\,dt$, whose mean is known in closed form.
- **Variance correction in the Python IV and Greeks scripts.** The Python engine now has the corrected low-rank sampler (`lowrank_sampler`), but no experiment uses it yet.
- **Timing hygiene on a fanless laptop.** Long $N = 4000$ runs throttle slightly ($\approx 8\%$ after 40 s), and background load distorts thread scaling. Cool-down pauses between configurations, or a desktop machine, would tighten the numbers.
- **Watch the Ubuntu 26 runner migration.** GitHub moves `ubuntu-latest` to Ubuntu 26 from October 19, 2026, with a newer GCC and Eigen. The code builds clean under GCC 16 locally; if CI breaks, fix it or pin `ubuntu-24.04`.

---

## Done

### 1. Variance-corrected low-rank sampler

`lowrank::price_corrected` and `lowrank::price_cv(..., corrected=true)` add independent noise with variance $\text{diag}(C - L_k L_k^\top)$ at each step. Measured by `./build/extensions` ($N = 500$, 100,000 paths per rank): the price error, between $-3.7\%$ and $-8.1\%$ for the plain sampler, falls to within one standard error ($\approx 0.56\%$) at every rank from 2 to 128, for about 40% more MC time at $k = 32$.

**Caveat found while testing.** The plain low-rank paths miss 93–100% of the increment (fGn) variance: they are far too smooth. The correction's noise is white in levels, so the increments become too rough instead (increment-variance error 119% at $k = 2$, 1.8% at $k = 64$, 16% at $k = 128$). The Asian price depends on volatility levels, so the fix works here. A payoff sensitive to path roughness would need the **banded correction** (a banded Cholesky of the residual), which remains open.

### 2. Conditional geometric control variate

`price_cv()` in all three samplers (shared driver `src/common/control_variate.hpp`) and `price_asian_call_cv()` in `data/rfsv_model.py`. $\beta$ is estimated on a separate pilot run of 10% of the paths, so the estimate is exactly unbiased. The variance reduction is 26–29 for every sampler at $N = 252$ and $1000$. Time to a 0.1% relative standard error at $N = 1000$ falls from 401 s to 17 s (Cholesky) and from 234 s to 10 s (FFT).

The reduction is set by $\text{corr}(V, C) = 0.981$, not by the raw payoff correlation (0.998). The control cancels price-shock noise but not volatility-path noise, which is most of the remaining 4% of the variance. **Open follow-up:** a second control on the volatility path (or quasi-Monte Carlo, item 9) to remove that part.

### 3. Continuous integration

`.github/workflows/ci.yml` builds the C++ code and runs `ctest` and `pytest` on Ubuntu (GCC, apt packages); it is the gating check and the README badge. `.github/workflows/ci-macos.yml` does the same on macOS (Apple Clang, Homebrew) as an informational check, because GitHub's macOS runners are capacity-constrained: the first post-merge run on `main` was cancelled after 15 minutes without ever getting a runner, and its re-run waited several hours. Both run on pushes to `main` and on pull requests, so a PR commit is tested once rather than twice. A second Ubuntu job runs the whole pipeline in fast mode and checks every figure (item 11). The full benchmarks and the live-data IV script are not run in CI.

### 4–6. Batched sampling, a faster generator, multithreading

`price_batched(N, M, batched::Config{batch, threads, rng})` in all three samplers, with the shared driver `src/common/batched_mc.hpp` and the generator `src/common/fast_rng.hpp` (xoshiro256++ and a 256-layer ziggurat). Measured by `./build/performance`, plotted by `plots/plot_performance.py`.

- **Batching (4).** Blocks of $B = 64$ paths: one triangular product $LZ$, one GEMM $L_k Z$, or one `fftw_plan_many_dft` call. The Cholesky transform gets $4.3\times$ faster at $N = 4000$ (about 42 GFlop/s), the low-rank product $2.6\times$, the FFT not at all (each transform is already in cache). Its local exponent between $N = 2000$ and 4000 falls from 2.33 to 1.74, which meets the done criterion (about 2, no cache penalty). All three samplers are benchmarked batched and unbatched.
- **Generator (5).** 3.8 ns per normal against 15.2 ns, $4.0\times$. The per-path breakdown (random numbers, transform, price path) is in `perf_breakdown.csv` and the report. The FFT path gets $2.1\times$ faster at every $N$, all from the generator. The tests check the distribution on $10^7$ draws, pin the xoshiro reference outputs, and pin a fixed-seed price (Clang and GCC agree to $10^{-15}$).
- **Threads (6).** `std::thread` over blocks rather than OpenMP, which Apple Clang lacks. Each block's stream is seeded by $(\text{seed}, b)$ and sums are reduced in block order, so prices are bitwise identical for any thread count. Thread scaling at $N = 500$ and 4000 for all methods, batched and unbatched, is in `perf_threads.csv`. At $N = 4000$ the per-path Cholesky loop peaks at $1.8\times$ and falls to $1.5\times$ at eight threads (bandwidth-bound), while batched Cholesky reaches $3.1\times$ on four performance cores and $4.1\times$ on eight.

Single-thread speedups over the original loop with $B = 64$ and the fast generator: Cholesky $2.2$–$4.0\times$ (growing with $N$), FFT $2.0$–$2.2\times$, rSVD $1.7$–$1.9\times$.

### 11. Smaller items (done)

- **Other variance normalizations.** `validate_asian.py` matches $\nu$ at $H = 0.5$ three ways. The roughness share of the ATM premium is 70% (integrated variance), 65% (integrated log-variance, closed form) and 58% (ATM implied volatility, by conditional Monte Carlo and Brent's method).
- **Greeks.** `data/greeks.py`: pathwise vega with a differentiated control variate ($2.900 \pm 0.035$) agrees with central finite differences under common random numbers ($2.892 \pm 0.035$ at $h = 0.005$); $\partial V/\partial H = -2.40 \pm 0.05$.
- **Timing robustness.** The main benchmark takes the median of 5 repeats and writes the interquartile range (`wall_time_q1_s`, `wall_time_q3_s`); `plot_scaling.py` draws it. The IQR caught a disturbed run (34% at one point) that was discarded.
- **Report polish.** No overfull boxes remain.
- **Quick mode.** `--quick` for `benchmark`, `extensions` and `performance`; `./run_pipeline.sh --fast` now runs everything in about two minutes.
- **CI smoke test.** A second job in `ci.yml` runs `./run_pipeline.sh --fast --no-iv` and checks that every figure is written.
- **Oxford-Man loader test.** Four tests on a synthetic long-format CSV. They found two bugs, now fixed: dates east of Greenwich moved back a day (UTC conversion), and zero RV values were kept (log of zero).
- **Control variate in the validation experiments.** `validate_asian.py`, `plot_sensitivity.py` and `validate_convergence.py` use `price_asian_call_cv`; the convergence study shows plain and control-variate curves side by side.
- **Variance correction in the Python engine.** `lowrank_sampler(..., corrected=True)` and `simulate_log_vol_paths_lowrank` in `data/rfsv_model.py`, mirroring the C++ sampler, with tests.
