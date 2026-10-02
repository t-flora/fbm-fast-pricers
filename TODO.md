# Further Work

A prioritized roadmap written after the October 2026 review (merged in `0f0744d`; see `CHANGELOG.md`). Each item gives the evidence for it from this repo, a concrete plan, and a criterion for when it is done. Effort is a rough guess: **S** is about a day, **M** a few days, **L** a week or more.

Items 1 and 2 have working prototypes in `experiments/prototype_further_work.py`; the numbers quoted for them come from that script ($N = 252$, $M = 200{,}000$ shared paths, the current parameters).

## Priorities at a glance

| # | Item | Why it matters | Effort |
|---|---|---|---|
| 1 | Variance-corrected low-rank sampler | Removes the rSVD's 4–8% underpricing for $O(N)$ extra work per path | S |
| 2 | Conditional geometric control variate | $27\times$ variance reduction in the prototype, for all three samplers | S–M |
| 3 | Continuous integration | Tests exist but only run locally | S |
| 4 | Batched sampling (GEMM, batched FFT, GPU) | Cholesky is bandwidth-bound at large $N$; a matrix-matrix product is not | S–M |
| 5 | Faster Gaussian generator | Random numbers dominate the FFT and rSVD per-path cost | S |
| 6 | Multithreading | Paths are independent; thread scaling also shows which methods are bandwidth-bound | M |
| 7 | Hierarchical (HODLR) sampler | Far-field blocks are rank $\leq 3$; works where the FFT cannot | L |
| 8 | Spot-vol correlation $\rho < 0$ | Needed for a real skew comparison against SPY | L |
| 9 | Randomized quasi-Monte Carlo | Faster than $M^{-1/2}$ convergence | M |
| 10 | Model extensions | Mean reversion (fOU), long memory ($H > \tfrac12$), surface calibration | M each |
| 11 | Smaller items | Normalizations, Greeks, hardware counters, timing, report polish | S each |

Suggested order: 3, then 1 and 2 (they change reported numbers, so do them together and re-run once), then 4–6 (performance), then 7–10.

---

## 1. Variance-corrected low-rank sampler (S)

**Evidence.** The rank-$k$ sampler underprices because it discards the variance in the small eigenvalues (13% of the total at $k = 128$, 40% at the first time step). That missing variance is concentrated near the diagonal. Adding independent noise with variance $\text{diag}(C - C_k)$ restores every marginal variance exactly:

$$W^H \approx L_k z + D^{1/2}\varepsilon, \qquad D = \text{diag}(C - L_k L_k^\top), \quad z \sim \mathcal{N}(0, I_k),\ \varepsilon \sim \mathcal{N}(0, I_N).$$

In the prototype this removes the bias:

| Rank $k$ | Low-rank | Low-rank + diagonal |
|---|---|---|
| 8 | −6.10% | +0.12% |
| 32 | −4.34% | +0.09% |
| 128 | −2.10% | −0.02% |

The standard error of an unpaired difference is about 0.57% (the runs share draws, so the true error is smaller), and the corrected prices are indistinguishable from exact. The extra cost is $N$ Gaussian draws and $N$ multiply-adds per path.

**Plan.**
- Add an option to `lowrank::lowrank_factor` that returns $D^{1/2}$ alongside $L_k$, and add the noise in `mc_price`.
- Report both variants in `error_vs_rank.csv`, plus the error in the lag-1 covariance. The diagonal fix leaves the near-diagonal covariances wrong, and that error is not visible in marginals.
- If the lag-1 error matters for some payoff, try a banded correction (bandwidth $b$, a banded Cholesky of the residual) and measure accuracy against $b$.

**Done when:** the corrected sampler's price bias is within 2 standard errors at every rank in the benchmark, a test checks that its marginal variances match $\text{diag}(C)$, and the report's Section 5.5 discusses it.

## 2. Conditional geometric control variate (S–M)

**Evidence.** Under constant volatility the geometric Asian call has a closed form (Kemna & Vorst 1990) and is the classic control variate for the arithmetic one. Under RFSV a closed form still exists *conditionally*. With $\rho = 0$ the price shocks are independent of the volatility path, so given that path the log of the geometric average $G = (\prod_n S_{t_n})^{1/N}$ is Gaussian, with

$$\mathbb{E}[\log G \mid \sigma] = \log S_0 - \frac{\Delta t}{2}\sum_{j=1}^{N} \sigma_j^2\,w_j, \qquad \text{Var}(\log G \mid \sigma) = \Delta t \sum_{j=1}^{N} \sigma_j^2\,w_j^2, \qquad w_j = \frac{N - j + 1}{N}.$$

So $\mathbb{E}[(G - K)^+ \mid \sigma]$ is a Black–Scholes formula, available per path in $O(N)$. With $Y = (G - K)^+$, the estimator $\hat{p} = \overline{V - \beta\,(Y - \mathbb{E}[Y \mid \sigma])}$ is unbiased for any fixed $\beta$. In the prototype the arithmetic and geometric payoffs correlate at 0.998, and the standard error falls from 0.0213 to 0.0041 ($27\times$ variance reduction). Equivalently, $27\times$ fewer paths reach the same accuracy.

**Plan.**
- Add a helper to `asian_payoff.hpp` that returns the geometric payoff and its conditional expectation from the same volatility path and shocks. Use it in all three pricers, estimating $\beta$ from a small pilot run so the main estimate stays exactly unbiased.
- Return the standard error from the pricers, which also removes the need to assume $\sigma_\text{payoff}$ in tests.
- Add the same estimator to `data/rfsv_model.py`.
- Re-frame the benchmark around *time to a target accuracy*. That comparison, a variance-reduction gain set against a faster sampler, fits the course theme well.

**Done when:** all pricers report a control-variate price and standard error, a test checks that $\mathbb{E}[Y \mid \sigma]$ averages to the Monte Carlo mean of $Y$ within error, and the benchmark reports time to reach a 0.1% standard error per method.

**Caveat.** The conditional closed form needs $\rho = 0$. With correlation (item 8), conditioning on the volatility path no longer leaves the shocks independent, so the control variate needs rework.

## 3. Continuous integration (S)

The benchmark no longer depends on the macOS `mach` API, so the whole project builds on Linux. Add a GitHub Actions workflow on `ubuntu-latest`:

- `apt install libeigen3-dev libfftw3-dev`
- `cmake -B build && cmake --build build`
- `ctest --test-dir build`
- `uv run pytest tests/`

Optionally add a smoke run of `./run_pipeline.sh --fast --no-iv`.

**Done when:** pull requests show a passing check, and a deliberately broken $\gamma(0)$ fails it.

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
- **Report polish.** Four overfull boxes predate the review (Sections 4, 5 and 7).
