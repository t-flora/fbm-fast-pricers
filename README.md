# Rough Volatility Asian Option Pricer

A C++ Monte Carlo pricer for an **arithmetic Asian call** under the **Rough Fractional Stochastic Volatility (RFSV)** model, built to compare three ways of sampling fractional Brownian motion (fBM):

| Sampler | Setup | Per path | Exact? |
|---|---|---|---|
| Dense Cholesky | $O(N^3)$ | $O(N^2)$ | Yes |
| Circulant embedding + FFT | $O(N \log N)$ | $O(N \log N)$ | Yes (embedding verified PSD) |
| Global low-rank rSVD, rank $k$ | $O(N^2 k)$ | $O(Nk)$ | No (rank-$k$ truncation) |

The project asks how far asymptotic complexity predicts real speedups on this problem, and *why* each method works or fails. Python scripts add calibration, validation, and structural/stability analysis.

**Headline results** ($M = 10{,}000$ paths, Apple M2):

- At $N = 1000$, FFT is only about 9% faster than a triangular Cholesky (1.04 s vs 1.13 s), though it needs $O(N)$ memory instead of $O(N^2)$. Each FFT path spends most of its time generating random numbers, not transforming.
- The FFT method is exact for every $H \leq \tfrac{1}{2}$. The smallest circulant eigenvalue has the closed form $\Delta t^{2H}(N^{2H} - (N-1)^{2H}) > 0$.
- Low-rank rSVD with $k = 32$ is fastest (0.43 s), but it is an *approximate* sampler. It leaves a 1.9% Frobenius error in the covariance, and that error decays slowly with $k$ because $H = 0.1$ makes the spectrum of $C$ heavy-tailed.
- Fitted exponents over $N \in \{252, 500, 1000\}$: Cholesky $\approx 1.33$, FFT $\approx 0.95$, rSVD $\approx 1.02$. The Cholesky exponent is far below 3 because the $O(MN^2)$ path loop, not the $O(N^3)$ factorization, dominates.

---

## Quick start

**Dependencies:** Eigen3, FFTW3, CMake $\geq$ 3.16, a C++17 compiler, and [uv](https://docs.astral.sh/uv/) for Python.

```bash
brew install eigen fftw cmake

# Build: cholesky_pricer, fft_pricer, rsvd_pricer, benchmark
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Each pricer prints price and wall time at N = 252, 500, 1000
./build/fft_pricer

# Full benchmark (~1.5 min): writes benchmarks/results/*.csv and reference_price.txt
./build/benchmark

# Plots from the benchmark CSVs
uv run python plots/plot_scaling.py
```

To run everything end to end (build, benchmark, all analysis scripts), use the pipeline script. It snapshots every figure and CSV into `plots/runs/<timestamp>/` with a manifest:

```bash
./run_pipeline.sh            # production parameters
./run_pipeline.sh --fast     # small M, quick smoke test
./run_pipeline.sh --no-iv    # skip the step that needs internet (live SPY chains)
```

All Python scripts write figures to `plots/figures/`. The PNGs committed directly under `plots/` are snapshots of those outputs.

> `benchmark` uses the macOS `mach` API for RSS measurement, so it builds only on macOS as written.

---

## The problem

### Model

Gatheral, Jaisson & Rosenbaum (2018) found that realized log-volatility of equity indices behaves like fBM with Hurst exponent $H \approx 0.1$. RFSV models this directly:

$$\log \sigma_t = \nu \, W_t^H, \qquad dS_t = r S_t \, dt + \sigma_t S_t \, dZ_t,$$

where $W^H$ is fBM and $\nu$ is the vol-of-vol. Because $H < \tfrac{1}{2}$, increments of $W^H$ are negatively correlated, so volatility paths are "rough".

Three simplifications matter when reading the numbers:

- **No log-vol level.** With $\log \sigma_t = \nu W_t^H$ we get $\sigma_0 = 1$, i.e. 100% annualized volatility. That is why the at-the-money Asian price is $\approx 23.6$ on $S_0 = 100$. The IV validation script adds a level $\mu_0$ (see below).
- **No spot-vol correlation** ($\rho = 0$). $Z$ is independent of $W^H$.
- **Time is in years** ($T = 1$, $\Delta t = 1/N$). The hardcoded $\nu = 0.30$ is the value Gatheral et al. report with time measured in *days*. In year units the same roughness corresponds to $\nu \approx 0.3 \cdot 252^{0.1} \approx 0.52$, so the engine runs at a lower vol-of-vol than the paper's fit. The benchmarks are valid for any $\nu$, but keep this in mind when comparing prices to market data.

Parameters live in `src/common/params.hpp`: $H = 0.10$, $\nu = 0.30$, $S_0 = K = 100$, $T = 1$, $r = 0$.

### Option

The arithmetic Asian call pays

$$V = \max\left(\frac{1}{N}\sum_{n=1}^{N} S_{t_n} - K,\; 0\right).$$

No closed form exists, so I estimate $p = e^{-rT}\,\mathbb{E}[V]$ by Monte Carlo over $M$ paths. The standard error is $\sigma_V / \sqrt{M}$. Here $\sigma_V \approx 61$ (measured from 200,000 paths), giving $\approx 0.6$ at $M = 10{,}000$. The payoff has a heavy right tail: $\sigma_0 = 1$ and volatility itself is lognormal.

### Why sampling fBM is the bottleneck

fBM is not Markov, so there is no step-by-step recursion. A path on $N$ grid points is one draw from $\mathcal{N}(0, C)$ with the dense covariance

$$C_{jk} = \tfrac{1}{2}\left(t_j^{2H} + t_k^{2H} - |t_j - t_k|^{2H}\right).$$

Each sampler is a different way to apply a "square root" of $C$ to white noise.

---

## The three samplers

### 1. Dense Cholesky: `src/cholesky/cholesky.hpp`

Build $C$, factor $C = LL^\top$ once in place with `Eigen::LLT<Eigen::Ref<MatrixXd>>` ($N^3/3$ flops, so $L$ overwrites $C$), then compute $W^H = Lz$ for each path through a lower-triangular view ($N^2$ flops).

An earlier version multiplied by a dense copy of $L$, zeros included: twice the flops, and three $N \times N$ matrices in memory. Fixing that cut the $N = 1000$ run from 1.61 s to 1.13 s with unchanged prices.

At $M = 10{,}000$ the $O(MN^2)$ loop dwarfs the factorization. At $N = 1000$, construction takes 26 ms and the MC loop 1108 ms. The fitted exponent of 1.33 (not 2) comes from the $O(N)$ per-path work (Gaussian draws, $N$ calls to `exp`, payoff accumulation), which is still a large share of runtime at these $N$.

### 2. Circulant embedding + FFT: `src/fft/fft.hpp`

This is the Davies–Harte (1987) / Wood–Chan (1994) method. fBM itself is non-stationary, but its increments (fractional Gaussian noise, fGn) are stationary, with autocovariance

$$\gamma(k) = \frac{\Delta t^{2H}}{2}\left(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}\right).$$

The fGn covariance is therefore Toeplitz. I embed it in a $2N \times 2N$ circulant with first row $c = [\gamma(0), \ldots, \gamma(N{-}1), 0, \gamma(N{-}1), \ldots, \gamma(1)]$. The DFT diagonalizes the circulant, and its eigenvalues are $\lambda = \text{FFT}(c)$.

Per path: scale complex white noise by $\sqrt{\lambda_j / 2N}$, apply one inverse FFT, take the first $N$ real parts as fGn, and cumulatively sum to get fBM. FFTW plans are created once and reused for every path.

**Exactness.** The method is exact iff every $\lambda_j \geq 0$. I check this numerically (`data/validate_stability.py`), and for every $H \in [0.05, 0.501]$ and $N \leq 1008$ tested all eigenvalues are strictly positive, so no clipping ever happens. For $H \leq \tfrac{1}{2}$ this is expected: $\gamma(k) \leq 0$ at every nonzero lag, the case covered by Craigmile (2003). `fft.hpp` throws instead of clipping if a negative eigenvalue ever shows up.

The proof is short. Because $\gamma(k) < 0$ for $k \geq 1$, every eigenvalue $\lambda_j = \gamma(0) + 2\sum_k \gamma(k)\cos(\pi jk/N)$ is at least $\lambda_0$, and the sum telescopes:

$$\lambda_{\min} = \lambda_0 = \Delta t^{2H}\left(N^{2H} - (N-1)^{2H}\right) \approx 2H\,\Delta t^{2H} N^{2H-1} > 0.$$

The margin is thin for rough $H$. At $H = 0.1$, $\min \lambda / \gamma(0)$ is $0.24\%$ at $N = 252$ and $0.08\%$ at $N = 1000$, matching the formula to every printed digit.

> **Pitfall:** embedding the fBM covariance directly does not work, since it is not Toeplitz. Only the increment covariance embeds.

### 3. Global low-rank rSVD: `src/rsvd/lowrank.hpp`, `src/rsvd/rsvd.hpp`

Approximate $C \approx U_k \,\text{diag}(s)\, U_k^\top$ with a randomized SVD (Halko, Martinsson & Tropp 2011, Algorithm 4.4: Gaussian sketch, $q = 2$ subspace iterations with re-orthonormalization, oversampling $p = 5$). Then sample with the $N \times k$ factor $L_k = U_k \,\text{diag}(\sqrt{s})$ at $O(Nk)$ per path.

This sampler is *approximate*: paths have covariance $\nu^2 C_k$, not $\nu^2 C$. How good the approximation is depends on how fast the spectrum of $C$ decays, and at $H = 0.1$ it decays slowly:

| Rank $k$ | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| $\lVert C - C_k \rVert_F / \lVert C \rVert_F$ ($N = 500$) | 8.5% | 5.4% | 3.5% | 2.5% | 1.9% | 1.5% | 1.2% |

The rSVD itself is not the problem: at every rank these errors are within about 2% (relative) of the optimal Eckart–Young truncation computed from the exact eigendecomposition. `plots/plot_structure.py` shows where the slow decay comes from. Well-separated off-diagonal blocks of $C$ compress extremely well even at $H = 0.1$: the singular values fall below 1% of $\sigma_1$ by rank 3. The full matrix does not, because the singularity of $|s-t|^{2H}$ sits on the diagonal. A hierarchical H-matrix compresses exactly the far-field blocks and keeps near-diagonal blocks dense, which makes it the natural next step (see Future work).

`price_freed_timed()` frees $C$ before the MC loop, so only $L_k$ (0.24 MB at $N = 1000$, $k = 32$) is resident during sampling.

---

## Results

All numbers come from `./build/benchmark` on an Apple M2 (`-O3 -march=native`, single-threaded) and from the CSVs in `benchmarks/results/`.

### Runtime

| Method | $N = 252$ | $N = 500$ | $N = 1000$ | Fit $t = cN^\alpha$ |
|---|---|---|---|---|
| Cholesky | 0.18 s | 0.42 s | 1.13 s | $\alpha = 1.33$, $R^2 = 0.998$ |
| FFT | 0.28 s | 0.52 s | 1.04 s | $\alpha = 0.95$, $R^2 = 0.999$ |
| rSVD, $k = 32$ | 0.10 s | 0.20 s | 0.43 s | $\alpha = 1.02$, $R^2 = 1.000$ |

Timings are from a single run; back-to-back repeats varied by up to about 15%.

The FFT method barely beats Cholesky at these sizes. Its inverse FFT is cheap (about 8 µs per path at $N = 1000$), but it draws $4N$ Gaussians per path against Cholesky's $N$, and those draws alone take 60–70 µs. On top of that, the price-path work is identical for all three methods. The asymptotic gap is real ($\alpha = 1.33$ vs $0.95$) but mostly hidden at $N \leq 1000$.

FFT's theoretical $\log N$ factor is invisible here. Over a $4\times$ range in $N$, $\log_2 N$ grows only from 8.0 to 10.0, and a 25% drift is within the noise of a three-point fit. Resolving it would need a range of $100\times$ or more. Construction is under 10% of runtime for every method at $M = 10{,}000$ (`plots/construction_breakdown.png`).

### Memory and cache

| Method ($N = 1000$) | Resident during MC loop | Notes |
|---|---|---|
| Cholesky | $8N^2 = 7.6$ MB | One matrix: $L$ overwrites $C$ in place (only its lower half is read per path) |
| FFT | $\approx 0.13$ MB | Four length-$2N$ complex buffers |
| rSVD, $C$ held | 7.6 MB + $L_k$ | |
| rSVD, $C$ freed | 0.24 MB | $L_k$ only ($8Nk$ bytes) |

The relevant cache on the M2 is the 16 MB L2 shared by the performance cores; the M2 has no L3. $L$ fits in that L2 for all $N \leq 1000$. The `est_bandwidth_GBs` column (14, 24, and 35 GB/s for the three $N$) is $4N(N+1) M$ bytes (the lower triangle of $L$, once per path) divided by wall time. It is an effective rate for streaming $L$, **not** measured DRAM traffic, so it does not by itself show that Cholesky is DRAM-bound at these sizes.

The `peak_rss_mb` column is the RSS change across a call. Buffers are freed before the function returns, so it reflects allocator retention rather than true peak usage, and it can read 0. Use `theoretical_peak_mb`.

### Accuracy of the low-rank sampler

The reference price is $p_\text{ref} = 23.582$ at $N = 500$: the average of Cholesky (23.622) and FFT (23.542), each with 500,000 paths. Each estimate has a standard error of $\approx 0.09$. Their difference ($0.08$) is well under the standard error of a difference ($0.12$), consistent with both samplers being exact. The pooled reference has a standard error of $\approx 0.06$.

`error_vs_rank.csv` also reports the rSVD price error at $M = 10{,}000$. The errors range from 0.05 to 0.74, all within about $1.2$ MC standard errors ($\approx 0.6$), so no rank's price bias is statistically resolved at this $M$. The noise-free Frobenius error above is the meaningful accuracy metric. Resolving price bias of order 0.1 would need $M \gtrsim 10^6$ per rank.

### Numerical stability (`data/validate_stability.py`)

- **FFT:** no negative eigenvalues anywhere (see above).
- **Cholesky:** $\kappa(C) = 789$ at $N = 252$ and $\approx 4 \times 10^3$ at $N = 1000$, fitting $\kappa \approx 1.08\,N^{1.19}$. Float64 Cholesky has enormous headroom; trouble would start near $N \sim 10^{12}$.
- **rSVD factor:** $\kappa(L_k) = 8.2$, 17.6, and 23.3 at $k = 8$, 32, 64 ($N = 252$), so it is well conditioned.

---

## Validation and sensitivity experiments

These use `data/rfsv_model.py`, a vectorized numpy port of the FFT sampler that matches the C++ normalization exactly.

| Script | What it does | Output (`plots/figures/`) |
|---|---|---|
| `data/validate_convergence.py` | Price $\pm 1\sigma$ vs $M$ over 5 seeds; checks $\sigma \propto M^{-1/2}$ | `convergence.png` |
| `data/validate_asian.py` | RFSV Asian price vs strike for several $H$, against the Lévy (1992) approximation | `validate_asian.png` |
| `plots/plot_sensitivity.py` | ATM price over $H \times \nu$; price vs strike for $H \in \{0.05, \ldots, 0.5\}$ | `sensitivity_surface.png`, `sensitivity_strike.png` |
| `data/validate_iv.py` | RFSV smile vs live SPY option IVs (needs internet; `--M 20000`) | `validate_iv.png` |
| `data/validate_stability.py` | FFT eigenvalues vs $H$; $\kappa(L_k)$ vs $k$; $\kappa(C)$ vs $N$ | `stability_report.png` |
| `plots/plot_structure.py` | Toeplitz structure of fGn; off-diagonal vs full spectrum of $C$ | `structure_analysis.png` |
| `data/profile_memory.py` | `tracemalloc` peak of the numpy fBM sampler vs $(N, M)$ | `memory_profile.png` |

Caveats to keep in mind when reading these figures:

- **Lévy is not a precise baseline at 100% vol.** At $\sigma = 1$, $K = 100$, Lévy's lognormal approximation gives 23.72, but exact discrete GBM by Monte Carlo gives $22.29 \pm 0.12$, so Lévy overprices by about 6%. The closeness of RFSV ($\approx 23.6$) to Lévy is a coincidence of two effects. Use a $\nu = 0$ Monte Carlo run as the GBM baseline.
- **Sensitivity sweep uses common random numbers.** Every $(H, \nu)$ cell reuses the same seed, so cell-to-cell differences are precise. At $\nu = 0.30$ the ATM price falls from 23.60 ($H = 0.10$) to 23.17 ($H = 0.50$). An earlier version used an independent seed per cell, and its differences were pure noise.
- **Lower $H$ at fixed $\nu$ is not purely "more roughness".** $\text{Var}(\nu W_t^H) = \nu^2 t^{2H}$ is *larger* for small $H$ when $t < 1$. Price differences across $H$ therefore mix a variance effect with a roughness effect. Isolating roughness would need $\nu$ rescaled to match integrated variance.
- **The IV comparison can only test smile curvature.** With $\rho = 0$ the model smile is symmetric in log-moneyness, so it cannot reproduce SPY's skew. A level $\mu_0$ ($\log \sigma_t = \mu_0 + \nu W_t^H$) is calibrated to the ATM market price. SPY options are American, which is treated as negligible for calls.
- **IV pricing uses moment matching.** All strikes share one set of paths, and the simulated $S_T$ is rescaled so its sample mean equals the forward. Without the rescaling, a 0.05% sampling error in the forward tilted the model curve into a fake upward skew. Deep-ITM market IVs are inflated by the $r = 0$, no-dividend assumption and are shown off scale.
- `validate_convergence.py` compares against the $N = 500$ C++ reference while simulating at $N = 252$. These are different quantities (the average is over a different grid), so the reference line is only approximate.
- `validate_asian.py` derives the price-path seed as `seed + 1`, which overlaps another run's log-vol seed. Each estimate stays unbiased, but errors across those runs are correlated.

---

## Calibration

`data/calibrate.py` fits the log-vol variogram $\mathbb{E}[(\log\sigma_{t+\Delta} - \log\sigma_t)^2] = \nu^2 \Delta^{2H}$ by log-log OLS. The slope gives $2H$ and the intercept gives $\nu$, with $\Delta$ in years to match the engine.

```bash
# Free proxy: 5-day realized variance from daily S&P 500 returns
uv run python data/calibrate.py --source yfinance
#   H = 0.117 (R^2 = 0.98), nu = 0.87 per year (0.46 per day)

# Oxford-Man Realized Library (5-min RV), if you have the CSV
uv run python data/calibrate.py --file data/raw/oxfordmanrealizedvolatilityindices.csv --ticker .SPX
```

The yfinance proxy recovers $H$ close to the published $0.1$. Its $\nu$ is inflated by measurement noise in weekly squared-return RV, which is why `params.hpp` keeps the literature values rather than this fit. The Oxford-Man library is no longer distributed from its original URL; place a copy in `data/raw/` if you have one.

---

## Future work

- **True hierarchical matrix.** Use a recursive block tree with H-Cholesky (Hackbusch 1999). The structure analysis shows the far-field blocks are low rank even at $H = 0.1$, so this targets exactly what limits the global rSVD.
- **Two fGn paths per inverse FFT.** The real and imaginary parts of the transform are independent fGn samples (checked numerically), but the pricer discards the imaginary part. Using both halves the Gaussian draws per path, which is the FFT pricer's dominant cost.
- **Wider $N$ range** ($64$ to $16{,}384$) to resolve FFT's $\log N$ factor and test rSVD rank requirements.
- **Spot-vol correlation $\rho < 0$**, to produce a skew and enable a real IV comparison.
- **Variance reduction.** The geometric Asian call (closed form) is a strong control variate.
- **Hybrid scheme** (Bennedsen, Lunde & Pakkanen 2017). This applies to Volterra/rough-Bergomi models, where a singular kernel must be discretized. It does *not* improve the present model: circulant embedding already samples fGn exactly on the grid, so there is no discretization error to correct.

---

## Repository layout

```
src/common/        params.hpp (model + run parameters), covariance.hpp (fBM kernel),
                   asian_payoff.hpp (vol path -> price path -> payoff), rng.hpp
src/cholesky/      cholesky.hpp + cholesky_pricer.cpp
src/fft/           fft.hpp + fft_pricer.cpp
src/rsvd/          lowrank.hpp (sampler), rsvd.hpp (Halko et al. Alg. 4.4), rsvd_pricer.cpp
benchmarks/        benchmark.cpp -> results/{time_vs_N,error_vs_rank}.csv, reference_price.txt
data/              calibrate.py, rfsv_model.py (numpy engine), validate_*.py, profile_memory.py
plots/             plot_scaling.py, plot_structure.py, plot_sensitivity.py; figures/ (generated)
report-files/      LaTeX report (main.tex + sec*.tex)
ALGORITHMS.md      Line-by-line walkthrough of the C++ samplers
run_pipeline.sh    Build + benchmark + every analysis script, snapshotted per run
```

Each sampler is a header-only `.hpp` with `inline` functions, so `benchmark.cpp` can include all three in one translation unit. `dataflow.svg` shows how the pieces connect (regenerate with `d2 dataflow.d2 dataflow.svg`).

---

## References

| Paper | Role here |
|---|---|
| Gatheral, Jaisson & Rosenbaum (2018). *Volatility is rough.* Quantitative Finance 18(6). DOI: 10.1080/14697688.2017.1393551 | Empirical $H \approx 0.1$; RFSV model; variogram calibration |
| Mandelbrot & Van Ness (1968). *Fractional Brownian motions, fractional noises and applications.* SIAM Review 10(4). DOI: 10.1137/1010093 | fBM and fGn definitions; covariance kernel |
| Davies & Harte (1987). *Tests for Hurst effect.* Biometrika 74(1). DOI: 10.1093/biomet/74.1.95 | Circulant embedding for exact stationary Gaussian simulation |
| Wood & Chan (1994). *Simulation of stationary Gaussian processes in $[0,1]^d$.* JCGS 3(4). DOI: 10.1080/10618600.1994.10474655 | General circulant embedding framework |
| Craigmile (2003). *Simulating a class of stationary Gaussian processes using the Davies–Harte algorithm, with application to long memory processes.* J. Time Series Analysis 24(5). DOI: 10.1111/1467-9892.00318 | Non-negativity of the embedding when autocovariances are non-positive at nonzero lags (fGn, $H \leq \tfrac{1}{2}$) |
| Halko, Martinsson & Tropp (2011). *Finding structure with randomness.* SIAM Review 53(2). DOI: 10.1137/090771806 | Algorithm 4.4 in `rsvd.hpp` |
| Candès, Demanet & Ying (2009). *A fast butterfly algorithm for the computation of Fourier integral operators.* Multiscale Model. Simul. 7(4). DOI: 10.1137/080734339 | Low-rank structure of smooth kernels on well-separated blocks |
| Lévy (1992). *Pricing European average rate currency options.* J. Int. Money Finance 11(5). DOI: 10.1016/0261-5606(92)90033-E | Lognormal approximation used in `validate_asian.py` |
| Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics 21(4). DOI: 10.1007/s00780-017-0335-5 | Future work, for Volterra-type rough-vol models |
