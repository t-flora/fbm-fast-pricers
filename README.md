# Rough Volatility Asian Option Pricer

[![CI](https://github.com/t-flora/fbm-fast-pricers/actions/workflows/ci.yml/badge.svg)](https://github.com/t-flora/fbm-fast-pricers/actions/workflows/ci.yml)

A C++ Monte Carlo pricer for an **arithmetic Asian call** under the **Rough Fractional Stochastic Volatility (RFSV)** model, built to compare three ways of sampling fractional Brownian motion (fBM):

| Sampler | Setup | Per path | Exact? |
|---|---|---|---|
| Dense Cholesky | $O(N^3)$ | $O(N^2)$ | Yes |
| Circulant embedding + FFT | $O(N \log N)$ | $O(N \log N)$ | Yes (embedding proven PSD) |
| Global low-rank rSVD, rank $k$ | $O(N^2 k)$ | $O(Nk)$ | No (rank-$k$ truncation) |

The project asks how far asymptotic complexity predicts real speedups on this problem, and *why* each method works or fails. Python scripts add calibration, validation, and structural and stability analysis. The full write-up is `report-files/main.pdf`.

**Headline results** ($M = 10{,}000$ paths, $N$ from 64 to 4000, Apple M2, one thread unless noted):

- **FFT wins at scale.** It ties Cholesky at $N = 252$ and is $1.7\times$ faster at $N = 1000$ and $6.7\times$ faster at $N = 4000$ (2.9 s vs 19.7 s), with $O(N)$ memory instead of $O(N^2)$.
- **The FFT method is exact** for every $H \leq \tfrac{1}{2}$: the smallest circulant eigenvalue has the closed form $\Delta t^{2H}(N^{2H} - (N-1)^{2H}) > 0$.
- **Low-rank rSVD is fastest but biased.** At $k = 32$ it is $9\times$ faster than Cholesky at $N = 4000$, but it underprices by about 4% even at $k = 128$. The bias follows the *variance* it discards (13% at $k = 128$), which the usual Frobenius error (1.2%) badly understates. Adding the discarded diagonal variance back removes the bias.
- **Scaling.** FFT and rSVD scale linearly in $N$ over the whole range. Cholesky's local exponent rises from about 1 to 2.3 as the $N^2$ mat-vec takes over and its factor outgrows the 16 MB cache.
- **Batching, a faster generator and threads** (opt-in `price_batched`) give Cholesky $4.0\times$ at $N = 4000$ on one thread (blocking $LZ$ removes the bandwidth bound) and FFT $2.1\times$. With threads, the per-path Cholesky loop stalls at $1.8\times$ while the batched one reaches $4.1\times$.
- **A conditional control variate** cuts the variance by 26–29$\times$ for every sampler.
- **Roughness premium.** At the money, $H = 0.1$ adds $0.47$ to the Asian price relative to $H = 0.5$ at the same $\nu$; 58–70% of it survives when the two models are matched on integrated variance, integrated log-variance, or ATM implied volatility.

---

## What this project demonstrates

- **Numerical linear algebra.** Dense Cholesky, circulant embedding of a Toeplitz covariance diagonalized by the FFT, and a randomized SVD with power iteration (Halko–Martinsson–Tropp Alg. 4.4), each checked against theory: a closed-form proof that the embedding is PSD, rSVD errors within 2% of the Eckart–Young optimum, and condition numbers of every factor.
- **Structure-exploiting algorithms.** Explaining *why* each method works: stationarity of the increments (Toeplitz, hence FFT), and the low-rank far-field versus full-rank near-diagonal structure of $C$, which shows why a global low-rank factor stalls and a hierarchical one would not.
- **Monte Carlo methods.** Exact fBM simulation, separating bias from variance (low-rank truncation), a conditional control variate with a separate pilot for $\beta$ so the estimate is unbiased, common random numbers, pathwise and finite-difference Greeks, and convergence studies with standard errors on every number.
- **Performance engineering.** Measured versus asymptotic scaling, cache and memory-bandwidth analysis, blocking a mat-vec into a matrix-matrix product, a xoshiro256++/ziggurat Gaussian generator, multithreading whose result is bitwise identical for any thread count, and benchmarks reported as median and IQR.
- **Quantitative finance.** The RFSV rough-volatility model, arithmetic Asian options, the Lévy approximation, implied-volatility smiles against live SPY quotes, and variogram calibration of $H$ and $\nu$ from realized variance.
- **Software engineering.** Header-only C++17 with Eigen and FFTW, a numpy engine that matches the C++ normalization exactly, 40 C++ and 27 Python tests, GitHub Actions CI (gating Ubuntu job plus a whole-pipeline smoke test), and a one-command pipeline that snapshots every figure with a manifest.
- **Experimental design and writing.** Each experiment has explicit control, independent and dependent variables, and the results are written up in a LaTeX report.

---

## Quick start

**Dependencies:** Eigen3, FFTW3, CMake $\geq$ 3.16, a C++17 compiler, and [uv](https://docs.astral.sh/uv/) for Python.

```bash
brew install eigen fftw cmake

cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

./build/fft_pricer             # price and wall time at N = 252, 500, 1000 (also cholesky_, rsvd_pricer)
./build/benchmark              # ~12 min, writes benchmarks/results/*.csv (--quick: ~20 s smoke test)
uv run python plots/plot_scaling.py

./build/test_samplers          # or: ctest --test-dir build
uv run pytest tests/
```

To run everything end to end (build, tests, benchmarks, every analysis script), use the pipeline. It snapshots every figure and CSV into `plots/runs/<timestamp>/` with a manifest:

```bash
./run_pipeline.sh            # production parameters
./run_pipeline.sh --fast     # small N and M everywhere: ~2 min smoke test (run in CI)
./run_pipeline.sh --no-iv    # skip the step that needs internet (live SPY chains)
```

Python scripts write figures to `plots/figures/` (gitignored); the PNGs committed under `plots/` are snapshots of them.

---

## The problem

Gatheral, Jaisson & Rosenbaum (2018) found that realized log-volatility of equity indices behaves like fBM with Hurst exponent $H \approx 0.1$. RFSV models this directly:

$$\log \sigma_t = \log \sigma_0 + \nu \, W_t^H, \qquad dS_t = r S_t \, dt + \sigma_t S_t \, dZ_t,$$

where $W^H$ is fBM and $\nu$ is the vol-of-vol. Because $H < \tfrac{1}{2}$, increments of $W^H$ are negatively correlated, so volatility paths are rough. The parameters ($H = 0.10$, $\nu = 0.52$, $\sigma_0 = 0.2$, $S_0 = K = 100$, $T = 1$, $r = 0$) live in `src/common/params.hpp` and `data/params.py`, and a test keeps them in sync. Two points matter when reading the numbers:

- **$\nu$ is in year units.** Gatheral et al. report $\nu \approx 0.3$ with time in days; with $\Delta t = 1/N$ years the same fit is $0.3 \cdot 252^{0.1} \approx 0.52$.
- **No spot-vol correlation** ($\rho = 0$): $Z$ is independent of $W^H$, so the model has a symmetric smile and no skew.

The option pays $V = \max\left(\tfrac{1}{N}\sum_{n=1}^{N} S_{t_n} - K,\; 0\right)$. There is no closed form, so I estimate $p = e^{-rT}\,\mathbb{E}[V]$ over $M$ paths. At the money $p \approx 5.32$ and $\sigma_V \approx 9.5$, so $M = 10{,}000$ gives a standard error of $\approx 0.095$ (1.8% of the price). Strictly, $\text{Var}(V)$ is infinite for any lognormal volatility, but the divergence sits far out in the tail and is never sampled.

fBM is not Markov, so a path on $N$ grid points is one draw from $\mathcal{N}(0, C)$ with the dense covariance

$$C_{jk} = \tfrac{1}{2}\left(t_j^{2H} + t_k^{2H} - |t_j - t_k|^{2H}\right).$$

Each sampler is a different way to apply a square root of $C$ to white noise.

---

## The three samplers

### 1. Dense Cholesky (`src/cholesky/cholesky.hpp`)

Factor $C = LL^\top$ once, in place ($N^3/3$ flops), then compute $W^H = Lz$ per path through a triangular view ($N^2$ flops). The factorization is a small share of runtime: 25 ms of 1.22 s at $N = 1000$, and 0.80 s of 19.7 s at $N = 4000$.

### 2. Circulant embedding + FFT (`src/fft/fft.hpp`)

This is the Davies–Harte (1987) / Wood–Chan (1994) method. fBM is non-stationary, but its increments (fractional Gaussian noise, fGn) are stationary, with autocovariance

$$\gamma(k) = \frac{\Delta t^{2H}}{2}\left(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}\right).$$

Their covariance is Toeplitz, so it embeds in a $2N \times 2N$ circulant with first row $[\gamma(0), \ldots, \gamma(N{-}1), 0, \gamma(N{-}1), \ldots, \gamma(1)]$, whose eigenvalues $\lambda$ are the FFT of that row. Scaling complex white noise by $\sqrt{\lambda_j / 2N}$ and applying one inverse FFT gives two *independent* fGn paths (the real and imaginary parts, uncorrelated because $\lambda_j = \lambda_{2N-j}$); cumulative sums turn them into fBM. FFTW plans are created once and reused.

> **Pitfall:** embedding the fBM covariance directly does not work, since it is not Toeplitz. Only the increment covariance embeds.

**Exactness.** The method is exact iff every $\lambda_j \geq 0$. For $H \leq \tfrac{1}{2}$, $\gamma(k) \leq 0$ at every nonzero lag (the case covered by Craigmile 2003), so every $\lambda_j = \gamma(0) + 2\sum_k \gamma(k)\cos(\pi jk/N)$ is at least $\lambda_0$, and the sum telescopes:

$$\lambda_{\min} = \lambda_0 = \Delta t^{2H}\left(N^{2H} - (N-1)^{2H}\right) \approx 2H\,\Delta t^{2H} N^{2H-1} > 0.$$

The margin is thin for rough $H$: at $H = 0.1$, $\lambda_{\min} / \gamma(0)$ is $0.24\%$ at $N = 252$ and $0.08\%$ at $N = 1000$, matching `data/validate_stability.py` (which finds no negative eigenvalue for any $H \in [0.05, 0.501]$). `fft.hpp` throws rather than clipping if one ever appears.

### 3. Global low-rank rSVD (`src/rsvd/lowrank.hpp`, `src/rsvd/rsvd.hpp`)

A randomized SVD (Halko, Martinsson & Tropp 2011, Alg. 4.4: Gaussian sketch, oversampling $p = 5$, $q = 2$ subspace iterations with QR after every product) gives $C \approx U_k \,\text{diag}(s)\, U_k^\top$, and paths are sampled with $L_k = U_k \,\text{diag}(\sqrt{s})$ at $O(Nk)$ each. This is not a true hierarchical matrix: one low-rank factor covers all of $C$.

The sampler is approximate, and two error measures ($N = 500$) tell very different stories:

| Rank $k$ | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| Frobenius $\lVert C - C_k \rVert_F / \lVert C \rVert_F$ | 8.5% | 5.4% | 3.5% | 2.5% | 1.9% | 1.5% | 1.2% |
| Variance lost $\text{tr}(C - C_k)/\text{tr}(C)$ | 37.9% | 32.8% | 28.3% | 24.4% | 20.8% | 17.2% | 13.2% |
| Price error vs reference ($\pm 1.1\%$ at 2 SE) | −8.1% | −7.3% | −6.8% | −5.9% | −4.3% | −3.7% | −3.9% |

The Frobenius norm is dominated by the few large eigenvalues and looks reassuring. The price depends on how much path variance the sampler keeps, and truncation discards the long tail of small eigenvalues that carry the short-scale roughness: at $k = 128$ it still drops 13% of the total variance (40% at the first time step). Less volatility means a lower price, so the bias decays slowly, like the variance lost.

The rSVD itself is not the problem: its Frobenius error is within about 2% of the optimal Eckart–Young truncation at every rank. `plots/plot_structure.py` shows where the slow decay comes from: well-separated off-diagonal blocks of $C$ fall below 1% of their top singular value by rank 3, but the full matrix does not, because the singularity of $|s-t|^{2H}$ sits on the diagonal. A hierarchical (HODLR) matrix would compress only the far field (see Future work).

---

## Results

From `./build/benchmark` on a fanless M2 MacBook Air (`-O3 -march=native`, one thread, AC power, Low Power Mode off). Each timing is the median of 5 repeats; the CSVs in `benchmarks/results/` also record the interquartile range, which is how disturbed runs are caught.

### Runtime

| Method | $N = 64$ | $252$ | $1000$ | $2000$ | $4000$ |
|---|---|---|---|---|---|
| Cholesky | 0.038 s | 0.18 s | 1.22 s | 4.10 s | 19.66 s |
| FFT | 0.046 s | 0.18 s | 0.71 s | 1.42 s | 2.93 s |
| rSVD, $k = 32$ | 0.031 s | 0.10 s | 0.44 s | 0.93 s | 2.13 s |

A single power law fits FFT and rSVD ($\alpha = 1.00$ and $1.02$, $R^2 \geq 0.997$) but not Cholesky ($\alpha = 1.48$, $R^2 = 0.983$), whose curve bends. Its local exponent between successive $N$ is 1.05, 1.19, 1.28, 1.52, 1.75 and 2.26 (from 64–128 up to 2000–4000), while FFT's stays between 0.87 and 1.09. At small $N$ the $O(N)$ per-path work (Gaussian draws, `exp`, payoff) hides the $N^2$ mat-vec; above $N \approx 1000$ the mat-vec dominates, and from 2000 to 4000 the exponent exceeds 2 because $L$ no longer fits in cache.

FFT's $\log N$ factor stays invisible: its MC cost per path per time step is flat at 70–76 ns (`per_path_cost.png`), because random number generation and price-path arithmetic, both $O(N)$, dominate the transform. Construction is negligible for FFT, at most 4% for Cholesky, and 24% of rSVD's runtime at $N = 4000$.

### Memory and cache

| Method ($N = 1000$) | Resident during MC loop | Measured lifetime peak |
|---|---|---|
| Cholesky | 7.6 MB ($8N^2$; $L$ overwrites $C$) | 11.1 MB |
| FFT | 0.08 MB ($2N$ scale factors, two length-$2N$ buffers) | 0.4 MB |
| rSVD, $C$ freed before the MC loop | 0.24 MB ($L_k$, $8Nk$ bytes) | 11.3 MB |

The measured peak runs each method in a forked child and reads its peak RSS via `wait4`, minus that of an idle child. Freeing $C$ (`price_freed_timed`) shrinks what is resident during sampling but not the lifetime peak, since $C$ must exist while $L_k$ is built. At $N = 4000$ the Cholesky and rSVD peaks reach 142–146 MB; FFT needs 1.2 MB.

The relevant cache is the M2's 16 MB L2 shared by the performance cores (there is no L3), which holds $L$ up to $N \approx 1450$. The effective Cholesky streaming rate (lower triangle of $L$ once per path, divided by wall time) rises to 33 GB/s at $N = 1000$ and 39 GB/s at $N = 2000$, then falls to 33 GB/s at $N = 4000$, where $L$ is about eight times the cache: the loop has become memory-bandwidth bound. The thread-scaling results below confirm it.

### Reference price and stability

The reference is $p_\text{ref} = 5.3144$ at $N = 500$, the average of Cholesky (5.3195) and FFT (5.3093) with 500,000 paths each. Each has a standard error of $\approx 0.013$, and their difference is about half the standard error of a difference, consistent with both being exact.

`data/validate_stability.py`: the FFT embedding has no negative eigenvalues; $\kappa(C) = 789$ at $N = 252$ and $\approx 4 \times 10^3$ at $N = 1000$ (fit $\kappa \approx 1.08\,N^{1.19}$), far from trouble in double precision at any $N$ where $C$ fits in memory; and $\kappa(L_k) = 8.2$, 17.6, 23.3 at $k = 8$, 32, 64.

---

## Validation and sensitivity experiments

These use `data/rfsv_model.py`, a vectorized numpy port of the samplers that matches the C++ FFT normalization exactly. Comparisons between configurations (different $H$, $\nu$ or $K$) use common random numbers, the Asian-price experiments use the control variate, and every price comes with its standard error.

| Script | What it does | Output (`plots/figures/`) |
|---|---|---|
| `data/validate_convergence.py` | Price $\pm 1\sigma$ vs $M$ over 20 seeds, plain and control variate, against a same-grid $10^6$-path reference | `convergence.png` |
| `data/validate_asian.py` | Price vs strike against Lévy (1992) and exact GBM; roughness premium under three normalizations of $\nu$ | `validate_asian.png`, `roughness_share.csv` |
| `data/greeks.py` | Vega in $\nu$ (pathwise vs central difference) and $\partial V/\partial H$ | `greeks.png` |
| `plots/plot_sensitivity.py` | ATM price over $H \times \nu$; price vs strike for $H \in \{0.05, \ldots, 0.5\}$ | `sensitivity_surface.png`, `sensitivity_strike.png` |
| `data/validate_iv.py` | RFSV smile vs live SPY option IVs (needs internet) | `validate_iv.png` |
| `data/validate_stability.py` | FFT eigenvalues vs $H$; $\kappa(L_k)$ vs $k$; $\kappa(C)$ vs $N$ | `stability_report.png` |
| `plots/plot_structure.py` | Toeplitz structure of fGn; off-diagonal vs full spectrum of $C$ | `structure_analysis.png` |
| `data/profile_memory.py` | `tracemalloc` peak of the numpy sampler vs $(N, M)$ | `memory_profile.png` |

- **Convergence.** The standard deviation of the plain estimate falls with fitted slope $-0.488$ ($R^2 = 0.986$) against the theoretical $-0.5$; the control variate keeps the rate ($-0.462$) with a $27\times$ smaller variance per path. The same-grid reference ($N = 252$) is $5.3217 \pm 0.0018$.
- **Lévy is accurate at realistic volatility.** At the money it gives 4.625 against $4.617 \pm 0.0005$ for exact GBM by Monte Carlo, an overprice of 0.2%.
- **Roughness premium.** At the money, $p(H{=}0.10) - p(H{=}0.50) = 0.471 \pm 0.004$ at $\nu = 0.52$. Lower $H$ also raises the variance of the volatility path, so a fair comparison gives the $H = 0.5$ model a larger $\nu$. The share of the premium that survives depends on what is matched:

  | Matched at $H = 0.5$ | $\nu(H{=}0.5)$ | ATM premium | Share |
  |---|---|---|---|
  | nothing (same $\nu$) | 0.520 | $0.471 \pm 0.004$ | 100% |
  | $\mathbb{E}\int_0^T \sigma_t^2\,dt$ | 0.651 | $0.332 \pm 0.005$ | $70.3 \pm 0.5$% |
  | $\int_0^T \text{Var}(\log\sigma_t)\,dt$ | 0.671 | $0.306 \pm 0.005$ | $65.0 \pm 0.6$% |
  | ATM implied volatility | 0.697 | $0.274 \pm 0.006$ | $58.0 \pm 0.8$% |

  The premium peaks at the money and is positive at every strike from 80 to 120.
- **Sensitivity.** At $\nu = 0.52$ the ATM price falls from 5.46 ($H = 0.05$) to 4.86 ($H = 0.5$). The gap between $H = 0.1$ and $H = 0.5$ vanishes at small $\nu$ (0.016 at $\nu = 0.1$) and grows to 0.91 at $\nu = 0.7$.
- **Greeks.** The pathwise vega $\partial V/\partial\nu = 2.900 \pm 0.035$ agrees with central finite differences ($2.892 \pm 0.035$ at $h = 0.005$); $\partial V/\partial H = -2.40 \pm 0.05$.
- **IV smile.** Against SPY on October 2, 2026, the model smile is a symmetric U with its minimum at the money: curvature, but no skew, since $\rho = 0$. All strikes share one set of paths, and the simulated $S_T$ is rescaled so its sample mean equals the forward; without that, a 0.05% forward error tilts the model curve into a fake skew.

---

## Extensions: variance correction and a control variate

Both are opt-in and measured by a separate program, so the main benchmark results stay unchanged:

```bash
./build/extensions                         # ~45 s; needs benchmarks/results/reference_price.txt
uv run python plots/plot_extensions.py
```

**Variance-corrected low-rank sampler** (`lowrank::price_corrected`). Adding independent noise $D^{1/2}\varepsilon$ with $D = \text{diag}(C - L_k L_k^\top)$ restores every marginal variance. At $N = 500$ (100,000 paths per rank, standard error $\approx 0.56\%$):

| Rank $k$ | 2 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Low-rank price error | −8.1% | −6.8% | −4.3% | −3.7% | −3.9% |
| Low-rank + diag price error | −0.1% | −0.2% | −0.0% | +0.0% | +0.1% |
| Increment variance error, low-rank | 100% | 100% | 100% | 99% | 93% |
| Increment variance error, low-rank + diag | 119% | 64% | 21% | 1.8% | 16% |

The bias disappears at every rank for about 40% more MC time at $k = 32$. The fix does *not* get the increments right: plain low-rank paths are far too smooth, and the correction's noise is white in levels, so its increments are too rough instead. The Asian price depends on volatility levels, not increments, which is why the fix works for this payoff.

**Conditional geometric control variate** (`price_cv` in all three samplers, `price_asian_call_cv` in Python). With $\rho = 0$ the log of the geometric average $G$ is Gaussian given the volatility path, so $\mathbb{E}[(G - K)^+ \mid \sigma]$ is a Black–Scholes formula. The control $(G - K)^+ - \mathbb{E}[(G - K)^+ \mid \sigma]$ has mean zero exactly, and $\beta$ comes from a separate 10% pilot run, so the estimator is unbiased.

| ($M = 10{,}000$) | Variance reduction, $N = 252$ / $1000$ | Time to 0.1% relative SE at $N = 1000$, plain / CV |
|---|---|---|
| Cholesky | 27 / 29 | 401 s / 16.9 s |
| FFT | 29 / 28 | 234 s / 10.0 s |
| rSVD $k = 32$ | 26 / 28 | 128 s / 5.7 s (but biased by ~4%) |
| rSVD $k = 32$ + diag | 27 / 26 | 179 s / 8.0 s |

The reduction is set by the correlation between $V$ and the control, 0.981 ($1/(1 - 0.981^2) \approx 27$). The control cancels the noise from the price shocks but not the noise from the volatility path, which is most of the remaining 4% of the variance.

---

## Performance: batching, a faster generator, threads

Opt-in `price_batched(N, M, batched::Config{batch, threads, rng})` in all three samplers (driver `src/common/batched_mc.hpp`):

```bash
./build/performance                        # ~10 min (--quick: ~10 s)
uv run python plots/plot_performance.py
```

- **Batching.** Blocks of $B = 64$ paths: one triangular product $LZ$ (Cholesky), one GEMM $L_k Z$ (low-rank), or one `fftw_plan_many_dft` call (FFT), so $L$ is read once per block instead of once per path.
- **Fast generator.** xoshiro256++ with a 256-layer ziggurat (`src/common/fast_rng.hpp`): 3.8 ns per normal against 15.2 ns for `std::mt19937` + `std::normal_distribution`.
- **Threads.** `std::thread` over blocks. Block $b$ uses its own stream seeded by $(\text{seed}, b)$ and block sums are reduced in order, so the price is bitwise identical for any thread count (tested).

Time per path, one thread:

| | $N = 1000$: original → batched + fast | $N = 4000$: original → batched + fast |
|---|---|---|
| Cholesky | 112 µs → 49 µs ($2.3\times$) | 1930 µs → 485 µs ($4.0\times$) |
| FFT | 71 µs → 34 µs ($2.1\times$) | 287 µs → 140 µs ($2.1\times$) |
| rSVD $k = 32$ | 40 µs → 23 µs ($1.8\times$) | 159 µs → 90 µs ($1.8\times$) |

Batching only helps where the transform is memory-bound. It speeds up the Cholesky transform $4.3\times$ at $N = 4000$ (to about 42 GFlop/s) and the low-rank product $2.6\times$, but leaves the FFT unchanged, since each transform already works in cache. In this run the Cholesky local exponent between $N = 2000$ and 4000 falls from 2.33 to 1.74, and batched FFT is still $3.5\times$ faster than batched Cholesky at $N = 4000$ (down from $6.7\times$).

Thread scaling is the cleanest test of the bandwidth argument. At $N = 4000$ the per-path Cholesky loop peaks at $1.8\times$ (three threads) and falls to $1.5\times$ at eight, while batched Cholesky reaches $3.1\times$ on the four performance cores and $4.1\times$ on all eight; FFT and rSVD reach $4.3$–$5.7\times$ on eight.

---

## Calibration

`data/calibrate.py` fits the log-vol variogram $\mathbb{E}[(\log\sigma_{t+\Delta} - \log\sigma_t)^2] = \nu^2 \Delta^{2H}$ by log-log OLS, with $\Delta$ in years to match the engine.

```bash
uv run python data/calibrate.py --source yfinance     # 5-day RV proxy: H = 0.117 (R^2 = 0.98)
uv run python data/calibrate.py --file data/raw/oxfordmanrealizedvolatilityindices.csv --ticker .SPX
```

The free yfinance proxy recovers $H$ close to the published 0.1, but its $\nu$ (0.87 per year) is inflated by measurement noise, so the parameter files keep the literature value. The Oxford-Man library is no longer distributed; place a copy in `data/raw/` if you have one.

---

## Future work

`TODO.md` is the prioritized roadmap, with evidence, plans and done-when criteria; items 1–6 and most of 11 are done. The main open items are a **hierarchical (HODLR) sampler**, which exploits the rank-3 far-field blocks and works on non-uniform grids where the FFT cannot, and **spot-vol correlation $\rho < 0$** for a real skew comparison, which moves to a Volterra (rough Bergomi) representation where the Hybrid Scheme of Bennedsen, Lunde & Pakkanen (2017) is the relevant fast method.

---

## Repository layout

```
src/common/        params, fBM covariance, Asian payoff, control-variate and batched MC drivers, RNGs
src/{cholesky,fft,rsvd}/   one header-only sampler each (+ rsvd.hpp) and a *_pricer.cpp main
benchmarks/        benchmark.cpp, extensions.cpp, performance.cpp -> results/*.csv; timing.hpp
data/              params.py, rfsv_model.py (numpy engine), calibrate.py, validate_*.py, greeks.py,
                   profile_memory.py
plots/             plot_*.py; committed figure snapshots (*.png); figures/ (generated)
tests/             test_samplers.cpp (ctest) and test_python.py (pytest)
report-files/      LaTeX report (main.tex, sec1–sec8) and the built main.pdf
experiments/       prototypes for roadmap items (not in the pipeline)
.github/workflows/ ci.yml (Ubuntu, gating) and ci-macos.yml (informational)
run_pipeline.sh    build, tests, benchmarks and every analysis script, snapshotted per run
ALGORITHMS.md      line-by-line walkthrough of the C++ samplers
CHANGELOG.md, TODO.md   history by branch; prioritized roadmap
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
| Craigmile (2003). *Simulating a class of stationary Gaussian processes using the Davies–Harte algorithm, with application to long memory processes.* J. Time Series Analysis 24(5). DOI: 10.1111/1467-9892.00318 | Non-negativity of the embedding when autocovariances are non-positive at nonzero lags |
| Halko, Martinsson & Tropp (2011). *Finding structure with randomness.* SIAM Review 53(2). DOI: 10.1137/090771806 | Algorithm 4.4 in `rsvd.hpp` |
| Candès, Demanet & Ying (2009). *A fast butterfly algorithm for the computation of Fourier integral operators.* Multiscale Model. Simul. 7(4). DOI: 10.1137/080734339 | Low-rank structure of smooth kernels on well-separated blocks |
| Lévy (1992). *Pricing European average rate currency options.* J. Int. Money Finance 11(5). DOI: 10.1016/0261-5606(92)90033-E | Lognormal approximation used in `validate_asian.py` |
| Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance and Stochastics 21(4). DOI: 10.1007/s00780-017-0335-5 | Future work, for Volterra-type rough-vol models |
