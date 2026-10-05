# Rough Volatility Asian Option Pricer

[![CI](https://github.com/t-flora/fbm-fast-pricers/actions/workflows/ci.yml/badge.svg)](https://github.com/t-flora/fbm-fast-pricers/actions/workflows/ci.yml)

A C++ Monte Carlo pricer for an **arithmetic Asian call** under the **Rough Fractional Stochastic Volatility (RFSV)** model, built to compare three ways of sampling fractional Brownian motion (fBM):

| Sampler | Setup | Per path | Exact? |
|---|---|---|---|
| Dense Cholesky | $O(N^3)$ | $O(N^2)$ | Yes |
| Circulant embedding + FFT | $O(N \log N)$ | $O(N \log N)$ | Yes (embedding verified PSD) |
| Global low-rank rSVD, rank $k$ | $O(N^2 k)$ | $O(Nk)$ | No (rank-$k$ truncation) |

The project asks how far asymptotic complexity predicts real speedups on this problem, and *why* each method works or fails. Python scripts add calibration, validation, and structural/stability analysis.

**Headline results** ($M = 10{,}000$ paths, $N$ from 64 to 4000, Apple M2, single-threaded):

- **FFT wins at scale.** It overtakes Cholesky at $N \approx 250$ and is $1.6\times$ faster at $N = 1000$ and $7.6\times$ faster at $N = 4000$ (2.9 s vs 22.3 s), with $O(N)$ memory instead of $O(N^2)$.
- **The FFT method is exact** for every $H \leq \tfrac{1}{2}$. The smallest circulant eigenvalue has the closed form $\Delta t^{2H}(N^{2H} - (N-1)^{2H}) > 0$.
- **Low-rank rSVD is fastest but biased.** At $k = 32$ it is $10\times$ faster than Cholesky at $N = 4000$, but it underprices by about 4% even at $k = 128$. The bias follows the *variance* it discards (13% of the total at $k = 128$), which the commonly reported Frobenius error (1.2%) badly understates.
- **Scaling.** FFT and rSVD scale linearly in $N$ over the whole range. Cholesky's local exponent rises from about 1 to 2.1 as the $N^2$ mat-vec takes over and its factor outgrows the 16 MB cache.
- **Roughness premium.** At the money, rough volatility ($H = 0.1$) adds $0.47$ to the Asian price relative to $H = 0.5$ at the same $\nu$. About 70% of that ($0.33$) survives when the two models are matched on integrated variance.

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

# Full benchmark (~8 min; N up to 4000, median of 3 timings):
# writes benchmarks/results/*.csv and reference_price.txt
./build/benchmark

# Plots from the benchmark CSVs
uv run python plots/plot_scaling.py

# Tests (~5 s): sampler covariances, FFT eigenvalue closed form, rSVD vs optimum,
# Cholesky/FFT price agreement, calibration recovery
./build/test_samplers          # or: ctest --test-dir build
uv run pytest tests/
```

To run everything end to end (build, benchmark, all analysis scripts), use the pipeline script. It snapshots every figure and CSV into `plots/runs/<timestamp>/` with a manifest:

```bash
./run_pipeline.sh            # production parameters
./run_pipeline.sh --fast     # small M, quick smoke test
./run_pipeline.sh --no-iv    # skip the step that needs internet (live SPY chains)
```

All Python scripts write figures to `plots/figures/`. The PNGs committed directly under `plots/` are snapshots of those outputs.

---

## The problem

### Model

Gatheral, Jaisson & Rosenbaum (2018) found that realized log-volatility of equity indices behaves like fBM with Hurst exponent $H \approx 0.1$. RFSV models this directly:

$$\log \sigma_t = \log \sigma_0 + \nu \, W_t^H, \qquad dS_t = r S_t \, dt + \sigma_t S_t \, dZ_t,$$

where $W^H$ is fBM, $\sigma_0$ is the base volatility level, and $\nu$ is the vol-of-vol. Because $H < \tfrac{1}{2}$, increments of $W^H$ are negatively correlated, so volatility paths are "rough".

Parameters live in `src/common/params.hpp` and `data/params.py` (a test keeps them in sync): $H = 0.10$, $\nu = 0.52$, $\sigma_0 = 0.2$, $S_0 = K = 100$, $T = 1$, $r = 0$. Three points matter when reading the numbers:

- **$\nu$ is in year units.** Gatheral et al. report $\nu \approx 0.3$ with time measured in days. The engine measures time in years ($T = 1$, $\Delta t = 1/N$), where the same fit is $0.3 \cdot 252^{0.1} \approx 0.52$.
- **The base level matters for Monte Carlo, not just for realism.** An earlier version had no $\sigma_0$ (so 100% volatility). At $\nu = 0.52$ that payoff is so heavy-tailed that a single path in a million carried 34% of the sample variance, and standard errors were meaningless. At $\sigma_0 = 0.2$ the payoff standard deviation is about 9.5 and stable across batches. (Strictly, the payoff variance is infinite for any lognormal volatility, since $\mathbb{E}[S^2]$ contains $\mathbb{E}[\exp(\Delta t\, e^{2\nu W})]$. In practice the divergence sits many standard deviations out and is never sampled.)
- **No spot-vol correlation** ($\rho = 0$). $Z$ is independent of $W^H$, so the model produces a symmetric smile and no skew.

### Option

The arithmetic Asian call pays

$$V = \max\left(\frac{1}{N}\sum_{n=1}^{N} S_{t_n} - K,\; 0\right).$$

No closed form exists, so I estimate $p = e^{-rT}\,\mathbb{E}[V]$ by Monte Carlo over $M$ paths. The standard error is $\sigma_V / \sqrt{M}$. At the money $p \approx 5.32$ and $\sigma_V \approx 9.5$ (measured from $10^6$ paths), so $M = 10{,}000$ gives a standard error of $\approx 0.095$, about 1.8% of the price.

### Why sampling fBM is the bottleneck

fBM is not Markov, so there is no step-by-step recursion. A path on $N$ grid points is one draw from $\mathcal{N}(0, C)$ with the dense covariance

$$C_{jk} = \tfrac{1}{2}\left(t_j^{2H} + t_k^{2H} - |t_j - t_k|^{2H}\right).$$

Each sampler is a different way to apply a "square root" of $C$ to white noise.

---

## The three samplers

### 1. Dense Cholesky: `src/cholesky/cholesky.hpp`

Build $C$, factor $C = LL^\top$ once in place with `Eigen::LLT<Eigen::Ref<MatrixXd>>` ($N^3/3$ flops, so $L$ overwrites $C$), then compute $W^H = Lz$ for each path through a lower-triangular view ($N^2$ flops).

An earlier version multiplied by a dense copy of $L$, zeros included: twice the flops, and three $N \times N$ matrices in memory. Fixing that cut the $N = 1000$ run by about 30% with unchanged prices.

The factorization is a small share of runtime: 24 ms of 1.15 s at $N = 1000$, and 0.74 s of 22.3 s at $N = 4000$. The $O(MN^2)$ loop dominates once $N^2$ outweighs the $O(N)$ per-path work (Gaussian draws, `exp` calls, payoff accumulation); see Results.

### 2. Circulant embedding + FFT: `src/fft/fft.hpp`

This is the Davies–Harte (1987) / Wood–Chan (1994) method. fBM itself is non-stationary, but its increments (fractional Gaussian noise, fGn) are stationary, with autocovariance

$$\gamma(k) = \frac{\Delta t^{2H}}{2}\left(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}\right).$$

The fGn covariance is therefore Toeplitz. I embed it in a $2N \times 2N$ circulant with first row $c = [\gamma(0), \ldots, \gamma(N{-}1), 0, \gamma(N{-}1), \ldots, \gamma(1)]$. The DFT diagonalizes the circulant, and its eigenvalues are $\lambda = \text{FFT}(c)$.

Per transform: scale complex white noise by $\sqrt{\lambda_j / 2N}$ and apply one inverse FFT. The first $N$ real parts and the first $N$ imaginary parts are two *independent* fGn samples (their cross-covariance vanishes because $\lambda_j = \lambda_{2N-j}$), and cumulative sums turn them into two fBM paths. Using both halves the Gaussian draws, which dominate the per-path cost; it cut the $N = 1000$ run from 1.04 s to 0.71 s. FFTW plans are created once and reused for every path.

**Exactness.** The method is exact iff every $\lambda_j \geq 0$. I check this numerically (`data/validate_stability.py`), and for every $H \in [0.05, 0.501]$ and $N \leq 1008$ tested all eigenvalues are strictly positive, so no clipping ever happens. For $H \leq \tfrac{1}{2}$ this is expected: $\gamma(k) \leq 0$ at every nonzero lag, the case covered by Craigmile (2003). `fft.hpp` throws instead of clipping if a negative eigenvalue ever shows up.

The proof is short. Because $\gamma(k) < 0$ for $k \geq 1$, every eigenvalue $\lambda_j = \gamma(0) + 2\sum_k \gamma(k)\cos(\pi jk/N)$ is at least $\lambda_0$, and the sum telescopes:

$$\lambda_{\min} = \lambda_0 = \Delta t^{2H}\left(N^{2H} - (N-1)^{2H}\right) \approx 2H\,\Delta t^{2H} N^{2H-1} > 0.$$

The margin is thin for rough $H$. At $H = 0.1$, $\min \lambda / \gamma(0)$ is $0.24\%$ at $N = 252$ and $0.08\%$ at $N = 1000$, matching the formula to every printed digit.

> **Pitfall:** embedding the fBM covariance directly does not work, since it is not Toeplitz. Only the increment covariance embeds.

### 3. Global low-rank rSVD: `src/rsvd/lowrank.hpp`, `src/rsvd/rsvd.hpp`

Approximate $C \approx U_k \,\text{diag}(s)\, U_k^\top$ with a randomized SVD (Halko, Martinsson & Tropp 2011, Algorithm 4.4: Gaussian sketch, $q = 2$ subspace iterations with re-orthonormalization, oversampling $p = 5$). Then sample with the $N \times k$ factor $L_k = U_k \,\text{diag}(\sqrt{s})$ at $O(Nk)$ per path.

This sampler is *approximate*: paths have covariance $\nu^2 C_k$, not $\nu^2 C$. Two error measures ($N = 500$) tell very different stories:

| Rank $k$ | 2 | 4 | 8 | 16 | 32 | 64 | 128 |
|---|---|---|---|---|---|---|---|
| Frobenius $\lVert C - C_k \rVert_F / \lVert C \rVert_F$ | 8.5% | 5.4% | 3.5% | 2.5% | 1.9% | 1.5% | 1.2% |
| Variance lost $\text{tr}(C - C_k)/\text{tr}(C)$ | 37.9% | 32.8% | 28.3% | 24.4% | 20.8% | 17.2% | 13.2% |
| Price error vs reference ($\pm 1.1\%$ at 2 SE) | −8.1% | −7.3% | −6.8% | −5.9% | −4.3% | −3.7% | −3.9% |

The Frobenius norm is dominated by the few large eigenvalues and looks reassuring. But the price depends on how much path variance the sampler keeps, and a rank-$k$ truncation discards the long tail of small eigenvalues, which carry the short-scale roughness. At $k = 128$ it still drops 13% of the total variance (40% at the first time step). Less volatility means a lower price, so the sampler underprices at every rank, and the bias decays slowly, like the variance lost.

The rSVD itself is not the problem: at every rank the Frobenius error is within about 2% (relative) of the optimal Eckart–Young truncation. `plots/plot_structure.py` shows where the slow decay comes from. Well-separated off-diagonal blocks of $C$ compress extremely well even at $H = 0.1$: the singular values fall below 1% of $\sigma_1$ by rank 3. The full matrix does not, because the singularity of $|s-t|^{2H}$ sits on the diagonal. A hierarchical H-matrix compresses exactly the far-field blocks and keeps near-diagonal blocks dense, which makes it the natural next step (see Future work).

`price_freed_timed()` frees $C$ before the MC loop, so only $L_k$ (0.24 MB at $N = 1000$, $k = 32$) is resident during sampling.

---

## Results

All numbers come from `./build/benchmark` on an Apple M2 (`-O3 -march=native`, single-threaded) and from the CSVs in `benchmarks/results/`. Each timing is the median of 3 repeats with the same seed.

### Runtime

| Method | $N = 64$ | $252$ | $1000$ | $2000$ | $4000$ |
|---|---|---|---|---|---|
| Cholesky | 0.040 s | 0.18 s | 1.15 s | 5.17 s | 22.25 s |
| FFT | 0.045 s | 0.17 s | 0.71 s | 1.49 s | 2.91 s |
| rSVD, $k = 32$ | 0.033 s | 0.10 s | 0.43 s | 0.94 s | 2.13 s |

A single power law fits FFT and rSVD well ($\alpha = 1.02$ for both, $R^2 \geq 0.995$) but not Cholesky ($\alpha = 1.52$, $R^2 = 0.978$), whose curve bends. Local exponents between successive $N$ show why:

| Method | 64–128 | 128–252 | 252–500 | 500–1000 | 1000–2000 | 2000–4000 |
|---|---|---|---|---|---|---|
| Cholesky | 0.95 | 1.22 | 1.29 | 1.44 | 2.17 | 2.10 |
| FFT | 0.92 | 1.02 | 1.04 | 1.02 | 1.06 | 0.97 |

At small $N$ the $O(N)$ per-path work hides the $N^2$ mat-vec. Above $N \approx 1000$ the mat-vec dominates and the exponent slightly exceeds 2, because $L$ no longer fits in cache (next section).

FFT's $\log N$ factor stays invisible even over this $64\times$ range: its MC cost per path per time step is flat at 67–73 ns (`plots/per_path_cost.png`). The transform is a small part of each path; random number generation and the price-path arithmetic dominate, and those are $O(N)$. Construction is negligible for FFT and at most a few percent for Cholesky, but 23% of rSVD's runtime at $N = 4000$ (`plots/construction_breakdown.png`).

### Memory and cache

| Method ($N = 1000$) | Resident during MC loop | Measured lifetime peak |
|---|---|---|
| Cholesky | $8N^2 = 7.6$ MB ($L$ overwrites $C$; the lower half is read per path) | 9.7 MB |
| FFT | 0.08 MB ($2N$ scale factors, two length-$2N$ complex buffers) | 0.4 MB |
| rSVD, $C$ held | 7.6 MB + $L_k$ | 10.7 MB |
| rSVD, $C$ freed | 0.24 MB ($L_k$ only, $8Nk$ bytes) | 12.8 MB |

The measured peak (`measured_peak_mb`) comes from running each method in a forked child process and reading its peak RSS via `wait4`, minus that of an idle child. It exceeds the dominant-array size because of allocator and matrix-multiply workspace. Freeing $C$ shrinks what is resident during the MC loop, but not the lifetime peak, since $C$ must exist while $L_k$ is built. At $N = 4000$ the Cholesky and rSVD peaks reach about 145–160 MB, while FFT needs 1.1 MB.

The relevant cache on the M2 is the 16 MB L2 shared by the performance cores; the M2 has no L3. $L$ fits in it up to $N \approx 1450$. The `est_bandwidth_GBs` column ($4N(N+1)M$ bytes, the lower triangle of $L$ once per path, divided by wall time) rises to 35 GB/s at $N = 1000$ and then *falls*, to 31 GB/s at $N = 2000$ and 29 GB/s at $N = 4000$, once $L$ spills out of cache. That turnover is the signature of the loop becoming memory-bandwidth bound. It is still an effective rate, not a direct measurement of DRAM traffic.

### Reference price

The reference is $p_\text{ref} = 5.3144$ at $N = 500$: the average of Cholesky (5.3195) and FFT (5.3093), each with 500,000 paths. Each estimate has a standard error of $\approx 0.013$, and their difference ($0.010$) is about half the standard error of a difference, consistent with both samplers being exact.

### Numerical stability (`data/validate_stability.py`)

- **FFT:** no negative eigenvalues anywhere (see above).
- **Cholesky:** $\kappa(C) = 789$ at $N = 252$ and $\approx 4 \times 10^3$ at $N = 1000$, fitting $\kappa \approx 1.08\,N^{1.19}$. Float64 Cholesky has enormous headroom; trouble would start near $N \sim 10^{12}$.
- **rSVD factor:** $\kappa(L_k) = 8.2$, 17.6, and 23.3 at $k = 8$, 32, 64 ($N = 252$), so it is well conditioned.

---

## Validation and sensitivity experiments

These use `data/rfsv_model.py`, a vectorized numpy port of the FFT sampler that matches the C++ normalization exactly.

| Script | What it does | Output (`plots/figures/`) |
|---|---|---|
| `data/validate_convergence.py` | Price $\pm 1\sigma$ vs $M$ over 20 seeds against a same-grid $10^6$-path reference; checks $\sigma \propto M^{-1/2}$ | `convergence.png` |
| `data/validate_asian.py` | Price vs strike for several $H$ against Lévy (1992) and exact GBM; roughness premium at fixed and at variance-matched $\nu$ | `validate_asian.png` |
| `plots/plot_sensitivity.py` | ATM price over $H \times \nu$; price vs strike for $H \in \{0.05, \ldots, 0.5\}$ | `sensitivity_surface.png`, `sensitivity_strike.png` |
| `data/validate_iv.py` | RFSV smile vs live SPY option IVs (needs internet; `--M 20000`) | `validate_iv.png` |
| `data/validate_stability.py` | FFT eigenvalues vs $H$; $\kappa(L_k)$ vs $k$; $\kappa(C)$ vs $N$ | `stability_report.png` |
| `plots/plot_structure.py` | Toeplitz structure of fGn; off-diagonal vs full spectrum of $C$ | `structure_analysis.png` |
| `data/profile_memory.py` | `tracemalloc` peak of the numpy fBM sampler vs $(N, M)$ | `memory_profile.png` |

All comparisons between configurations (different $H$, $\nu$ or $K$) use common random numbers, so differences are far more precise than the absolute prices. Key results:

- **Convergence.** With 20 seeds the standard deviation of the estimate falls with fitted slope $-0.488$ ($R^2 = 0.986$) against the theoretical $-0.5$. The same-grid reference ($N = 252$) is $5.3252 \pm 0.0095$.
- **Lévy is accurate at realistic volatility.** At the money it gives 4.625, against $4.606 \pm 0.027$ for exact GBM by Monte Carlo. (At the old 100% volatility it overpriced by about 6%.)
- **Roughness premium.** At the money, $p(H{=}0.10) - p(H{=}0.50) = 0.468 \pm 0.008$ at $\nu = 0.52$. But lower $H$ also raises the expected integrated variance $\mathbb{E}\int_0^T \sigma_t^2\,dt$ by 19%. Matching it (by solving for $\nu = 0.651$ at $H = 0.5$) leaves $0.330 \pm 0.009$, so about 70% of the premium is due to roughness itself. The premium peaks at the money and is positive at every strike from 80 to 120.
- **Sensitivity.** At $\nu = 0.52$ the ATM price falls from 5.37 ($H = 0.05$) to 4.74 ($H = 0.5$). The gap between $H = 0.1$ and $H = 0.5$ vanishes at small $\nu$ (0.02 at $\nu = 0.1$) and grows to 0.94 at $\nu = 0.7$.
- **IV smile.** For a snapshot taken October 2, 2026 the model smile is a symmetric U with its minimum at the money: curvature, but no skew. SPY's smile falls steeply below the money. With $\rho = 0$ the model cannot produce skew, so the comparison tests curvature only. Two implementation details matter here. All strikes share one set of paths, and the simulated $S_T$ is rescaled so that its sample mean equals the forward; without that, a 0.05% forward error tilted the model curve into a fake skew. Deep in-the-money market IVs are inflated by the $r = 0$, no-dividend assumption and are shown off scale.

---

## Extensions: variance correction and a control variate

Two opt-in additions, measured by a separate program so the main benchmark results above stay unchanged:

```bash
./build/extensions                         # ~45 s; needs benchmarks/results/reference_price.txt
uv run python plots/plot_extensions.py     # -> plots/figures/{variance_corrected_rank,control_variate}.png
```

### Variance-corrected low-rank sampler (`lowrank::price_corrected`, `lowrank::price_cv(..., corrected=true)`)

The rank-$k$ sampler underprices because it drops the variance held by small eigenvalues. Adding independent noise at each step restores every marginal variance exactly:

$$W^H \approx L_k z + D^{1/2}\varepsilon, \qquad D = \text{diag}(C - L_k L_k^\top).$$

At $N = 500$ with 100,000 paths per rank (standard error $\approx 0.56\%$):

| Rank $k$ | 2 | 8 | 32 | 64 | 128 |
|---|---|---|---|---|---|
| Low-rank price error | −8.1% | −6.8% | −4.3% | −3.7% | −3.9% |
| Low-rank + diag price error | −0.1% | −0.2% | −0.0% | +0.0% | +0.1% |
| Increment variance error, low-rank | 100% | 100% | 100% | 99% | 93% |
| Increment variance error, low-rank + diag | 119% | 64% | 21% | 1.8% | 16% |

The price bias disappears at every rank, for about 40% more MC time at $k = 32$ (the extra $N$ Gaussian draws per path). The structural rows show what the fix does *not* do. Plain low-rank paths are far too smooth: they miss almost all of the increment (fGn) variance, which is where the roughness lives. The correction's per-step noise is white in levels, so the increments become too rough instead, and they are only close to right around $k = 64$. The Asian price depends on volatility levels, not increments, which is why the fix works for this payoff. A payoff sensitive to path roughness itself (for example realized vol-of-vol) would need a banded correction.

### Conditional geometric control variate (`price_cv` in all three samplers, `price_asian_call_cv` in Python)

With $\rho = 0$, the log of the geometric average $G$ is Gaussian given the volatility path, so $\mathbb{E}[(G - K)^+ \mid \sigma]$ is a Black–Scholes formula, computed per path in $O(N)$. With $V$ the arithmetic payoff and $C = (G - K)^+ - \mathbb{E}[(G - K)^+ \mid \sigma]$ (mean zero exactly), the estimator $e^{-rT}\,\overline{V - \beta C}$ is unbiased. $\beta$ comes from a separate pilot run of 10% of the paths (`src/common/control_variate.hpp`).

| ($M = 10{,}000$) | Variance reduction, $N = 252$ / $1000$ | Time to 0.1% relative SE at $N = 1000$, plain / CV |
|---|---|---|
| Cholesky | 27 / 29 | 355 s / 15.0 s |
| FFT | 29 / 28 | 218 s / 9.6 s |
| rSVD $k = 32$ | 26 / 28 | 125 s / 5.5 s (but biased by ~4%) |
| rSVD $k = 32$ + diag | 27 / 26 | 173 s / 7.8 s |

The variance falls by a factor of 26–29 for every sampler, and the time to a given accuracy falls by about the same factor, pilot included. The reduction is set by the correlation between $V$ and the control $C$, which is 0.981 ($1/(1 - 0.981^2) \approx 27$); the raw arithmetic and geometric payoffs correlate more strongly (0.998). The gap is informative. Because $C$ has mean zero given the volatility path, the control cancels the noise from the price shocks but not the noise from the volatility path itself. The remaining variance, about 4% of the original, is roughly the size of that path-driven part ($\mathbb{E}[(G - K)^+ \mid \sigma]$ alone accounts for 2.7% of $\text{Var}(V)$). Removing it would take a second control on the volatility path, or quasi-Monte Carlo. A smaller variance does not fix bias: the plain rSVD sampler converges quickly to the wrong price. The closed form needs $\rho = 0$, since conditioning on the volatility path must leave the price shocks independent.

Both extensions are covered by the test suites: the closed form matches a Monte Carlo average for a fixed volatility path, the control-variate prices of the exact samplers agree, the corrected sampler's marginal variances equal $\text{diag}(C)$ to $10^{-16}$, and its price matches the exact one, while the uncorrected sampler's bias is detected.

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

The yfinance proxy recovers $H$ close to the published $0.1$. Its $\nu$ is inflated by measurement noise in weekly squared-return RV, which is why the parameter files keep the literature value ($0.3$ in days, i.e. $0.52$ in years) rather than this fit. The Oxford-Man library is no longer distributed from its original URL; place a copy in `data/raw/` if you have one.

---

## Future work

`TODO.md` has the full prioritized roadmap, with evidence, plans and done-when criteria. The top items:

Items 1–3 of the roadmap are done: the two extensions above, and CI on every push (Ubuntu with GCC, macOS with Apple Clang). The top remaining items:

1. **Performance:** batch paths into matrix-matrix products (Cholesky is bandwidth-bound at large $N$), use a faster Gaussian generator (it dominates the FFT per-path cost), and add multithreading.
2. **A hierarchical (HODLR) sampler**, which exploits the rank-3 far-field blocks and works on non-uniform grids where the FFT cannot.
3. **Spot-vol correlation $\rho < 0$**, needed for a real skew comparison. This moves to a Volterra (rough Bergomi) representation, where the Hybrid Scheme of Bennedsen, Lunde & Pakkanen (2017) becomes the relevant fast method.

---

## Repository layout

```
src/common/        params.hpp (model + run parameters), covariance.hpp (fBM kernel),
                   asian_payoff.hpp (vol path -> price path -> payoff), rng.hpp,
                   control_variate.hpp (geometric control-variate MC driver)
src/cholesky/      cholesky.hpp + cholesky_pricer.cpp
src/fft/           fft.hpp + fft_pricer.cpp
src/rsvd/          lowrank.hpp (sampler), rsvd.hpp (Halko et al. Alg. 4.4), rsvd_pricer.cpp
benchmarks/        benchmark.cpp -> results/{time_vs_N,error_vs_rank}.csv, reference_price.txt;
                   extensions.cpp -> results/{variance_corrected_rank,control_variate}.csv
data/              params.py (shared model parameters), calibrate.py, rfsv_model.py (numpy engine),
                   validate_*.py, profile_memory.py
plots/             plot_scaling.py, plot_structure.py, plot_sensitivity.py, plot_extensions.py;
                   figures/ (generated)
experiments/       prototype_further_work.py (prototypes for roadmap items)
.github/workflows/ ci.yml (build + both test suites on every push and pull request)
report-files/      LaTeX report (main.tex + sec*.tex)
ALGORITHMS.md      Line-by-line walkthrough of the C++ samplers
tests/             test_samplers.cpp (C++, via ctest) and test_python.py (pytest)
run_pipeline.sh    Build + tests + benchmark + every analysis script, snapshotted per run
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
