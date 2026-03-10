# Rough Volatility Asian Option Pricer

A high-performance Monte Carlo pricer for an **Arithmetic Asian Call Option** under the **Rough Fractional Stochastic Volatility (RFSV)** model, benchmarking three algorithms for generating fractional Brownian motion (fBM) paths. The project is a practical study in how complexity theory translates to real speedups on a computationally demanding financial problem.

---

## The Problem

### Why rough volatility?

Gatheral, Jaisson & Rosenbaum (2014) showed by estimating volatility from high-frequency data that **log-volatility behaves essentially as fractional Brownian motion with Hurst exponent $H \approx 0.1$**. This is empirically robust across many equity indices and time scales.

The **RFSV model** specifies log-volatility as:

$$\log \sigma_t = \nu \, W_t^H$$

where $W^H$ is fBM with $H \approx 0.1$ and $\nu$ is the volatility-of-volatility. Because $H < \tfrac{1}{2}$, increments are **anti-persistent** ("rough") — each volatility spike is likely to reverse. This contradicts classical long-memory models ($H > \tfrac{1}{2}$) and matches observed implied volatility smiles far better.

### Why is this computationally hard?

fBM is **non-Markovian**: the future of $W^H_t$ depends on the entire past path. There is no recursion to step forward. Exact path generation requires sampling from the full $N \times N$ multivariate Gaussian:

$$C_{ij} = \mathbb{E}\left[W^H(t_i)\, W^H(t_j)\right] = \tfrac{1}{2}\left(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H}\right)$$

This matrix is dense — every pair of time points is correlated — and dense Cholesky costs $O(N^3)$.

### The option

**Arithmetic Asian Call**: payoff at maturity is

$$V = \max\left(\frac{1}{N}\sum_{i=1}^{N} S_i - K,\; 0\right)$$

The path-dependent average eliminates any closed-form pricing formula. Monte Carlo is the standard method.

---

## Three Algorithms

### Algorithm 1 — Dense Cholesky

**File:** `src/cholesky/cholesky.hpp` | **Cost:** $O(N^3)$ setup, $O(N^2)$ per path

Build the full $N \times N$ fBM covariance matrix $C$ and factorize it as $C = LL^\top$ once via `Eigen::LLT`. For each Monte Carlo path, draw $z \sim \mathcal{N}(0, I_N)$ and compute $\log \sigma = \nu L z$.

The per-path $O(MN^2)$ term dominates for $M = 10{,}000$ paths, yielding an observed scaling exponent of $\approx 1.54$ rather than 3.

### Algorithm 2 — Circulant Embedding + FFT

**File:** `src/fft/fft.hpp` | **Cost:** $O(N \log N)$ setup, $O(N \log N)$ per path

Based on the **Davies–Harte / Wood–Chan** exact method. The key insight: fBM **increments** (fractional Gaussian noise, fGn) are stationary even though fBM is not, so the increment covariance is Toeplitz. A Toeplitz matrix embeds into a circulant whose eigenvalues are the DFT of its first row.

**fGn autocovariance** at lag $k$:

$$\gamma(k) = \frac{\mathrm{d}t^{2H}}{2}\!\left(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}\right)$$

**Embedding:** size-$2N$ circulant first row $c = [\gamma(0), \ldots, \gamma(N{-}1), 0, \gamma(N{-}1), \ldots, \gamma(1)]$

**Eigenvalues:** $\lambda = \text{FFT}(c)$ — all $\lambda_j \geq 0$ for $H \leq \tfrac{1}{2}$ (Wood & Chan 1994).

**Per path:** draw complex white noise $w_j$, scale by $\sqrt{\lambda_j / 2N}$, apply IFFT, take real part → fGn increments. Cumulative sum gives the fBM log-vol path. FFTW plans are created once and reused.

> **Critical:** the fBM covariance itself is non-Toeplitz and produces negative eigenvalues if embedded directly. Only the increment covariance embeds into a valid PSD circulant.

### Algorithm 3 — H-matrix + Randomized SVD

**Files:** `src/hmatrix/hmatrix.hpp`, `src/hmatrix/rsvd.hpp` | **Cost:** $O(Nk^2)$ setup, $O(Nk)$ per path

The fBM covariance kernel is smooth away from its diagonal, so off-diagonal blocks are numerically low-rank (Candès, Demanet & Ying 2008). A global rank-$k$ randomized SVD (Halko, Martinsson & Tropp 2011) approximates:

$$C \approx U \text{diag}(S) U^\top, \qquad U \in \mathbb{R}^{N \times k}$$

The approximate Cholesky factor $L_k = U \text{diag}(\sqrt{S})$ satisfies $L_k L_k^\top \approx C$, reducing per-path cost to $O(Nk)$.

**rSVD** (Algorithm 4.4 from Halko et al.): random sketch $Y = C\Omega$, $q = 2$ power iterations (critical for slowly-decaying spectra at $H = 0.1$), QR decomposition, thin SVD on the small projected matrix.

Two variants implemented: `price_timed()` (holds $C$ in memory throughout) and `price_freed_timed()` (releases $C$ before the MC loop, reducing peak RSS from $O(N^2)$ to $O(Nk)$).

---

## Computational Results

Benchmarked at $M = 10{,}000$ paths, $N \in \{252, 500, 1000\}$:

| Method | Complexity | Fitted $\alpha$ | $c$ | $R^2$ |
|--------|-----------|-----------------|-----|-------|
| Dense Cholesky | $O(MN^2 + N^3)$ | 1.54 | $4.1 \times 10^{-5}$ | 0.999 |
| Circulant+FFT | $O(MN \log N)$ | 1.01 | $1.0 \times 10^{-3}$ | 1.000 |
| H-matrix+rSVD $(k=32)$ | $O(MNk)$ | 1.06 | $3.0 \times 10^{-4}$ | 1.000 |

FFT and H-matrix both scale as $O(N)$ in this regime (the $\log N$ factor is invisible across a $4\times$ range of $N$).

**Memory** (Apple M2, $N = 1000$):

| Method | Peak memory | Cache pressure (L3 = 16 MB) |
|--------|------------|------------------------------|
| Cholesky | 7.6 MB ($N^2 \cdot 8$) | $0.48\times$ |
| FFT | 0.015 MB ($2N \cdot 8$) | $0.001\times$ |
| H-matrix (C held) | 7.6 MB | $0.48\times$ |
| H-matrix (C freed) | 0.24 MB ($Nk \cdot 8$) | $0.015\times$ |

Cholesky memory-bandwidth utilization: $\approx 47$ GB/s at $N = 1000$ (47% of M2's rated 100 GB/s), confirming the MC loop is memory-bound.

**H-matrix approximation quality** ($\|C - C_k\|_F / \|C\|_F$):

| Rank $k$ | Frobenius error |
|----------|----------------|
| 4 | 5.4% |
| 16 | 2.5% |
| 64 | 1.5% |
| 128 | 1.2% |

Slow convergence is fundamental: the singular spectrum of $C$ decays slowly for rough $H = 0.1$.

**Reference price:** $p_\text{ref} = 23.58$ (average of 500k-path Cholesky and 500k-path FFT; $H = 0.10$, $\nu = 0.30$, $S_0 = K = 100$, $T = 1$).

---

## Validation

### Phase 1 — Implied volatility smile vs SPY option chains

`data/validate_iv.py` fetches live SPY European option chains via yfinance, computes market implied volatilities via Black-Scholes inversion, and overlays RFSV model IVs. A log-vol drift $\mu_0$ is auto-calibrated to match the ATM market IV, isolating the smile *shape* rather than level.

```bash
uv run python data/validate_iv.py [--M 3000] [--N 63]
# → plots/validate_iv.png
```

### Phase 2 — Lévy benchmark + roughness premium

`data/validate_asian.py` benchmarks the RFSV Asian price against the Lévy (1992) log-normal approximation and computes the roughness premium: the price increase when H is reduced from 0.5 (standard GBM) to 0.1 (rough volatility). Runs 3 seeds for ±1σ error bars.

```bash
uv run python data/validate_asian.py [--M 5000] [--N 252]
# → plots/validate_asian.png
```

### Phase 3 — Sensitivity: H × nu heatmap + price vs strike

`plots/plot_sensitivity.py` sweeps $H \in \{0.05, 0.10, 0.15, 0.20\}$ and $\nu \in \{0.1, 0.2, 0.3, 0.4\}$, producing an ATM price heatmap and price-vs-strike curves for varying H.

```bash
uv run python plots/plot_sensitivity.py [--M 10000] [--N 252]
# → plots/sensitivity_surface.png, plots/sensitivity_strike.png
```

---

## Structural Analysis

`plots/plot_structure.py` provides the geometric motivation for each algorithm:

- **Toeplitz structure of fGn** (why FFT works): fGn covariance heatmap shows constant diagonals; fBM covariance does not. The translational invariance of increments is exactly what enables the circulant embedding.
- **Low-rank off-diagonal blocks** (why H-matrix/rSVD works): singular value decay of the off-diagonal block is faster for $H = 0.5$ than $H = 0.1$, connecting to Candès-Demanet-Ying (2008).

```bash
uv run python plots/plot_structure.py [--N-small 64] [--N-large 128]
# → plots/structure_analysis.png
```

---

## Additional Experiments

| Script | What it measures | Output |
|--------|-----------------|--------|
| `data/validate_convergence.py` | Price ±1σ vs M paths; log-log slope confirms σ ∝ 1/√M | `plots/convergence.png` |
| `data/validate_stability.py` | FFT eigenvalue clipping near H=0.5; rSVD condition number vs rank; Cholesky κ(C) vs N | `plots/stability_report.png` |
| `data/profile_memory.py` | Python engine tracemalloc peak vs N×M | `plots/memory_profile.png` |
| `plots/plot_scaling.py` | Timing + complexity fits; construction vs MC breakdown; memory vs N | `plots/time_vs_N.png`, `plots/construction_breakdown.png`, `plots/memory_vs_N.png`, `plots/error_vs_rank.png` |

---

## A Fourth Method (future work): Hybrid Scheme

Bennedsen, Lunde & Pakkanen (2017) introduce a simulation scheme better suited to rough kernels. The fBM kernel $g(x) = x^{H - 1/2}$ has a power-law singularity near zero that circulant embedding handles poorly (the first step misses the spike). The hybrid scheme approximates $g$ analytically near zero and with step functions elsewhere, keeping $O(N \log N)$ complexity while reducing RMSE by more than 80% for $H = 0.1$.

---

## Build and Run

```bash
# Build all binaries
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Calibrate H and nu from market data
uv run python data/calibrate.py --source yfinance   # free; H ≈ 0.12
# (or place Oxford-Man CSV in data/raw/ and omit --source)
# Paste printed H and nu into src/common/params.hpp, then rebuild.

# Run C++ benchmark (writes CSVs to benchmarks/results/)
./build/benchmark

# All plots
uv run python plots/plot_scaling.py
uv run python plots/plot_structure.py
uv run python data/validate_convergence.py
uv run python data/validate_stability.py
uv run python data/validate_iv.py
uv run python data/validate_asian.py
uv run python plots/plot_sensitivity.py
```

---

## Project Structure

```
papers-bg/              Source papers (references below)
data/
  calibrate.py          Estimates H, nu from Oxford-Man or yfinance
  rfsv_model.py         Vectorized Python fBM Monte Carlo engine
  validate_iv.py        Phase 1: IV smile vs live SPY chains
  validate_asian.py     Phase 2: Lévy benchmark + roughness premium
  validate_convergence.py  MC convergence study (price vs M)
  validate_stability.py    Stability: clipping, conditioning, κ(C)
  profile_memory.py     Python engine tracemalloc profiling
src/common/
  params.hpp            Calibrated H, nu, S0, K, T, r (update after calibration)
  covariance.hpp        fBM kernel + Eigen matrix builder
  asian_payoff.hpp      Payoff + log-vol → price path
  rng.hpp               Seeded RNG helpers
src/cholesky/           Algorithm 1: Dense Cholesky
src/fft/                Algorithm 2: Circulant embedding + FFTW
src/hmatrix/            Algorithm 3: H-matrix + rSVD (rsvd.hpp = Halko et al. Alg 4.4)
benchmarks/
  benchmark.cpp         Timing + accuracy + memory runner; writes CSVs
  results/              time_vs_N.csv, error_vs_rank.csv, reference_price.txt
plots/
  plot_scaling.py       Timing fits + construction breakdown + memory vs N
  plot_structure.py     Toeplitz heatmaps + SVD decay
  plot_sensitivity.py   Phase 3: H×nu heatmap + price vs strike
ALGORITHMS.md           Beginner's guide to each algorithm with code annotations
```

---

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| Eigen3 | Dense matrix ops, Cholesky | `brew install eigen` |
| FFTW3 | Fast Fourier transforms | `brew install fftw` |
| CMake 3.16+ | Build system | `brew install cmake` |
| Python (uv) | Calibration, validation, plotting | `uv add pandas numpy scipy matplotlib seaborn yfinance` |

---

## References

| Paper | Role in this project |
|-------|---------------------|
| Gatheral, Jaisson & Rosenbaum (2014). *Volatility is rough.* | Empirical $H \approx 0.1$; RFSV model definition; calibration method |
| Halko, Martinsson & Tropp (2011). *Finding structure with randomness.* SIAM Review 53(2) | Algorithm 4.4 (rSVD with power iteration) in `rsvd.hpp` |
| Candès, Demanet & Ying (2008). *A fast butterfly algorithm for Fourier integral operators.* | Theoretical basis for low-rank off-diagonal blocks in smooth kernels |
| Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* | Better simulation for rough kernels; identified as future work |
