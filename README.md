# Rough Volatility Asian Option Pricer

A high-performance **option pricer** for an **Arithmetic Asian Call Option** under the **Rough Fractional Stochastic Volatility (RFSV)** model. The output is a single number: the fair price of the option (expected discounted payoff under the model). Because no closed-form formula exists, we estimate it via Monte Carlo — averaging the payoff over many independently simulated stock-price paths. The project benchmarks three algorithms for generating those paths using fractional Brownian motion (fBM), and studies how complexity theory translates to real speedups on this computationally demanding problem.

---

## The Problem

### Why rough volatility?

Gatheral, Jaisson & Rosenbaum (2014) showed by estimating volatility from high-frequency data that **log-volatility behaves essentially as fractional Brownian motion with Hurst exponent $H \approx 0.1$**. This is empirically robust across many equity indices and time scales.

The **RFSV model** specifies log-volatility as:

$$\log \sigma_t = \nu W_t^H$$

where $W^H$ is fBM with $H \approx 0.1$ and $\nu$ is the volatility-of-volatility. The full Gatheral RFSV model includes a log-vol drift $\mu$: $\log\sigma_t = \mu + \nu W_t^H$. This implementation sets $\mu = 0$, so $\sigma_0 = e^{\mu + \nu W_0^H} = 1$ — a deliberate simplification giving 100% base annualized volatility, consistent with the reference price $p_\text{ref} = 23.58$. When comparing against market data (Phase 1), $\mu_0 = \log\sigma_\text{target}$ is auto-calibrated to match the ATM implied volatility level. Because $H < \tfrac{1}{2}$, increments are **anti-persistent** ("rough") — each volatility spike is likely to reverse. This contradicts classical long-memory models ($H > \tfrac{1}{2}$) and matches observed implied volatility smiles far better.

### Why is this computationally hard?

fBM is **non-Markovian**: the future of $W^H_t$ depends on the entire past path. There is no recursion to step forward. Sampling an fBM path at $N$ discrete times requires drawing from an $N$-dimensional multivariate Gaussian whose covariance matrix $C$ is given analytically by the fBM kernel:

$$C_{ij} = \mathbb{E}\left[W^H(t_i) W^H(t_j)\right] = \tfrac{1}{2}\left(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H}\right)$$

$C$ is not estimated from data — it is computed exactly from this formula. The challenge is that $C$ is dense (every pair of time points is correlated), so computing the Cholesky factorization $C = LL^\top$ — the standard way to sample from $\mathcal{N}(0, C)$ — costs $O(N^3)$.

### The option

**Arithmetic Asian Call**: payoff at maturity is

$$V = \max\left(\frac{1}{N}\sum_{i=1}^{N} S_i - K,\; 0\right)$$

The path-dependent average eliminates any closed-form pricing formula. **Monte Carlo** is the standard method: simulate $M$ independent stock-price paths (each driven by a fresh fBM sample), compute the payoff $V$ on each, and average. The option price is $\hat{p} = e^{-rT} \cdot \frac{1}{M}\sum_{m=1}^{M} V^{(m)}$, with standard error $e^{-rT} \sigma_V / \sqrt{M}$.

---

## Three Algorithms

### Algorithm 1 — Dense Cholesky

**File:** `src/cholesky/cholesky.hpp` | **Cost:** $O(N^3)$ setup, $O(N^2)$ per path

Build the full $N \times N$ fBM covariance matrix $C$ analytically (from the formula above) and factorize it as $C = LL^\top$ **once** via `Eigen::LLT` (the $O(N^3)$ step). Then, for each of the $M$ Monte Carlo paths: draw $z \sim \mathcal{N}(0, I_N)$ and compute $\log \sigma = \nu L z$ — a single triangular matrix-vector multiply ($O(N^2)$ flops) that produces one correlated fBM log-volatility path. Exponentiate to get instantaneous volatility, simulate stock prices, compute the Asian payoff, and average over all $M$ paths.

The per-path triangular solve costs $O(N^2)$, so the full MC loop costs $O(MN^2)$, which at $M = 10{,}000$ dominates the one-time $O(N^3)$ factorization (explaining why the slope is not 3). The pure $O(N^2)$ per-path cost would predict an empirical exponent of $\approx 2$, but the observed exponent is $\approx 1.54$ because each Monte Carlo iteration also carries significant $O(N)$ overhead: $N$ Gaussian draws, $N$ exponentials to recover $\sigma$ from $\log\sigma$, and $N$ steps of the Asian payoff accumulation. In the tested range $N \in \{252, 1000\}$ these $O(N)$ operations are large enough relative to the $O(N^2)$ triangular multiply to pull the effective scaling exponent below 2.

### Algorithm 2 — Circulant Embedding + FFT

**File:** `src/fft/fft.hpp` | **Cost:** $O(N \log N)$ setup, $O(N \log N)$ per path

Based on the **Davies–Harte** (1987) / **Wood–Chan** (1994) exact method. The key insight: fBM **increments** $\delta W_k = W_{(k+1)\Delta t} - W_{k\Delta t}$ (fractional Gaussian noise, fGn) are stationary even though fBM itself is not. Stationarity means $\operatorname{Cov}(\delta W_i, \delta W_{i+k})$ depends only on the lag $k$, not on the absolute position $i$. A covariance matrix whose $(i,j)$ entry depends only on $|i-j|$ has constant diagonals — by definition, it is **Toeplitz**. A Toeplitz matrix can be embedded into a circulant, and circulant matrices are diagonalized by the DFT, whose eigenvalues are the FFT of the circulant's first row.

**fGn autocovariance** at lag $k$ (with time step $\Delta t = T/N$):

$$\gamma(k) = \frac{\Delta t^{2H}}{2}\!\left(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}\right)$$

**Embedding:** size-$2N$ circulant first row $c = [\gamma(0), \ldots, \gamma(N{-}1), 0, \gamma(N{-}1), \ldots, \gamma(1)]$

**Eigenvalues:** $\lambda = \text{FFT}(c)$ — all $\lambda_j \geq 0$ is guaranteed for $H \geq \tfrac{1}{2}$ at any $N$ (Wood & Chan 1994). For the rough case $H < \tfrac{1}{2}$, the fGn autocovariance is negative at all lags $k \geq 1$, so the finite-$N$ circulant can produce negative eigenvalues; non-negativity holds only in the large-$N$ limit where the spectral density is bounded away from zero. See the stability section for empirical clipping rates at $H = 0.1$.

**Per path:** draw complex white noise $w_j$, scale by $\sqrt{\lambda_j / 2N}$, apply IFFT, take real part → fGn increments. Cumulative sum gives the fBM log-vol path.

**FFTW** ("Fastest Fourier Transform in the West") is the C library used for all FFT computation. A **plan** is a one-time strategy computed at startup: FFTW profiles several algorithm variants and memory layouts for the given array size and CPU, then stores the winning recipe. Executing the plan is then near-optimal. Creating a plan takes ~1 ms; reusing it across all $M$ paths amortizes that cost to nothing. Two plans are created once: one for the forward FFT (eigenvalue computation) and one for the inverse FFT (per-path synthesis).

> **Critical:** the fBM covariance itself is non-Toeplitz and produces negative eigenvalues if embedded directly. Only the increment (fGn) covariance embeds into a valid PSD circulant.

### Algorithm 3 — Global Low-Rank Approximation via rSVD

**Files:** `src/hmatrix/hmatrix.hpp`, `src/hmatrix/rsvd.hpp` | **Cost:** $O(N^2k)$ setup, $O(Nk)$ per path

The fBM covariance kernel is smooth away from its diagonal, which motivates approximating $C$ as globally low-rank (Candès, Demanet & Ying 2008). For $H = 0.1$, however, the singularity of $|s-t|^{2H} = |s-t|^{0.2}$ near the diagonal is severe, and its influence spreads into off-diagonal blocks — causing the singular values of $C$ to decay slowly rather than rapidly. The algorithm therefore works as an *attempt* to exploit smoothness that is ultimately bottlenecked by the roughness of the kernel: moderate rank $k$ is needed to achieve even 1–2% error (see approximation quality table below), and power iterations in the rSVD are critical to handle the flat spectrum. A global rank-$k$ randomized SVD (Halko, Martinsson & Tropp 2011) approximates:

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
| Low-rank rSVD $(k=32)$ | $O(MNk + N^2k)$ | 1.06 | $3.0 \times 10^{-4}$ | 1.000 |

The full FFT loop theoretically scales as $O(MN \log N)$ and low-rank rSVD as $O(MNk + N^2k)$, but both **appear linear in $N$** in the fitted data. This is an artifact of the narrow $N$ range tested: going from $N = 252$ to $N = 1000$ is only a $4\times$ increase, over which $\log_2 N$ grows from $\approx 7.97$ to $\approx 9.97$ — a factor of 1.25. A 25% multiplicative drift in the prefactor is smaller than the noise in a three-point log-log regression, so the fitted exponent comes out as 1.01 rather than something distinguishably above 1. The $\log N$ factor has not vanished; it is simply unresolvable at this scale. To observe it cleanly, you would need to benchmark over a range of $100\times$ or more in $N$.

**Memory** ($N = 1000$, numbers measured on Apple M2 with L3 = 16 MB, peak DRAM bandwidth $\approx 100$ GB/s; L3 size and bandwidth vary by platform):

| Method | Peak memory | Cache pressure (fraction of L3) |
|--------|------------|----------------------------------|
| Cholesky | 7.6 MB ($N^2 \cdot 8$) | $0.48\times$ |
| FFT | 0.015 MB ($2N \cdot 8$) | $< 0.001\times$ |
| Low-rank rSVD (C held) | 7.6 MB | $0.48\times$ |
| Low-rank rSVD (C freed) | 0.24 MB ($Nk \cdot 8$) | $0.015\times$ |

The Cholesky MC loop is **memory-bandwidth limited**, not compute-limited. Each path reads the full $N \times N$ lower-triangular $L$ matrix ($\approx 4$ MB at $N = 1000$) to compute $Lz$. With $M = 10{,}000$ paths this is $\approx 40$ GB of sequential reads. On the M2 test machine this consumed $\approx 47$ GB/s — roughly half of rated peak bandwidth.

At **larger $N$**, the $L$ matrix grows as $N^2$. Once it no longer fits in the last-level cache, every path becomes a full DRAM read and the operation stays saturated at the bandwidth ceiling. There is no sudden failure: the MC loop continues to work correctly, it just remains bandwidth-bound. The implication is that Cholesky's wall time is increasingly dominated by memory access, not arithmetic, and a hardware with higher memory bandwidth (e.g. a newer GPU or a machine with wider memory buses) would close the performance gap with FFT more than a faster CPU would.

**Low-rank approximation quality** ($\|C - C_k\|_F / \|C\|_F$):

| Rank $k$ | Frobenius error |
|----------|----------------|
| 4 | 5.4% |
| 16 | 2.5% |
| 64 | 1.5% |
| 128 | 1.2% |

Slow convergence is fundamental: the singular spectrum of $C$ decays slowly for rough $H = 0.1$.

**Reference price:** $p_\text{ref} = 23.58$. There is no closed-form formula for this option, so we need a Monte Carlo ground truth to measure low-rank approximation error against. We run both Cholesky and FFT — two independent simulation methods (one strictly exact, one asymptotically exact) — with 500,000 paths each and average the results. The Monte Carlo standard error at 500k paths is $\sigma_V / \sqrt{500{,}000} \approx 35 / 707 \approx 0.05$ price units, much smaller than the low-rank pricing errors being measured (which range from ~0.1 to ~2 at low rank). Both methods draw from the identical $N$-dimensional Gaussian, so pooling their results is equivalent to a single $1{,}000{,}000$-path run, reducing the Monte Carlo standard error by a factor of $\sqrt{2}$ relative to either method alone. The parameters $H = 0.10$, $\nu = 0.30$, $S_0 = K = 100$, $T = 1$, $r = 0$ are the calibrated RFSV model values stored in `src/common/params.hpp`; $S_0 = K = 100$ and $r = 0$ means the option is struck exactly at-the-money forward.

---

## Validation

### Phase 1 — Implied volatility smile vs SPY option chains

`data/validate_iv.py` fetches live SPY European option chains via yfinance, computes market implied volatilities via Black-Scholes inversion, and overlays RFSV model IVs. A log-vol drift $\mu_0$ is auto-calibrated to match the ATM market IV, isolating the smile *shape* rather than level.

```bash
uv run python data/validate_iv.py [--M 3000] [--N 63]
# → plots/validate_iv.png
```

### Phase 2 — Lévy benchmark + roughness premium

`data/validate_asian.py` benchmarks the RFSV Asian price against the Lévy (1992) log-normal approximation and computes the roughness premium: the price increase when $H$ is reduced from $H = 0.5$ (standard GBM) to $H = 0.1$ (rough volatility). Runs 3 seeds for $\pm 1\sigma$ error bars. Note that $\nu$ is held fixed across $H$ values; since the log-volatility path-integrated variance $\int_0^T \nu^2 t^{2H}\,dt = \nu^2 \frac{T^{2H+1}}{2H+1}$ differs by $H$, this comparison conflates roughness with a change in average instantaneous variance — a variance-normalized comparison (rescaling $\nu$ to match integrated variance) would isolate the pure roughness premium.

```bash
uv run python data/validate_asian.py [--M 5000] [--N 252]
# → plots/validate_asian.png
```

### Phase 3 — Sensitivity: $H \times \nu$ heatmap + price vs strike

`plots/plot_sensitivity.py` sweeps $H \in \{0.05, 0.10, 0.15, 0.20\}$ and $\nu \in \{0.1, 0.2, 0.3, 0.4\}$, producing an ATM price heatmap and price-vs-strike curves for varying $H$.

```bash
uv run python plots/plot_sensitivity.py [--M 10000] [--N 252]
# → plots/sensitivity_surface.png, plots/sensitivity_strike.png
```

---

## Structural Analysis

`plots/plot_structure.py` provides the geometric motivation for each algorithm:

- **Toeplitz structure of fGn** (why FFT works): fGn covariance heatmap shows constant diagonals; fBM covariance does not. The translational invariance of increments is exactly what enables the circulant embedding.
- **Low-rank off-diagonal blocks** (why global low-rank rSVD works): singular value decay of the off-diagonal block is faster for $H = 0.5$ than $H = 0.1$, connecting to Candès-Demanet-Ying (2008).

```bash
uv run python plots/plot_structure.py [--N-small 64] [--N-large 128]
# → plots/structure_analysis.png
```

---

## Additional Experiments

| Script | What it measures | Output |
|--------|-----------------|--------|
| `data/validate_convergence.py` | Price $\pm 1\sigma$ vs $M$ paths; log-log slope confirms $\sigma \propto 1/\sqrt{M}$ | `plots/convergence.png` |
| `data/validate_stability.py` | FFT eigenvalue clipping near $H = 0.5$; rSVD condition number vs rank; Cholesky $\kappa(C)$ vs $N$ | `plots/stability_report.png` |
| `data/profile_memory.py` | Python engine tracemalloc peak vs $N \times M$ | `plots/memory_profile.png` |
| `plots/plot_scaling.py` | Timing + complexity fits; construction vs MC breakdown; memory vs N | `plots/time_vs_N.png`, `plots/construction_breakdown.png`, `plots/memory_vs_N.png`, `plots/error_vs_rank.png` |

---

## A Fourth Method (future work): Hybrid Scheme

Bennedsen, Lunde & Pakkanen (2017) introduce a simulation scheme for **Brownian Semistationary (BSS) processes**, which underlie the rough Bergomi model. In that model class, log-volatility is driven by a Volterra integral $\int_0^t (t-s)^{H-\tfrac{1}{2}}\,dW_s$ that must be numerically discretized. The singular kernel $g(x) = x^{H-\tfrac{1}{2}}$ has a power-law spike near $x = 0$ that an Euler scheme skips entirely (the step nearest $s = t$ misses the bulk of the contribution); the Hybrid Scheme corrects this by treating the near-zero piece of $g$ analytically and the remainder with step functions, keeping $O(N \log N)$ complexity and reducing discretization RMSE by more than 80% for $H = 0.1$.

This improvement does not apply to the RFSV model implemented here. Our model drives log-volatility with fBM $W_t^H$ directly, and circulant embedding simulates fGn increments **exactly in distribution** on the discrete grid — no kernel is being integrated numerically, so there is no discretization error to correct. Benefiting from the Hybrid Scheme would require switching from RFSV to a Volterra/BSS model (i.e., rough Bergomi), a change of model class rather than just sampling routine.

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

## Data Flow

![Data flow diagram](dataflow.svg)

> Regenerate after changes: `d2 dataflow.d2 dataflow.svg`

---

## Project Structure

```
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
src/hmatrix/            Algorithm 3: Global Low-Rank rSVD (rsvd.hpp = Halko et al. Alg 4.4)
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
| Gatheral, Jaisson & Rosenbaum (2014). *Volatility is rough.* Quantitative Finance 18(6). DOI: 10.1080/14697688.2017.1393551 | Empirical $H \approx 0.1$; RFSV model definition; calibration method |
| Davies, R.B. & Harte, D.S. (1987). *Tests for Hurst effect.* Biometrika 74(1), 95–101. DOI: 10.1093/biomet/74.1.95 | Original circulant embedding method for exact stationary Gaussian simulation |
| Wood, A.T.A. & Chan, G. (1994). *Simulation of stationary Gaussian processes in $[0,1]^d$.* JCGS 3(4), 409–432. DOI: 10.1080/10618600.1994.10474655 | Proves the circulant embedding is PSD for $H \in (0,1)$; theoretical guarantee for the FFT pricer |
| Halko, Martinsson & Tropp (2011). *Finding structure with randomness.* SIAM Review 53(2). DOI: 10.1137/090771806 | Algorithm 4.4 (rSVD with power iteration) in `rsvd.hpp` |
| Candès, Demanet & Ying (2008). *A fast butterfly algorithm for Fourier integral operators.* Multiscale Model. Simul. 7(4). DOI: 10.1137/080734339 | Theoretical basis for low-rank off-diagonal blocks in smooth kernels |
| Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* Finance Stoch. 21(4). DOI: 10.1007/s00780-017-0335-5 | Hybrid Scheme for Volterra/BSS rough volatility models (e.g. rough Bergomi); corrects singular kernel discretization error near $s = t$; relevant as future work if switching from the RFSV/fBM model class |
| Lévy, E. (1992). *Pricing European average rate currency options.* J. Int. Money Finance 11(5). DOI: 10.1016/0261-5606(92)90033-E | Analytical approximation for arithmetic Asian calls; used as benchmark in Phase 2 validation |
| Mandelbrot, B.B. & Van Ness, J.W. (1968). *Fractional Brownian motions, fractional noises and applications.* SIAM Review 10(4). DOI: 10.1137/1010093 | Original definition of fBM and fGn; covariance kernel in `covariance.hpp` |
