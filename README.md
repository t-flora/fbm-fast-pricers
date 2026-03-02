# Rough Volatility Asian Option Pricer

A high-performance Monte Carlo pricer for an **Arithmetic Asian Call Option** under the **Rough Fractional Stochastic Volatility (RFSV)** model, benchmarking three methods for generating fractional Brownian motion (fBM) paths. The project is a practical study in how algorithmic complexity theory translates to real speedups on a computationally demanding financial problem.

---

## The Problem

### Why rough volatility?

Gatheral, Jaisson & Rosenbaum (2014) — [*Volatility is rough*](papers-bg/vol-is-rough.pdf) — showed by estimating volatility from high-frequency data that **log-volatility behaves essentially as fractional Brownian motion with Hurst exponent H ≈ 0.1**. This finding is empirically robust across many equity indices and time scales.

The **RFSV model** specifies log-volatility as:

```
log σ_t = ν · W^H_t
```

where `W^H` is fBM with `H ≈ 0.1` and `ν` is the volatility-of-volatility. Because `H < 1/2`, increments are **anti-persistent** ("rough"), meaning each spike in volatility is likely to reverse. This is the opposite of the classical long-memory assumption (`H > 1/2`) and contradicts much of the prior literature.

### Why is this computationally hard?

fBM is **non-Markovian**: the future distribution of `W^H_t` depends on the entire past path. There is no recursion or SDE to step forward from. Exact path generation requires constructing the full `N×N` covariance matrix:

```
C_ij = E[W^H(t_i) · W^H(t_j)] = ½(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})
```

This is dense — every pair of time points is correlated — and its Cholesky factorization costs `O(N³)`. For `N = 1000` timesteps and `M = 10,000` Monte Carlo paths this is the dominant bottleneck.

### The option

**Arithmetic Asian Call Option**: the payoff at maturity is

```
max( (1/N) · Σ S_i − K, 0 )
```

The averaging over the price path `S_1, ..., S_N` makes this path-dependent and eliminates any closed-form pricing formula. Monte Carlo simulation is the only general method.

---

## Three Algorithms

### Algorithm 1 — Dense Cholesky  `O(N³)` factorization, `O(N²)` per path

**File:** `src/cholesky/cholesky.hpp`

Build the full `N×N` fBM covariance matrix `C` and compute its Cholesky factor `L = chol(C)` once via Eigen's `LLT`. For each Monte Carlo path, draw `z ~ N(0, I_N)` and form `log_vol = ν · L · z`. Exact, but the per-path matrix-vector multiply `L·z` costs `O(N²)`, so total cost scales as `O(N³ + M·N²)`.

In practice for `M = 10,000` the per-path `O(M·N²)` term dominates for `N < 10,000`, giving an observed exponent of ~1.5 rather than 3.

### Algorithm 2 — Circulant Embedding + FFT  `O(N log N)` per path

**File:** `src/fft/fft.hpp`

Based on the **Davies-Harte / Wood-Chan** exact method. The key insight is that fBM **increments** (fractional Gaussian noise, fGn) *are* stationary, even though fBM itself is not — so the increment covariance matrix is Toeplitz. A Toeplitz matrix embeds into a larger circulant matrix, whose eigenvalues are just the DFT of the first row. This enables exact sampling:

1. Build the fGn autocovariance: `γ(k) = (dt^{2H}/2) · (|k+1|^{2H} + |k-1|^{2H} − 2|k|^{2H})`
2. Embed into `2N×2N` circulant: `c = [γ(0), γ(1), ..., γ(N-1), 0, γ(N-1), ..., γ(1)]`
3. Eigenvalues: `λ = FFT(c)` — all `λ ≥ 0` for `H ≤ 0.5` (Wood & Chan, 1994)
4. Per path: draw complex white noise, scale by `√(λ/M)`, apply IFFT, take real part → fGn increments
5. Cumsum of increments → fBM path

**Critical note:** embedding the *fBM* covariance directly (as a Toeplitz matrix) fails because fBM is non-stationary. Only the *increments* are stationary and embed correctly.

FFTW plans are created once outside the Monte Carlo loop (plans are reusable for fixed `N`).

### Algorithm 3 — H-matrix + Randomized SVD  `O(N·k²)` construction, `O(N·k)` per path

**File:** `src/hmatrix/hmatrix.hpp`, `src/hmatrix/rsvd.hpp`

Motivated by two observations:
- The fBM covariance kernel `C(t,s)` is smooth away from the diagonal `t ≈ s`; off-diagonal (far-field) blocks are approximately low-rank — the theoretical basis comes from [*A Fast Butterfly Algorithm for Fourier Integral Operators*](papers-bg/fast-butterfly-fft.pdf) (Candès, Demanet & Ying, 2008), which proves low-rank structure for smooth kernels under dyadic partitioning.
- Randomized SVD can exploit this efficiently: Halko, Martinsson & Tropp (2011) — [*Finding Structure with Randomness*](papers-bg/2010_HMT_random_review.pdf) — prove that a rank-`k` sketch plus `q` power iterations recovers the dominant singular vectors of any matrix in `O(mn·log k)` operations with high probability.

**Algorithm (as implemented):** Compute the global rank-`k` rSVD of `C`:
```
C ≈ U · diag(S) · U^T     (U is N×k, all S_i > 0)
```
Then the approximate Cholesky factor is:
```
L_k = U · diag(√S)         (N×k matrix)
```
because `L_k · L_k^T = U · S · U^T ≈ C`. Per-path cost drops from `O(N²)` to `O(N·k)`.

**rSVD implementation** (`rsvd.hpp`, Algorithm 4.4 from Halko et al.):
1. Draw Gaussian sketch `Ω ∈ R^{N×(k+p)}` (`p=5` oversampling)
2. Form `Y = C · Ω`, apply `q=2` power iterations: `Y ← (C·C^T)^q · C · Ω`
3. QR factorization: `Y = Q·R`
4. Project: `B = Q^T · C`, compute thin SVD of small matrix `B`

Power iteration is important here: with `H = 0.1` the singular values of `C` decay slowly (the rough spectrum), so without power iteration the sketch quality degrades. With `q = 2` the approximation is much better.

---

## Computational Results

Benchmarked at `M = 10,000` paths, `N = 252, 500, 1000`:

| Method | Fitted exponent α | Constant c | R² |
|---|---|---|---|
| Dense Cholesky | 1.54 | 4.0×10⁻⁵ | 0.999 |
| Circulant+FFT | 1.03 | 9.2×10⁻⁴ | 1.000 |
| H-matrix+rSVD (k=32) | 1.06 | 2.9×10⁻⁴ | 1.000 |

FFT and H-matrix both scale as `O(N)` in this regime (the `log N` factor is invisible across 4×N range). Cholesky's exponent of 1.54 reflects the mixed `O(M·N² + N³)` cost, where the per-path term dominates for `M = 10,000`.

**H-matrix approximation quality** (Frobenius norm `||C − C_k|| / ||C||`):

| Rank k | Frobenius error |
|---|---|
| 2 | 8.5% |
| 16 | 2.5% |
| 128 | 1.2% |

Errors decay slowly because the singular value spectrum of `C` decays slowly for rough `H = 0.1` — a fundamental property of rough processes. At `H = 0.4` (closer to Brownian) convergence would be much faster.

**Reference price:** `p_ref = 23.58` (average of 500k-path Cholesky and 500k-path FFT runs, parameters: `H = 0.10`, `ν = 0.30`, `S₀ = K = 100`, `T = 1`).

---

## A Fourth Method (not implemented): Hybrid Scheme

Bennedsen, Lunde & Pakkanen (2017) — [*Hybrid scheme for Brownian semistationary processes*](papers-bg/hybrid-scheme-brownian-semistationary-process.pdf) — introduce a simulation scheme that is especially suited to rough kernels. The key insight: the fBM kernel `g(x) = x^{H−1/2} = x^{−0.4}` has a power-law singularity near zero that circulant embedding handles poorly (the first step function cell misses the spike). The hybrid scheme approximates `g(x)` as:
- **near zero:** a power function (analytically integrable, exact near the singularity)
- **elsewhere:** step functions (standard Riemann sum)

This gives `O(N log N)` complexity with substantially lower MSE than circulant embedding for `H < 1/2`. For `α = H − 1/2 ≈ −0.4`, the hybrid scheme reduces RMSE by >80% versus the plain Riemann scheme. Implementing it as Algorithm 4 would provide a more accurate baseline for small `N` and is left as a natural extension.

---

## Workflow

### 1. Calibrate parameters from data

```bash
# Place Oxford-Man realized variance CSV in data/raw/
# https://realized.oxford-man.ox.ac.uk/data/download
python data/calibrate.py
# Copy printed H and nu values into src/common/params.hpp
```

### 2. Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 3. Run benchmarks

```bash
./build/benchmark
# Outputs: benchmarks/results/time_vs_N.csv, error_vs_rank.csv, reference_price.txt
```

### 4. Visualize

```bash
uv run python plots/plot_scaling.py
# Outputs: plots/time_vs_N.png, plots/error_vs_rank.png
```

---

## Dependencies

| Package | Purpose | Install |
|---|---|---|
| Eigen3 | Dense matrix ops, Cholesky | `brew install eigen` |
| FFTW3 | Fast Fourier transforms | `brew install fftw` |
| CMake 3.16+ | Build system | `brew install cmake` |
| Python (uv) | Calibration + plotting | `uv add pandas numpy scipy matplotlib seaborn` |

---

## Project Structure

```
papers-bg/          Source papers (see references below)
data/               Oxford-Man data + calibration script
src/common/         Shared C++ headers: params, fBM covariance, payoff, RNG
src/cholesky/       Algorithm 1: Dense Cholesky
src/fft/            Algorithm 2: Circulant embedding + FFTW
src/hmatrix/        Algorithm 3: H-matrix + rSVD (rsvd.hpp = Halko et al. Alg 4.4)
benchmarks/         Timing + accuracy runner; CSV results
plots/              Python visualization with fitted complexity constants
```

---

## References

| Paper | Role in this project |
|---|---|
| Gatheral, Jaisson & Rosenbaum (2014). *Volatility is rough.* arXiv:1410.3394 | Empirical basis for H ≈ 0.1; RFSV model definition; calibration method |
| Halko, Martinsson & Tropp (2011). *Finding structure with randomness.* SIAM Review 53(2) | Algorithm 4.4 (rSVD with power iteration) implemented in `rsvd.hpp` |
| Candès, Demanet & Ying (2008). *A fast butterfly algorithm for Fourier integral operators.* arXiv:0809.0719 | Theoretical basis for low-rank structure of smooth off-diagonal kernel blocks |
| Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* arXiv:1507.03004 | Better simulation scheme for rough kernels; identified as future work |
