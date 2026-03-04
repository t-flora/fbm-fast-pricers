# A Beginner's Guide to the Three fBM Simulation Algorithms

This document walks through the three algorithms implemented in `src/` for
simulating fractional Brownian Motion (fBM) paths and pricing an Arithmetic
Asian Call Option under the Rough Fractional Stochastic Volatility (RFSV) model.

---

## Background: What Problem Are We Solving?

The RFSV model (Gatheral, Jaisson & Rosenbaum 2014, `papers-bg/vol-is-rough.pdf`) says
that log-volatility evolves as a **fractional Brownian motion**:

```
log σ_t = ν · W_t^H
```

where `W_t^H` is fBM with Hurst exponent `H ≈ 0.10` (empirically estimated from
S&P 500 realized variance data). The key difficulty: fBM paths are **correlated across
all time steps**. You can't generate `W_{t+1}` independently of `W_1, ..., W_t` the
way you would with standard Brownian motion.

**The core challenge**: given `N` time steps, we need to sample a vector
`(W_{dt}, W_{2dt}, ..., W_{T})` from a **multivariate Gaussian** with covariance matrix
`C` where `C[i,j] = ½(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})`.

Each of the three algorithms below solves this sampling problem at a different
computational cost.

The shared setup is in `src/common/`:
- `params.hpp`: `H=0.10`, `ν=0.30`, `S0=100`, `K=100`, `T=1`, `r=0`
- `covariance.hpp`: the kernel `C(t,s) = ½(|t|^{2H} + |s|^{2H} - |t-s|^{2H})`
- `asian_payoff.hpp`: path → prices → `max(mean(S) - K, 0)`

---

## Algorithm 1: Dense Cholesky (`src/cholesky/cholesky.hpp`)

### The idea

The textbook method for sampling `x ~ N(0, C)` is:
1. Factorize `C = L · Lᵀ` (Cholesky decomposition)
2. Sample `z ~ N(0, I_N)` (i.i.d. Gaussians)
3. Compute `x = L · z`

This works because `Cov(Lz) = L · Cov(z) · Lᵀ = L · I · Lᵀ = C`. ✓

### Key lines

**Build the covariance matrix** (`cholesky.hpp:20`):
```cpp
Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
```
This fills the N×N matrix with `C[i,j] = ½(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})`
where `t_i = i·dt`. Cost: O(N²).

**Cholesky factorization** (`cholesky.hpp:21-23`):
```cpp
Eigen::LLT<Eigen::MatrixXd> llt(C);
if (llt.info() != Eigen::Success)
    throw std::runtime_error("Cholesky: matrix not positive-definite");
Eigen::MatrixXd L = llt.matrixL();
```
`Eigen::LLT` computes the lower-triangular factor `L` such that `C = L·Lᵀ`.
Cost: O(N³). Done **once** before the Monte Carlo loop.

**Per-path sampling** (`cholesky.hpp:30-33`):
```cpp
for (int i = 0; i < N; ++i) z(i) = norm(rng);
Eigen::VectorXd lv = nu * (L * z);
```
`L * z` is an N×N lower-triangular matrix times an N-vector: cost O(N²) per path.
`lv[i] = log σ_{t_i}` — the log-volatility path.

**Price simulation** (`cholesky.hpp:34-35`):
```cpp
auto inno = randn(N, rng);
payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
```
`inno` is a fresh set of N i.i.d. Gaussians for the **price process** (independent of
the vol process). `log_vol_to_prices` steps `S_i = S_{i-1}·exp((r - ½σ²)dt + σ√dt·Z_i)`.

### Complexity

| Phase | Cost | N=252 | N=1000 |
|-------|------|-------|--------|
| Factorize C | O(N³) | ~0.1ms | ~26ms |
| MC loop (M paths) | O(M·N²) | ~200ms | ~1670ms |

The factorization cost is dominated by the per-path cost at M=10k.
The fitted exponent α ≈ 1.54 (from `benchmarks/results/time_vs_N.csv`)
reflects the O(M·N²) regime — confirmed by the memory bandwidth calculation:
N=1000 reads 8MB × 10k paths = 80 GB at ~47 GB/s ≈ 1.7s. ✓

### Memory
Peak: N×N matrix `L` lives on the heap throughout = N²·8 bytes.
At N=1000: 7.6 MB (well below M2's L3=16 MB, so L matrix stays in cache at small N).

---

## Algorithm 2: Circulant Embedding + FFT (`src/fft/fft.hpp`)

### The key insight: fBM increments are stationary

fBM itself is **non-stationary**: `Cov(W_s, W_t)` depends on both `s` and `t` separately,
not just `|s-t|`. The fBM covariance matrix is therefore **not Toeplitz**.

But the **increments** `δW_i = W_{i·dt} - W_{(i-1)·dt}` (called fractional Gaussian
noise, fGn) ARE stationary. Their covariance depends only on the lag:

```
γ(k) = Cov(δW_i, δW_{i+k}) = (dt^{2H}/2) · ((k+1)^{2H} + (k-1)^{2H} - 2k^{2H})
```

A stationary covariance means the matrix is **Toeplitz** (constant along diagonals).
Toeplitz matrices can be embedded in **circulant** matrices, which are diagonalized
by the DFT. This is the Wood & Chan (1994) / Davies-Harte method.

You can verify this structure visually in `plots/structure_analysis.png` (panel b):
the fGn heatmap shows perfectly flat diagonals, while the fBM heatmap (panel a) does not.

### The algorithm (Steps 1–5 in `fft.hpp:39-82`)

**Step 1 — fGn autocovariance** (`fft.hpp:26-31`):
```cpp
static inline double fgn_cov(int k, double H, double dt) {
    double h2 = 2.0 * H;
    if (k == 0) return std::pow(dt, h2);
    double km1 = (k == 1) ? 0.0 : std::pow(k - 1.0, h2);
    return 0.5 * std::pow(dt, h2) * (std::pow(k + 1.0, h2) + km1 - 2.0 * std::pow(k, h2));
}
```
This is γ(k). Note `γ(0) = dt^{2H}` (variance of each increment).

**Step 2 — Build the circulant first row** (`fft.hpp:44-50`):
```cpp
std::vector<std::complex<double>> c_emb(M, 0.0);  // M = 2N
for (int j = 0; j < N; ++j)
    c_emb[j] = fgn_cov(j, H, dt);
for (int j = 1; j < N; ++j)
    c_emb[M - j] = c_emb[j];  // symmetric reflection
```
This builds the size-2N circulant embedding:
```
c = [γ(0), γ(1), ..., γ(N-1), 0, γ(N-1), ..., γ(1)]
```
The reflection makes the circulant symmetric, which guarantees real eigenvalues.
The critical property (proved in Wood & Chan 1994): for H ≤ 0.5, all eigenvalues
are **non-negative** — the matrix is PSD and the embedding is exact.

**Step 3 — FFT to get eigenvalues** (`fft.hpp:53-64`):
```cpp
fftw_plan p = fftw_plan_dft_1d(M, ..., FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
```
The DFT of a circulant's first row gives its eigenvalues `λ`. This is the core
mathematical fact: **a circulant matrix C is diagonalized by the DFT matrix F**,
meaning `C = F·diag(λ)·F*`. Cost: O(N log N). Done once.

**Step 4 — Per-path synthesis** (`fft.hpp:72-79`):
```cpp
for (int j = 0; j < M; ++j) {
    double s = std::sqrt(std::max(lam[j].real(), 0.0) / M);
    w_buf[j] = s * std::complex<double>(norm(rng), norm(rng));
}
fftw_execute(plan_inv);
```
This samples `w[j] = √(λ[j]/M) · (a_j + ib_j)` with `a_j, b_j ~ N(0,1)`,
then computes the IFFT. Why does this work?

If `C = F·diag(λ)·F*`, then we want `x` such that `Cov(x) = C`. Let `x = F·w`.
Then `Cov(x) = F·Cov(w)·F* = F·diag(λ/M)·diag(M)·F* = F·diag(λ)·F* = C`. ✓

The `/ M` comes from FFTW's unnormalized convention (IFFT multiplies by M).
The `std::max(..., 0.0)` clips any tiny negative eigenvalues from floating-point error.

**Step 5 — Cumsum to recover fBM** (`fft.hpp:82-87`):
```cpp
double acc = 0.0;
for (int i = 0; i < N; ++i) {
    acc += out_buf[i].real();
    log_vol[i] = nu * acc;
}
```
We generated fGn increments (first N entries of the IFFT output). A cumulative sum
reconstructs the fBM path `W_{t_i} = Σ_{j≤i} δW_j`. Then `log σ_i = ν · W_{t_i}`.

### Complexity

| Phase | Cost |
|-------|------|
| Build eigenvalues | O(N log N) |
| Per-path IFFT | O(N log N) |
| Full MC | O(M·N log N) |

At N=1000, FFT is 1.5× faster than Cholesky despite both running M=10k paths,
because O(N log N) ≈ 10k vs O(N²) = 1M per path. The fitted exponent α ≈ 1.01 ≈ 1.

**Memory**: only the size-2N complex vector `c_emb` and `lam` are needed (not N×N).
At N=1000: ~16 KB — fits in L1 cache. This is the critical memory advantage.

---

## Algorithm 3: H-matrix + Randomized SVD (`src/hmatrix/hmatrix.hpp`)

### The key insight: covariance has low-rank off-diagonal blocks

The fBM covariance matrix is **smooth** away from its diagonal. Intuitively:
far-apart time points have a slowly-varying, well-approximated covariance.
This means the **off-diagonal blocks** are numerically low-rank.

Candès, Demanet & Ying (2008) (`papers-bg/fast-butterfly-fft.pdf`) formalize this:
a kernel `C(s,t)` that is smooth away from the diagonal has off-diagonal blocks with
singular values decaying rapidly. The **H-matrix** data structure exploits this by
recursively compressing off-diagonal blocks.

Our implementation uses a simpler global version: approximate `C ≈ U·diag(S)·Uᵀ`
using a **randomized SVD**, then use `L_k = U·diag(√S)` as an approximate Cholesky factor.

You can see the SVD decay in `plots/structure_analysis.png` (panel d): for H=0.1 the
singular values decay more slowly than H=0.5 — the rough spectrum is harder to compress.

### The rSVD algorithm (`src/hmatrix/rsvd.hpp`)

This implements Halko, Martinsson & Tropp (2011) Algorithm 4.4
(`papers-bg/2010_HMT_random_review.pdf`).

**Stage A — Random sketch** (`rsvd.hpp:22-29`):
```cpp
Eigen::MatrixXd Omega(n, l);  // l = k + p (oversampling p=5)
// ... fill Omega with N(0,1) entries ...
Eigen::MatrixXd Y = A * Omega;   // N × l  (sketch of A's column space)
```
`Y = A·Ω` captures the dominant directions of `A`. If `A` has rank `k`, then `Y`'s
column space captures it perfectly; for approximate rank-k, it captures the `k`
largest singular value directions. `l = k + p` with oversampling `p=5` reduces
the failure probability to near zero.

**Power iteration** (`rsvd.hpp:31-34`):
```cpp
for (int iter = 0; iter < q; ++iter) {
    Y = A * (A.transpose() * Y);
}
```
This replaces `A` with `(A·Aᵀ)^q · A` for `q=2` iterations. The singular values
of the iterated matrix are `σ_i^{2q+1}`, so the ratio between large and small
singular values is amplified: `(σ_1/σ_2)^5` instead of `σ_1/σ_2`.

**Why this matters for H=0.1**: the fBM covariance at small H has **slowly-decaying**
singular values (rough spectrum). Without power iteration, the random sketch can't
distinguish the top-k directions. With q=2, the sketch quality improves dramatically.
This is the key insight from Section 4.3 of Halko et al.

**Stage B — QR + small SVD** (`rsvd.hpp:36-43`):
```cpp
Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(m, l);
Eigen::MatrixXd B = Q.transpose() * A;   // l × n  (small!)
Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, ...);
```
`Q` is an orthonormal basis for the range of `Y`. Projecting `A` onto `Q` gives
the small `l×n` matrix `B`. The SVD of `B` costs O(l²·n) — much cheaper than O(N³)
for the full SVD.

**Reassemble** (`rsvd.hpp:45-48`):
```cpp
result.U  = Q * svd.matrixU().leftCols(k);
result.S  = svd.singularValues().head(k);
result.Vt = svd.matrixV().leftCols(k).transpose();
```
The final approximation is `A ≈ U·diag(S)·Vᵀ` with `U` being N×k.

### Using rSVD for path generation (`hmatrix.hpp:35-52`)

**Approximate Cholesky factor** (`hmatrix.hpp:38-39`):
```cpp
Eigen::VectorXd sqrt_S = decomp.S.cwiseMax(0.0).cwiseSqrt();
Eigen::MatrixXd Lk = decomp.U * sqrt_S.asDiagonal();  // N × k
```
For a symmetric PSD matrix, `C ≈ U·S·Uᵀ`, so
`C ≈ (U·√S) · (U·√S)ᵀ = L_k · L_kᵀ`.
The `cwiseMax(0.0)` clips any tiny negative singular values from numerical error.

**Per-path sampling** (`hmatrix.hpp:44-49`):
```cpp
for (int i = 0; i < k; ++i) z(i) = norm(rng);
Eigen::VectorXd lv = nu * (Lk * z);  // O(N*k)
```
Instead of an N×N triangular solve, we compute an **N×k matrix times a k-vector**.
At k=32, N=1000: 32k operations vs 1M for Cholesky. This is the per-path speedup.

### The freed variant (`hmatrix.hpp:93-131`)

The standard `price_timed()` keeps the full N×N matrix `C` alive for the function's
entire scope (Eigen matrices are RAII — they're freed when the variable leaves scope).
This means peak RSS is O(N²) even though the MC loop only needs `L_k` (N×k):

```cpp
// price_freed_timed: C destroyed before MC
Eigen::MatrixXd Lk;
{
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);   // N×N
    // ... rSVD ...
    Lk = ...;   // N×k
}  // <-- C destroyed here by RAII
// MC loop: only Lk (N×k) needed
```

At N=1000, k=32: `C` is 7.6 MB; `L_k` is only 0.24 MB. This is the difference
between the theoretical O(N²) and O(N·k) memory profiles visible in
`plots/memory_vs_N.png`.

### Complexity

| Phase | Cost |
|-------|------|
| Build C | O(N²) |
| rSVD | O(N·k² + N·k·q) ≈ O(N·k²) |
| Per-path MC | O(N·k) |
| Full MC | O(M·N·k) |

Fitted exponent α ≈ 1.06 (close to linear in N at fixed k=32). The crossover
point vs Cholesky: at k=32, O(M·N·32) < O(M·N²) when 32 < N, which is always true.

### Accuracy tradeoff

The approximation error depends on how many singular values of `C` we keep.
From `benchmarks/results/error_vs_rank.csv`:

| rank k | Frobenius error | Price error |
|--------|----------------|-------------|
| 4  | 5.4% | 0.05 |
| 16 | 2.5% | 0.13 |
| 32 | 1.9% | 0.34 |
| 64 | 1.5% | 0.61 |

Paradoxically, price error increases at higher rank — this is MC noise dominating
(the noise floor is ~0.3% at M=10k). The Frobenius error (which is noise-free)
correctly shows monotone improvement. This is why we need both metrics.

---

## The Shared Payoff (`src/common/asian_payoff.hpp`)

All three algorithms produce a `log_vol` vector and call the same payoff function:

```cpp
// log_vol[i] = log σ_i
// Step 1: simulate price path
double S = S0;
for (int i = 0; i < N; ++i) {
    double sigma = std::exp(log_vol[i]);
    S *= std::exp((r - 0.5*sigma*sigma)*dt + sigma*sqrt(dt)*Z[i]);
    prices[i] = S;
}
// Step 2: arithmetic Asian payoff
double mean = accumulate(prices) / N;
return max(mean - K, 0.0);
```

The GBM step `exp((r - ½σ²)dt + σ√dt·Z)` is the Euler–Maruyama discretization of
`dS = rS dt + σS dW`. The `-½σ²dt` is the Itô correction that makes `E[S_T] = S_0·e^{rT}`.

The arithmetic mean of prices (not geometric) is what makes Asian options hard to
price analytically — no closed-form exists under stochastic volatility.

---

## Comparison Summary

| | Cholesky | FFT | H-matrix (k=32) |
|---|---|---|---|
| **Math foundation** | L·Lᵀ = C exactly | Circulant embedding of fGn | Low-rank approx C ≈ L_k·L_kᵀ |
| **Construction** | O(N³) | O(N log N) | O(N·k²) with rSVD |
| **Per path** | O(N²) | O(N log N) | O(N·k) |
| **Memory** | O(N²) | O(N) | O(N²) or O(N·k) if freed |
| **Exact?** | Yes | Yes | No (truncation error) |
| **Key paper** | — | Wood & Chan 1994 | Halko et al. 2011 |
| **Bottleneck at N=1000** | Memory bandwidth | Compute | Compute |
| **Fitted α** | 1.54 | 1.01 | 1.06 |

The FFT method wins on memory (O(N) vs O(N²)) and is exact — it is the recommended
method for production use. The H-matrix method is valuable pedagogically because it
exposes the low-rank structure of the problem and provides a tunable accuracy-speed
tradeoff. Cholesky is the baseline that makes all other methods' improvements concrete.
