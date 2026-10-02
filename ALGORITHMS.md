# A Beginner's Guide to the Three fBM Simulation Algorithms

This document walks through the three algorithms implemented in `src/` for
simulating fractional Brownian Motion (fBM) paths and pricing an Arithmetic
Asian Call Option under the Rough Fractional Stochastic Volatility (RFSV) model.

---

## Background: What Problem Are We Solving?

The RFSV model (Gatheral, Jaisson & Rosenbaum 2014) says
that log-volatility evolves as a fractional Brownian motion:

```
log σ_t = log σ_0 + ν · W_t^H
```

where `W_t^H` is fBM with Hurst exponent `H ≈ 0.10` (empirically estimated from
realized variance data), base level $\sigma_0 = 0.2$, and vol-of-vol $\nu = 0.52$ (Gatheral
et al.'s $0.3$ with time in days, converted to the engine's years). The samplers below
produce $\nu W^H$; the shared payoff code multiplies by $\sigma_0$. An earlier version had no
$\sigma_0$ (100% volatility), which made the payoff so heavy-tailed that MC error bars were
meaningless. For the IV comparison, $\log\sigma_0$ is instead calibrated to the ATM market price.

The key difficulty simulating fBM paths is that they're correlated across
all time steps; fBM is non-Markovian. In standard Brownian motion, increments $W_{t+1} - W_t$ are
independent of the past, so you can step forward from the current value alone by adding independent
Gaussian noise. In fBM, increments are correlated with the entire history — you cannot simply step
forward by adding independent noise; the distribution of the next increment depends on all previous ones.

The core challenge: given `N` time steps, we need to sample a vector
`(W_{dt}, W_{2dt}, ..., W_{T})` from a multivariate Gaussian with covariance matrix
`C` where $C[i,j] = \frac{1}{2}(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})$.

Each of the three algorithms below solves this sampling problem at a different
computational cost.

The shared setup is in `src/common/`:
- `params.hpp`: `H=0.10`, `nu=0.52`, `sigma0=0.20`, `S0=100`, `K=100`, `T=1`, `r=0`
  (mirrored by `data/params.py`; a test keeps them in sync)
- `covariance.hpp`: the kernel $C(t,s) = \frac{1}{2}(|t|^{2H} + |s|^{2H} - |t-s|^{2H})$
- `asian_payoff.hpp`: path → prices → `max(mean(S) - K, 0)`

---

## Algorithm 1: Dense Cholesky (`src/cholesky/cholesky.hpp`)

### The idea

The textbook method for sampling `x ~ N(0, C)` is:
1. Factorize $C = LL^\top$ (Cholesky decomposition)
2. Sample $z \sim \mathcal{N}(0, I_N)$ (i.i.d. Gaussians)
3. Compute $x = L \cdot z$

This works because $\text{Cov}(Lz) = L \cdot \text{Cov}(z) \cdot L^\top = L \cdot I \cdot L^\top = C$.

### Key lines

**Build the covariance matrix** (inside `fbm_cholesky_factor`):
```cpp
Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
```
This fills the $N \times N$ matrix with $C[i,j] = \frac{1}{2}(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})$
where $t_i = i \cdot dt$. Cost: $O(N^2)$.

**Cholesky factorization**:
```cpp
Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(C);
if (llt.info() != Eigen::Success)
    throw std::runtime_error("Cholesky: matrix not positive-definite");
auto L = C.triangularView<Eigen::Lower>();
```
`Eigen::LLT` computes the lower-triangular factor `L` such that $C = L L^\top$. Because it
wraps an `Eigen::Ref` to `C`, it factors **in place**: `L` overwrites the lower triangle of `C`,
and `L` is just a triangular *view* of that storage, not a copy.
Cost: $N^3/3$ flops. Done **once** before the Monte Carlo loop.

**Per-path sampling**:
```cpp
for (int i = 0; i < N; ++i) z(i) = norm(rng);
lv.noalias() = L * z;   // triangular mat-vec
lv *= nu;
```
`L * z` is a lower-triangular matrix times an $N$-vector: $N^2$ flops per path, reading only the
lower half of the matrix. (Eigen will not scale a triangular product by a scalar inside one
expression, hence the separate `lv *= nu`.)
`lv[i] = log σ_{t_i}` — the log-volatility path.

**Price simulation**:
```cpp
auto inno = randn(N, rng);
payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
```
`inno` is a fresh set of N i.i.d. Gaussians for the **price process** (independent of
the vol process). `log_vol_to_prices` steps $S_i = S_{i-1} \cdot \exp\!\bigl((r - \tfrac{1}{2}\sigma^2)\,dt + \sigma\sqrt{dt}\,Z_i\bigr)$.

### Complexity

| Phase | Cost | $N=252$ | $N=1000$ |
|-------|------|---------|----------|
| Factorize $C$ | $O(N^3)$ | ~1.2ms | ~24ms |
| MC loop ($M$ paths) | $O(MN^2)$ | ~175ms | ~1130ms |

The factorization cost is dominated by the per-path cost at $M = 10{,}000$.
A single power law fits poorly ($\alpha \approx 1.52$, $R^2 = 0.978$ over $N = 64$ to 4000) because the
curve bends. The local exponent rises from about 1.0 at small $N$ to 2.1 above $N = 1000$. At small
$N$ each MC iteration is dominated by $O(N)$ work ($N$ Gaussian draws, $2N$ exponentials, $N$ payoff
steps). Once $N^2$ takes over, the exponent slightly exceeds 2 because `L` spills out of the 16 MB
cache at $N \approx 1450$ and the loop becomes memory-bandwidth bound.

An earlier version copied `L` into a separate dense `MatrixXd` and computed `L * z`, a
$2N^2$-flop dense mat-vec that also multiplied the zero upper triangle. Switching to the
in-place factorization and triangular view halved the mat-vec (about 115 µs to 62 µs at
$N = 1000$) and cut the $N = 1000$ run by about 30%, with unchanged prices.

### Memory
One $N \times N$ matrix is alive during the MC loop, because `L` overwrites `C`: $8N^2$ bytes,
7.6 MB at $N = 1000$, of which each path reads the lower half. That fits in the M2's 16 MB
performance-core L2 (the M2 has no L3).

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

### The algorithm (Steps 1–5 in `fft.hpp`)

The code is split in two: `circulant_eigenvalues()` does the one-time setup (Steps 1–3), and
the `FbmSampler` class holds the eigenvalue scaling and the inverse-FFT plan and draws one path
per call to `sample()` (Steps 4–5).

**Step 1 — fGn autocovariance** (`fgn_cov`):
```cpp
inline double fgn_cov(int k, double H, double dt) {
    double h2 = 2.0 * H;
    if (k == 0) return std::pow(dt, h2);
    double km1 = (k == 1) ? 0.0 : std::pow(k - 1.0, h2);
    return 0.5 * std::pow(dt, h2) * (std::pow(k + 1.0, h2) + km1 - 2.0 * std::pow(k, h2));
}
```
This is $\gamma(k)$. Note `γ(0) = dt^{2H}` (variance of each increment).

**Step 2 — Build the circulant first row** (`circulant_eigenvalues`):
```cpp
std::vector<std::complex<double>> c_emb(M, 0.0), lam(M);  // M = 2N
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
The method is exact iff all eigenvalues are **non-negative**. For $H \leq \tfrac{1}{2}$ every
$\gamma(k)$ with $k \geq 1$ is negative (Craigmile 2003), so every eigenvalue
$\lambda_j = \gamma(0) + 2\sum_k \gamma(k)\cos(\pi jk/N)$ is at least $\lambda_0$, which telescopes
to $\Delta t^{2H}(N^{2H} - (N-1)^{2H}) > 0$. The margin is thin for rough $H$: at $H = 0.1$,
$\min\lambda/\gamma(0) = 0.24\%$ at $N = 252$. `tests/test_samplers.cpp` checks this closed form.

> **Pitfall:** $\gamma(0)$ must be $\Delta t^{2H}$. Writing the $|k-1|^{2H}$ term as 0 at $k = 0$
> halves $\gamma(0)$ and spuriously makes about 28% of eigenvalues negative at $H = 0.1$. An earlier
> version of the stability script had exactly this bug; both test suites now catch it.

**Step 3 — FFT to get eigenvalues** (`circulant_eigenvalues`):
```cpp
fftw_plan p = fftw_plan_dft_1d(M, ..., FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
// then: throw if any lam[k].real() < -1e-8, return the real parts
```
The DFT of a circulant's first row gives its eigenvalues `λ`. This is the core
mathematical fact: **a circulant matrix C is diagonalized by the DFT matrix F**,
meaning `C = F·diag(λ)·F*`. Cost: $O(N \log N)$. Done once.

**Step 4 — Per-path synthesis** (`FbmSampler::sample`):
```cpp
// constructor: scale_[j] = sqrt(max(lam[j], 0) / M), plus a reusable FFTW_BACKWARD plan
for (int j = 0; j < M_; ++j) {
    double a = norm(rng);
    double b = norm(rng);
    w_[j] = scale_[j] * std::complex<double>(a, b);
}
fftw_execute(plan_);
```
This samples $w_j = \sqrt{\lambda_j / M}\,(a_j + i b_j)$ with $a_j, b_j \sim \mathcal{N}(0,1)$ and
applies FFTW's unnormalized inverse transform, $x_n = \sum_j w_j e^{2\pi i jn/M}$. Why does this
give the right covariance? Write $\theta_{jn} = 2\pi jn/M$. Then
$\operatorname{Re} x_n = \sum_j \sqrt{\lambda_j/M}\,(a_j\cos\theta_{jn} - b_j\sin\theta_{jn})$, and
because the $a_j, b_j$ are independent standard normals,

$$\operatorname{Cov}(\operatorname{Re} x_n, \operatorname{Re} x_m) = \sum_j \frac{\lambda_j}{M}\cos\frac{2\pi j(n-m)}{M} = c_{n-m},$$

the inverse DFT of the eigenvalues, i.e. the circulant's first row. For $|n - m| < N$ that is
exactly $\gamma(|n-m|)$. The same calculation gives $\operatorname{Im} x$ the same covariance, and
$\operatorname{Cov}(\operatorname{Re} x_n, \operatorname{Im} x_m) = \sum_j (\lambda_j/M)\sin(\cdot) = 0$
because $\lambda_j = \lambda_{M-j}$. So the real and imaginary parts are two *independent* fGn
samples. `FbmSampler::sample_pair()` uses both, giving two fBM paths per transform.

The `/ M` comes from FFTW's unnormalized convention, and the `std::max(..., 0.0)` is defensive
(the eigenvalues are already checked to be non-negative).

**Step 5 — Cumsum to recover fBM** (`FbmSampler::sample`):
```cpp
double acc = 0.0;
for (int i = 0; i < N_; ++i) {
    acc += out_[i].real();
    path[i] = nu * acc;
}
```
We generated fGn increments (first N entries of the IFFT output). A cumulative sum
reconstructs the fBM path $W_{t_n} = \sum_{j \leq n} \delta W_j$. Then $\log\sigma_n = \nu \cdot W_{t_n}$.

### Complexity

| Phase | Cost |
|-------|------|
| Build eigenvalues | $O(N \log N)$ |
| Per-path IFFT | $O(N \log N)$ |
| Full MC | $O(MN \log N)$ |

At $N = 1000$, FFT is $1.6\times$ faster than Cholesky (0.71 s vs 1.15 s), and $7.6\times$ at
$N = 4000$. That is less than the arithmetic suggests at moderate $N$: a size-$2N$ complex FFT is
about $10^5$ flops against $N^2 = 10^6$ for the triangular mat-vec. Two reasons, both Amdahl's Law:

1. The price-path simulation ($N$ Gaussian draws for the innovations, $2N$ exponentials, and
   the payoff accumulation) is identical for every method and is a large share of each path.
2. Random numbers dominate the FFT sampler. Each transform needs $4N$ Gaussians (real and
   imaginary parts of $2N$ complex variates); measured in isolation at $N = 1000$, those draws
   take 60–70 µs, while the inverse FFT itself takes about 8 µs.

The cheapest win was therefore not a faster FFT but fewer random numbers per path. The real and
imaginary parts of the IFFT output are *independent* fGn samples (their cross-covariance
vanishes because $\lambda_j = \lambda_{2N-j}$), so the pricer takes two paths per transform. That
cut the $N = 1000$ run from 1.04 s to 0.71 s.
The fitted exponent is $\alpha \approx 1.02$, and the per-path cost per time step is flat
(67–73 ns) from $N = 64$ to 4000, so the $\log N$ factor never becomes visible.

**Memory**: during the MC loop the sampler holds the $2N$ scale factors and two length-$2N$
complex buffers for the inverse FFT, not an $N \times N$ matrix. At $N = 1000$ that is 80 KB
(the setup arrays `c_emb` and `lam` are freed after construction). This is the critical memory
advantage.

---

## Algorithm 3: Global Low-Rank Approximation via rSVD (`src/rsvd/lowrank.hpp`)

### The key insight: covariance has low-rank off-diagonal blocks

The fBM covariance matrix is **smooth** away from its diagonal. Intuitively:
far-apart time points have a slowly-varying, well-approximated covariance.
This means the **off-diagonal blocks** are numerically low-rank.

Candès, Demanet & Ying (2009) formalize this:
a kernel `C(s,t)` that is smooth away from the diagonal has off-diagonal blocks with
singular values decaying rapidly. A true **H-matrix** (Hierarchical matrix) exploits this
recursively: it partitions the matrix into a quad-tree, keeps near-diagonal blocks dense
(where the singularity lives), and compresses each smooth far-field block with a small local
rank — isolating the rough $H = 0.1$ singularity so it cannot contaminate the off-diagonal
compression.

Our implementation is simpler: a **global low-rank approximation** $C \approx U \operatorname{diag}(S) U^\top$
via randomized SVD applied to the entire matrix at once. This is not a true H-matrix.
Because the global decomposition cannot isolate the diagonal singularity, it must represent
the rough near-diagonal behavior with the same rank-$k$ budget as the smooth far field —
which is exactly why singular values decay slowly and large $k$ is needed for low error.
$L_k = U \operatorname{diag}(\sqrt{S})$ serves as the approximate Cholesky factor.

You can see this in `plots/structure_analysis.png` (panel d). The off-diagonal block's singular
values fall below 1% of $\sigma_1$ by rank 3 even at $H = 0.1$ (at $H = 0.5$ the block is exactly
rank 1), but the full matrix's spectrum decays only algebraically. The far field compresses
well; the diagonal singularity is what a global rank-$k$ approximation cannot absorb.

### The rSVD algorithm (`src/rsvd/rsvd.hpp`)

This implements Halko, Martinsson & Tropp (2011) Algorithm 4.4.

**Stage A — Random sketch**:
```cpp
Eigen::MatrixXd Omega(n, l);  // l = k + p (oversampling p=5)
// ... fill Omega with N(0,1) entries ...
Eigen::MatrixXd Q = orth(A * Omega);   // N × l orthonormal basis (sketch of A's column space)
```
$A\Omega$ captures the dominant directions of `A`. If `A` has rank `k`, then `Y`'s
column space captures it perfectly; for approximate rank-k, it captures the `k`
largest singular value directions. `l = k + p` with oversampling `p=5` reduces
the failure probability to near zero.

**Subspace iteration**:
```cpp
for (int iter = 0; iter < q; ++iter) {
    Eigen::MatrixXd W = orth(A.transpose() * Q);
    Q = orth(A * W);
}
```
This replaces `A` with $(AA^\top)^q \cdot A$ for $q = 2$ iterations. The singular values
of the iterated matrix are $\sigma_j^{2q+1}$, so the ratio between large and small
singular values is amplified: $(\sigma_1/\sigma_2)^5$ instead of $\sigma_1/\sigma_2$.
The `orth` (thin QR) after every product is what distinguishes Algorithm 4.4 from the plain
power iteration of Algorithm 4.3. Without it, at $H = 0.1$ and $k = 128$ the columns of
$(AA^\top)^2 A\Omega$ span a dynamic range of $(\sigma_1/\sigma_{k+p})^5 \approx 2 \times 10^{15}$,
right at the limit of double precision, and the small directions are lost to round-off.

**Why this matters for $H = 0.1$**: the fBM covariance at small $H$ has **slowly-decaying**
singular values (rough spectrum). Without power iteration, the random sketch can't
distinguish the top-$k$ directions. With $q = 2$, the sketch quality improves dramatically.
This is the key insight from Section 4.3 of Halko et al.

**Stage B — Project + small SVD**:
```cpp
Eigen::MatrixXd B = Q.transpose() * A;   // l × n  (small!)
Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, ...);
```
`Q` is an orthonormal basis for the range of $(AA^\top)^q A\Omega$. Projecting `A` onto `Q` gives
the small $l \times n$ matrix `B`. The SVD of `B` costs $O(l^2 n)$ — much cheaper than $O(N^3)$
for the full SVD.

**Reassemble**:
```cpp
result.U  = Q * svd.matrixU().leftCols(k);
result.S  = svd.singularValues().head(k);
result.Vt = svd.matrixV().leftCols(k).transpose();
```
The final approximation is $A \approx U \operatorname{diag}(S) V^\top$ with $U \in \mathbb{R}^{N \times k}$.

### Using rSVD for path generation (`lowrank.hpp`, `price()`)

**Approximate Cholesky factor**:
```cpp
Eigen::VectorXd sqrt_S = decomp.S.cwiseMax(0.0).cwiseSqrt();
Eigen::MatrixXd Lk = decomp.U * sqrt_S.asDiagonal();  // N × k
```
For a symmetric PSD matrix, $C \approx U S U^\top$, so
$C \approx (U\sqrt{S})(U\sqrt{S})^\top = L_k L_k^\top$.
The `cwiseMax(0.0)` clips any tiny negative singular values from numerical error.

**Per-path sampling**:
```cpp
for (int i = 0; i < k; ++i) z(i) = norm(rng);
Eigen::VectorXd lv = nu * (Lk * z);  // O(N*k)
```
Instead of an $N \times N$ mat-vec, we compute an **$N \times k$ matrix times a $k$-vector**.
At $k = 32$, $N = 1000$: $32{,}000$ operations vs $10^6$ for Cholesky. This is the per-path speedup.

### The freed variant (`lowrank.hpp`, `price_freed_timed()`)

The standard `price_timed()` keeps the full $N \times N$ matrix `C` alive for the function's
entire scope (Eigen matrices are RAII — they're freed when the variable leaves scope).
This means peak RSS is $O(N^2)$ even though the MC loop only needs `L_k` ($N \times k$):

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

At $N = 1000$, $k = 32$: `C` is 7.6 MB; `L_k` is only 0.24 MB. This is the difference
between the theoretical $O(N^2)$ and $O(Nk)$ memory profiles visible in
`plots/memory_vs_N.png`.

### Complexity

| Phase | Cost |
|-------|------|
| Build $C$ | $O(N^2)$ |
| rSVD | $O(N^2 k)$ |
| Per-path MC | $O(Nk)$ |
| Full MC | $O(MNk)$ |

**Why rSVD costs $O(N^2k)$, not $O(Nk^2)$.** Halko et al. quote $O(Nk^2)$ for the rSVD setup, but that assumes the matrix $A$ is never materialized — matrix-vector products $Av$ are computed in $O(N)$ via sparsity or a fast multipole method. Here we explicitly build the dense $N \times N$ matrix $C$. The random sketch `Y = C * Omega` multiplies an $N \times N$ matrix by an $N \times k$ matrix: that is $N^2 k$ multiply-adds, costing $O(N^2 k)$. Each of the $q = 2$ power iterations (`A * (A.transpose() * Y)`) is also $O(N^2 k)$. The QR decomposition and small SVD are $O(Nk^2)$, but they are dominated by the sketch. The total rSVD setup is therefore $O(N^2 k)$.

This does not undercut the algorithm's advantage. The setup is $O(N^2 k)$ vs Cholesky's $O(N^3)$ — a factor of $N/k$ cheaper — and the per-path MC loop is $O(Nk)$ vs $O(N^2)$, so the full cost $O(N^2 k + MNk)$ is substantially less than Cholesky's $O(N^3 + MN^2)$ for any $k \ll N$.

Fitted exponent $\alpha \approx 1.02$ (close to linear in $N$ at fixed $k = 32$). The crossover
point vs Cholesky: at $k = 32$, $O(MN \cdot 32) < O(MN^2)$ when $32 < N$, which is always true.

### Accuracy tradeoff

The approximation error depends on how many singular values of `C` we keep.
From `benchmarks/results/error_vs_rank.csv`:

| rank k | Frobenius error | Variance lost $\text{tr}(C - C_k)/\text{tr}(C)$ | Price error |
|--------|----------------|------|-------------|
| 2   | 8.5% | 37.9% | −8.1% |
| 8   | 3.5% | 28.3% | −6.8% |
| 32  | 1.9% | 20.8% | −4.3% |
| 128 | 1.2% | 13.2% | −3.9% |

($N = 500$, 100,000 paths per rank, price standard error $\approx 0.6\%$.)

The Frobenius error looks reassuring, but it is the wrong metric for pricing. It is dominated by
the few large eigenvalues of `C`. The price depends instead on how much *variance* the sampled
paths carry, and a rank-$k$ truncation throws away the long tail of small eigenvalues, which hold
the short-scale, rough part of the path. At $k = 128$ that is still 13% of the total variance
(40% of the variance at the first time step). Less volatility means a lower price, so the
low-rank sampler underprices at every rank, and the bias decays slowly, like the variance lost.

An earlier version of this guide read the noisy price errors of a 10,000-path run as a
"systematic" accidental cancellation. That was wrong: those errors were all within about one
standard error of zero. With 100,000 paths and a well-behaved payoff (base volatility 0.2 instead
of 1.0), the bias is resolved and has the expected sign.

---

## The Shared Payoff (`src/common/asian_payoff.hpp`)

All three algorithms produce a `log_vol` vector and call the same payoff function:

```cpp
// log_vol[i] = nu * W^H(t_i);  sigma_i = sigma0 * exp(log_vol[i])
// Step 1: simulate price path
double S = S0;
for (int i = 0; i < N; ++i) {
    double sigma = sigma0 * std::exp(log_vol[i]);
    S *= std::exp((r - 0.5*sigma*sigma)*dt + sigma*sqrt(dt)*Z[i]);
    prices[i] = S;
}
// Step 2: arithmetic Asian payoff
double mean = accumulate(prices) / N;
return max(mean - K, 0.0);
```

The GBM step $\exp\!\bigl((r - \tfrac{1}{2}\sigma^2)\,dt + \sigma\sqrt{dt}\,Z\bigr)$ is the **exact solution** to the GBM SDE $dS = rS\,dt + \sigma S\,dW$ over one step (treating $\sigma$ as constant over $dt$), or equivalently the Euler–Maruyama discretization of the **log-price** dynamics $d(\log S) = (r - \tfrac{1}{2}\sigma^2)\,dt + \sigma\,dW$. It is not the Euler–Maruyama scheme on $dS$ directly, which would give the purely linear step $S_{t+dt} = S_t(1 + r\,dt + \sigma\sqrt{dt}\,Z)$. The $-\tfrac{1}{2}\sigma^2 dt$ Itô correction ensures $\mathbb{E}[S_T] = S_0 e^{rT}$.

The arithmetic mean of prices (not geometric) is what makes Asian options hard to
price analytically — no closed-form exists under stochastic volatility.

---

## Comparison Summary

| | Cholesky | FFT | Low-Rank rSVD ($k=32$) |
|---|---|---|---|
| **Math foundation** | $LL^\top = C$ exactly | Circulant embedding of fGn | Low-rank approx $C \approx L_k L_k^\top$ |
| **Construction** | $O(N^3)$ | $O(N \log N)$ | $O(N^2k)$ with rSVD |
| **Per path** | $O(N^2)$ | $O(N \log N)$ | $O(Nk)$ |
| **Memory** | $O(N^2)$ | $O(N)$ | $O(N^2)$ or $O(Nk)$ if freed |
| **Exact?** | Yes | Yes (all circulant eigenvalues verified $> 0$) | No (truncation error) |
| **Key paper** | — | Wood & Chan 1994 | Halko et al. 2011 |
| **Bottleneck at $N=1000$** | $N^2$ mat-vec per path | Shared $O(N)$ price-path work | Shared $O(N)$ price-path work |
| **Fitted $\alpha$** (N = 64–4000) | 1.52 global; local 1.0 → 2.1 | 1.02 | 1.02 |

The FFT method wins on memory ($O(N)$ vs $O(N^2)$) and is exact on the grid, so it is the recommended
method for production use. The global rSVD method is valuable pedagogically because it
exposes the low-rank structure of the problem and provides a tunable accuracy-speed
tradeoff. Cholesky is the baseline that makes all other methods' improvements concrete.

---

## Limitations and Further Work

Each algorithm has structural limitations that explain both why it was chosen for this
benchmark and what a production-quality system would do differently.

### Algorithm 1 — Dense Cholesky

**Scaling wall.** The $O(N^3)$ factorization and $O(N^2)$ memory scale poorly: at
$N = 10{,}000$ (minute-resolution paths over one year), the lower-triangular factor $L$
alone requires $\approx 800$ MB and the factorization takes several minutes. These are
hard limits — no constant-factor optimization can fix cubic growth.

**Memory-bandwidth ceiling at larger $N$.** Each path re-reads all of $L$. Up to $N = 1000$
the 7.6 MB matrix fits in the M2's 16 MB L2. Beyond $N \approx 1450$ it no longer fits, and
every path then streams $8N^2$ bytes from DRAM, so the loop becomes bandwidth-bound and a
faster core helps little. The benchmark's `est_bandwidth_GBs` column (up to 50 GB/s) is an
effective streaming rate, not a DRAM measurement, so it does not show this regime directly.

**Further directions.** Quasi-Monte Carlo (QMC) methods — replacing pseudo-random draws
with low-discrepancy Sobol or Halton sequences — can reduce MC standard error from
$O(1/\sqrt{M})$ toward $O(1/M)$, extracting more value from each expensive exact path.
For very large $N$, a sparse-direct or hierarchical Cholesky factorization would be
necessary.

### Algorithm 2 — Circulant Embedding + FFT

**Thin positivity margin for $H < \tfrac{1}{2}$.** No eigenvalue is ever negative here, but
the smallest one shrinks like $N^{-(1-2H)}$ relative to $\gamma(0)$: $0.24\%$ at $N = 252$ and
$0.08\%$ at $N = 1000$ for $H = 0.1$ (see `plots/stability_report.png`). This is harmless in
double precision, but it is why the code checks for negative eigenvalues and throws instead of
silently clipping.

**Restricted to stationary processes.** The embedding exploits the Toeplitz structure
of the fGn covariance, which follows from stationarity of the increments. Any departure
from stationarity — time-varying parameters, non-homogeneous volatility grids, or switching
to the Volterra-integral representation used in the rough Bergomi model — breaks the
Toeplitz structure and invalidates the method entirely. For those models the Hybrid Scheme
(Bennedsen, Lunde & Pakkanen 2017) is the natural replacement; see the "A Fourth Method"
entry under "Future work" in `README.md`.

**Further directions.** For processes whose minimal embedding is *not* PSD (e.g. some
long-memory kernels), padding to a $4N$ or $8N$ circulant is the standard fix. For
non-stationary settings, the Hybrid Scheme or a direct Gaussian
simulation via Cholesky on a reduced grid are the alternatives.

### Algorithm 3 — Global Low-Rank rSVD

**Not a true H-matrix — and that is the root cause of the slow convergence.** A
true Hierarchical matrix partitions the covariance matrix into a recursive block-tree:
near-diagonal blocks (where the $|s-t|^{0.2}$ singularity is concentrated) are kept
dense, while off-diagonal blocks are compressed with small *local* ranks. The local rank
required for a genuinely smooth far-field block is much smaller than the global rank $k$
needed here, because those blocks are isolated from the singularity.

By applying one global rSVD to the entire $N \times N$ matrix, the rank-$k$ budget must
represent both the rough near-diagonal behavior and the smooth far field simultaneously.
This is why $k = 128$ is needed for 1.2% Frobenius error on a $500 \times 500$ matrix —
roughly 10–16$\times$ larger than the per-block ranks a true H-matrix would require for
comparable accuracy. It also explains the $O(N^2 k)$ construction cost: a true H-matrix
can be built in $O(N \log N)$ or $O(N \log^2 N)$ precisely because it never forms the
full $N \times N$ matrix.

**No triangular structure.** The approximate factor $L_k = U \operatorname{diag}(\sqrt{S})$
is $N \times k$ but has no triangular structure. For tasks beyond sampling — solving linear
systems $Cx = b$, computing conditional distributions, or evaluating log-likelihoods — a
true hierarchical factorization (H-Cholesky) would be needed.

**Further directions.** A full H-matrix implementation (e.g., using HLIBpro or h2lib)
would give far better accuracy-vs-cost for rough $H$. A practical intermediate step is a
*block-diagonal + global low-rank* decomposition: keep a banded near-diagonal block dense
to capture the singularity, then apply the global rSVD only to the remaining off-diagonal
part, which is genuinely smooth and compresses well.

### Why These Three Algorithms Together

The three methods form a deliberate progression that spans the main structural properties
a fast-algorithms course aims to teach:

- **Cholesky** is the exact, structure-blind baseline. It asks only "is $C$ positive
  definite?" and applies the general dense factorization. Its cost reflects the full
  $O(N^3)$ price of ignoring structure.
- **FFT** exploits the *stationarity* of fGn increments — a property of the process
  itself — to reduce the covariance matrix to a Toeplitz form embeddable in a circulant.
  The $O(N \log N)$ cost is a direct reward for recognizing and using that structure.
  It stays exact for rough $H$ as well, since every circulant eigenvalue is positive.
- **Global rSVD** exploits the *smoothness* of the kernel away from the diagonal — a
  property of the geometry of the covariance function — to compress the matrix into a
  low-rank factor. It is cheaper per path than FFT for small $k$, but trades away
  exactness for a tunable approximation.

Together they illustrate three fundamental techniques: exploiting stationarity (FFT /
circulant diagonalization), exploiting low-rank structure (rSVD), and paying the full
dense price as a correctness baseline (Cholesky). The natural next steps — a true
H-matrix and the Hybrid Scheme — represent the frontier where multiple structural
exploitations are combined: hierarchical compression that isolates the singularity
(H-matrix) and an analytically corrected kernel discretization for Volterra models
(Hybrid Scheme).
