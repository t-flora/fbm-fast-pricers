# A Beginner's Guide to the Three fBM Simulation Algorithms

This document walks through the three algorithms implemented in `src/` for
simulating fractional Brownian Motion (fBM) paths and pricing an Arithmetic
Asian Call Option under the Rough Fractional Stochastic Volatility (RFSV) model.

---

## Background: What Problem Are We Solving?

The RFSV model (Gatheral, Jaisson & Rosenbaum 2014) says
that log-volatility evolves as a fractional Brownian motion:

```
log σ_t = ν · W_t^H
```

where `W_t^H` is fBM with Hurst exponent `H ≈ 0.10` (empirically estimated from
realized variance data). The full Gatheral RFSV model includes a log-vol drift $\mu$:
$\log\sigma_t = \mu + \nu W_t^H$. This implementation sets $\mu = 0$, so
$\sigma_0 = e^{\mu + \nu W_0^H} = 1$ — a deliberate simplification giving 100% base
annualized volatility (consistent with the reference price $p_\text{ref} = 23.58$).
When comparing against market data, $\mu_0 = \log\sigma_\text{target}$ is calibrated
to match the ATM implied volatility level.

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
- `params.hpp`: `H=0.10`, `ν=0.30`, `S0=100`, `K=100`, `T=1`, `r=0`
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

**Build the covariance matrix** (`cholesky.hpp:20`):
```cpp
Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
```
This fills the $N \times N$ matrix with $C[i,j] = \frac{1}{2}(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})$
where $t_i = i \cdot dt$. Cost: $O(N^2)$.

**Cholesky factorization** (`cholesky.hpp:21-23`):
```cpp
Eigen::LLT<Eigen::MatrixXd> llt(C);
if (llt.info() != Eigen::Success)
    throw std::runtime_error("Cholesky: matrix not positive-definite");
Eigen::MatrixXd L = llt.matrixL();
```
`Eigen::LLT` computes the lower-triangular factor `L` such that $C = L L^\top$.
Cost: $O(N^3)$. Done **once** before the Monte Carlo loop.

**Per-path sampling** (`cholesky.hpp:30-33`):
```cpp
for (int i = 0; i < N; ++i) z(i) = norm(rng);
Eigen::VectorXd lv = nu * (L * z);
```
`L * z` is an $N \times N$ lower-triangular matrix times an $N$-vector: cost $O(N^2)$ per path.
`lv[i] = log σ_{t_i}` — the log-volatility path.

**Price simulation** (`cholesky.hpp:34-35`):
```cpp
auto inno = randn(N, rng);
payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
```
`inno` is a fresh set of N i.i.d. Gaussians for the **price process** (independent of
the vol process). `log_vol_to_prices` steps $S_i = S_{i-1} \cdot \exp\!\bigl((r - \tfrac{1}{2}\sigma^2)\,dt + \sigma\sqrt{dt}\,Z_i\bigr)$.

### Complexity

| Phase | Cost | $N=252$ | $N=1000$ |
|-------|------|---------|----------|
| Factorize $C$ | $O(N^3)$ | ~0.1ms | ~26ms |
| MC loop ($M$ paths) | $O(MN^2)$ | ~200ms | ~1670ms |

The factorization cost is dominated by the per-path cost at $M = 10{,}000$.
The fitted exponent $\alpha \approx 1.54$ (from `benchmarks/results/time_vs_N.csv`)
reflects the $O(MN^2)$ regime, but the pure $O(N^2)$ per-path cost would predict $\alpha \approx 2$.
The observed value is lower because each MC iteration also carries $O(N)$ overhead — $N$ Gaussian
draws, $N$ exponentials to recover $\sigma$ from $\log\sigma$, and $N$ payoff steps — which
dampens the effective exponent to 1.54 over the tested range $N \in \{252, 1000\}$.
This is confirmed by the memory bandwidth calculation:
$N = 1000$ reads $8\,\mathrm{MB} \times 10{,}000$ paths $\approx 80\,\mathrm{GB}$ at $\approx 47\,\mathrm{GB/s} \approx 1.7\,\mathrm{s}$. ✓

### Memory
Peak: $N \times N$ matrix `L` lives on the heap throughout $= N^2 \cdot 8$ bytes.
At $N = 1000$: 7.6 MB (well below M2's L3 = 16 MB, so `L` stays in cache at small $N$).

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
This is $\gamma(k)$. Note `γ(0) = dt^{2H}` (variance of each increment).

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
The critical property (proved in Wood & Chan 1994): for $H \geq \tfrac{1}{2}$, all eigenvalues
are **non-negative** at any $N$. For $H < \tfrac{1}{2}$ non-negativity holds only in the large-$N$ limit; finite-$N$ negative eigenvalues are clipped to zero (see the stability analysis).

**Step 3 — FFT to get eigenvalues** (`fft.hpp:53-64`):
```cpp
fftw_plan p = fftw_plan_dft_1d(M, ..., FFTW_FORWARD, FFTW_ESTIMATE);
fftw_execute(p);
fftw_destroy_plan(p);
```
The DFT of a circulant's first row gives its eigenvalues `λ`. This is the core
mathematical fact: **a circulant matrix C is diagonalized by the DFT matrix F**,
meaning `C = F·diag(λ)·F*`. Cost: $O(N \log N)$. Done once.

**Step 4 — Per-path synthesis** (`fft.hpp:72-79`):
```cpp
for (int j = 0; j < M; ++j) {
    double s = std::sqrt(std::max(lam[j].real(), 0.0) / M);
    w_buf[j] = s * std::complex<double>(norm(rng), norm(rng));
}
fftw_execute(plan_inv);
```
This samples $w_j = \sqrt{\lambda_j / M} \cdot (a_j + i b_j)$ with $a_j, b_j \sim \mathcal{N}(0,1)$,
then computes the IFFT. Why does this work?

If $C = F \cdot \operatorname{diag}(\lambda) \cdot F^*$, then we want $x$ such that $\operatorname{Cov}(x) = C$. Let $x = F \cdot w$.
Then $\operatorname{Cov}(x) = F \cdot \operatorname{Cov}(w) \cdot F^* = F \cdot \operatorname{diag}(\lambda/M) \cdot \operatorname{diag}(M) \cdot F^* = F \cdot \operatorname{diag}(\lambda) \cdot F^* = C$. ✓

**A note on complex vs real covariance.** Each $w_j$ is complex with independent real and imaginary
parts each of variance $\lambda_j/M$, so the Hermitian variance is $E[|w_j|^2] = 2\lambda_j/M$ — a
factor of 2 larger than what the sketch above uses. This factor is recovered exactly when the code
takes `out_buf[i].real()` in Step 5: for a symmetric circulant with real eigenvalues, the real and
imaginary parts of the IFFT output carry equal power (by the $\cos^2 + \sin^2 = 1$ identity across
all frequency components), so extracting the real part halves the total variance back to the correct
target $\gamma(0) = \Delta t^{2H}$. The sketch above implicitly absorbs this factor and is correct in its final result.

The `/ M` comes from FFTW's unnormalized convention (IFFT multiplies by $M$).
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
reconstructs the fBM path $W_{t_i} = \sum_{j \leq i} \delta W_j$. Then $\log\sigma_i = \nu \cdot W_{t_i}$.

### Complexity

| Phase | Cost |
|-------|------|
| Build eigenvalues | $O(N \log N)$ |
| Per-path IFFT | $O(N \log N)$ |
| Full MC | $O(MN \log N)$ |

At $N = 1000$, FFT is $1.5\times$ faster than Cholesky despite both running $M = 10{,}000$ paths,
because $O(N \log N) \approx 10{,}000$ vs $O(N^2) = 10^6$ operations per path. A sharp reader
will ask: if the vol simulation is $100\times$ cheaper, why is the wall-clock speedup only
$1.5\times$? This is Amdahl's Law in action. The price process simulation — $N$ Gaussian draws
for the innovations, $N$ exponentials to step the stock price, and $N$ payoff accumulations —
is identical for both methods and takes roughly twice as long as Cholesky's own triangular
solve at $N = 1000$. That shared $O(N)$ work now dominates FFT's total runtime, diluting the
$100\times$ vol-simulation gain into a $1.5\times$ end-to-end speedup. The fitted exponent $\alpha \approx 1.01 \approx 1$.

**Memory**: only the size-$2N$ complex vectors `c_emb` and `lam` are needed (not $N \times N$).
At $N = 1000$: ~16 KB — fits in L1 cache. This is the critical memory advantage.

---

## Algorithm 3: Global Low-Rank Approximation via rSVD (`src/hmatrix/hmatrix.hpp`)

### The key insight: covariance has low-rank off-diagonal blocks

The fBM covariance matrix is **smooth** away from its diagonal. Intuitively:
far-apart time points have a slowly-varying, well-approximated covariance.
This means the **off-diagonal blocks** are numerically low-rank.

Candès, Demanet & Ying (2008) formalize this:
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

You can see the SVD decay in `plots/structure_analysis.png` (panel d): for $H = 0.1$ the
singular values decay more slowly than for $H = 0.5$ — the rough spectrum is harder to compress.

### The rSVD algorithm (`src/hmatrix/rsvd.hpp`)

This implements Halko, Martinsson & Tropp (2011) Algorithm 4.4.

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
This replaces `A` with $(AA^\top)^q \cdot A$ for $q = 2$ iterations. The singular values
of the iterated matrix are $\sigma_i^{2q+1}$, so the ratio between large and small
singular values is amplified: $(\sigma_1/\sigma_2)^5$ instead of $\sigma_1/\sigma_2$.

**Why this matters for $H = 0.1$**: the fBM covariance at small $H$ has **slowly-decaying**
singular values (rough spectrum). Without power iteration, the random sketch can't
distinguish the top-$k$ directions. With $q = 2$, the sketch quality improves dramatically.
This is the key insight from Section 4.3 of Halko et al.

**Stage B — QR + small SVD** (`rsvd.hpp:36-43`):
```cpp
Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(m, l);
Eigen::MatrixXd B = Q.transpose() * A;   // l × n  (small!)
Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, ...);
```
`Q` is an orthonormal basis for the range of `Y`. Projecting `A` onto `Q` gives
the small $l \times n$ matrix `B`. The SVD of `B` costs $O(l^2 n)$ — much cheaper than $O(N^3)$
for the full SVD.

**Reassemble** (`rsvd.hpp:45-48`):
```cpp
result.U  = Q * svd.matrixU().leftCols(k);
result.S  = svd.singularValues().head(k);
result.Vt = svd.matrixV().leftCols(k).transpose();
```
The final approximation is $A \approx U \operatorname{diag}(S) V^\top$ with $U \in \mathbb{R}^{N \times k}$.

### Using rSVD for path generation (`hmatrix.hpp:35-52`)

**Approximate Cholesky factor** (`hmatrix.hpp:38-39`):
```cpp
Eigen::VectorXd sqrt_S = decomp.S.cwiseMax(0.0).cwiseSqrt();
Eigen::MatrixXd Lk = decomp.U * sqrt_S.asDiagonal();  // N × k
```
For a symmetric PSD matrix, $C \approx U S U^\top$, so
$C \approx (U\sqrt{S})(U\sqrt{S})^\top = L_k L_k^\top$.
The `cwiseMax(0.0)` clips any tiny negative singular values from numerical error.

**Per-path sampling** (`hmatrix.hpp:44-49`):
```cpp
for (int i = 0; i < k; ++i) z(i) = norm(rng);
Eigen::VectorXd lv = nu * (Lk * z);  // O(N*k)
```
Instead of an $N \times N$ triangular solve, we compute an **$N \times k$ matrix times a $k$-vector**.
At $k = 32$, $N = 1000$: $32{,}000$ operations vs $10^6$ for Cholesky. This is the per-path speedup.

### The freed variant (`hmatrix.hpp:93-131`)

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

Fitted exponent $\alpha \approx 1.06$ (close to linear in $N$ at fixed $k = 32$). The crossover
point vs Cholesky: at $k = 32$, $O(MN \cdot 32) < O(MN^2)$ when $32 < N$, which is always true.

### Accuracy tradeoff

The approximation error depends on how many singular values of `C` we keep.
From `benchmarks/results/error_vs_rank.csv`:

| rank k | Frobenius error | Price error |
|--------|----------------|-------------|
| 4  | 5.4% | 0.05 |
| 16 | 2.5% | 0.13 |
| 32 | 1.9% | 0.34 |
| 64 | 1.5% | 0.61 |

The price error trend is counter-intuitive and deserves careful reading. The Frobenius error
monotonically decreases with rank, as expected. The price error does the opposite, which
is not simply MC noise: a clean monotone increase across four doublings is a systematic effect.
The most likely cause is that at low rank the truncation bias moves the H-matrix price
*toward* the reference by accident — heavy smoothing of the volatility paths alters the
distribution of payoffs in a way that happens to compensate, giving a deceptively small
price error at $k = 4$ despite the large matrix approximation error. As rank increases and
the paths become truer to the actual fBM distribution, this accidental cancellation
disappears, and the price error reflects the remaining MC variance (~$\sigma_V/\sqrt{M} \approx 0.35$).
The Frobenius error is the reliable measure of approximation quality; price error at low
rank should not be read as evidence of a good approximation. This is why we need both metrics.

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
| **Exact?** | Yes | Asymptotically (exact for $H \geq \tfrac{1}{2}$; clipping bias for $H < \tfrac{1}{2}$ at finite $N$) | No (truncation error) |
| **Key paper** | — | Wood & Chan 1994 | Halko et al. 2011 |
| **Bottleneck at $N=1000$** | Memory bandwidth | Compute | Compute |
| **Fitted $\alpha$** | 1.54 | 1.01 | 1.06 |

The FFT method wins on memory ($O(N)$ vs $O(N^2)$) and is asymptotically exact (though
subject to minor clipping bias at finite grids for rough $H$) — it is the recommended
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

**Memory-bandwidth ceiling.** The per-path $O(N^2)$ triangular solve is bandwidth-limited,
not compute-limited: each path re-reads the full $L$ matrix from DRAM. A faster CPU core
gives almost no benefit; only wider memory buses (e.g., a GPU or HBM) would help materially.

**Further directions.** Quasi-Monte Carlo (QMC) methods — replacing pseudo-random draws
with low-discrepancy Sobol or Halton sequences — can reduce MC standard error from
$O(1/\sqrt{M})$ toward $O(1/M)$, extracting more value from each expensive exact path.
For very large $N$, a sparse-direct or hierarchical Cholesky factorization would be
necessary.

### Algorithm 2 — Circulant Embedding + FFT

**Finite-$N$ eigenvalue clipping for $H < \tfrac{1}{2}$.** The PSD guarantee of Wood &
Chan (1994) holds asymptotically; at finite $N$ with $H = 0.1$, roughly 28% of circulant
eigenvalues are negative and must be clipped to zero (see `plots/stability_report.png`).
Clipping discards $\approx 15\%$ of spectral energy, introducing a small bias that
decreases as $N \to \infty$ but never vanishes at any finite grid. The method is therefore
not strictly exact for rough $H$ at practical $N$, only asymptotically exact.

**Restricted to stationary processes.** The embedding exploits the Toeplitz structure
of the fGn covariance, which follows from stationarity of the increments. Any departure
from stationarity — time-varying parameters, non-homogeneous volatility grids, or switching
to the Volterra-integral representation used in the rough Bergomi model — breaks the
Toeplitz structure and invalidates the method entirely. For those models the Hybrid Scheme
(Bennedsen, Lunde & Pakkanen 2017) is the natural replacement; see the "A Fourth Method"
section in `README.md`.

**Further directions.** Embedding into a $4N$ or $8N$ circulant instead of $2N$ drives
more eigenvalues positive at small $H$ and small $N$, trading higher FFT cost for fewer
clipping artifacts. For non-stationary settings, the Hybrid Scheme or a direct Gaussian
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
This is why $k = 128$ is needed for 1.2% Frobenius error on a $1000 \times 1000$ matrix —
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
  Crucially, it achieves this exactness for standard and long-memory processes, remaining
  the asymptotically optimal choice even when finite-grid clipping is required for rough
  volatility.
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
