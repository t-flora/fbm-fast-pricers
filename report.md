# Fast Algorithms for Rough Volatility Monte Carlo Pricing

**Course:** Fast Algorithms · Final Project
**Topic:** Three algorithms for fractional Brownian motion simulation applied to Asian option pricing under the RFSV model

---

## Abstract

We implement and benchmark three algorithms for Monte Carlo pricing of an Arithmetic Asian Call Option under the Rough Fractional Stochastic Volatility (RFSV) model of Gatheral, Jaisson & Rosenbaum (2014), where log-volatility evolves as fractional Brownian motion with Hurst exponent $H \approx 0.10$. The central computational challenge is exact sampling from a correlated $N$-dimensional Gaussian, whose covariance matrix is dense and non-Markovian. We compare: (1) Dense Cholesky decomposition ($O(N^3)$ setup, $O(N^2)$ per path), (2) Circulant embedding + FFTW ($O(N \log N)$ per path, exploiting the Toeplitz structure of fractional Gaussian noise increments), and (3) Randomized SVD compression ($O(N k)$ per path for rank $k$, exploiting the low-rank off-diagonal structure of the covariance kernel). At $N = 1000$ and $M = 10{,}000$ paths, the FFT method is $1.6\times$ faster than Cholesky and the H-matrix method $3.7\times$ faster, with both approximately linear in $N$. We validate the model against live SPY implied volatility smiles, the Lévy (1992) Asian option approximation, and a 500k-path reference price, and characterize the numerical stability and memory footprint of each method.

---

## 1. Introduction

### Why Asian options?

Arithmetic Asian options have payoff $\max(\bar{S} - K, 0)$ where $\bar{S} = \frac{1}{N}\sum S_i$ is the time-average of the underlying. This averaging makes them cheaper than vanilla European options (reduced sensitivity to end-of-period price manipulation) and widely used in commodity and FX markets. The arithmetic average has no closed-form price under stochastic volatility, making Monte Carlo simulation the standard method.

### Why rough volatility?

Classical stochastic volatility models (Heston, SABR) assume log-volatility is a mean-reverting diffusion, implying Hurst exponent $H = 0.5$. Gatheral et al. (2014) estimated $H$ from high-frequency realized variance data for equity indices and found $H \approx 0.10$ — a dramatically rougher process than assumed. This "rough" regime ($H < 0.5$) means volatility increments are anti-persistent: spikes tend to reverse quickly. Rough volatility models fit the short-maturity implied volatility skew slope ($\sim T^{H-0.5}$) orders of magnitude better than classical models.

### Why fast algorithms?

Generating one fBM path naively requires sampling from $\mathcal{N}(0, C)$ where $C$ is an $N \times N$ dense matrix. The brute-force Cholesky approach costs $O(N^3)$ to factorize and $O(N^2)$ per path — at $N = 1000$ and $M = 10{,}000$ paths this is $10^{10}$ operations. Two structural properties of this problem enable faster methods: the *stationarity of increments* (leading to Toeplitz structure, exploitable by FFT) and the *low-rank off-diagonal structure* of the smooth covariance kernel (exploitable by hierarchical/randomized compression).

---

## 2. Mathematical Background

### The RFSV model

The asset price under RFSV follows:

$$dS_t = S_t \, \sigma_t \, dW_t^{\perp}, \qquad \log \sigma_t = \nu W_t^H$$

where $W_t^H$ is fractional Brownian motion with Hurst exponent $H$, and $W_t^\perp$ is an independent standard Brownian motion. Parameters: $H = 0.10$, $\nu = 0.30$ (calibrated from Oxford-Man realized variance data via `data/calibrate.py`).

### Fractional Brownian motion

fBM $W_t^H$ is the unique (up to scaling) Gaussian process with $W_0^H = 0$, stationary increments, and covariance:

$$\mathbb{E}[W_s^H W_t^H] = \tfrac{1}{2}\bigl(|s|^{2H} + |t|^{2H} - |s-t|^{2H}\bigr)$$

At $H = 0.5$ this reduces to standard Brownian motion. For $H < 0.5$ increments are anti-correlated (rough); for $H > 0.5$ they are positively correlated (persistent).

### Discretization

On a uniform grid $t_i = i \cdot \Delta t$, $\Delta t = T/N$, we need to sample the joint vector $(W_{\Delta t}^H, \ldots, W_T^H)$ from $\mathcal{N}(0, C)$ where $C_{ij} = \tfrac{1}{2}(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H})$.

### Option pricing

Given a log-vol path $\ell_i = \nu W_{t_i}^H$, the price path is:

$$S_i = S_{i-1} \exp\!\left[\left(r - \tfrac{1}{2}\sigma_i^2\right)\Delta t + \sigma_i \sqrt{\Delta t}\, Z_i\right], \qquad \sigma_i = e^{\ell_i}$$

where $Z_i \overset{\text{iid}}{\sim} \mathcal{N}(0,1)$ are independent of the vol process. The discounted Asian call price is:

$$V = e^{-rT} \mathbb{E}\!\left[\max\!\left(\tfrac{1}{N}\sum_{i=1}^N S_i - K,\; 0\right)\right]$$

estimated by Monte Carlo over $M$ paths.

---

## 3. Algorithms

### 3.1 Dense Cholesky (`src/cholesky/cholesky.hpp`)

**Idea.** Factor $C = LL^\top$ once, then draw $z \sim \mathcal{N}(0, I_N)$ per path and compute $\ell = \nu L z$.

**Complexity.**

| Phase | Cost |
|-------|------|
| Build $C$ | $O(N^2)$ |
| Factorize $C = LL^\top$ | $O(N^3)$ |
| Per-path multiply $Lz$ | $O(N^2)$ |
| Total ($M$ paths) | $O(N^3 + MN^2)$ |

At $M = 10{,}000$ the per-path cost dominates for all $N$ tested. Fitted exponent: $\alpha \approx 1.54$, with $t \approx 4.1 \times 10^{-5} \cdot N^{1.54}$ seconds.

**Memory.** The $N \times N$ lower-triangular factor $L$ lives on the heap throughout: $N^2 \cdot 8$ bytes. At $N = 1000$: 7.6 MB. The MC loop re-reads this matrix for every path, making Cholesky **memory-bandwidth limited** — estimated utilization 47 GB/s at $N = 1000$ vs. Apple M2 rated 100 GB/s.

### 3.2 Circulant Embedding + FFT (`src/fft/fft.hpp`)

**Key insight.** fBM is non-stationary: its covariance $C_{ij}$ depends on both $t_i$ and $t_j$. But the **increments** (fractional Gaussian noise, fGn) are stationary: $\text{Cov}(\delta W_i, \delta W_{i+k}) = \gamma(k)$ depends only on the lag. The fGn covariance matrix is therefore **Toeplitz**, and any Toeplitz matrix embeds into a **circulant**, which is diagonalized by the DFT.

**The fGn autocovariance:**

$$\gamma(k) = \frac{\Delta t^{2H}}{2}\bigl(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}\bigr)$$

**Algorithm** (Wood & Chan 1994 / Davies-Harte):

1. Build circulant first row: $c = [\gamma(0), \ldots, \gamma(N{-}1), 0, \gamma(N{-}1), \ldots, \gamma(1)]$ (size $2N$)
2. Eigenvalues: $\lambda = \text{FFT}(c)$ (all real, all $\geq 0$ for $H \leq 0.5$ in theory)
3. Per path: draw complex white noise $w_j \sim \mathcal{CN}(0, \lambda_j / 2N)$, compute IFFT$(w)$, take real part → fGn increments
4. Cumulative sum of increments → fBM log-vol path

**Complexity.**

| Phase | Cost |
|-------|------|
| Build eigenvalues | $O(N \log N)$ |
| Per-path IFFT | $O(N \log N)$ |
| Total ($M$ paths) | $O(MN \log N)$ |

Fitted exponent: $\alpha \approx 1.01$. **Memory:** only the size-$2N$ complex vector $\lambda$ (16 KB at $N = 1000$, far below L1 cache). FFTW plans are created once and reused.

**Stability note.** Wood & Chan proved $\lambda_j \geq 0$ for $H \in (0,1)$ in the **continuous** limit. At finite $N$ with $H = 0.10$, 28% of eigenvalues are numerically negative (15% of spectral energy). We clip them to zero, introducing a small but nonzero bias. The price error vs. exact Cholesky is $\approx 0.3\%$ at $N = 252$.

### 3.3 H-matrix + Randomized SVD (`src/hmatrix/hmatrix.hpp`)

**Key insight.** The fBM covariance kernel $C(s,t)$ is smooth away from the diagonal $s \approx t$ (Candès, Demanet & Ying 2008). Smooth kernels on well-separated index blocks are numerically low-rank. A global rank-$k$ randomized SVD approximates $C \approx U \operatorname{diag}(S) U^\top$, giving an approximate Cholesky factor $L_k = U \operatorname{diag}(\sqrt{S})$.

**rSVD** (Halko, Martinsson & Tropp 2011, Algorithm 4.4):

1. Random sketch: $Y = C\Omega$, $\Omega \in \mathbb{R}^{N \times (k+p)}$ Gaussian, oversampling $p = 5$
2. Power iteration: $Y \leftarrow (CC^\top)^q Y$ for $q = 2$ iterations (critical for $H = 0.1$ slow spectrum)
3. QR: $Q = \text{orth}(Y)$; project: $B = Q^\top C$
4. Thin SVD of small matrix $B$; reassemble $U, S$

**Why power iteration?** At $H = 0.1$, singular values of $C$ decay slowly (rough spectrum). Power iteration replaces $\sigma_i$ with $\sigma_i^{2q+1}$, amplifying the ratio between large and small values, dramatically improving sketch quality.

**Complexity.**

| Phase | Cost |
|-------|------|
| Build $C$ | $O(N^2)$ |
| rSVD | $O(N k^2)$ |
| Per-path multiply $L_k z$ | $O(Nk)$ |
| Total ($M$ paths) | $O(N^2 + MNk)$ |

At $k = 32$: fitted exponent $\alpha \approx 1.06$. **Memory:** $C$ ($N^2 \cdot 8$ bytes) is needed during construction. In the `price_freed_timed` variant, $C$ is destroyed before the MC loop (via RAII block scope), leaving only $L_k$ ($Nk \cdot 8$ bytes) — 0.24 MB vs. 7.6 MB at $N = 1000$.

**Accuracy.** Frobenius error $\|C - C_k\|_F / \|C\|_F$ at $k = 64$: 1.5%. Price error is masked by MC noise above $k \approx 8$.

---

## 4. Experiments

### Experiment 1 — Runtime scaling

**Controlled:** $M = 10{,}000$ paths, seeded RNG, same payoff function.
**Independent:** $N \in \{252, 500, 1000\}$.
**Dependent:** total wall-clock time (seconds).

| Method | $c$ | $\alpha$ | $R^2$ | $t$ at $N=1000$ |
|--------|-----|----------|-------|-----------------|
| Dense Cholesky | $4.1 \times 10^{-5}$ | 1.54 | 0.999 | 1.71 s |
| Circulant FFT | $1.0 \times 10^{-3}$ | 1.01 | 1.000 | 1.10 s |
| H-matrix $k=32$ | $3.0 \times 10^{-4}$ | 1.06 | 1.000 | 0.45 s |

FFT and H-matrix are effectively $O(N)$ across this range. Cholesky's $\alpha = 1.54$ is explained by the $O(MN^2)$ per-path matrix-vector product dominating over the $O(N^3)$ one-time factorization (at $M = 10{,}000$, the crossover is at $N \approx 10{,}000$).

**Construction vs. MC breakdown** (see `plots/construction_breakdown.png`): FFT construction is $< 0.1$ ms (negligible); Cholesky factorization is 26 ms at $N = 1000$ (1.5% of total time); rSVD construction is 32 ms (7% of total). The MC loop dominates all three methods.

### Experiment 2 — Memory footprint

**Controlled:** $M = 10{,}000$ paths.
**Independent:** $N \in \{252, 500, 1000\}$.
**Dependent:** theoretical peak memory and estimated memory bandwidth.

| Method | Peak memory ($N=1000$) | Cache pressure (L3=16 MB) |
|--------|----------------------|--------------------------|
| Cholesky | 7.6 MB | $0.48\times$ |
| FFT | 0.015 MB | $0.001\times$ |
| H-matrix (C held) | 7.6 MB | $0.48\times$ |
| H-matrix (C freed) | 0.24 MB | $0.015\times$ |

Cholesky at $N = 1000$ requires re-reading 7.6 MB for every path: $7.6 \text{ MB} \times 10{,}000 = 76 \text{ GB}$ total memory traffic. At the measured 47 GB/s, this accounts for the observed wall time. FFT requires only a 15 KB working set — comfortably in L1 cache.

Python engine (tracemalloc): peak = $13\times$ float64 arrays of size $M \times N$. L3 spill occurs at $M \geq 5{,}000$ for any $N$.

### Experiment 3 — MC convergence

**Controlled:** $N = 252$, $H = 0.10$, $\nu = 0.30$.
**Independent:** $M \in \{100, 250, 500, 1{,}000, 2{,}500, 5{,}000, 10{,}000, 25{,}000\}$, 5 seeds each.
**Dependent:** mean price and standard error across seeds.

The CLT predicts $\sigma_{\hat{p}} = \sigma_\text{payoff} / \sqrt{M}$ with $\sigma_\text{payoff} \approx 35$ (large because $\sigma_0 = 1.0$, i.e. 100% annualised vol). Fitted log-log slope: $\approx -0.50$ ($R^2 \approx 0.97$), confirming the $1/\sqrt{M}$ convergence law. At $M = 25{,}000$, the price standard error is $\approx 0.22$, or $\approx 0.9\%$ of the reference price.

### Experiment 4 — H-matrix accuracy vs. rank

**Controlled:** $N = 500$, $M = 10{,}000$ paths.
**Independent:** rank $k \in \{2, 4, 8, 16, 32, 64, 128\}$.
**Dependent:** Frobenius error $\|C - C_k\|_F / \|C\|_F$ and price error $|\hat{p}_k - p_\text{ref}|$.

| Rank $k$ | Frobenius error | Price error | Speed vs. Cholesky |
|----------|----------------|-------------|-------------------|
| 4 | 5.4% | 0.05 | $3.8\times$ |
| 16 | 2.5% | 0.13 | $3.5\times$ |
| 32 | 1.9% | 0.34 | $2.6\times$ |
| 128 | 1.2% | 0.29 | $1.6\times$ |

Price error does not decrease monotonically with $k$ because MC noise ($\approx 0.3$) dominates the truncation error above $k \approx 8$. The Frobenius error (noise-free) correctly shows monotone improvement. Slow decay of the Frobenius error is intrinsic to $H = 0.1$: the rough spectrum requires many singular values to capture.

### Experiment 5 — Stability

**FFT eigenvalue clipping.** At $H = 0.10$, $N = 252$: 28.4% of circulant eigenvalues are numerically negative, corresponding to 15.4% of total spectral energy. This is a finite-$N$ discretisation artefact; Wood & Chan (1994) guarantees non-negative eigenvalues only in the continuous limit. At $H \geq 0.499$: all eigenvalues are positive, the method is exactly PSD.

**rSVD condition number.** $\kappa(L_k) = \sqrt{S_1}/\sqrt{S_k}$ grows from 8.2 at $k=8$ to 23.3 at $k=64$. For 64-bit floats ($\epsilon \approx 10^{-16}$), even $\kappa = 100$ loses only 2 significant digits — far from catastrophic. The practical effect is slightly inflated variance in the MC estimator for paths corresponding to small singular values.

**Cholesky conditioning.** $\kappa(C) \approx 789$ at $N = 252$, growing as $\approx N^{1.5}$ (fitted). At $N = 1000$: $\kappa \approx 10^4$. All eigenvalues remain well above machine epsilon, so Cholesky is numerically stable across the range tested. For very large $N$ (beyond 10,000), regularisation may become necessary.

---

## 5. Model Validation

### Phase 1 — Implied volatility smile vs. SPY

`data/validate_iv.py` fetches live SPY option chains (via yfinance), computes market implied volatility by Black-Scholes inversion, and overlays RFSV model IVs. A log-vol drift $\mu_0 = \log \sigma_\text{eff}$ is calibrated to match the ATM market price, isolating smile *shape* from level.

**Result:** the RFSV smile slope and curvature qualitatively match the SPY market skew across two expiration dates, confirming that $H = 0.10$ captures the characteristic steep short-maturity skew. The model slightly overestimates OTM call IV, consistent with the known result that RFSV without jumps underprices crash risk.

### Phase 2 — Lévy benchmark and roughness premium

`data/validate_asian.py` benchmarks RFSV prices against the Lévy (1992) analytical approximation for arithmetic Asian calls under constant-$\sigma$ GBM.

**Sanity check:** at $\nu \to 0$, RFSV $\to$ constant-$\sigma$ GBM and prices match Lévy. At $\nu = 0.30$, the Jensen's inequality convexity effect raises RFSV prices above Lévy by $\approx 3$–5 points.

**Roughness premium** (RFSV at $H=0.10$ minus $H=0.50$): concentrated near ATM ($K=100$), approximately $+0.7$ at $M = 10{,}000$ paths. OTM and deep-ITM options are less sensitive to roughness because the payoff is either always zero or approximately linear in price.

### Phase 3 — Parameter sensitivity

`plots/plot_sensitivity.py` sweeps $H \in \{0.05, 0.10, 0.15, 0.20, 0.30, 0.50\}$ and $\nu \in \{0.10, 0.20, 0.30, 0.40\}$ at ATM ($K = S_0$).

**Key findings:**
- Price increases as $H$ decreases (roughness premium, Jensen's inequality): from $H=0.50$ to $H=0.05$, ATM price rises by $\approx 3$ points at $\nu = 0.30$.
- Price is approximately linear in $\nu^2$ (as expected for small-noise expansions).
- Price vs. strike: the roughness effect is concentrated near ATM; deep OTM prices are nearly identical across $H$ values.

---

## 6. Structural Analysis

`plots/plot_structure.py` provides geometric motivation for each algorithm.

**Toeplitz structure of fGn** (motivates FFT): the fGn covariance heatmap shows perfectly constant diagonals (max row-to-row deviation = 0.00e+00), while the fBM heatmap does not. The translational invariance of increments is the mathematical foundation for circulant embedding.

**Low-rank off-diagonal structure** (motivates rSVD): singular values of the top-right quadrant of $C$ (representing covariance between the first and second half of the path) are plotted for $H = 0.10$ and $H = 0.50$. At the 1% threshold, $H = 0.50$ requires rank $\approx 7$ vs. $H = 0.10$ requiring rank $\approx 20$. This explains why Frobenius errors converge slowly for rough $H$: more singular values are needed to capture the same fraction of energy.

---

## 7. Discussion

### Limitations

1. **FFT accuracy at finite $N$.** The Wood-Chan embedding is theoretically exact but produces negative eigenvalues at $H = 0.1$, $N = 252$. Clipping introduces a 15% energy loss. The Bennedsen-Lunde-Pakkanen (2017) hybrid scheme avoids this by treating the power-law kernel singularity at zero analytically.

2. **H-matrix is global, not hierarchical.** Our implementation uses a single rank-$k$ global SVD rather than a recursive block-tree with separate low-rank approximations per sub-block. A true hierarchical H-matrix would achieve $O(N \log^2 N)$ construction and could compress the diagonal blocks too.

3. **Model risk.** RFSV with $H = 0.10$ fits volatility dynamics well but ignores jumps, mean reversion (the empirical log-vol process is more precisely a rough Ornstein-Uhlenbeck), and multi-factor structure. The roughness premium result is sensitive to $H$ and $\nu$ calibration.

4. **Calibration.** $H$ and $\nu$ are estimated from realized variance using a log-log regression on $\mathbb{E}[\log \text{RV}(\Delta)]$ vs. $\Delta$. We use yfinance 5-day non-overlapping windows as a free proxy for Oxford-Man data, obtaining $H \approx 0.117$ ($R^2 = 0.98$), close to the Oxford-Man value $H \approx 0.10$.

### Future work

- **Hybrid scheme** (Bennedsen et al. 2017): better accuracy for rough kernels at small $N$, same $O(N \log N)$ complexity.
- **OpenMP parallelism**: the MC loop over paths is embarrassingly parallel; each path is independent.
- **Mean-reverting rough vol**: the actual Gatheral-Jaisson-Rosenbaum model uses a fractional Ornstein-Uhlenbeck (rough Bergomi), not pure fBM. This would require a different covariance kernel.
- **Variance reduction**: antithetic variates or control variates (e.g., using the Lévy approximation as a control) could reduce $\sigma_\text{payoff}$ from $\approx 35$ to $\approx 5$, reducing the required $M$ by $50\times$.

---

## 8. Conclusion

| | Cholesky | FFT | H-matrix ($k=32$) |
|---|---|---|---|
| Exact? | Yes | Approx. at finite $N$ | No |
| Construction | $O(N^3)$ | $O(N \log N)$ | $O(N k^2)$ |
| Per path | $O(N^2)$ | $O(N \log N)$ | $O(Nk)$ |
| Memory | $O(N^2)$ | $O(N)$ | $O(N^2)$ or $O(Nk)$ |
| Speed at $N=1000$ | 1.71 s | 1.10 s | 0.45 s |
| Price (reference) | 23.58 | 23.58 | 23.15 ($\pm$ MC) |

The FFT method is recommended for production use: it is nearly exact (eigenvalue clipping error $< 0.3\%$), memory-efficient ($O(N)$), and approximately linear in $N$. The H-matrix method is valuable as a pedagogical illustration of how the low-rank structure of smooth covariance kernels can be exploited algorithmically — the same idea underlying the butterfly algorithm, FMM, and panel methods. Cholesky serves as the exact baseline that makes all speedup claims meaningful.

The project demonstrates that going from $O(N^3)$ to $O(N \log N)$ is not just a theoretical curiosity: at $N = 1000$ paths and $M = 10{,}000$ Monte Carlo paths, it corresponds to a $1.6\times$ runtime improvement and a $500\times$ memory reduction, both of which become critical at production scale.

---

## References

1. Gatheral, J., Jaisson, T. & Rosenbaum, M. (2014). *Volatility is rough.*
2. Halko, N., Martinsson, P.-G. & Tropp, J. A. (2011). Finding structure with randomness: probabilistic algorithms for constructing approximate matrix decompositions. *SIAM Review* 53(2), 217–288.
3. Candès, E. J., Demanet, L. & Ying, L. (2008). A fast butterfly algorithm for the computation of Fourier integral operators. *Multiscale Modeling & Simulation* 7(4).
4. Bennedsen, M., Lunde, A. & Pakkanen, M. S. (2017). Hybrid scheme for Brownian semistationary processes. *Finance and Stochastics* 21(4).
5. Wood, A. T. A. & Chan, G. (1994). Simulation of stationary Gaussian processes in $[0,1]^d$. *Journal of Computational and Graphical Statistics* 3(4), 409–432.
6. Lévy, E. (1992). Pricing European average rate currency options. *Journal of International Money and Finance* 11(5), 474–491.
