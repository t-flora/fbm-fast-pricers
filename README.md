# Rough Volatility Asian Option Pricer

A high-performance Monte Carlo pricer for an **Arithmetic Asian Call Option** under the **Rough Fractional Stochastic Volatility (RFSV)** model, benchmarking three methods for generating fractional Brownian motion (fBM) paths. The project is a practical study in how algorithmic complexity theory translates to real speedups on a computationally demanding financial problem.

---

## The Problem

### Why rough volatility?

Gatheral, Jaisson & Rosenbaum (2014) — [*Volatility is rough*](papers-bg/vol-is-rough.pdf) — showed by estimating volatility from high-frequency data that **log-volatility behaves essentially as fractional Brownian motion with Hurst exponent $H \approx 0.1$**. This finding is empirically robust across many equity indices and time scales.

The **RFSV model** specifies log-volatility as:

$$\log \sigma_t = \nu \, W_t^H$$

where $W^H$ is fBM with $H \approx 0.1$ and $\nu$ is the volatility-of-volatility. Because $H < \tfrac{1}{2}$, increments are **anti-persistent** ("rough"), meaning each spike in volatility is likely to reverse. This is the opposite of the classical long-memory assumption ($H > \tfrac{1}{2}$) and contradicts much of the prior literature.

The fBM $W^H$ is characterised by its scaling law: for any $q > 0$,

$$\mathbb{E}\bigl[|W^H_{t+\Delta} - W^H_t|^q\bigr] = K_q \,\Delta^{qH}$$

so increments have Hölder regularity $r$ for any $r < H$.

### Why is this computationally hard?

fBM is **non-Markovian**: the future distribution of $W^H_t$ depends on the entire past path. There is no recursion or SDE to step forward from. Exact path generation requires constructing the full $N \times N$ covariance matrix:

$$C_{ij} = \mathbb{E}\!\left[W^H(t_i)\, W^H(t_j)\right] = \tfrac{1}{2}\!\left(t_i^{2H} + t_j^{2H} - |t_i - t_j|^{2H}\right)$$

This is dense — every pair of time points is correlated — and its Cholesky factorisation costs $O(N^3)$. For $N = 1\,000$ timesteps and $M = 10\,000$ Monte Carlo paths this is the dominant bottleneck.

### The option

**Arithmetic Asian Call Option**: the payoff at maturity is

$$V = \max\!\left(\frac{1}{N}\sum_{i=1}^{N} S_i - K,\; 0\right)$$

The averaging over the full price path $S_1, \ldots, S_N$ makes this path-dependent and eliminates any closed-form pricing formula. Monte Carlo simulation is the standard method.

---

## Three Algorithms

### Algorithm 1 — Dense Cholesky &nbsp; $O(N^3)$ factorisation, $O(N^2)$ per path

**File:** `src/cholesky/cholesky.hpp`

Build the full $N \times N$ fBM covariance matrix $C$ and compute its Cholesky factor $L = \operatorname{chol}(C)$ once via Eigen's `LLT`. For each Monte Carlo path, draw $z \sim \mathcal{N}(0, I_N)$ and form

$$\log \sigma = \nu L z$$

The per-path matrix-vector multiply $Lz$ costs $O(N^2)$, so total cost scales as $O(N^3 + M N^2)$.

In practice for $M = 10\,000$ the per-path $O(MN^2)$ term dominates for $N < 10\,000$, giving an observed scaling exponent of $\approx 1.54$ rather than $3$.

### Algorithm 2 — Circulant Embedding + FFT &nbsp; $O(N \log N)$ per path

**File:** `src/fft/fft.hpp`

Based on the **Davies–Harte / Wood–Chan** exact method. The key insight is that fBM **increments** (fractional Gaussian noise, fGn) *are* stationary, even though fBM itself is not — so the increment covariance matrix is Toeplitz. A Toeplitz matrix embeds into a larger circulant matrix whose eigenvalues are the DFT of its first row, enabling exact sampling via FFT.

**fGn autocovariance** at lag $k$:

$$\gamma(k) = \frac{\mathrm{d}t^{2H}}{2}\!\left(|k+1|^{2H} + |k-1|^{2H} - 2|k|^{2H}\right)$$

**Embedding:** form the $2N$-periodic circulant first row

$$c = \bigl[\gamma(0),\, \gamma(1),\, \ldots,\, \gamma(N-1),\, 0,\, \gamma(N-1),\, \ldots,\, \gamma(1)\bigr]$$

**Eigenvalues:** $\lambda = \operatorname{FFT}(c)$ — all $\lambda_j \geq 0$ for $H \leq \tfrac{1}{2}$ (Wood & Chan 1994).

**Per path:** draw complex white noise $w_j$, scale by $\sqrt{\lambda_j / M}$, apply IFFT, take real part → fGn increments. Cumsum gives the fBM log-vol path.

> **Critical:** embedding the *fBM* covariance directly fails because fBM is non-stationary; only the *increments* have a Toeplitz covariance that embeds into a PSD circulant. Using the fBM covariance produces negative eigenvalues and a runtime crash.

FFTW plans are created once outside the Monte Carlo loop and reused for all paths.

### Algorithm 3 — H-matrix + Randomized SVD &nbsp; $O(Nk^2)$ construction, $O(Nk)$ per path

**Files:** `src/hmatrix/hmatrix.hpp`, `src/hmatrix/rsvd.hpp`

Motivated by two observations:

1. The fBM covariance kernel $C(t,s)$ is smooth away from the diagonal $t \approx s$; off-diagonal (far-field) blocks are approximately low-rank. The theoretical basis comes from [*A Fast Butterfly Algorithm for Fourier Integral Operators*](papers-bg/fast-butterfly-fft.pdf) (Candès, Demanet & Ying, 2008), which proves that smooth kernels restricted to well-separated dyadic boxes have numerically low rank.

2. Randomized SVD can exploit this efficiently: Halko, Martinsson & Tropp (2011) — [*Finding Structure with Randomness*](papers-bg/2010_HMT_random_review.pdf) — prove that a rank-$k$ sketch with $q$ power iterations recovers the dominant singular subspace of any matrix in $O(mn \log k)$ operations with high probability.

**Rank-$k$ approximation:**

$$C \approx U \operatorname{diag}(S) U^\top \qquad (U \in \mathbb{R}^{N \times k},\; S_i > 0)$$

The approximate Cholesky factor $L_k = U \operatorname{diag}(\sqrt{S})$ satisfies $L_k L_k^\top \approx C$, so per-path cost drops from $O(N^2)$ to $O(Nk)$:

$$\log \sigma \approx \nu L_k z_k, \qquad z_k \sim \mathcal{N}(0, I_k)$$

**rSVD algorithm** (`rsvd.hpp`, Algorithm 4.4 from Halko et al.):

1. Draw Gaussian sketch $\Omega \in \mathbb{R}^{N \times (k+p)}$ with oversampling $p = 5$
2. Form $Y = C\Omega$; apply $q = 2$ power iterations: $Y \leftarrow (CC^\top)^q C\Omega$
3. QR: $Y = QR$
4. Project: $B = Q^\top C$; compute thin SVD of the small $(k{+}p) \times N$ matrix $B$

Power iteration is critical: with $H = 0.1$ the singular values of $C$ decay slowly, so without it the sketch quality degrades significantly.

---

## Computational Results

Benchmarked at $M = 10\,000$ paths, $N \in \{252, 500, 1000\}$:

| Method | Fitted $t \approx c \cdot N^\alpha$ | $\alpha$ | $c$ | $R^2$ |
|---|---|---|---|---|
| Dense Cholesky | $O(MN^2 + N^3)$ | 1.54 | $4.0 \times 10^{-5}$ | 0.999 |
| Circulant+FFT | $O(N \log N)$ | 1.03 | $9.2 \times 10^{-4}$ | 1.000 |
| H-matrix+rSVD ($k=32$) | $O(Nk)$ per path | 1.06 | $2.9 \times 10^{-4}$ | 1.000 |

FFT and H-matrix both scale as $O(N)$ in this regime — the $\log N$ factor is invisible across a $4\times$ range of $N$. Cholesky's exponent of 1.54 reflects the mixed $O(MN^2 + N^3)$ cost where the per-path term dominates for $M = 10\,000$.

**H-matrix approximation quality** (relative Frobenius norm $\|C - C_k\|_F \,/\, \|C\|_F$):

| Rank $k$ | Frobenius error |
|---|---|
| 2 | 8.5% |
| 16 | 2.5% |
| 128 | 1.2% |

Errors decay slowly because the singular value spectrum of $C$ decays slowly for rough $H = 0.1$ — a fundamental property of rough processes. At $H = 0.4$ (closer to Brownian) convergence would be much faster.

**Reference price:** $p_\text{ref} = 23.58$ (average of 500k-path Cholesky and 500k-path FFT runs; $H = 0.10$, $\nu = 0.30$, $S_0 = K = 100$, $T = 1$).

---

## A Fourth Method (not implemented): Hybrid Scheme

Bennedsen, Lunde & Pakkanen (2017) — [*Hybrid scheme for Brownian semistationary processes*](papers-bg/hybrid-scheme-brownian-semistationary-process.pdf) — introduce a simulation scheme better suited to rough kernels. The fBM kernel

$$g(x) = x^{H - 1/2} = x^{-0.4} \qquad (H = 0.1)$$

has a power-law singularity near zero that circulant embedding handles poorly: the first step-function cell misses the spike. The hybrid scheme approximates $g(x)$ as a power function near zero (analytically integrable, exact near the singularity) and step functions elsewhere, while keeping the same $O(N \log N)$ complexity.

For $\alpha = H - \tfrac{1}{2} \approx -0.4$, the hybrid scheme reduces RMSE by more than 80% versus the plain Riemann scheme. Implementing it as Algorithm 4 would provide a more accurate baseline for small $N$ and is the most natural extension of this project.

---

## Validation Against Market Data

Confirming the model against traded prices requires three layers of validation, from easiest to hardest data access.

### Phase 0 — RFSV Calibration (already implemented)

Run `data/calibrate.py` on the [Oxford-Man Realized Library](https://realized.oxford-man.ox.ac.uk/data/download) (free, manual download). The script estimates $H$ via variogram regression

$$\mathbb{E}\!\left[|\log\sigma_{t+h} - \log\sigma_t|^2\right] \;\approx\; c \cdot h^{2H}$$

and $\nu$ as the standard deviation of log-vol increments.

```bash
python data/calibrate.py --rv-col rv5
# paste printed H and nu into src/common/params.hpp, then rebuild
```

### Phase 1 — Validate against vanilla implied-volatility surface (free)

Vanilla (European) options on SPX/SPY are liquid and exchange-traded. The RFSV model should reproduce their implied volatility smile, particularly the characteristic power-law ATM skew:

$$\partial_k \sigma_\text{BS}(k, T) \;\sim\; T^{H - 1/2} \quad \text{as } T \to 0$$

**Implementation plan:**

```python
# data/validate_iv.py
import yfinance as yf
from py_vollib.black_scholes.implied_volatility import implied_volatility

# 1. Fetch SPY option chains for several expiries
ticker = yf.Ticker("SPY")
S = ticker.history(period="1d")["Close"].iloc[-1]
r = 0.05  # risk-free rate (current Fed funds)

results = []
for expiry in ["2026-06-20", "2026-09-19", "2026-12-18"]:
    chain = ticker.option_chain(expiry).calls
    T = (pd.Timestamp(expiry) - pd.Timestamp.today()).days / 365
    chain["market_iv"] = chain.apply(
        lambda row: implied_volatility(row["lastPrice"], S, row["strike"], T, r, "c"),
        axis=1
    )
    # 2. Price same contracts with RFSV Monte Carlo
    chain["model_price"] = chain["strike"].apply(
        lambda K: rfsv_european_price(S, K, T, H, nu, M=50_000)
    )
    chain["model_iv"] = chain.apply(
        lambda row: implied_volatility(row["model_price"], S, row["strike"], T, r, "c"),
        axis=1
    )
    results.append(chain[["strike", "market_iv", "model_iv"]])

# 3. Plot IV smile: model vs market
```

**Dependencies to add:** `uv add yfinance py-vollib`

**Expected output:** Two plots per expiry — market IV smile (curved) vs RFSV model IV smile. The RFSV model is known to produce a steep ATM skew that matches SPX options better than classical Heston for short maturities.

### Phase 2 — Compare to traded Asian option prices

**Exchange-traded Asian options:** CME's **Average Price Options (APO)** on WTI crude oil (ticker `CL`) settle on the arithmetic average of daily front-month futures over the contract month — exactly the payoff we price.

**Data access options (cheapest first):**

| Source | Cost | Method |
|---|---|---|
| CME daily settlement prices | Free (delayed) | Manual download from [cmegroup.com](https://www.cmegroup.com/markets/energy/crude-oil/light-sweet-crude.html) |
| CME DataMine | ~$300–500/month | REST API, historical tick + settlement |
| Quandl/Nasdaq Data Link | ~$50/month | `CHRIS/CME_CL1` series (spot only); APO separate |
| Academic data request | Free | CME academic data program for universities |

**Implementation plan:**

```python
# data/validate_asian.py

# 1. Load CME WTI APO settlement prices (manual CSV download)
#    Columns: date, expiry, strike, option_type, settlement_price, underlying_avg_price
apo_df = pd.read_csv("data/raw/wti_apo_settlements.csv", parse_dates=["date"])

# 2. Calibrate RFSV to WTI realized variance (not equity — use energy RV)
#    Oxford-Man has WTI realized variance in some datasets
#    Alternative: compute from daily WTI futures prices via yfinance
wti = yf.download("CL=F", start="2020-01-01", end="2025-12-31")
log_returns = np.log(wti["Close"]).diff().dropna()
# estimate H, nu from log-returns (approximate; ideally use intraday data)

# 3. For each option in the dataset, compute RFSV model price
for _, row in apo_df.iterrows():
    model_price = price_asian_call(
        S0=row["underlying_price"],
        K=row["strike"],
        T=row["days_to_expiry"] / 252,
        H=H_wti, nu=nu_wti, M=50_000
    )
    row["model_price"] = model_price
    row["price_error"] = row["settlement_price"] - model_price

# 4. Report: mean absolute error, relative error, by moneyness bucket
```

**Alternative (no market data):** Benchmark against the **Turnbull-Wakeman approximation** — a closed-form formula for arithmetic Asian calls under log-normal dynamics. Use it as a sanity check: at $H = 0.5$ the RFSV model reduces to standard GBM, so the two should agree.

### Phase 3 — Sensitivity analysis

Vary $H \in \{0.05, 0.10, 0.15, 0.20\}$ and $\nu \in \{0.1, 0.2, 0.3, 0.4\}$, compute model prices for a grid of strikes/maturities, and produce a surface plot. This answers: *how sensitive is the Asian option price to the roughness parameter?*

```python
# plots/plot_sensitivity.py
H_grid  = [0.05, 0.10, 0.15, 0.20]
nu_grid = [0.10, 0.20, 0.30, 0.40]
K_grid  = [90, 95, 100, 105, 110]

# Run ./build/benchmark with modified params, or call Python FFI if wrapped
prices = {(H, nu, K): price_fft(H, nu, K) for H in H_grid for nu in nu_grid for K in K_grid}
```

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
| yfinance | SPX/SPY option chains for IV validation | `uv add yfinance` |
| py-vollib | Black-Scholes implied volatility inversion | `uv add py-vollib` |

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
| Gatheral, Jaisson & Rosenbaum (2014). *Volatility is rough.* arXiv:1410.3394 | Empirical basis for $H \approx 0.1$; RFSV model definition; calibration method |
| Halko, Martinsson & Tropp (2011). *Finding structure with randomness.* SIAM Review 53(2) | Algorithm 4.4 (rSVD with power iteration) implemented in `rsvd.hpp` |
| Candès, Demanet & Ying (2008). *A fast butterfly algorithm for Fourier integral operators.* arXiv:0809.0719 | Theoretical basis for low-rank structure of smooth off-diagonal kernel blocks |
| Bennedsen, Lunde & Pakkanen (2017). *Hybrid scheme for Brownian semistationary processes.* arXiv:1507.03004 | Better simulation scheme for rough kernels ($H < \tfrac{1}{2}$); identified as future work |
