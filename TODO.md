# Further Work

Ideas for extending this project beyond its current scope.

---

## Simulation accuracy

**Hybrid scheme (Bennedsen, Lunde & Pakkanen 2017)**
The fBM kernel $g(x) = x^{H-1/2}$ has a power-law singularity near zero that circulant embedding misses — the first time step approximates a spike with a constant. The hybrid scheme handles the singularity analytically and approximates the rest with step functions, keeping $O(N \log N)$ complexity while reducing RMSE by >80% for $H = 0.1$. This is the most impactful accuracy improvement available.

**Quasi-Monte Carlo (QMC)**
Replace pseudo-random $z \sim \mathcal{N}(0, I_N)$ with Sobol sequences. QMC improves convergence from $O(1/\sqrt{M})$ to $O((\log M)^d / M)$ in smooth integrands. The Asian payoff is not smooth (the $\max$ kink), but randomized QMC (RQMC) still typically halves the effective $M$ needed for a given accuracy.

**Variance reduction**
The geometric Asian option has a closed-form price (Kemna & Vorst 1990) and is highly correlated with the arithmetic Asian. Using it as a control variate can reduce $\sigma_V$ by 10–50×, making the MC noise floor much lower without increasing $M$.

---

## Algorithms

**True H-matrix with recursive block-tree**
The current "H-matrix" is a global rank-$k$ rSVD — a useful approximation but not a true hierarchical matrix. A proper recursive $\mathcal{H}$-matrix with admissibility conditions and $\mathcal{H}$-Cholesky factorization would achieve $O(N \log N)$ storage and $O(N \log^2 N)$ factorization, replacing the $O(N^2)$ construction cost. See Hackbusch (1999).

**Resolve the $\log N$ factor empirically**
The benchmark only covers $N \in \{252, 500, 1000\}$ — a $4\times$ range where $\log N$ is indistinguishable from a constant. Extending to $N \in \{64, 128, \ldots, 16384\}$ would cleanly separate $O(N)$ from $O(N \log N)$ on a log-log plot and stress-test the rSVD rank requirements at large $N$.

---

## Performance

**GPU acceleration**
The Cholesky MC loop is memory-bandwidth limited (~47 GB/s on M2, ~50% utilization). A GPU with HBM (e.g. H100: ~3 TB/s) would reduce Cholesky MC time by ~30× on memory bandwidth alone. The FFT pricer maps directly to cuFFT. Path-level parallelism is embarrassingly parallel.

**OpenMP path parallelism**
Each of the $M$ MC paths is independent. Adding `#pragma omp parallel for reduction(+:price_sum)` to the MC loop in all three pricers would give near-linear speedup on multi-core hardware at zero algorithmic cost.

---

## Model extensions

**Mean reversion**
The current model $\log \sigma_t = \nu W_t^H$ has no mean reversion — volatility drifts without bound over long horizons. The rough Bergomi model adds a mean-reverting drift: $\log \sigma_t = \xi_0(t) + \nu \int_0^t (t-s)^{H-1/2} dW_s$, which is more realistic for long-dated options.

**Calibration to the full IV surface**
Currently $H$ and $\nu$ are estimated from realized volatility time series. A richer calibration would jointly fit the entire implied volatility surface (smile + term structure) by minimizing $\sum_{K,T} (\sigma^{\text{RFSV}}_{K,T} - \sigma^{\text{market}}_{K,T})^2$, requiring the pricer to be called in a loop over $(K, T)$ pairs.

**Path-wise Greeks**
Delta ($\partial p / \partial S_0$), Vega ($\partial p / \partial \nu$), and sensitivity to $H$ are needed for hedging. Under fBM simulation these require either finite-difference bump-and-reprice (doubling the MC cost per Greek) or Malliavin calculus weights (complex but unbiased).

---

## Diagnostics

**Long-memory regime ($H > 0.5$)**
All experiments use $H = 0.1$ (rough/anti-persistent). Testing at $H = 0.7$–$0.9$ would exercise the long-memory regime where increments are positively correlated and volatility clustering is stronger. The circulant embedding is guaranteed PSD for all $H \in (0,1)$ in the large-$N$ limit, but the rSVD rank requirements change significantly.

**Conditioning at large $N$**
The stability analysis shows $\kappa(C) \sim c \cdot N^\alpha$ with $\alpha \approx 1.5$. At $N = 10{,}000$ this implies $\kappa \sim 10^6$, suggesting Cholesky may need regularization ($C \leftarrow C + \epsilon I$) to remain numerically stable. Quantifying the threshold $N$ is practically useful.
