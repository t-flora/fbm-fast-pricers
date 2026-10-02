# Context Document: High-Performance Monte Carlo Pricing under Rough Volatility (18-Hour Sprint)

## 1. Project Objective & Theoretical Foundation
**Goal:** Build a highly performant C++ Monte Carlo pricer for an Arithmetic Asian Call Option under the Rough Fractional Stochastic Volatility (RFSV) model, benchmarking three linear algebraic methods for their time-efficiency and accuracy.

**The Theory:**
* Empirical analysis of high-frequency data demonstrates that log-volatility behaves essentially as a fractional Brownian motion (fBM) with a Hurst exponent of order $H \approx 0.1$.
* The RFSV model defines log-volatility using fBM with $H < 1/2$. This implies volatility is microscopically anti-persistent and highly "rough," contradicting historical assumptions of long memory.
* Because fBM is non-Markovian, exact path generation requires dealing with a dense $N \times N$ covariance matrix, creating a massive computational bottleneck for Monte Carlo pricing.

## 2. The 18-Hour / 8-Day Execution Plan

### Block 1: Data Extraction & Calibration (2 Hours)
* **Dataset:** Oxford-Man Institute of Quantitative Finance Realized Library (contains daily non-parametric estimates of volatility). URL: `https://realized.oxford-man.ox.ac.uk/data/download`
* **Task:** Ingest the raw CSV data using Python (Pandas/NumPy) to calculate log-volatility increments.
* **Output:** Run a fractional linear regression to extract the Hurst exponent ($H$) and volatility of volatility ($\nu$). Hardcode these calibrated parameters directly into the C++ engine to save time.

### Block 2: Baseline C++ Engine & Asian Payoff (4 Hours)
* **Task:** Initialize the high-performance C++ environment and integrate the Eigen library.
* **Algorithm 1 (Baseline):** Exact Dense Cholesky Decomposition. Construct the dense $N \times N$ covariance matrix and compute $L L^T$.
* **Complexity:** $O(N^3)$ factorization, $O(N^2)$ path generation.
* **Pricing:** Implement the Monte Carlo loop for a path-dependent Arithmetic Asian Call Option payoff: $\max(\frac{1}{N}\sum P_i - K, 0)$.

### Block 3: Circulant Embedding / FFT (4 Hours)
* **Task:** Implement the exact fast generation method.
* **Algorithm 2:** The fBM covariance matrix is Toeplitz. Embed this into a $2N \times 2N$ circulant matrix. Use the FFTW C++ library to perform the Discrete Fourier Transform, apply the square root of the eigenvalues to complex normal noise, and execute an inverse FFT.
* **Complexity:** $O(N \log N)$.

### Block 4: Hierarchical Matrices ($\mathcal{H}$-matrices) & rSVD (6 Hours)
* **Task:** Implement the algorithmic centerpiece using concepts from the Fast Multipole Method (FMM) and randomized linear algebra.
* **Algorithm 3:** * Recursively partition the covariance matrix into a quad-tree structure.
    * Keep the near-field diagonal blocks (containing the singularity) dense.
    * Apply the Randomized SVD (rSVD) algorithm to compress the smooth far-field off-diagonal blocks into a low-rank format.
* **Complexity:** $O(N \log^2 N)$ approximation.

### Block 5: Benchmarking & Visualization (2 Hours)
* **Task:** Profile the C++ code using `std::chrono` across varying path resolutions ($N = 252, 500, 1000$).
* **Output:** Export results to CSV. Write a Python script (Matplotlib/Seaborn) to generate scaling graphs:
    1. Wall-clock Time vs. Path Resolution ($N$).
    2. L2 Pricing Error vs. rSVD Target Rank ($k$).

---

## 3. References
1. Gatheral, J., Jaisson, T., & Rosenbaum, M. (2014). *Volatility is rough*. arXiv:1410.3394 [q-fin.ST].
2. Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions*. SIAM Review, 53(2), 217-288.
