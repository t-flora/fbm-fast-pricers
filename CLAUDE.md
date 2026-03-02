# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance C++ Monte Carlo pricer for an **Arithmetic Asian Call Option** under the **Rough Fractional Stochastic Volatility (RFSV)** model. Benchmarks three methods for generating fractional Brownian motion (fBM) paths: Dense Cholesky (O(N³)), Circulant Embedding + FFTW (O(N log N)), and H-matrix + Randomized SVD (O(N·k) per path). See README.md for the full mathematical background.

## Build Commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Individual pricers
./build/cholesky_pricer
./build/fft_pricer
./build/hmatrix_pricer

# Full benchmark (writes CSVs to benchmarks/results/)
./build/benchmark
```

## Data & Calibration

```bash
# Option A: Place Oxford-Man CSV in data/raw/ then run:
uv run python data/calibrate.py
# → prints H and nu; paste into src/common/params.hpp

# Option B: Free alternative using yfinance (proxy RV from 5-day squared returns)
uv run python data/calibrate.py --source yfinance
# → H ≈ 0.12 (close to Oxford-Man 0.10), R² ≈ 0.98
# Oxford-Man: https://realized.oxford-man.ox.ac.uk/data/download
```

## Visualization & Validation

```bash
# Scaling benchmarks (requires ./build/benchmark to have been run)
uv run python plots/plot_scaling.py
# → plots/time_vs_N.png, plots/error_vs_rank.png

# Phase 1: IV smile vs live SPY option chains
uv run python data/validate_iv.py [--M 3000] [--N 63]
# → plots/validate_iv.png

# Phase 2: Levy benchmark + roughness premium vs strike
uv run python data/validate_asian.py [--M 5000] [--N 252]
# → plots/validate_asian.png

# Phase 3: H × nu sensitivity heatmap + price vs K curves
uv run python plots/plot_sensitivity.py [--M 3000] [--N 252]
# → plots/sensitivity_surface.png, plots/sensitivity_strike.png
```

## Architecture

```
final-project/
├── papers-bg/            Source papers (see references in README)
├── data/
│   ├── raw/              Oxford-Man CSV downloaded here
│   ├── calibrate.py      Estimates H and nu (Oxford-Man or --source yfinance)
│   ├── rfsv_model.py     Vectorized numpy RFSV Monte Carlo engine (all phases)
│   ├── validate_iv.py    Phase 1: SPY IV smile vs RFSV European calls
│   └── validate_asian.py Phase 2: Levy benchmark + roughness premium
├── src/
│   ├── common/
│   │   ├── params.hpp        Hardcoded H, nu, K, T, S0 (update after calibration)
│   │   ├── covariance.hpp    fBM kernel + Eigen matrix builder
│   │   ├── asian_payoff.hpp  Payoff + log-vol → price path
│   │   └── rng.hpp           Seeded RNG helpers
│   ├── cholesky/
│   │   ├── cholesky.hpp      price() in namespace cholesky:: (anonymous inner ns)
│   │   └── cholesky_pricer.cpp  main()
│   ├── fft/
│   │   ├── fft.hpp           price() in namespace fft_pricer:: (anonymous inner ns)
│   │   └── fft_pricer.cpp    main()
│   └── hmatrix/
│       ├── hmatrix.hpp       price() in namespace hmatrix:: (anonymous inner ns)
│       ├── hmatrix_pricer.cpp  main()
│       └── rsvd.hpp          Halko et al. Algorithm 4.4
├── benchmarks/
│   ├── benchmark.cpp     Calls all three price() functions; exports two CSVs
│   └── results/          time_vs_N.csv, error_vs_rank.csv, reference_price.txt
├── plots/
│   ├── plot_scaling.py       Two-panel time scaling + fitted exponents; error vs rank
│   └── plot_sensitivity.py   Phase 3: H×nu ATM heatmap + price vs K curves
└── CMakeLists.txt
```

## Key Design Decisions

**Header-only pricers with anonymous namespaces.** Each algorithm lives entirely in a `.hpp` with an inner anonymous namespace. This lets `benchmark.cpp` include all three headers in one translation unit without ODR violations, while each `_pricer.cpp` also includes its header independently. The pattern is:
```cpp
namespace cholesky { namespace { double price(...) { ... } } }
```

**Calibrated parameters hardcoded in `params.hpp`.** H and nu come from running `data/calibrate.py` once on Oxford-Man data. Keeping them as `constexpr` avoids runtime I/O in the hot path and makes the C++ engine self-contained.

**fGn (not fBM) for the FFT embedding.** fBM itself is non-stationary — its covariance matrix is NOT Toeplitz. Only the *increments* (fractional Gaussian noise, fGn) are stationary, giving a Toeplitz covariance that embeds into a PSD circulant for H ≤ 0.5 (Wood & Chan 1994). The FFT pricer uses fGn autocovariance `γ(k) = (dt^{2H}/2)·((k+1)^{2H} + (k-1)^{2H} − 2k^{2H})`, then cumsums to get fBM. Using fBM covariance directly gives negative eigenvalues and crashes.

**rSVD with power iteration for slow spectra.** With H = 0.1 the fBM covariance matrix has slowly-decaying singular values (rough spectrum). `rsvd.hpp` uses `q = 2` power iterations — `Y ← (C·C^T)^q · C · Ω` before QR — to improve the sketch quality, as prescribed by Halko et al. Without power iteration the approximation degrades badly for small rank.

**H-matrix via global rSVD rather than block-tree.** A full recursive H-matrix with an H-Cholesky factorization is complex to implement correctly. Instead we use a global rank-k rSVD: `C ≈ U·S·U^T`, then `L_k = U·diag(√S)`, giving `L_k·L_k^T ≈ C`. Per-path cost drops from O(N²) to O(N·k). The accuracy-vs-speed tradeoff (controlled by k) is the point of the `error_vs_rank.csv` benchmark.

**Dual accuracy metrics for error_vs_rank.** Monte Carlo noise (~0.4 with M=10k) swamps the truncation error at high ranks. We therefore report both: (1) Frobenius norm `‖C − C_k‖_F / ‖C‖_F` (exact, noise-free), and (2) price error vs a 500k-path reference (average of Cholesky and FFT, neither privileged).

**FFTW plan reuse.** The FFT pricer creates the c2c forward plan (for eigenvalue computation) and the c2c backward plan (for per-path synthesis) once, then calls `fftw_execute` in the MC loop.

**Reference price is the average of two exact methods.** Both Cholesky (dense LL^T) and FFT (Davies-Harte circulant embedding) are exact fBM simulators — prices from both methods converge to the same value with sufficient paths. The benchmark uses their average over 500k paths each as the "ground truth" for the error-vs-rank experiment, avoiding any bias toward either method.

**Python RFSV engine matches C++ FFTW convention exactly.** `data/rfsv_model.py` reimplements the circulant-FFT path generator in pure numpy for use in the validation scripts. The normalization must match FFTW's unnormalized backward transform: `sqrt_lam = sqrt(max(λ,0) / (2N))` then `increments = Re(np.fft.ifft(W) * 2N)[:N]`. The `* 2N` undoes numpy's automatic `1/(2N)` normalization. This exact match is verified by comparing the Python ATM price (~23.56) to the C++ benchmark (~23.58).

**Log-vol drift `μ₀` for IV validation.** The RFSV model as implemented uses `σ_t = exp(ν W_t^H)`, giving `σ_0 = 1.0` (100% annualized vol). For comparing IV *smiles* against SPY (σ ≈ 15%), `validate_iv.py` auto-calibrates a log-vol drift `μ₀ = log(σ_target)` by matching the ATM RFSV price to the market ATM option price. This decouples vol-of-vol level from smile curvature.

**Oxford-Man alternative calibration.** The Oxford-Man Realized Library CSV is needed for accurate H estimation. Without it, `calibrate.py --source yfinance` uses non-overlapping 5-day window realized variance from squared daily returns. This gives H ≈ 0.12 with R² ≈ 0.98 — close to the Oxford-Man value of H ≈ 0.10. Single-day RV or rolling-window RV are both unreliable (too noisy or over-smoothed).

## Dependencies

- **Eigen3** — dense matrix ops, Cholesky (`brew install eigen`)
- **FFTW3** — fast Fourier transforms (`brew install fftw`)
- C++17, CMake 3.16+
- Python (via `uv`): pandas, numpy, scipy, matplotlib, seaborn, yfinance, pdfminer.six

## Benchmarking Notes

`benchmarks/results/time_vs_N.csv` columns: `method, N, M_paths, wall_time_s, price`
`benchmarks/results/error_vs_rank.csv` columns: `rank_k, N, reference_price, hmatrix_price, abs_price_error, rel_price_error, frob_error, construction_time_s, mc_time_s`
`benchmarks/results/reference_price.txt` — documents reference price inputs

Fitted complexity constants (log-log regression over N = {252, 500, 1000}):
- Cholesky: `t = 4.0e-5 · N^1.54` (dominated by O(M·N²) per-path cost, not O(N³) factorization)
- FFT: `t = 9.2e-4 · N^1.03`
- H-matrix k=32: `t = 2.9e-4 · N^1.06`

## Background Papers

All in `papers-bg/`:
- `vol-is-rough.pdf` — Gatheral, Jaisson & Rosenbaum (2014): empirical H ≈ 0.1, RFSV model definition
- `2010_HMT_random_review.pdf` — Halko, Martinsson & Tropp (2011): rSVD Algorithm 4.4 implemented in `rsvd.hpp`
- `fast-butterfly-fft.pdf` — Candès, Demanet & Ying (2008): theoretical basis for low-rank off-diagonal blocks in smooth kernel matrices (motivates H-matrix structure)
- `hybrid-scheme-brownian-semistationary-process.pdf` — Bennedsen, Lunde & Pakkanen (2017): hybrid scheme for BSS processes — better simulation accuracy for rough kernels (H < 0.5), identified as future work
