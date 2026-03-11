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
uv run python plots/plot_sensitivity.py [--M 10000] [--N 252]
# → plots/sensitivity_surface.png, plots/sensitivity_strike.png

# Structural analysis: Toeplitz property + off-diagonal SVD decay
uv run python plots/plot_structure.py [--N-small 64] [--N-large 128]
# → plots/structure_analysis.png

# MC convergence study: price ± 1σ vs M paths
uv run python data/validate_convergence.py [--n-seeds 5] [--max-M 25000]
# → plots/convergence.png

# Construction vs MC time breakdown (requires benchmark CSV with new columns)
uv run python plots/plot_scaling.py
# → plots/time_vs_N.png, plots/error_vs_rank.png, plots/construction_breakdown.png
```

## Architecture

```
final-project/
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

## Project Evaluation Criteria

This is a final project for a fast-algorithms course. Every experiment should be framed with explicit control/independent/dependent variables. The rubric (`include.md`) requires:

1. **Runtime efficiency** — scaling benchmarks with fitted exponents (already done in `plot_scaling.py`)
2. **Memory use** — peak RSS vs N for all three C++ methods + Python engine (`benchmarks/memory_benchmark.cpp` + `data/profile_memory.py`, planned)
3. **Accuracy** — MC convergence: price ± 1σ vs M paths, confirm σ ∝ 1/√M log-log slope ≈ −0.5 (`data/validate_convergence.py`, planned)
4. **Stability** — FFT eigenvalue clipping near H=0.5, rSVD condition number vs rank, Cholesky κ(C) vs N (`data/validate_stability.py`, planned)
5. **Structural analysis** — *why* each method works: Toeplitz structure of fGn (→ FFT), low-rank off-diagonal structure of C(t,s) (→ H-matrix), singular value decay comparison H=0.1 vs H=0.5 (`plots/plot_structure.py`, planned)

## Planned Scripts (not yet implemented)

See `TODO.md` for full task descriptions and priority ordering. Summary of new files to create:

| Script | Status | Purpose |
|---|---|---|
| `data/validate_convergence.py` | ✅ done | Price ± 1σ vs M ∈ {100..25k}, 5 seeds → `plots/convergence.png` |
| `plots/plot_structure.py` | ✅ done | fGn Toeplitz heatmap + off-diagonal SVD decay → `plots/structure_analysis.png` |
| `plots/construction_breakdown.png` | ✅ done | Stacked bar: setup vs MC time (via `price_timed()` in each .hpp) |
| `data/validate_stability.py` | pending | FFT clipping vs H, rSVD conditioning vs rank, Cholesky κ(C) vs N |
| `benchmarks/memory_benchmark.cpp` | pending | `getrusage` peak RSS added to `time_vs_N.csv` |
| `data/profile_memory.py` | pending | `tracemalloc` peak allocation vs (N, M) for Python engine |

**Production-quality run parameters** (use for final report plots):
- `validate_asian.py`: `--M 10000 --N 252`, 5 seeds, mean ± std error bars
- `validate_iv.py`: `--M 3000 --N 63`, calibration M=1000
- `plot_sensitivity.py`: `--M 10000 --N 252` (noise floor ≈ 0.3)
- `validate_convergence.py`: M up to 25000

## Documentation Formatting

All mathematical content in `.md` files must use LaTeX, not Unicode approximations:

- Use `$O(N^2)$`, not `O(N²)` or `O(N^2)` in plain text
- Use `$\times$`, `$\cdot$`, not `×`, `·`
- Use `$\alpha$`, `$\sigma$`, `$\kappa$`, `$\gamma$`, not `α`, `σ`, `κ`, `γ`
- Use `$\pm$`, `$\approx$`, `$\propto$`, `$\leq$`, not `±`, `≈`, `∝`, `≤`
- Use `$\frac{1}{2}$` or `$\tfrac{1}{2}$`, not `½` or `\frac 1 2` (unbraced)
- Use `$L^\top$`, not `Lᵀ`
- Use `$\sqrt{M}$`, not `√M`
- Use `$\sum_{i}$`, not `Σ_i`
- Complexity annotations in tables and prose: always wrap in `$...$`
- Backtick code spans are for actual C++ identifiers and code output only — mathematical formulas that appear in backticks should be converted to inline LaTeX

## Commit Convention

Use `type: short description` (one line, no period). Common types:

- `feat` — new script, function, or C++ feature
- `fix` — bug fix
- `docs` — README, CLAUDE.md, comments, ALGORITHMS.md
- `refactor` — restructuring without behavior change
- `bench` — benchmark runner or CSV output changes
- `style` — formatting, plot aesthetics
