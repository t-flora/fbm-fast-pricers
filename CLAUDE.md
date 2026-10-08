# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

High-performance C++ Monte Carlo pricer for an **Arithmetic Asian Call Option** under the **Rough Fractional Stochastic Volatility (RFSV)** model. Benchmarks three methods for generating fractional Brownian motion (fBM) paths: Dense Cholesky (O(N³)), Circulant Embedding + FFTW (O(N log N)), and global low-rank randomized SVD (O(N·k) per path; not a true H-matrix). See README.md for the full mathematical background.

## Build Commands

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Individual pricers
./build/cholesky_pricer
./build/fft_pricer
./build/rsvd_pricer

# Full benchmark (writes CSVs to benchmarks/results/; ~12 min, median of 5 repeats)
./build/benchmark            # --quick: N <= 500, one repeat (~20 s; smoke tests only)

# Tests — run after any change to a sampler, the Python engine, or calibrate.py
./build/test_samplers      # or: ctest --test-dir build
uv run pytest tests/       # CI runs both on pushes to main and on PRs (.github/workflows/)

# Extensions (variance-corrected low-rank, control variate); needs reference_price.txt
./build/extensions           # --quick available
uv run python plots/plot_extensions.py

# Performance: batched sampling, fast Gaussian generator, thread scaling (~10 min)
./build/performance          # --quick available
uv run python plots/plot_performance.py

# Whole pipeline; --fast runs everything at small sizes in ~2 min (CI smoke test)
./run_pipeline.sh [--fast] [--no-iv] [--no-build]
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
# → plots/figures/time_vs_N.png, plots/figures/error_vs_rank.png

# Phase 1: IV smile vs live SPY option chains
uv run python data/validate_iv.py [--M 20000] [--N 63]
# → plots/figures/validate_iv.png

# Phase 2: Levy benchmark + roughness premium vs strike
uv run python data/validate_asian.py [--M 10000] [--N 252] [--n-seeds 10]
# → plots/figures/validate_asian.png

# Phase 3: H × nu sensitivity heatmap + price vs K curves
uv run python plots/plot_sensitivity.py [--M 10000] [--N 252]
# → plots/figures/sensitivity_surface.png, plots/figures/sensitivity_strike.png

# Structural analysis: Toeplitz property + off-diagonal SVD decay
uv run python plots/plot_structure.py [--N-small 64] [--N-large 128]
# → plots/figures/structure_analysis.png

# MC convergence study: price ± 1σ vs M paths, plain and control variate
uv run python data/validate_convergence.py [--n-seeds 20] [--max-M 25000]
# → plots/figures/convergence.png

# Greeks: vega in nu (pathwise vs central difference) and dV/dH, with the control variate
uv run python data/greeks.py [--M 20000] [--N 252]
# → plots/figures/greeks.png

# Construction vs MC time breakdown (requires benchmark CSV with new columns)
uv run python plots/plot_scaling.py
# → plots/figures/time_vs_N.png, plots/figures/error_vs_rank.png, plots/figures/construction_breakdown.png
```

## Architecture

```
final-project/
├── src/
│   ├── common/
│   │   ├── params.hpp           H, nu, sigma0, S0, K, T, r (kept in sync with data/params.py)
│   │   ├── covariance.hpp       fBM kernel + Eigen matrix builder
│   │   ├── asian_payoff.hpp     log-vol path → price path → payoff; asian_sample() for the CV
│   │   ├── control_variate.hpp  mc_control_variate(): conditional geometric CV driver
│   │   ├── batched_mc.hpp       batched::run(): blocked, multithreaded MC driver (Config, Result)
│   │   ├── fast_rng.hpp         xoshiro256++ + 256-layer ziggurat (FastNormal)
│   │   └── rng.hpp              Seeded RNG helpers
│   ├── cholesky/  cholesky.hpp (fbm_cholesky_factor, price, price_timed, price_cv,
│   │              BatchWorker, price_batched), cholesky_pricer.cpp
│   ├── fft/       fft.hpp (circulant_eigenvalues, FbmSampler, price, price_timed, price_cv,
│   │              circulant_scale, BatchWorker, price_batched), fft_pricer.cpp
│   └── rsvd/      rsvd.hpp (Halko et al. Alg. 4.4), lowrank.hpp (lowrank_factor, residual_sd,
│                  price, price_corrected, price_freed_timed, price_cv, BatchWorker,
│                  price_batched), rsvd_pricer.cpp
├── benchmarks/
│   ├── benchmark.cpp     main benchmark → results/{time_vs_N,error_vs_rank}.csv, reference_price.txt
│   ├── extensions.cpp    variance correction + CV → results/{variance_corrected_rank,control_variate}.csv
│   ├── performance.cpp   batching, generator, threads → results/perf_{breakdown,batched,batch_size,threads}.csv
│   └── timing.hpp        repeat_timed() (median + IQR), has_flag() for --quick
├── data/
│   ├── params.py         Python mirror of params.hpp (pytest checks they agree)
│   ├── rfsv_model.py     Vectorized numpy RFSV engine (FFT and low-rank samplers, CV pricer,
│   │                     BS/Lévy helpers, fbm_cov_matrix, rsvd)
│   ├── calibrate.py      Variogram fit of H, nu (Oxford-Man CSV or --source yfinance)
│   ├── validate_*.py     convergence, stability, asian (Lévy + premium), iv (live SPY)
│   ├── greeks.py         vega (pathwise vs FD) and dV/dH
│   └── profile_memory.py tracemalloc profile of the numpy engine
├── plots/                plot_scaling, plot_structure, plot_sensitivity, plot_extensions,
│                         plot_performance;
│                         figures/ (generated, gitignored); *.png committed snapshots
├── tests/                test_samplers.cpp (ctest), test_python.py (pytest)
├── experiments/          prototypes for roadmap items (not in the pipeline)
├── report-files/         LaTeX report: main.tex + sec1–sec8 section files
├── .github/workflows/    ci.yml (Ubuntu, gating: build + both test suites, and a
│                         `run_pipeline.sh --fast --no-iv` smoke job that checks every
│                         figure) + ci-macos.yml (macOS, informational: tests only)
├── run_pipeline.sh       build → tests → benchmarks → every analysis script, snapshotted
└── CMakeLists.txt
```

## Key Design Decisions

**Header-only pricers with `inline` functions.** Each algorithm lives entirely in a `.hpp` (namespaces `cholesky`, `fft_pricer`, `lowrank`) with `inline` functions, so `benchmark.cpp` can include all three headers in one translation unit while each `_pricer.cpp` also includes its header independently, without ODR violations.

**Model parameters in two synced files.** `src/common/params.hpp` (C++) and `data/params.py` (Python) hold H = 0.10 and ν = 0.52; a pytest fails if they disagree. ν = 0.52 is Gatheral et al.'s ν ≈ 0.30 (time in days) converted to the engine's year units (0.30·252^0.1). Never hardcode H or ν in a script — import from `data/params.py`. Prices at ν = 0.52 are heavy-tailed (σ_payoff ≈ 90 at N = 252; the variance is mathematically infinite for lognormal vol but the divergence is ~18σ out, never sampled).

**fGn (not fBM) for the FFT embedding.** fBM itself is non-stationary — its covariance matrix is NOT Toeplitz. Only the *increments* (fractional Gaussian noise, fGn) are stationary, giving a Toeplitz covariance that embeds into a PSD circulant for H ≤ 0.5 (γ(k) ≤ 0 at all nonzero lags; Craigmile 2003). `validate_stability.py` verifies all 2N eigenvalues are strictly positive for H ∈ [0.05, 0.501], so the FFT sampler is exact — no clipping. γ(0) must equal dt^{2H}; an earlier bug that halved it produced a spurious "28% negative eigenvalues" result that also leaked into the report. The FFT pricer uses fGn autocovariance `γ(k) = (dt^{2H}/2)·((k+1)^{2H} + (k-1)^{2H} − 2k^{2H})`, then cumsums to get fBM. Using fBM covariance directly gives negative eigenvalues and crashes.

**rSVD with power iteration for slow spectra.** With H = 0.1 the fBM covariance matrix has slowly-decaying singular values (rough spectrum). `rsvd.hpp` uses `q = 2` subspace iterations — range of `(C·C^T)^q · C · Ω`, re-orthonormalized with QR after every product (Alg. 4.4; without the QR it is Alg. 4.3 and loses precision since (σ₁/σ_{k+p})^5 ≈ 2e15 at k=128). The resulting Frobenius errors are within ~2% of optimal Eckart–Young truncation.

**H-matrix via global rSVD rather than block-tree.** A full recursive H-matrix with an H-Cholesky factorization is complex to implement correctly. Instead we use a global rank-k rSVD: `C ≈ U·S·U^T`, then `L_k = U·diag(√S)`, giving `L_k·L_k^T ≈ C`. Per-path cost drops from O(N²) to O(N·k). The accuracy-vs-speed tradeoff (controlled by k) is the point of the `error_vs_rank.csv` benchmark.

**Accuracy metrics for error_vs_rank.** Report three: (1) Frobenius `‖C − C_k‖_F / ‖C‖_F`, (2) variance lost `tr(C − C_k)/tr(C)` (computed in plot_scaling.py from the exact eigendecomposition), and (3) price error vs a 500k+500k-path Cholesky/FFT reference, with 100k paths per rank (SE ≈ 0.03). The price bias (−8.1% at k=2, −3.9% at k=128) follows the variance lost (37.9% → 13.2%), NOT the Frobenius error (8.5% → 1.2%) — Frobenius is dominated by large eigenvalues and understates what drives the price.

**FFT sampler yields two paths per transform.** `FbmSampler::sample_pair` returns the real and imaginary parts of one inverse FFT as two independent fGn paths (Cov(Re, Im) = 0 since λ_j = λ_{2N−j}); this halves the Gaussian draws, which dominate the FFT per-path cost. Tests check both parts' covariance and their independence.

**Python RNG streams.** Log-vol uses `default_rng(seed)`, price shocks `default_rng([seed, 1])` — never `seed + 1`, which collides with another run's log-vol stream. Experiments comparing configurations use common random numbers (same seeds across H, ν, K).

**Extensions are opt-in, never defaults.** `lowrank::price_corrected` (adds diag(C − L_k L_kᵀ) noise; fixes the price but makes increments too rough) and `price_cv()` in all three samplers (conditional geometric control variate via `src/common/control_variate.hpp`; β from a separate 10% pilot so the estimate is exactly unbiased; needs ρ = 0). The plain `price()`/`price_timed()` paths and RNG order are untouched, so the main benchmark CSVs and the report's numbers stay valid. Extension results live in their own CSVs (`variance_corrected_rank.csv`, `control_variate.csv`) from `./build/extensions`.

**Batched, multithreaded pricers are opt-in too.** `price_batched(N, M, batched::Config{batch, threads, rng})` in all three samplers (driver `src/common/batched_mc.hpp`) generates paths in blocks of B: each sampler's `BatchWorker::log_vol` fills an N×B block with one product L·Z (Cholesky TRMM, low-rank GEMM) or one `fftw_plan_many_dft` call, so L is read once per block instead of once per path. Blocks are also the unit of thread work: block b draws from its own stream seeded by (seed, b) and block sums are reduced in order, so **the price is bitwise identical for any thread count** (tested), but depends on B and on the generator. Threads are `std::thread` (no OpenMP: Apple Clang ships without it). Workers are constructed on the calling thread because FFTW planning is not thread-safe. For the FFT, B counts paths and B = 2 (one transform) is "unbatched". `Rng::Fast` = xoshiro256++ + 256-layer ziggurat (`fast_rng.hpp`, ~3 ns/draw vs ~26 ns for mt19937 + normal_distribution); `Rng::Std` keeps mt19937 per block. The plain `price()`/`price_timed()` loops keep mt19937 and are unchanged. A fixed-seed test pins the fast stream end to end (Clang and GCC agree to 1e-15).

**FFTW plan reuse.** The FFT pricer creates the c2c forward plan (for eigenvalue computation) and the c2c backward plan (for per-path synthesis) once, then calls `fftw_execute` in the MC loop.

**Reference price is the average of two exact methods.** Both Cholesky (dense LL^T) and FFT (Davies-Harte circulant embedding) are exact fBM simulators — prices from both methods converge to the same value with sufficient paths. The benchmark uses their average over 500k paths each as the "ground truth" for the error-vs-rank experiment, avoiding any bias toward either method.

**Python RFSV engine matches C++ FFTW convention exactly.** `data/rfsv_model.py` reimplements the circulant-FFT path generator in pure numpy for use in the validation scripts. The normalization must match FFTW's unnormalized backward transform: `sqrt_lam = sqrt(max(λ,0) / (2N))` then `increments = Re(np.fft.ifft(W) * 2N)[:N]`. The `* 2N` undoes numpy's automatic `1/(2N)` normalization. Agreement is checked by tests/ (covariance, eigenvalues) and by the prices: Python same-grid reference 5.325 ± 0.010 at N=252 vs C++ 5.314 at N=500.

**Log-vol drift `μ₀` for IV validation.** The RFSV model as implemented uses `σ_t = exp(ν W_t^H)`, giving `σ_0 = 1.0` (100% annualized vol). For comparing IV *smiles* against SPY (σ ≈ 15%), `validate_iv.py` auto-calibrates a log-vol drift `μ₀ = log(σ_target)` by matching the ATM RFSV price to the market ATM option price. This decouples vol-of-vol level from smile curvature.

**Oxford-Man alternative calibration.** The Oxford-Man Realized Library CSV is needed for accurate H estimation. Without it, `calibrate.py --source yfinance` uses non-overlapping 5-day window realized variance from squared daily returns. This gives H ≈ 0.12 with R² ≈ 0.98 — close to the Oxford-Man value of H ≈ 0.10. Single-day RV or rolling-window RV are both unreliable (too noisy or over-smoothed).

## Dependencies

- **Eigen3** — dense matrix ops, Cholesky (`brew install eigen`)
- **FFTW3** — fast Fourier transforms (`brew install fftw`)
- C++17, CMake 3.16+
- Python (via `uv`): pandas, numpy, scipy, matplotlib, seaborn, yfinance, pdfminer.six

## Benchmarking Notes

`benchmarks/results/time_vs_N.csv` columns: `method, N, M_paths, wall_time_s, price, construction_time_s, mc_time_s, measured_peak_mb, theoretical_peak_mb, cache_pressure, est_bandwidth_GBs, wall_time_q1_s, wall_time_q3_s` (timings are the median of 5 repeats, with the interquartile range; N ∈ {64, 128, 252, 500, 1000, 2000, 4000})
`benchmarks/results/perf_batched.csv` columns: `method, N, M_paths, config, batch, rng, threads, price, se, construction_time_s, mc_time_s, mc_time_q1_s, mc_time_q3_s`; config ∈ {legacy, unbatched_fast, batched_std, batched_fast}. Also `perf_breakdown.csv` (ns per path of each stage), `perf_batch_size.csv`, `perf_threads.csv`.
`benchmarks/results/error_vs_rank.csv` columns: `rank_k, N, reference_price, rsvd_price, abs_price_error, rel_price_error, frob_error, construction_time_s, mc_time_s`
`benchmarks/results/reference_price.txt` — documents reference price inputs

Benchmark facts (σ0 = 0.2, ν = 0.52, M = 10k, N = 64…4000, current CSV of 2026-10-08, AC power):
- Global fits: Cholesky `t = 5.8e-5 · N^1.48` (R² 0.983 — curve bends; local exponent 1.05 → 2.26, >2 only for 2000→4000), FFT `t = 7.2e-4 · N^1.00`, rSVD k=32 `t = 4.0e-4 · N^1.02`
- Cholesky/FFT speedup: 1.0× at N=252, 1.7× at 1000, 6.7× at 4000 (19.66 s vs 2.93 s). FFT per-path cost per step is flat (70–76 ns): RNG and price-path work dominate, the transform is ~8 µs of a path at N=1000, so log N is invisible
- Reference price (N=500): 5.3144 (Cholesky 5.3195, FFT 5.3093). σ_payoff ≈ 9.5 → SE ≈ 0.095 at M=10k
- Cholesky effective streaming rate: 14/24/33/39/33 GB/s at N=252/500/1000/2000/4000 (the N=2000 point varies between runs; an earlier run gave 31)
- Performance (perf_*.csv, one thread unless noted): B=64 + fast generator vs the original loop — Cholesky 2.2× (N=64) → 4.0× (N=4000; transform 1621 → 379 µs/path ≈ 42 GFlop/s), FFT 2.0–2.2× (all from the generator; batching does nothing for the FFT), rSVD 1.7–1.9×. Generator: 15.2 → 3.8 ns per normal. B=2 is slower than B=1 for Cholesky; gain saturates at B≈64. Threads (perf_threads.csv is from an idle-machine run): unbatched Cholesky at N=4000 peaks at 1.8× (3 threads) and falls to 1.5× at 8 — bandwidth-bound; batched reaches 3.1× (4 threads) / 4.1× (8); FFT and rSVD 4.3–5.7× at 8
- Timing hygiene: the dev machine is a **fanless MacBook Air M2**. Run benchmarks on AC power with Low Power Mode off (`pmset -g | grep lowpowermode`; it slowed everything ~1.8×) and nothing else busy. Check the IQR columns: a disturbed point shows up as a large IQR (one N=4000 Cholesky run had 34% and was 72% slow)

Memory notes: the "L3 = 16 MB" constant in benchmark.cpp / plot scripts is really the M2 P-cluster L2 (M2 has no L3). `measured_peak_mb` is a true lifetime peak (forked child, `wait4` ru_maxrss minus an idle child); it replaced the old meaningless `peak_rss_mb`. `est_bandwidth_GBs` = lower-triangle bytes × M / wall time — an effective rate, not DRAM traffic (L fits in L2 at N ≤ 1000).

## Project Evaluation Criteria

This is a final project for a fast-algorithms course. Every experiment should be framed with explicit control/independent/dependent variables. The rubric (`include.md`) requires:

1. **Runtime efficiency** — scaling benchmarks with fitted exponents (already done in `plot_scaling.py`)
2. **Memory use** — peak RSS vs N for all three C++ methods + Python engine (`benchmarks/memory_benchmark.cpp` + `data/profile_memory.py`, planned)
3. **Accuracy** — MC convergence: price ± 1σ vs M paths, confirm σ ∝ 1/√M log-log slope ≈ −0.5 (`data/validate_convergence.py`, planned)
4. **Stability** — FFT eigenvalue positivity vs H, rSVD condition number vs rank, Cholesky κ(C) vs N (`data/validate_stability.py`)
5. **Structural analysis** — *why* each method works: Toeplitz structure of fGn (→ FFT), low-rank off-diagonal structure of C(t,s) (→ H-matrix), singular value decay comparison H=0.1 vs H=0.5 (`plots/plot_structure.py`, planned)

## Analysis Scripts

`TODO.md` is the prioritized roadmap for further work (evidence, plan, and done-when criteria per item). `experiments/` holds prototypes for roadmap items; they are not part of `run_pipeline.sh`. Every analysis script writes to `plots/figures/`; `run_pipeline.sh` runs all of them.

| Script | Purpose |
|---|---|
| `plots/plot_scaling.py` | Timing fits, per-path cost, construction breakdown, memory, error vs rank (from `benchmark` CSVs) |
| `plots/plot_extensions.py` | Variance correction vs rank; control-variate variance reduction and time to 0.1% SE (from `extensions` CSVs) |
| `plots/plot_structure.py` | fGn Toeplitz heatmaps; off-diagonal block vs full-matrix spectrum |
| `plots/plot_sensitivity.py` | ATM price over H × nu and price vs K, with common random numbers |
| `data/validate_convergence.py` | Price ± 1σ vs M over 20 seeds against a same-grid 10^6-path reference |
| `data/validate_stability.py` | FFT eigenvalue positivity vs H, rSVD κ(L_k) vs rank, Cholesky κ(C) vs N |
| `data/validate_asian.py` | Lévy and exact-GBM baselines; roughness premium at fixed and variance-matched nu |
| `data/validate_iv.py` | RFSV smile vs live SPY chain (needs internet; martingale-corrected, CRN) |
| `data/profile_memory.py` | `tracemalloc` peak allocation vs (N, M) for the numpy engine |
| `data/greeks.py` | Vega in $\nu$ (pathwise and central difference, both with the CV) and $dV/dH$ |
| `plots/plot_performance.py` | Per-path cost breakdown, batched vs unbatched, block size, thread scaling (from `performance` CSVs) |

**Production-quality run parameters** (use for final report plots):
- `validate_asian.py`: `--M 10000 --N 252 --n-seeds 10` (common random numbers; paired SEs on the premium)
- `validate_iv.py`: `--M 20000 --N 63` (runs in seconds; uses common random numbers + martingale correction of S_T — without it, forward sampling error tilts the IV curve into a fake skew)
- `plot_sensitivity.py`: `--M 10000 --N 252` (absolute SE ≈ 0.1; cell differences precise via common random numbers)
- `validate_convergence.py`: `--n-seeds 20 --max-M 25000`
- `greeks.py`: `--M 20000 --N 252`

`validate_asian.py`, `plot_sensitivity.py` and `validate_convergence.py` price with the control variate (`price_asian_call_cv`; convergence shows plain and CV side by side). `price_asian_call_cv` takes an array of strikes (one β per strike), an optional `sampler` (e.g. `lowrank_sampler(..., corrected=True)`) and `return_samples=True` for paired SEs of differences under common random numbers.

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

## Report Writing Style

The report (`report-files/`) uses a distinctive style. Maintain it when drafting new sections.

**Voice and structure**
- First-person technical voice: "I implement", "I test", not "it is implemented"
- Motivate before defining: state *why* something is needed, then define it formally
- Short paragraphs (3–5 sentences each), one clear point per paragraph
- Lead each subsection with the key claim; supporting detail follows

**Equations and notation**
- Every displayed equation is introduced in the preceding sentence and interpreted in the following one — never dropped naked
- Use `\tfrac` for inline fractions in running text, `\frac` in displayed math
- `\mathbb{E}^{\mathbb{Q}}` for risk-neutral expectation; `\mathcal{N}(0,C)` for distributions
- Use `\text{Cov}`, `\text{Var}`, `\max`, `\min` (roman operators), not bare `Cov`, `max`
- Complexity in prose: `$O(N^3)$` (not `O(N³)` or `O(N^3)` plain text)

**Numbers and concreteness**
- Every complexity claim is accompanied by a concrete number at $N = 252$ or $M = 10{,}000$
- Leading constants matter: $N^3/3$, not $N^3$; always state the constant when it affects interpretation
- Benchmark figures are cited numerically inline (e.g., "$\approx 1.54$"), not just described

**Citations**
- Use `\citep{}` for parenthetical references, `\citet{}` when the author is the grammatical subject
- Include a short inline gloss on first citation: "Gatheral et al.~\citep{gatheral2018volatility} show that …"
- Multiple related cites compressed: `\citep{gatheral2018volatility, mandelbrot1968fractional}`

**Limitations and contested points**
- Call out traps and non-obvious choices directly: "The key pitfall is…", "Note that setting $H=0.5$ alone does *not* recover GBM…"
- Quantify the impact of every limitation (e.g., "1.9\% Frobenius error at $k=32$, decaying only to 1.2\% at $k=128$")
- End subsections with a one-sentence forward pointer when the limitation is addressed elsewhere

**LaTeX mechanics (report-files/)**
- `\usepackage{booktabs}` — always use `\toprule`, `\midrule`, `\bottomrule`; never `\hline`
- `\usepackage{nicefrac}` for `\nicefrac{1}{2}` in compact inline fractions
- Section files are `\input`-ed from `main.tex`; each file contains only `\section{}`…`\subsection{}` content, no preamble
- Bibliography key: `\bibliography{bibliography}` (no `.bib` extension in the `\bibliography` command)
- Do NOT use `i` as a matrix or time-step index in any section — it clashes with $i = \sqrt{-1}$ in the FFT sections. Use `j`, `k`, `m`, `n` instead.

## Report Build

```bash
# From report-files/
pdflatex -interaction=nonstopmode main.tex

# Version every successful build — always do this after a compile
ts=$(date +%Y%m%d_%H%M) && cp main.pdf versions/main_${ts}.pdf
```

Figures live in `plots/figures/` (gitignored); `main.tex` uses `\graphicspath{{../plots/figures/}}`.
If figures are missing, copy from `plots/*.png`: `cp plots/*.png plots/figures/`.
The `.bbl` file is generated once by `bibtex main`; after that, single `pdflatex` passes suffice.

**Section files:** `sec1-intro.tex` through `sec8-futurework.tex` in `report-files/` (`sec7-extensions.tex` covers the variance correction and control variate).
Track drafting status and all post-draft corrections in `report-plan.md` (gitignored).

## Commit Convention

Use `type: short description` (one line, no period). Common types:

- `feat` — new script, function, or C++ feature
- `fix` — bug fix
- `docs` — README, CLAUDE.md, comments, ALGORITHMS.md
- `refactor` — restructuring without behavior change
- `bench` — benchmark runner or CSV output changes
- `style` — formatting, plot aesthetics
