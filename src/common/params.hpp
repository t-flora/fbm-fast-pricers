#pragma once

// ──────────────────────────────────────────────────────────────────────────────
// Calibrated model parameters
// `uv run python data/calibrate.py` prints H and nu (time in years) for comparison.
// ──────────────────────────────────────────────────────────────────────────────
namespace params {

// Fractional Brownian Motion parameters (Gatheral, Jaisson & Rosenbaum estimates).
// Time is in years here (T = 1, dt = 1/N). Gatheral et al. quote nu ~ 0.30 with time in
// days; in year units that is 0.30 * 252^0.10 ~ 0.52. Keep data/params.py in sync.
constexpr double H   = 0.10;   // Hurst exponent
constexpr double nu  = 0.52;   // Vol-of-vol (time in years)
// Base volatility level: log sigma_t = log(sigma0) + nu * W_t^H. Without it sigma_0 = 1
// (100% vol), and at nu = 0.52 the payoff becomes so heavy-tailed that a single path in
// 10^6 carries a third of the sample variance.
constexpr double sigma0 = 0.20;

// Option parameters
constexpr double S0  = 100.0;  // Initial spot price
constexpr double K   = 100.0;  // Strike price (at-the-money)
constexpr double T   = 1.0;    // Maturity (years)
constexpr double r   = 0.0;    // Risk-free rate

// Benchmark path resolutions
constexpr int N_SMALL  = 252;
constexpr int N_MEDIUM = 500;
constexpr int N_LARGE  = 1000;

// Default Monte Carlo paths
constexpr int M_PATHS  = 10000;

} // namespace params
