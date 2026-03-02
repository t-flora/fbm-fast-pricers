#pragma once

// ──────────────────────────────────────────────────────────────────────────────
// Calibrated model parameters
// Run `python data/calibrate.py` and paste the printed values here.
// ──────────────────────────────────────────────────────────────────────────────
namespace params {

// Fractional Brownian Motion parameters (calibrated from Oxford-Man data)
constexpr double H   = 0.10;   // Hurst exponent  (update after calibration)
constexpr double nu  = 0.30;   // Vol-of-vol      (update after calibration)

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
