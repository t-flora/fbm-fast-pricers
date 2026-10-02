"""
Model parameters shared by the Python scripts.

Mirrors src/common/params.hpp; tests/test_python.py fails if the two disagree.
"""

H = 0.10    # Hurst exponent (Gatheral, Jaisson & Rosenbaum)
# Vol-of-vol with time in years, as used by both engines (T = 1, dt = 1/N).
# Gatheral et al.'s nu ~ 0.30 is quoted with time in days: 0.30 * 252**0.10 ~ 0.52.
NU = 0.52
# Base volatility: log sigma_t = log(SIGMA0) + NU * W_t^H.  Without it sigma_0 = 1 (100%
# vol) and the payoff is so heavy-tailed that MC standard errors are unreliable.
SIGMA0 = 0.20
MU0 = float(__import__("math").log(SIGMA0))

S0 = 100.0  # spot
K = 100.0   # strike (at the money)
T = 1.0     # maturity (years)
R = 0.0     # risk-free rate
