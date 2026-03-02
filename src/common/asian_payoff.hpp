#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

// Arithmetic Asian call payoff: max(mean(prices) - K, 0)
inline double asian_call_payoff(const std::vector<double>& prices, double K) {
    double mean = std::accumulate(prices.begin(), prices.end(), 0.0) / prices.size();
    return std::max(mean - K, 0.0);
}

// Convert a log-volatility path to a price path under GBM dynamics.
// log_vol[i] = log(sigma_i);  prices[i] = S0 * exp(integral of sigma dW)
// Simplified discrete version: S_i = S_{i-1} * exp(sigma_i * sqrt(dt) * Z_i - 0.5*sigma_i^2*dt)
inline std::vector<double> log_vol_to_prices(
    const std::vector<double>& log_vol,
    const std::vector<double>& Z,   // i.i.d. N(0,1) innovations for price process
    double S0, double r, double dt)
{
    int N = static_cast<int>(log_vol.size());
    std::vector<double> prices(N);
    double S = S0;
    for (int i = 0; i < N; ++i) {
        double sigma = std::exp(log_vol[i]);
        S *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z[i]);
        prices[i] = S;
    }
    return prices;
}
