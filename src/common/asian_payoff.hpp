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
// sigma_i = sigma0 * exp(log_vol[i]), where log_vol = nu * W^H is the simulated fBM path;
// prices[i] = S0 * exp(integral of sigma dW)
// Simplified discrete version: S_i = S_{i-1} * exp(sigma_i * sqrt(dt) * Z_i - 0.5*sigma_i^2*dt)
inline std::vector<double> log_vol_to_prices(
    const std::vector<double>& log_vol,
    const std::vector<double>& Z,   // i.i.d. N(0,1) innovations for price process
    double S0, double r, double dt, double sigma0)
{
    int N = static_cast<int>(log_vol.size());
    std::vector<double> prices(N);
    double S = S0;
    for (int i = 0; i < N; ++i) {
        double sigma = sigma0 * std::exp(log_vol[i]);
        S *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z[i]);
        prices[i] = S;
    }
    return prices;
}

// ── Control-variate support ──────────────────────────────────────────────────
// One path of the arithmetic Asian payoff together with the geometric-average payoff and
// its conditional expectation given the volatility path.
//
// With rho = 0 the price shocks Z are independent of the volatility path, so given sigma,
//   log G = (1/N) sum_n log S_n = log S0 + sum_j w_j [(r - sigma_j^2/2) dt + sigma_j sqrt(dt) Z_j],
//   w_j = (N - j) / N  (0-based j),
// is Gaussian with mean m = log S0 + sum_j w_j (r - sigma_j^2/2) dt and variance
// v = dt sum_j w_j^2 sigma_j^2, and E[(G - K)^+ | sigma] = e^{m + v/2} Phi(d1) - K Phi(d1 - sqrt v),
// d1 = (m - log K + v) / sqrt v.  Undiscounted; the caller applies e^{-rT}.
struct AsianSample {
    double arith;      // max(A - K, 0), A = arithmetic average of the price path
    double geom;       // max(G - K, 0), G = geometric average of the same path
    double geom_mean;  // E[max(G - K, 0) | sigma path]  (closed form, see above)
};

inline double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }

inline AsianSample asian_sample(const std::vector<double>& log_vol,
                                const std::vector<double>& Z,
                                double S0, double r, double dt, double sigma0, double K)
{
    const int N = static_cast<int>(log_vol.size());
    double S = S0, sum_S = 0.0;
    double logS = std::log(S0), sum_logS = 0.0;
    double m = std::log(S0), v = 0.0;
    for (int i = 0; i < N; ++i) {
        double sigma = sigma0 * std::exp(log_vol[i]);
        double drift = (r - 0.5 * sigma * sigma) * dt;
        double shock = sigma * std::sqrt(dt) * Z[i];
        S *= std::exp(drift + shock);           // same step as log_vol_to_prices()
        sum_S += S;
        logS += drift + shock;
        sum_logS += logS;
        double w = static_cast<double>(N - i) / N;
        m += w * drift;
        v += w * w * sigma * sigma * dt;
    }
    double sd = std::sqrt(v);
    double d1 = (m - std::log(K) + v) / sd;
    double geom_mean = std::exp(m + 0.5 * v) * norm_cdf(d1) - K * norm_cdf(d1 - sd);
    return { std::max(sum_S / N - K, 0.0), std::max(std::exp(sum_logS / N) - K, 0.0), geom_mean };
}
