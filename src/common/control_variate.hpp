#pragma once
// Conditional geometric-Asian control variate, shared by all three samplers.
//
// Estimator:  p_cv = e^{-rT} * mean(V - beta * C),   C = Y - E[Y | sigma]
//   V = arithmetic Asian payoff, Y = geometric Asian payoff on the same path,
//   E[Y | sigma] = its closed-form conditional expectation (asian_sample()).
// C has mean zero exactly, so p_cv is unbiased for any fixed beta. beta is estimated from
// a separate pilot run (its own seed), so the main estimate stays exactly unbiased.
// The variance-optimal beta is Cov(V, C) / Var(C); the variance reduction factor is
// 1 / (1 - corr(V, C)^2) at that beta.
//
// Requires rho = 0 (price shocks independent of the volatility path).
#include <chrono>
#include <cmath>
#include <random>
#include <vector>
#include "common/asian_payoff.hpp"
#include "common/params.hpp"
#include "common/rng.hpp"

struct CVResult {
    double price, se;              // control-variate estimate and its standard error
    double price_plain, se_plain;  // plain Monte Carlo estimate from the same paths
    double beta;                   // control coefficient (estimated on the pilot run)
    double t_mc;                   // seconds, pilot included
    double t_construct = 0.0;      // seconds of sampler setup (set by the pricer)
};

namespace cv_detail {

// Running sums of V and C over a batch of paths
struct Sums {
    double n = 0, V = 0, VV = 0, C = 0, CC = 0, VC = 0;
    void add(double v, double c) { n += 1; V += v; VV += v * v; C += c; CC += c * c; VC += v * c; }
    double var_V() const { return (VV - V * V / n) / (n - 1); }
    double var_C() const { return (CC - C * C / n) / (n - 1); }
    double cov_VC() const { return (VC - V * C / n) / (n - 1); }
};

// next_path(rng, log_vol) fills log_vol (size N) with nu * W^H for one path
template <class NextPath>
Sums run(NextPath& next_path, int N, int M, std::mt19937& rng) {
    using namespace params;
    double dt = T / N;
    std::vector<double> log_vol(N);
    Sums s;
    for (int m = 0; m < M; ++m) {
        next_path(rng, log_vol);
        auto inno = randn(N, rng);
        AsianSample a = asian_sample(log_vol, inno, S0, r, dt, sigma0, K);
        s.add(a.arith, a.geom - a.geom_mean);
    }
    return s;
}

}  // namespace cv_detail

// Pilot size: 10% of the main run, at least 2000 paths, even (so pair-generating
// samplers never carry a half-used pair from the pilot into the main run)
inline int cv_pilot_size(int M_paths) {
    int p = std::max(2000, M_paths / 10);
    return p + (p % 2);
}

template <class NextPath>
CVResult mc_control_variate(NextPath&& next_path, int N, int M_paths, unsigned seed) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;
    const double disc = std::exp(-r * T);
    auto t0 = Clock::now();

    auto rng_pilot = make_rng(seed + 1000003u);
    cv_detail::Sums pilot = cv_detail::run(next_path, N, cv_pilot_size(M_paths), rng_pilot);
    double beta = pilot.cov_VC() / pilot.var_C();

    auto rng = make_rng(seed);
    cv_detail::Sums s = cv_detail::run(next_path, N, M_paths, rng);
    double mean_V = s.V / s.n, mean_C = s.C / s.n;
    double var_cv = s.var_V() - 2.0 * beta * s.cov_VC() + beta * beta * s.var_C();

    double t_mc = std::chrono::duration<double>(Clock::now() - t0).count();
    return { disc * (mean_V - beta * mean_C), disc * std::sqrt(var_cv / s.n),
             disc * mean_V, disc * std::sqrt(s.var_V() / s.n), beta, t_mc };
}
