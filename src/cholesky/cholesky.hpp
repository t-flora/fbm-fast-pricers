#pragma once
// Block 2: Dense Cholesky path generator.
// O(N^3) factorization once (N^3/3 flops); O(N^2) per path (N^2 flops, triangular).
#include <Eigen/Dense>
#include <chrono>
#include <stdexcept>
#include "common/params.hpp"
#include "common/covariance.hpp"
#include "common/asian_payoff.hpp"
#include "common/rng.hpp"

namespace cholesky {

inline double price(int N, int M_paths, unsigned seed = 42) {
    using namespace params;
    double dt = T / N;
    auto rng = make_rng(seed);

    // Factor in place: L overwrites the lower triangle of C, so only one N×N matrix is alive
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(C);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky: matrix not positive-definite");
    auto L = C.triangularView<Eigen::Lower>();

    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::VectorXd z(N), lv(N);
    double payoff_sum = 0.0;

    for (int m = 0; m < M_paths; ++m) {
        for (int i = 0; i < N; ++i) z(i) = norm(rng);
        lv.noalias() = L * z;  // triangular mat-vec: N^2 flops, reads only the lower half
        lv *= nu;
        std::vector<double> log_vol(lv.data(), lv.data() + N);
        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
    }
    return std::exp(-r * T) * payoff_sum / M_paths;
}

// Construction vs MC timing breakdown
struct CholeskyTimed { double price, t_construct, t_mc; };

inline CholeskyTimed price_timed(int N, int M_paths, unsigned seed = 42) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;
    double dt = T / N;
    auto rng = make_rng(seed);

    auto t0 = Clock::now();
    // Factor in place: L overwrites the lower triangle of C, so only one N×N matrix is alive
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    Eigen::LLT<Eigen::Ref<Eigen::MatrixXd>> llt(C);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky: matrix not positive-definite");
    auto L = C.triangularView<Eigen::Lower>();
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();

    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::VectorXd z(N), lv(N);
    double payoff_sum = 0.0;

    t0 = Clock::now();
    for (int m = 0; m < M_paths; ++m) {
        for (int i = 0; i < N; ++i) z(i) = norm(rng);
        lv.noalias() = L * z;
        lv *= nu;
        std::vector<double> log_vol(lv.data(), lv.data() + N);
        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
    }
    double t_mc = std::chrono::duration<double>(Clock::now() - t0).count();
    return { std::exp(-r * T) * payoff_sum / M_paths, t_construct, t_mc };
}

} // namespace cholesky
