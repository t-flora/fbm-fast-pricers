#pragma once
// Block 2: Dense Cholesky path generator.
// O(N^3) factorization once; O(N^2) per path.
#include <Eigen/Dense>
#include <stdexcept>
#include "common/params.hpp"
#include "common/covariance.hpp"
#include "common/asian_payoff.hpp"
#include "common/rng.hpp"

namespace cholesky {
namespace {

double price(int N, int M_paths, unsigned seed = 42) {
    using namespace params;
    double dt = T / N;
    auto rng = make_rng(seed);

    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    Eigen::LLT<Eigen::MatrixXd> llt(C);
    if (llt.info() != Eigen::Success)
        throw std::runtime_error("Cholesky: matrix not positive-definite");
    Eigen::MatrixXd L = llt.matrixL();

    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::VectorXd z(N);
    double payoff_sum = 0.0;

    for (int m = 0; m < M_paths; ++m) {
        for (int i = 0; i < N; ++i) z(i) = norm(rng);
        Eigen::VectorXd lv = nu * (L * z);
        std::vector<double> log_vol(lv.data(), lv.data() + N);
        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
    }
    return std::exp(-r * T) * payoff_sum / M_paths;
}

} // anonymous namespace
} // namespace cholesky
