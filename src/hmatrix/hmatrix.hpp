#pragma once
// Block 4: Randomized low-rank (H-matrix) path generator.
//
// Approach: global rank-k approximation of the fBM covariance matrix C via rSVD.
// Since C is symmetric PD, rSVD yields C ≈ U * diag(S) * U^T (U is N×k).
// Approximate Cholesky factor: L_k = U * diag(sqrt(S))  →  L_k * L_k^T ≈ C.
// Path generation: log_vol = nu * L_k * z   (z ~ N(0, I_k))  →  O(N*k) per path.
//
// This captures the H-matrix insight that the fBM covariance is effectively
// low-rank away from the diagonal; rank k controls the accuracy-speed tradeoff.
// Construction cost: O(N*k^2) via power-iteration rSVD.
// Per-path cost: O(N*k) vs O(N^2) for dense Cholesky.
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "common/params.hpp"
#include "common/covariance.hpp"
#include "common/asian_payoff.hpp"
#include "common/rng.hpp"
#include "hmatrix/rsvd.hpp"

namespace hmatrix {
namespace {

double price(int N, int M_paths, int rank_k = 16, unsigned seed = 42) {
    using namespace params;
    double dt = T / N;

    // Build full covariance once, then compress
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);

    // rSVD: C ≈ U * diag(S) * Vt  (for SPD matrix U ≈ V, but rsvd gives general form)
    int k = std::min(rank_k, N);
    RSVD decomp = rsvd(C, k, /*oversampling=*/5, /*power_iters=*/2, seed);

    // Approximate sqrt factor L_k = U * diag(sqrt(max(S,0)))  [N × k]
    Eigen::VectorXd sqrt_S = decomp.S.cwiseMax(0.0).cwiseSqrt();
    Eigen::MatrixXd Lk = decomp.U * sqrt_S.asDiagonal();  // N × k

    auto rng = make_rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::VectorXd z(k);
    double payoff_sum = 0.0;

    for (int m = 0; m < M_paths; ++m) {
        for (int i = 0; i < k; ++i) z(i) = norm(rng);
        Eigen::VectorXd lv = nu * (Lk * z);  // O(N*k)
        std::vector<double> log_vol(lv.data(), lv.data() + N);
        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
    }
    return std::exp(-r * T) * payoff_sum / M_paths;
}

} // anonymous namespace
} // namespace hmatrix
