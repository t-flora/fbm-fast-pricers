#pragma once
// Block 4: Global low-rank rSVD path generator.
//
// Approach: global rank-k approximation of the fBM covariance matrix C via rSVD.
// Since C is symmetric PD, rSVD yields C ≈ U * diag(S) * U^T (U is N×k).
// Approximate Cholesky factor: L_k = U * diag(sqrt(S))  →  L_k * L_k^T ≈ C.
// Path generation: log_vol = nu * L_k * z   (z ~ N(0, I_k))  →  O(N*k) per path.
//
// Rank k controls the accuracy-speed tradeoff. Note this is an approximate sampler:
// paths have covariance nu^2 * C_k, not nu^2 * C, and the truncated variance is lost.
// Construction cost: O(N^2*k) via power-iteration rSVD.
// Per-path cost: O(N*k) vs O(N^2) for dense Cholesky.
#include <Eigen/Dense>
#include <chrono>
#include <vector>
#include <cmath>
#include "common/params.hpp"
#include "common/covariance.hpp"
#include "common/asian_payoff.hpp"
#include "common/rng.hpp"
#include "rsvd/rsvd.hpp"
#include "common/control_variate.hpp"

namespace lowrank {

// Approximate factor L_k = U * diag(sqrt(max(S, 0)))  [N × k]  with L_k L_k^T ≈ C
inline Eigen::MatrixXd lowrank_factor(const Eigen::MatrixXd& C, int rank_k, unsigned seed) {
    int k = std::min(rank_k, static_cast<int>(C.rows()));
    RSVD decomp = rsvd(C, k, /*oversampling=*/5, /*power_iters=*/2, seed);
    Eigen::VectorXd sqrt_S = decomp.S.cwiseMax(0.0).cwiseSqrt();
    return decomp.U * sqrt_S.asDiagonal();
}

// Variance correction: per-step standard deviation of the variance the rank-k factor misses,
// d_i = sqrt(max(C_ii - (L_k L_k^T)_ii, 0)). Adding independent N(0, d_i^2) noise at step i
// restores every marginal variance Var(W^H(t_i)) exactly; off-diagonal covariances remain
// those of C_k. Costs N extra Gaussian draws and multiply-adds per path.
inline Eigen::VectorXd residual_sd(const Eigen::MatrixXd& C, const Eigen::MatrixXd& Lk) {
    return (C.diagonal() - Lk.rowwise().squaredNorm()).cwiseMax(0.0).cwiseSqrt();
}

// Monte Carlo loop shared by all variants. If resid is given, the variance correction
// is applied (its draws come after z, so uncorrected runs are unchanged).
inline double mc_price(const Eigen::MatrixXd& Lk, int M_paths, unsigned seed,
                       const Eigen::VectorXd* resid = nullptr) {
    using namespace params;
    int N = Lk.rows(), k = Lk.cols();
    double dt = T / N;
    auto rng = make_rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::VectorXd z(k);
    double payoff_sum = 0.0;

    for (int m = 0; m < M_paths; ++m) {
        for (int i = 0; i < k; ++i) z(i) = norm(rng);
        Eigen::VectorXd lv = nu * (Lk * z);  // O(N*k)
        if (resid)
            for (int i = 0; i < N; ++i) lv(i) += nu * (*resid)(i) * norm(rng);
        std::vector<double> log_vol(lv.data(), lv.data() + N);
        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt, sigma0), K);
    }
    return std::exp(-r * T) * payoff_sum / M_paths;
}

// Construction vs MC timing breakdown. C stays alive (O(N^2)) through the MC loop.
struct LowRankTimed { double price, t_construct, t_mc; };

inline LowRankTimed price_timed(int N, int M_paths, int rank_k = 16, unsigned seed = 42) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;

    auto t0 = Clock::now();
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    Eigen::MatrixXd Lk = lowrank_factor(C, rank_k, seed);
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();

    t0 = Clock::now();
    double p = mc_price(Lk, M_paths, seed);
    double t_mc = std::chrono::duration<double>(Clock::now() - t0).count();
    return { p, t_construct, t_mc };
}

inline double price(int N, int M_paths, int rank_k = 16, unsigned seed = 42) {
    return price_timed(N, M_paths, rank_k, seed).price;
}

// Variance-corrected low-rank sampler: rank-k paths plus independent per-step noise that
// restores the marginal variances (see residual_sd()).
inline LowRankTimed price_corrected_timed(int N, int M_paths, int rank_k = 16, unsigned seed = 42) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;

    auto t0 = Clock::now();
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    Eigen::MatrixXd Lk = lowrank_factor(C, rank_k, seed);
    Eigen::VectorXd resid = residual_sd(C, Lk);
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();

    t0 = Clock::now();
    double p = mc_price(Lk, M_paths, seed, &resid);
    double t_mc = std::chrono::duration<double>(Clock::now() - t0).count();
    return { p, t_construct, t_mc };
}

inline double price_corrected(int N, int M_paths, int rank_k = 16, unsigned seed = 42) {
    return price_corrected_timed(N, M_paths, rank_k, seed).price;
}

// Low-rank sampler (optionally variance-corrected), priced with the conditional geometric
// control variate (common/control_variate.hpp).
inline CVResult price_cv(int N, int M_paths, int rank_k = 16, unsigned seed = 42,
                         bool corrected = false) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    Eigen::MatrixXd Lk = lowrank_factor(C, rank_k, seed);
    Eigen::VectorXd resid = corrected ? residual_sd(C, Lk) : Eigen::VectorXd();
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();

    const int k = Lk.cols();
    std::normal_distribution<double> norm(0.0, 1.0);
    Eigen::VectorXd z(k), lv(N);
    auto next_path = [&](std::mt19937& rng, std::vector<double>& log_vol) {
        for (int i = 0; i < k; ++i) z(i) = norm(rng);
        lv.noalias() = Lk * z;
        for (int i = 0; i < N; ++i) {
            double x = lv(i);
            if (corrected) x += resid(i) * norm(rng);
            log_vol[i] = nu * x;
        }
    };
    CVResult res = mc_control_variate(next_path, N, M_paths, seed);
    res.t_construct = t_construct;
    return res;
}

// C freed before MC loop: only Lk (N×k) is resident during sampling.
// Compare to price_timed() where C stays alive (O(N^2)) throughout.
// The lifetime peak is the same for both (C must exist while Lk is built).
struct LowRankFreedTimed { double price, t_construct, t_mc; };

inline LowRankFreedTimed price_freed_timed(int N, int M_paths, int rank_k = 16, unsigned seed = 42) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;

    // Construction block: C destroyed when this scope exits.
    Eigen::MatrixXd Lk;
    double t_construct;
    {
        auto t0 = Clock::now();
        Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
        Lk = lowrank_factor(C, rank_k, seed);
        t_construct = std::chrono::duration<double>(Clock::now() - t0).count();
    }  // C destroyed here; only Lk (N×k) remains on the heap

    auto t1 = Clock::now();
    double p = mc_price(Lk, M_paths, seed);
    double t_mc = std::chrono::duration<double>(Clock::now() - t1).count();
    return { p, t_construct, t_mc };
}

} // namespace lowrank
