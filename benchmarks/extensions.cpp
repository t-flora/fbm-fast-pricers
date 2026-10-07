// Benchmarks for the two extensions (TODO.md items 1-2). Separate from benchmark.cpp so
// the main results stay untouched.
//
// (a) Variance-corrected low-rank sampler vs rank (N = 500, 100k paths per rank):
//     price error of the plain and corrected samplers against the main benchmark's
//     reference price, plus exact structural errors of each sampler's covariance:
//       var_lost        tr(C - Cov) / tr(C)              (0 for corrected, by construction)
//       frob            ||C - Cov||_F / ||C||_F
//       incr_var_err    mean relative error of the increment (fGn) variances
//       incr_lag1_err   mean relative error of the lag-1 increment covariances
//     where Cov = L_k L_k^T (plain) or L_k L_k^T + diag(d^2) (corrected).
//
// (b) Conditional geometric control variate for each sampler at N = 252 and 1000
//     (M = 10k): plain vs CV price and standard error, beta, variance reduction, and the
//     time each needs to reach a 0.1% relative standard error.
//
// Outputs:
//   benchmarks/results/variance_corrected_rank.csv
//   benchmarks/results/control_variate.csv
//
// Usage: ./build/extensions [--quick]
//   --quick: 10k paths and ranks {2, 8, 32} in (a), N = 252 with 5k paths in (b).
//            For smoke tests only; it overwrites the same output files.

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "cholesky/cholesky.hpp"
#include "fft/fft.hpp"
#include "rsvd/lowrank.hpp"
#include "common/params.hpp"
#include "common/covariance.hpp"
#include "timing.hpp"

// Reference price written by ./build/benchmark (average of 500k Cholesky + 500k FFT paths)
static double read_reference_price() {
    std::ifstream f("benchmarks/results/reference_price.txt");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("reference_price=", 0) == 0) return std::stod(line.substr(16));
    std::cerr << "reference_price.txt not found; run ./build/benchmark first\n";
    std::exit(1);
}

struct Structure { double var_lost, frob, incr_var_err, incr_lag1_err; };

// Exact covariance errors of a sampler with covariance Cov against the fBM covariance C
static Structure structure_errors(const Eigen::MatrixXd& C, const Eigen::MatrixXd& Cov) {
    const int N = C.rows();
    // Increment covariance: D Cov D^T with (D W)_0 = W_0, (D W)_i = W_i - W_{i-1}
    Eigen::MatrixXd D = Eigen::MatrixXd::Identity(N, N);
    for (int i = 1; i < N; ++i) D(i, i - 1) = -1.0;
    Eigen::MatrixXd G = D * C * D.transpose(), Gs = D * Cov * D.transpose();
    double var_err = 0.0, lag1_err = 0.0;
    for (int i = 0; i < N; ++i) var_err += std::abs(Gs(i, i) - G(i, i)) / G(i, i);
    for (int i = 0; i + 1 < N; ++i) lag1_err += std::abs(Gs(i, i + 1) - G(i, i + 1)) / std::abs(G(i, i + 1));
    return { (C - Cov).trace() / C.trace(), (C - Cov).norm() / C.norm(),
             var_err / N, lag1_err / (N - 1) };
}

int main(int argc, char** argv) {
    using namespace params;
    const bool quick = has_flag(argc, argv, "--quick");
    std::cout << "Mode: " << (quick ? "QUICK (smoke test)" : "FULL") << "\n";

    // ── (a) Variance-corrected low-rank sampler vs rank ─────────────────────
    {
        const int N = N_MEDIUM, M = quick ? 10000 : 100000;
        const double p_ref = read_reference_price();
        Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
        std::ofstream csv("benchmarks/results/variance_corrected_rank.csv");
        csv << "rank_k,N,M_paths,reference_price,"
               "plain_price,plain_rel_error,corrected_price,corrected_rel_error,"
               "plain_var_lost,plain_frob,plain_incr_var_err,plain_incr_lag1_err,"
               "corrected_frob,corrected_incr_var_err,corrected_incr_lag1_err,"
               "plain_mc_time_s,corrected_mc_time_s\n";
        std::cout << "── (a) variance correction vs rank (N=" << N << ", M=" << M
                  << ", reference " << p_ref << ") ──\n";
        for (int k : quick ? std::vector<int>{2, 8, 32} : std::vector<int>{2, 4, 8, 16, 32, 64, 128}) {
            auto plain = lowrank::price_timed(N, M, k, /*seed=*/42);
            auto corr = lowrank::price_corrected_timed(N, M, k, /*seed=*/42);
            Eigen::MatrixXd Lk = lowrank::lowrank_factor(C, k, 42);
            Eigen::VectorXd d = lowrank::residual_sd(C, Lk);
            Eigen::MatrixXd Cov_plain = Lk * Lk.transpose();
            Eigen::MatrixXd Cov_corr = Cov_plain;
            Cov_corr.diagonal() += d.cwiseAbs2();
            Structure sp = structure_errors(C, Cov_plain), sc = structure_errors(C, Cov_corr);
            csv << k << "," << N << "," << M << "," << p_ref << ","
                << plain.price << "," << (plain.price - p_ref) / p_ref << ","
                << corr.price << "," << (corr.price - p_ref) / p_ref << ","
                << sp.var_lost << "," << sp.frob << "," << sp.incr_var_err << "," << sp.incr_lag1_err << ","
                << sc.frob << "," << sc.incr_var_err << "," << sc.incr_lag1_err << ","
                << plain.t_mc << "," << corr.t_mc << "\n";
            std::cout << std::fixed << std::setprecision(4)
                      << "  k=" << std::setw(3) << k
                      << "  plain " << 100 * (plain.price - p_ref) / p_ref << "%"
                      << "  corrected " << 100 * (corr.price - p_ref) / p_ref << "%"
                      << "  incr var err plain " << 100 * sp.incr_var_err
                      << "% corrected " << 100 * sc.incr_var_err << "%\n";
        }
    }

    // ── (b) Control variate per sampler ─────────────────────────────────────
    {
        const int M = quick ? 5000 : M_PATHS, RANK_K = 32;
        const double target = 1e-3;  // relative standard error
        std::ofstream csv("benchmarks/results/control_variate.csv");
        csv << "method,N,M_paths,price_plain,se_plain,price_cv,se_cv,beta,variance_reduction,"
               "construction_time_s,mc_time_plain_s,mc_time_cv_s,"
               "time_to_target_plain_s,time_to_target_cv_s\n";
        std::cout << "\n── (b) control variate (M=" << M << ", target relative SE "
                  << target * 100 << "%) ──\n";
        for (int N : quick ? std::vector<int>{N_SMALL} : std::vector<int>{N_SMALL, N_LARGE}) {
            struct Row { std::string name; double t_plain_mc; CVResult cv; };
            std::vector<Row> rows;
            rows.push_back({ "cholesky", cholesky::price_timed(N, M).t_mc, cholesky::price_cv(N, M) });
            rows.push_back({ "fft", fft_pricer::price_timed(N, M).t_mc, fft_pricer::price_cv(N, M) });
            rows.push_back({ "rsvd", lowrank::price_timed(N, M, RANK_K).t_mc,
                             lowrank::price_cv(N, M, RANK_K, 42, false) });
            rows.push_back({ "rsvd_corrected", lowrank::price_corrected_timed(N, M, RANK_K).t_mc,
                             lowrank::price_cv(N, M, RANK_K, 42, true) });
            for (const Row& row : rows) {
                const CVResult& c = row.cv;
                double vr = (c.se_plain / c.se) * (c.se_plain / c.se);
                // Paths needed for the target, scaled from the observed SE; the pilot is
                // included in the CV per-path cost
                double M_plain = M * std::pow(c.se_plain / (target * c.price_plain), 2);
                double M_cv = M * std::pow(c.se / (target * c.price), 2);
                double t_plain = c.t_construct + row.t_plain_mc / M * M_plain;
                double t_cv = c.t_construct + c.t_mc / M * M_cv;
                csv << row.name << "," << N << "," << M << ","
                    << c.price_plain << "," << c.se_plain << "," << c.price << "," << c.se << ","
                    << c.beta << "," << vr << "," << c.t_construct << ","
                    << row.t_plain_mc << "," << c.t_mc << "," << t_plain << "," << t_cv << "\n";
                std::cout << std::fixed << std::setprecision(4)
                          << "  N=" << std::setw(4) << N << "  " << std::setw(15) << row.name
                          << "  plain " << c.price_plain << " ± " << c.se_plain
                          << "   cv " << c.price << " ± " << c.se
                          << "   VR " << std::setprecision(1) << vr
                          << "   time to 0.1%: " << std::setprecision(2) << t_plain << " s -> "
                          << t_cv << " s\n";
            }
        }
    }
    std::cout << "\nResults written to benchmarks/results/{variance_corrected_rank,control_variate}.csv\n";
}
