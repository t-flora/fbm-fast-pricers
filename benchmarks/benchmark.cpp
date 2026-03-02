// Unified benchmark: times all three pricers, exports CSV results.
//
// "Exact" methods: Cholesky and FFT both produce exact fBM samples.
// "Approximate" method: H-matrix + rSVD — truncation error controlled by rank k.
//
// Two accuracy metrics for H-matrix:
//   1. Frobenius: ||C - Ck||_F / ||C||_F  (exact, no Monte Carlo noise)
//   2. Price error: |p_hmatrix - p_reference|  (high-accuracy reference from 500k paths)
//
// Outputs:
//   benchmarks/results/time_vs_N.csv
//   benchmarks/results/error_vs_rank.csv
//   benchmarks/results/reference_price.txt

#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include "cholesky/cholesky.hpp"
#include "fft/fft.hpp"
#include "hmatrix/hmatrix.hpp"
#include "hmatrix/rsvd.hpp"
#include "common/params.hpp"
#include "common/covariance.hpp"

using Clock = std::chrono::high_resolution_clock;
static double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int main() {
    using namespace params;

    const std::vector<int> Ns    = {N_SMALL, N_MEDIUM, N_LARGE};
    const std::vector<int> ranks = {2, 4, 8, 16, 32, 64, 128};

    std::ofstream csv_time("benchmarks/results/time_vs_N.csv");
    csv_time << "method,N,M_paths,wall_time_s,price\n";

    std::ofstream csv_err("benchmarks/results/error_vs_rank.csv");
    csv_err << "rank_k,N,reference_price,hmatrix_price,abs_price_error,"
               "rel_price_error,frob_error,construction_time_s,mc_time_s\n";

    // ── Wall-clock time vs N ─────────────────────────────────────────────────
    for (int N : Ns) {
        std::cout << "\n── N = " << N << " (M=" << M_PATHS << " paths) ──\n";

        auto t0 = Clock::now();
        double p_chol = cholesky::price(N, M_PATHS);
        double t_chol = elapsed_s(t0);
        csv_time << "cholesky," << N << "," << M_PATHS << "," << t_chol << "," << p_chol << "\n";
        std::cout << "  cholesky : price=" << p_chol << "  time=" << t_chol << "s\n";

        t0 = Clock::now();
        double p_fft = fft_pricer::price(N, M_PATHS);
        double t_fft = elapsed_s(t0);
        csv_time << "fft," << N << "," << M_PATHS << "," << t_fft << "," << p_fft << "\n";
        std::cout << "  fft      : price=" << p_fft << "  time=" << t_fft << "s\n";

        t0 = Clock::now();
        double p_hmat = hmatrix::price(N, M_PATHS, 32);
        double t_hmat = elapsed_s(t0);
        csv_time << "hmatrix," << N << "," << M_PATHS << "," << t_hmat << "," << p_hmat << "\n";
        std::cout << "  hmatrix  : price=" << p_hmat << "  time=" << t_hmat << "s\n";
    }

    // ── High-accuracy reference (average of two exact simulators) ────────────
    constexpr int M_REF = 500000;
    std::cout << "\n── Computing reference price (N=" << N_MEDIUM
              << ", M=" << M_REF << " per method) ──\n";

    auto t0 = Clock::now();
    double p_chol_ref = cholesky::price(N_MEDIUM, M_REF, /*seed=*/1001);
    std::cout << "  Cholesky  (" << M_REF << " paths): " << p_chol_ref
              << "  (" << elapsed_s(t0) << "s)\n";

    t0 = Clock::now();
    double p_fft_ref = fft_pricer::price(N_MEDIUM, M_REF, /*seed=*/1002);
    std::cout << "  FFT       (" << M_REF << " paths): " << p_fft_ref
              << "  (" << elapsed_s(t0) << "s)\n";

    double p_ref = 0.5 * (p_chol_ref + p_fft_ref);
    std::cout << "  Reference (avg): " << p_ref << "\n";
    {
        std::ofstream f("benchmarks/results/reference_price.txt");
        f << "# Reference price: average of two exact simulators\n"
          << "# N=" << N_MEDIUM << ", M=" << M_REF << " paths each\n"
          << "p_cholesky=" << p_chol_ref << "\np_fft=" << p_fft_ref
          << "\nreference_price=" << p_ref << "\n";
    }

    // ── Error vs rSVD rank ───────────────────────────────────────────────────
    // Precompute the full covariance matrix once for Frobenius norm comparison.
    std::cout << "\n── Building C(" << N_MEDIUM << "x" << N_MEDIUM << ") for Frobenius norm ──\n";
    Eigen::MatrixXd C_full = build_fbm_cov_matrix(N_MEDIUM, H, T);
    double frob_C = C_full.norm();

    std::cout << "\n── Error vs rank (N=" << N_MEDIUM << ", M=" << M_PATHS << ") ──\n";
    std::cout << std::left << std::setw(8)  << "rank_k"
              << std::setw(12) << "frob_err%"
              << std::setw(12) << "|price_err|"
              << std::setw(12) << "rel_price%"
              << "time\n";

    for (int k : ranks) {
        if (k > N_MEDIUM) break;

        // Frobenius norm of approximation error (exact, no MC noise)
        t0 = Clock::now();
        RSVD decomp = rsvd(C_full, k, 5, 2, /*seed=*/42);
        Eigen::MatrixXd Ck = decomp.U * decomp.S.asDiagonal() * decomp.Vt;
        double frob_err = (C_full - Ck).norm() / frob_C;
        double t_construct = elapsed_s(t0);

        // Price approximation error
        t0 = Clock::now();
        double p_approx = hmatrix::price(N_MEDIUM, M_PATHS, k, /*seed=*/42);
        double t_mc = elapsed_s(t0);

        double abs_err = std::abs(p_approx - p_ref);
        double rel_err = abs_err / p_ref;

        csv_err << k << "," << N_MEDIUM << "," << p_ref << ","
                << p_approx << "," << abs_err << "," << rel_err << ","
                << frob_err << "," << t_construct << "," << t_mc << "\n";

        std::cout << std::setw(8)  << k
                  << std::setw(12) << frob_err * 100
                  << std::setw(12) << abs_err
                  << std::setw(12) << rel_err * 100
                  << (t_construct + t_mc) << "s\n";
    }

    csv_time.close();
    csv_err.close();
    std::cout << "\nResults written to benchmarks/results/\n";
}
