// Unified benchmark: times all three pricers, exports CSV results.
//
// "Exact" methods: Cholesky and FFT both produce exact fBM samples.
// "Approximate" method: H-matrix + rSVD — truncation error controlled by rank k.
//
// Memory metrics (macOS mach API):
//   peak_rss_mb       — resident set size after the call (proxy for peak allocation)
//   theoretical_peak_mb — analytical formula: Cholesky N^2*8, FFT N*8, hmatrix N^2*8 or N*k*8
//   cache_pressure    — peak_rss_mb / L3_MB  (>1 means L3 spill)
//   est_bandwidth_GBs — Cholesky only: bytes_accessed / wall_time_s (memory-bound check)
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
#include <mach/mach.h>

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

// Current resident set size in bytes (macOS mach API).
// Unlike getrusage ru_maxrss (cumulative peak), this reflects the current RSS
// after heap allocations have been freed by RAII / leaving scope.
static size_t current_rss_bytes() {
    mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        return 0;
    return info.resident_size;
}

static double rss_mb() { return current_rss_bytes() / (1024.0 * 1024.0); }

// L3 cache size on this Mac (Apple M2 = 16 MB).
static constexpr double L3_MB = 16.0;
// M2 rated memory bandwidth (GB/s) — from Apple spec sheet.
static constexpr double BANDWIDTH_GBS = 100.0;

int main() {
    using namespace params;

    const std::vector<int> Ns    = {N_SMALL, N_MEDIUM, N_LARGE};
    const std::vector<int> ranks = {2, 4, 8, 16, 32, 64, 128};

    std::ofstream csv_time("benchmarks/results/time_vs_N.csv");
    csv_time << "method,N,M_paths,wall_time_s,price,construction_time_s,mc_time_s,"
                "peak_rss_mb,theoretical_peak_mb,cache_pressure,est_bandwidth_GBs\n";

    std::ofstream csv_err("benchmarks/results/error_vs_rank.csv");
    csv_err << "rank_k,N,reference_price,hmatrix_price,abs_price_error,"
               "rel_price_error,frob_error,construction_time_s,mc_time_s\n";

    // ── Wall-clock time vs N ─────────────────────────────────────────────────
    for (int N : Ns) {
        std::cout << "\n── N = " << N << " (M=" << M_PATHS << " paths) ──\n";

        // Cholesky: peak = N x N lower-triangular L matrix = N*(N+1)/2 doubles
        double rss_before = rss_mb();
        auto rc = cholesky::price_timed(N, M_PATHS);
        double rss_chol = rss_mb();
        double t_chol = rc.t_construct + rc.t_mc;
        double theory_chol_mb = static_cast<double>(N) * N * 8.0 / (1024.0 * 1024.0);
        double cache_chol = theory_chol_mb / L3_MB;
        // Memory-bandwidth proxy: L matrix re-read for each of M paths → N^2 * 8 * M bytes
        double bytes_accessed = static_cast<double>(N) * N * 8.0 * M_PATHS;
        double bw_chol = bytes_accessed / t_chol / 1e9;

        csv_time << "cholesky," << N << "," << M_PATHS << "," << t_chol << "," << rc.price
                 << "," << rc.t_construct << "," << rc.t_mc << ","
                 << (rss_chol - rss_before) << "," << theory_chol_mb << ","
                 << cache_chol << "," << bw_chol << "\n";
        std::cout << "  cholesky : price=" << rc.price << "  time=" << t_chol
                  << "s  theory=" << theory_chol_mb << " MB"
                  << "  bw_est=" << bw_chol << " GB/s\n";

        // FFT: peak = circulant row c_emb (2N doubles) + work arrays ~ O(N)
        rss_before = rss_mb();
        auto rf = fft_pricer::price_timed(N, M_PATHS);
        double rss_fft = rss_mb();
        double t_fft = rf.t_construct + rf.t_mc;
        double theory_fft_mb = static_cast<double>(N) * 2 * 8.0 / (1024.0 * 1024.0);
        double cache_fft = theory_fft_mb / L3_MB;

        csv_time << "fft," << N << "," << M_PATHS << "," << t_fft << "," << rf.price
                 << "," << rf.t_construct << "," << rf.t_mc << ","
                 << (rss_fft - rss_before) << "," << theory_fft_mb << ","
                 << cache_fft << ",\n";
        std::cout << "  fft      : price=" << rf.price << "  time=" << t_fft
                  << "s  theory=" << theory_fft_mb << " MB\n";

        // H-matrix (C held): peak = N x N covariance matrix (O(N^2)) + Lk (N x k)
        constexpr int RANK_K = 32;
        rss_before = rss_mb();
        auto rh = hmatrix::price_timed(N, M_PATHS, RANK_K);
        double rss_hmat = rss_mb();
        double t_hmat = rh.t_construct + rh.t_mc;
        double theory_hmat_mb = static_cast<double>(N) * N * 8.0 / (1024.0 * 1024.0);
        double cache_hmat = theory_hmat_mb / L3_MB;

        csv_time << "hmatrix," << N << "," << M_PATHS << "," << t_hmat << "," << rh.price
                 << "," << rh.t_construct << "," << rh.t_mc << ","
                 << (rss_hmat - rss_before) << "," << theory_hmat_mb << ","
                 << cache_hmat << ",\n";
        std::cout << "  hmatrix  : price=" << rh.price << "  time=" << t_hmat
                  << "s  theory=" << theory_hmat_mb << " MB\n";

        // H-matrix (C freed before MC): peak during MC = Lk only (N x k)
        rss_before = rss_mb();
        auto rhf = hmatrix::price_freed_timed(N, M_PATHS, RANK_K);
        double rss_hmat_f = rss_mb();
        double t_hmat_f = rhf.t_construct + rhf.t_mc;
        // After C freed: only Lk (N * k doubles) remains
        double theory_freed_mb = static_cast<double>(N) * RANK_K * 8.0 / (1024.0 * 1024.0);
        double cache_freed = theory_freed_mb / L3_MB;

        csv_time << "hmatrix_freed," << N << "," << M_PATHS << "," << t_hmat_f << ","
                 << rhf.price << "," << rhf.t_construct << "," << rhf.t_mc << ","
                 << (rss_hmat_f - rss_before) << "," << theory_freed_mb << ","
                 << cache_freed << ",\n";
        std::cout << "  hmat_free: price=" << rhf.price << "  time=" << t_hmat_f
                  << "s  theory_mc=" << theory_freed_mb << " MB\n";
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

        t0 = Clock::now();
        RSVD decomp = rsvd(C_full, k, 5, 2, /*seed=*/42);
        Eigen::MatrixXd Ck = decomp.U * decomp.S.asDiagonal() * decomp.Vt;
        double frob_err = (C_full - Ck).norm() / frob_C;
        double t_construct = elapsed_s(t0);

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
    std::cout << "\nMemory notes:\n"
              << "  Rated M2 bandwidth: " << BANDWIDTH_GBS << " GB/s\n"
              << "  L3 cache: " << L3_MB << " MB  (cache_pressure > 1 => L3 spill)\n"
              << "  hmatrix_freed keeps only Lk (N*k*8 bytes) during MC loop\n";
}
