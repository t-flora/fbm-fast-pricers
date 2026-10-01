// Unified benchmark: times all three pricers, exports CSV results.
//
// "Exact" methods: Cholesky and FFT both produce exact fBM samples.
// "Approximate" method: low-rank rSVD — truncation error controlled by rank k.
//
// Memory metrics:
//   measured_peak_mb  — lifetime peak RSS of one call, measured by running it (with a few
//                       paths; memory does not depend on M) in a forked child and reading
//                       ru_maxrss via wait4, minus the peak of an idle child. POSIX.
//   theoretical_peak_mb — size of the dominant matrix held during the MC loop:
//                       Cholesky N^2*8 (L), FFT 2N*40, rsvd N^2*8 (C held) or N*k*8 (C freed)
//   cache_pressure    — theoretical_peak_mb / CACHE_MB  (>1 means it no longer fits on-chip)
//   est_bandwidth_GBs — Cholesky only: (N(N+1)/2)*8*M bytes / total wall time. An effective streaming
//                       rate for L, not measured DRAM traffic (L fits in cache for N <= 1000).
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
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cholesky/cholesky.hpp"
#include "fft/fft.hpp"
#include "rsvd/lowrank.hpp"
#include "rsvd/rsvd.hpp"
#include "common/params.hpp"
#include "common/covariance.hpp"

using Clock = std::chrono::high_resolution_clock;
static double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Peak RSS (MB) of a forked child that runs f() and exits.
template <class F>
static double child_maxrss_mb(F f) {
    pid_t pid = fork();
    if (pid == 0) { f(); _exit(0); }
    int status = 0;
    struct rusage ru {};
    if (pid < 0 || wait4(pid, &status, 0, &ru) < 0) return 0.0;
#ifdef __APPLE__
    return ru.ru_maxrss / (1024.0 * 1024.0);  // macOS reports bytes
#else
    return ru.ru_maxrss / 1024.0;             // Linux reports kilobytes
#endif
}

// Lifetime peak memory of f() beyond what an idle child (same inherited state) uses.
template <class F>
static double measured_peak_mb(F f) {
    double base = child_maxrss_mb([] {});
    return child_maxrss_mb(f) - base;
}

static constexpr int M_MEMORY = 100;  // paths per memory probe

// Largest on-chip cache on the test machine: Apple M2 performance-cluster L2 = 16 MB
// (`sysctl hw.perflevel0.l2cachesize`). The M2 has no L3; the name is kept for the CSV.
static constexpr double L3_MB = 16.0;
// M2 rated memory bandwidth (GB/s) — from Apple spec sheet.
static constexpr double BANDWIDTH_GBS = 100.0;

int main() {
    using namespace params;

    const std::vector<int> Ns    = {N_SMALL, N_MEDIUM, N_LARGE};
    const std::vector<int> ranks = {2, 4, 8, 16, 32, 64, 128};

    std::ofstream csv_time("benchmarks/results/time_vs_N.csv");
    csv_time << "method,N,M_paths,wall_time_s,price,construction_time_s,mc_time_s,"
                "measured_peak_mb,theoretical_peak_mb,cache_pressure,est_bandwidth_GBs\n";

    std::ofstream csv_err("benchmarks/results/error_vs_rank.csv");
    csv_err << "rank_k,N,reference_price,rsvd_price,abs_price_error,"
               "rel_price_error,frob_error,construction_time_s,mc_time_s\n";

    // ── Wall-clock time vs N ─────────────────────────────────────────────────
    for (int N : Ns) {
        std::cout << "\n── N = " << N << " (M=" << M_PATHS << " paths) ──\n";

        // Cholesky: C is factored in place, so one N x N matrix (N^2 doubles) holds L
        auto rc = cholesky::price_timed(N, M_PATHS);
        double peak_chol = measured_peak_mb([N] { cholesky::price_timed(N, M_MEMORY); });
        double t_chol = rc.t_construct + rc.t_mc;
        double theory_chol_mb = static_cast<double>(N) * N * 8.0 / (1024.0 * 1024.0);
        double cache_chol = theory_chol_mb / L3_MB;
        // Memory-bandwidth proxy: lower triangle of L re-read for each of M paths
        double bytes_accessed = static_cast<double>(N) * (N + 1) / 2.0 * 8.0 * M_PATHS;
        double bw_chol = bytes_accessed / t_chol / 1e9;

        csv_time << "cholesky," << N << "," << M_PATHS << "," << t_chol << "," << rc.price
                 << "," << rc.t_construct << "," << rc.t_mc << ","
                 << peak_chol << "," << theory_chol_mb << ","
                 << cache_chol << "," << bw_chol << "\n";
        std::cout << "  cholesky : price=" << rc.price << "  time=" << t_chol
                  << "s  theory=" << theory_chol_mb << " MB"
                  << "  bw_est=" << bw_chol << " GB/s\n";

        // FFT: during MC the sampler holds the 2N scale factors (doubles) and two 2N complex
        // buffers for the inverse FFT: 2N * (8 + 16 + 16) bytes
        auto rf = fft_pricer::price_timed(N, M_PATHS);
        double peak_fft = measured_peak_mb([N] { fft_pricer::price_timed(N, M_MEMORY); });
        double t_fft = rf.t_construct + rf.t_mc;
        double theory_fft_mb = static_cast<double>(N) * 2 * 40.0 / (1024.0 * 1024.0);
        double cache_fft = theory_fft_mb / L3_MB;

        csv_time << "fft," << N << "," << M_PATHS << "," << t_fft << "," << rf.price
                 << "," << rf.t_construct << "," << rf.t_mc << ","
                 << peak_fft << "," << theory_fft_mb << ","
                 << cache_fft << ",\n";
        std::cout << "  fft      : price=" << rf.price << "  time=" << t_fft
                  << "s  theory=" << theory_fft_mb << " MB\n";

        // rSVD (C held): peak = N x N covariance matrix (O(N^2)) + Lk (N x k)
        constexpr int RANK_K = 32;
        auto rh = lowrank::price_timed(N, M_PATHS, RANK_K);
        double peak_hmat = measured_peak_mb([N] { lowrank::price_timed(N, M_MEMORY, RANK_K); });
        double t_hmat = rh.t_construct + rh.t_mc;
        double theory_hmat_mb = static_cast<double>(N) * N * 8.0 / (1024.0 * 1024.0);
        double cache_hmat = theory_hmat_mb / L3_MB;

        csv_time << "rsvd," << N << "," << M_PATHS << "," << t_hmat << "," << rh.price
                 << "," << rh.t_construct << "," << rh.t_mc << ","
                 << peak_hmat << "," << theory_hmat_mb << ","
                 << cache_hmat << ",\n";
        std::cout << "  rsvd     : price=" << rh.price << "  time=" << t_hmat
                  << "s  theory=" << theory_hmat_mb << " MB\n";

        // rSVD (C freed before MC): peak during MC = Lk only (N x k)
        auto rhf = lowrank::price_freed_timed(N, M_PATHS, RANK_K);
        double peak_hmat_f = measured_peak_mb([N] { lowrank::price_freed_timed(N, M_MEMORY, RANK_K); });
        double t_hmat_f = rhf.t_construct + rhf.t_mc;
        // After C freed: only Lk (N * k doubles) remains
        double theory_freed_mb = static_cast<double>(N) * RANK_K * 8.0 / (1024.0 * 1024.0);
        double cache_freed = theory_freed_mb / L3_MB;

        csv_time << "rsvd_freed," << N << "," << M_PATHS << "," << t_hmat_f << ","
                 << rhf.price << "," << rhf.t_construct << "," << rhf.t_mc << ","
                 << peak_hmat_f << "," << theory_freed_mb << ","
                 << cache_freed << ",\n";
        std::cout << "  rsvd_free: price=" << rhf.price << "  time=" << t_hmat_f
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
        double p_approx = lowrank::price(N_MEDIUM, M_PATHS, k, /*seed=*/42);
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
              << "  Largest cache (M2 P-cluster L2): " << L3_MB << " MB  (cache_pressure > 1 => spill)\n"
              << "  rsvd_freed keeps only Lk (N*k*8 bytes) during MC loop\n";
}
