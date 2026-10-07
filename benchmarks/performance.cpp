// Performance benchmark for the batched, multithreaded pricers (TODO.md items 4-6).
// Separate from benchmark.cpp so the main results stay untouched.
//
// Control variables: model parameters (params.hpp), seed 42, rank k = 32 for the
// low-rank sampler, M paths per run (below). Independent variables: N, block size B,
// generator (std = mt19937 + normal_distribution, fast = xoshiro256++ + ziggurat),
// thread count. Dependent variables: Monte Carlo time (median of REPS repeats, with
// interquartile range), per-path cost of each stage, price and standard error.
//
// (a) Per-path cost breakdown: drawing the normals (std vs fast), the transform (mat-vec,
//     FFT or L_k z; unbatched vs batched; normals replayed from a buffer), the price path
//     and payoff.                                                  → perf_breakdown.csv
// (b) Time vs N for four configurations of each sampler, one thread:
//       legacy          the per-path loop of price_timed() (as in time_vs_N.csv)
//       unbatched_fast  B = 1 (FFT: 2, one transform), fast generator
//       batched_std     B = 64, std generator
//       batched_fast    B = 64, fast generator                     → perf_batched.csv
// (c) MC time vs block size B, fast generator, one thread.         → perf_batch_size.csv
// (d) Thread scaling, fast generator, unbatched and B = 64.        → perf_threads.csv
//
// Usage: ./build/performance [--quick]   (--quick: small N, few paths, one repeat)

#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <Eigen/Dense>

#include "cholesky/cholesky.hpp"
#include "fft/fft.hpp"
#include "rsvd/lowrank.hpp"
#include "common/batched_mc.hpp"
#include "common/params.hpp"
#include "timing.hpp"

static constexpr int RANK_K = 32;
static constexpr int BATCH = 64;
static const char* METHODS[] = { "cholesky", "fft", "rsvd" };

// Smallest block that is "unbatched" for each sampler: the FFT always pairs two paths
static int unbatched(const std::string& method) { return method == "fft" ? 2 : 1; }

static batched::Result run_batched(const std::string& method, int N, int M, const batched::Config& cfg) {
    if (method == "cholesky") return cholesky::price_batched(N, M, cfg);
    if (method == "fft") return fft_pricer::price_batched(N, M, cfg);
    return lowrank::price_batched(N, M, RANK_K, cfg);
}

static batched::Result run_legacy(const std::string& method, int N, int M) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (method == "cholesky") { auto r = cholesky::price_timed(N, M); return { r.price, nan, r.t_construct, r.t_mc }; }
    if (method == "fft") { auto r = fft_pricer::price_timed(N, M); return { r.price, nan, r.t_construct, r.t_mc }; }
    auto r = lowrank::price_timed(N, M, RANK_K);
    return { r.price, nan, r.t_construct, r.t_mc };
}

static double mc_time(const batched::Result& r) { return r.t_mc; }

// Median over 3 runs of the seconds per call of f(), each run lasting at least `budget`
template <class F>
static double seconds_per_call(F f, double budget) {
    std::vector<double> t;
    for (int rep = 0; rep < 3; ++rep) {
        long n = 0;
        auto t0 = Clock::now();
        do { f(); ++n; } while (elapsed_s(t0) < budget);
        t.push_back(elapsed_s(t0) / n);
    }
    std::sort(t.begin(), t.end());
    return t[1];
}

// Normal source that replays a fixed buffer, so a worker's log_vol() can be timed
// without the cost of generating its normals
struct ReplaySrc {
    const std::vector<double>& buf;
    void fill(double* out, int n) { std::copy(buf.begin(), buf.begin() + n, out); }
};

// Seconds per path of a worker's log_vol() on blocks of n paths (normals replayed)
template <class Worker>
static double transform_per_path(Worker& worker, int N, int n, const std::vector<double>& normals,
                                 double budget) {
    Eigen::MatrixXd LV(N, n);
    ReplaySrc src{normals};
    return seconds_per_call([&] { worker.log_vol(src, n, LV); }, budget) / n;
}

int main(int argc, char** argv) {
    using namespace params;
    const bool quick = has_flag(argc, argv, "--quick");
    const int REPS = quick ? 1 : 3;
    const double budget = quick ? 0.02 : 0.15;  // seconds per breakdown measurement
    const std::vector<int> Ns = quick ? std::vector<int>{64, 252, 500}
                                      : std::vector<int>{64, 128, N_SMALL, N_MEDIUM, N_LARGE, 2000, 4000};
    const int M = quick ? 2000 : M_PATHS;
    std::printf("Mode: %s\n", quick ? "QUICK" : "FULL");

    // ── (a) Per-path cost breakdown ──────────────────────────────────────────
    {
        std::ofstream csv("benchmarks/results/perf_breakdown.csv");
        csv << "method,N,normals_per_path,rng_std_ns,rng_fast_ns,transform_unbatched_ns,"
               "transform_batched_ns,payoff_ns\n";
        std::printf("\n── (a) per-path cost breakdown (ns per path) ──\n");
        for (int N : quick ? std::vector<int>{252} : std::vector<int>{N_SMALL, N_LARGE, 4000}) {
            const double dt = T / N;
            std::vector<double> normals(static_cast<size_t>(4 * N) * BATCH);
            fastrng::FastNormal(1, 0).fill(normals.data(), static_cast<int>(normals.size()));
            const Eigen::MatrixXd F = cholesky::fbm_cholesky_factor(N, H, T);
            const std::vector<double> scale = fft_pricer::circulant_scale(N, H, dt);
            Eigen::MatrixXd Lk;
            { Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T); Lk = lowrank::lowrank_factor(C, RANK_K, 42); }

            // Price path + payoff, shared by every sampler
            Eigen::MatrixXd LV(N, 1);
            ReplaySrc src{normals};
            { cholesky::BatchWorker w(F, 1); w.log_vol(src, 1, LV); }
            double payoff_s = seconds_per_call([&] {
                volatile double p = batched::asian_payoff(LV.data(), normals.data(), N, S0, r, dt, sigma0, K);
                (void)p;
            }, budget);

            for (const std::string method : METHODS) {
                // Normals per path: volatility draws + N price shocks
                const int n_vol = method == "cholesky" ? N : method == "fft" ? 2 * N : RANK_K;
                const int D = n_vol + N;
                std::vector<double> out(D);
                batched::StdNormal s_std(1, 0);
                fastrng::FastNormal s_fast(1, 0);
                double rng_std = seconds_per_call([&] { s_std.fill(out.data(), D); }, budget);
                double rng_fast = seconds_per_call([&] { s_fast.fill(out.data(), D); }, budget);

                double t_unb, t_bat;
                const int u = unbatched(method);
                if (method == "cholesky") {
                    cholesky::BatchWorker w1(F, u), w2(F, BATCH);
                    t_unb = transform_per_path(w1, N, u, normals, budget);
                    t_bat = transform_per_path(w2, N, BATCH, normals, budget);
                } else if (method == "fft") {
                    fft_pricer::BatchWorker w1(scale, u), w2(scale, BATCH);
                    t_unb = transform_per_path(w1, N, u, normals, budget);
                    t_bat = transform_per_path(w2, N, BATCH, normals, budget);
                } else {
                    lowrank::BatchWorker w1(Lk, u), w2(Lk, BATCH);
                    t_unb = transform_per_path(w1, N, u, normals, budget);
                    t_bat = transform_per_path(w2, N, BATCH, normals, budget);
                }
                csv << method << "," << N << "," << D << "," << rng_std * 1e9 << "," << rng_fast * 1e9
                    << "," << t_unb * 1e9 << "," << t_bat * 1e9 << "," << payoff_s * 1e9 << "\n";
                std::printf("  %-8s N=%4d  rng std %8.0f fast %7.0f | transform unbatched %8.0f batched %8.0f"
                            " | payoff %6.0f\n", method.c_str(), N, rng_std * 1e9, rng_fast * 1e9,
                            t_unb * 1e9, t_bat * 1e9, payoff_s * 1e9);
            }
        }
    }

    // ── (b) Time vs N, four configurations ──────────────────────────────────
    {
        std::ofstream csv("benchmarks/results/perf_batched.csv");
        csv << "method,N,M_paths,config,batch,rng,threads,price,se,construction_time_s,"
               "mc_time_s,mc_time_q1_s,mc_time_q3_s\n";
        std::printf("\n── (b) MC time vs N, one thread (M=%d, median of %d) ──\n", M, REPS);
        for (int N : Ns) {
            for (const std::string method : METHODS) {
                struct Cfg { const char* name; int batch; batched::Rng rng; };
                const Cfg cfgs[] = { { "unbatched_fast", unbatched(method), batched::Rng::Fast },
                                     { "batched_std", BATCH, batched::Rng::Std },
                                     { "batched_fast", BATCH, batched::Rng::Fast } };
                auto rl = repeat_timed(REPS, [&] { return run_legacy(method, N, M); }, mc_time);
                csv << method << "," << N << "," << M << ",legacy,," << "std,1," << rl.median.price << ",,"
                    << rl.median.t_construct << "," << rl.median.t_mc << "," << rl.q1 << "," << rl.q3 << "\n";
                std::printf("  %-8s N=%4d  legacy %8.3fs", method.c_str(), N, rl.median.t_mc);
                for (const Cfg& c : cfgs) {
                    batched::Config cfg{ c.batch, 1, c.rng };
                    auto rb = repeat_timed(REPS, [&] { return run_batched(method, N, M, cfg); }, mc_time);
                    csv << method << "," << N << "," << M << "," << c.name << "," << c.batch << ","
                        << (c.rng == batched::Rng::Fast ? "fast" : "std") << ",1," << rb.median.price << ","
                        << rb.median.se << "," << rb.median.t_construct << "," << rb.median.t_mc << ","
                        << rb.q1 << "," << rb.q3 << "\n";
                    std::printf("  %s %7.3fs", c.name, rb.median.t_mc);
                }
                std::printf("\n");
                csv.flush();
            }
        }
    }

    // ── (c) MC time vs block size ───────────────────────────────────────────
    {
        std::ofstream csv("benchmarks/results/perf_batch_size.csv");
        csv << "method,N,M_paths,batch,price,mc_time_s,mc_time_q1_s,mc_time_q3_s\n";
        const int M_sweep = quick ? 1000 : 4000;
        std::printf("\n── (c) MC time vs block size B (M=%d, fast generator) ──\n", M_sweep);
        for (int N : quick ? std::vector<int>{N_MEDIUM} : std::vector<int>{N_LARGE, 4000}) {
            for (const std::string method : METHODS) {
                std::printf("  %-8s N=%4d ", method.c_str(), N);
                for (int B : {1, 2, 4, 8, 16, 32, 64, 128, 256}) {
                    if (B < unbatched(method) || (quick && B > 64)) continue;
                    batched::Config cfg{ B, 1, batched::Rng::Fast };
                    auto rb = repeat_timed(REPS, [&] { return run_batched(method, N, M_sweep, cfg); }, mc_time);
                    csv << method << "," << N << "," << M_sweep << "," << B << "," << rb.median.price << ","
                        << rb.median.t_mc << "," << rb.q1 << "," << rb.q3 << "\n";
                    std::printf(" B=%d %.0fus", B, rb.median.t_mc / M_sweep * 1e6);
                }
                std::printf("\n");
                csv.flush();
            }
        }
    }

    // ── (d) Thread scaling ──────────────────────────────────────────────────
    {
        std::ofstream csv("benchmarks/results/perf_threads.csv");
        csv << "method,N,M_paths,batch,threads,price,mc_time_s,mc_time_q1_s,mc_time_q3_s\n";
        const int max_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
        std::printf("\n── (d) thread scaling (fast generator; %d hardware threads) ──\n", max_threads);
        for (int N : quick ? std::vector<int>{N_SMALL} : std::vector<int>{N_MEDIUM, 4000}) {
            const int M_thr = quick ? 2000 : (N >= 4000 ? 2000 : M_PATHS);
            for (const std::string method : METHODS) {
                for (int B : {unbatched(method), BATCH}) {
                    std::printf("  %-8s N=%4d B=%2d ", method.c_str(), N, B);
                    for (int th = 1; th <= max_threads; ++th) {
                        if (quick && th != 1 && th != 2 && th != 4) continue;
                        batched::Config cfg{ B, th, batched::Rng::Fast };
                        auto rb = repeat_timed(REPS, [&] { return run_batched(method, N, M_thr, cfg); }, mc_time);
                        csv << method << "," << N << "," << M_thr << "," << B << "," << th << ","
                            << rb.median.price << "," << rb.median.t_mc << "," << rb.q1 << "," << rb.q3 << "\n";
                        std::printf(" %d:%.3fs", th, rb.median.t_mc);
                    }
                    std::printf("\n");
                    csv.flush();
                }
            }
        }
    }
    std::cout << "\nResults written to benchmarks/results/perf_{breakdown,batched,batch_size,threads}.csv\n";
}
