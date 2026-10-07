#pragma once
// Batched, multithreaded Monte Carlo driver shared by the three samplers (TODO items 4-6).
//
// Paths are generated in blocks of B. Each sampler supplies a worker that fills an N x B
// matrix of log-volatility paths at once, so its core operation becomes a matrix-matrix
// product (Cholesky: triangular L Z; low-rank: L_k Z) or a batch of FFTs, instead of one
// mat-vec or transform per path. A product L Z reads each element of L once per block
// rather than once per path, which removes the memory-bandwidth bound of the per-path loop.
//
// Blocks are also the unit of parallel work. Block b always draws from its own stream,
// seeded by (seed, b), and per-block sums are reduced in block order, so the price is
// bitwise identical for any number of threads. It does depend on B (blocks own different
// paths) and on the generator.
//
// Opt-in, like the other extensions: price() and price_timed() keep their original
// per-path loop and std::mt19937 stream, so the main benchmark CSVs stay valid.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <random>
#include <thread>
#include <vector>
#include <Eigen/Dense>
#include "common/fast_rng.hpp"
#include "common/params.hpp"

namespace batched {

enum class Rng { Std, Fast };

struct Config {
    int batch = 64;       // paths per block (1 = unbatched; the FFT pairs paths, so use 2)
    int threads = 1;      // 0 = std::thread::hardware_concurrency()
    Rng rng = Rng::Fast;  // Std: mt19937 + normal_distribution, Fast: xoshiro256++ + ziggurat
};

struct Result { double price, se, t_construct, t_mc; };

// Normal sources, one per block. Both expose fill(out, n).
class StdNormal {
public:
    StdNormal(uint32_t seed, uint32_t stream) {
        std::seed_seq seq{seed, stream};
        g_.seed(seq);
    }
    void fill(double* out, int n) {
        for (int j = 0; j < n; ++j) out[j] = d_(g_);
    }

private:
    std::mt19937 g_;
    std::normal_distribution<double> d_{0.0, 1.0};
};
using fastrng::FastNormal;

// Arithmetic Asian call payoff of one path, without allocating. Same arithmetic, in the
// same order, as asian_call_payoff(log_vol_to_prices(...)), so the results are identical.
inline double asian_payoff(const double* log_vol, const double* Z, int N,
                           double S0, double r, double dt, double sigma0, double K) {
    double S = S0, sum = 0.0;
    for (int n = 0; n < N; ++n) {
        double sigma = sigma0 * std::exp(log_vol[n]);
        S *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z[n]);
        sum += S;
    }
    return std::max(sum / N - K, 0.0);
}

inline int resolve_threads(int threads) {
    if (threads > 0) return threads;
    return std::max(1u, std::thread::hardware_concurrency());
}

// Runs the Monte Carlo loop. make_worker() returns a std::unique_ptr to a per-thread worker
// with   template <class Src> void log_vol(Src& src, int n, Eigen::MatrixXd& LV)
// that writes nu * W^H for n paths into the first n columns of LV (N x B). Workers are
// created here, on the calling thread, because FFTW planning is not thread-safe.
template <class Src, class MakeWorker>
Result run_with(int N, int M_paths, unsigned seed, const Config& cfg, MakeWorker& make_worker) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;
    const double dt = T / N;
    const int B = std::max(1, cfg.batch);
    const int n_blocks = (M_paths + B - 1) / B;
    const int n_threads = std::min(resolve_threads(cfg.threads), n_blocks);

    std::vector<decltype(make_worker())> workers;
    for (int t = 0; t < n_threads; ++t) workers.push_back(make_worker());

    std::vector<double> sums(n_blocks), sums2(n_blocks);
    std::atomic<int> next_block{0};
    auto body = [&](int t) {
        auto& worker = *workers[t];
        Eigen::MatrixXd LV(N, B), Z(N, B);
        for (int b; (b = next_block.fetch_add(1)) < n_blocks;) {
            const int n = std::min(B, M_paths - b * B);
            Src src(seed, static_cast<uint32_t>(b));
            worker.log_vol(src, n, LV);
            src.fill(Z.data(), N * n);  // price shocks, after the volatility draws
            double s = 0.0, s2 = 0.0;
            for (int j = 0; j < n; ++j) {
                double p = asian_payoff(LV.col(j).data(), Z.col(j).data(), N, S0, r, dt, sigma0, K);
                s += p;
                s2 += p * p;
            }
            sums[b] = s;
            sums2[b] = s2;
        }
    };

    auto t0 = Clock::now();
    if (n_threads == 1) {
        body(0);
    } else {
        std::vector<std::thread> pool;
        for (int t = 0; t < n_threads; ++t) pool.emplace_back(body, t);
        for (auto& th : pool) th.join();
    }
    double sum = 0.0, sum2 = 0.0;
    for (int b = 0; b < n_blocks; ++b) { sum += sums[b]; sum2 += sums2[b]; }
    double t_mc = std::chrono::duration<double>(Clock::now() - t0).count();

    const double disc = std::exp(-r * T), mean = sum / M_paths;
    const double var = (sum2 - M_paths * mean * mean) / (M_paths - 1);
    return { disc * mean, disc * std::sqrt(var / M_paths), 0.0, t_mc };
}

template <class MakeWorker>
Result run(int N, int M_paths, unsigned seed, const Config& cfg, MakeWorker make_worker) {
    return cfg.rng == Rng::Fast ? run_with<FastNormal>(N, M_paths, seed, cfg, make_worker)
                                : run_with<StdNormal>(N, M_paths, seed, cfg, make_worker);
}

}  // namespace batched
