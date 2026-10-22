#pragma once
// Block 3: Circulant Embedding + FFTW path generator  (Davies-Harte / Wood-Chan)
// O(N log N) per path.
//
// Key insight: fBM itself is non-stationary, so its covariance matrix is NOT Toeplitz.
// However, fBM *increments* (fractional Gaussian noise, fGn) ARE stationary, so their
// covariance IS Toeplitz and embeds into a circulant. For H ≤ ½ the fGn autocovariance is
// non-positive at every nonzero lag and the embedding is PSD (Craigmile 2003); the smallest
// eigenvalue is λ_0 = dt^{2H} (N^{2H} − (N−1)^{2H}) > 0, so sampling is exact.
//
// Algorithm:
//   1. Compute fGn autocovariance γ(k) = (dt^{2H}/2)*((k+1)^{2H} - 2k^{2H} + (k-1)^{2H})
//   2. Embed into 2N circulant: c = [γ(0)..γ(N-1), 0, γ(N-1)..γ(1)]
//   3. FFT(c) → eigenvalues λ (all > 0; we throw rather than clip if any is negative)
//   4. Per transform: w[j] = sqrt(λ[j]/M) * (a+ib); Re and Im of IFFT(w) are two
//      independent fGn paths (first N entries each)
//   5. log_vol = cumsum(x[0..N-1])  → fBM path; then simulate GBM prices

#include <fftw3.h>
#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <complex>
#include <memory>
#include <random>
#include <vector>
#include <cmath>
#include <stdexcept>
#include "common/params.hpp"
#include "common/asian_payoff.hpp"
#include "common/rng.hpp"
#include "common/control_variate.hpp"
#include "common/batched_mc.hpp"

namespace fft_pricer {

// fGn autocovariance at lag k (Var = dt^{2H}, negative for H<0.5 at k≥1)
inline double fgn_cov(int k, double H, double dt) {
    double h2 = 2.0 * H;
    if (k == 0) return std::pow(dt, h2);
    double km1 = (k == 1) ? 0.0 : std::pow(k - 1.0, h2);
    return 0.5 * std::pow(dt, h2) * (std::pow(k + 1.0, h2) + km1 - 2.0 * std::pow(k, h2));
}

// Eigenvalues of the 2N circulant embedding of the fGn autocovariance.
// Throws if any is negative (the embedding would not be a valid covariance).
inline std::vector<double> circulant_eigenvalues(int N, double H, double dt) {
    int M = 2 * N;
    // Complex arrays for a full c2c transform (avoids Hermitian bookkeeping)
    std::vector<std::complex<double>> c_emb(M, 0.0), lam(M);
    for (int j = 0; j < N; ++j)
        c_emb[j] = fgn_cov(j, H, dt);
    for (int j = 1; j < N; ++j)
        c_emb[M - j] = c_emb[j];  // symmetric reflection; c_emb[N] stays 0

    fftw_plan p = fftw_plan_dft_1d(
        M,
        reinterpret_cast<fftw_complex*>(c_emb.data()),
        reinterpret_cast<fftw_complex*>(lam.data()),
        FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    // Imaginary parts are ≈ 0 (real symmetric input)
    std::vector<double> out(M);
    for (int k = 0; k < M; ++k) {
        if (lam[k].real() < -1e-8)
            throw std::runtime_error("fGn circulant embedding not PSD");
        out[k] = lam[k].real();
    }
    return out;
}

// Draws fBM paths nu * W^H(t_1..t_N) by circulant embedding. The eigenvalue scaling,
// the inverse-FFT plan and its buffers are set up once and reused for every path.
class FbmSampler {
public:
    FbmSampler(int N, double H, double dt)
        : N_(N), M_(2 * N), scale_(M_), w_(M_), out_(M_)
    {
        std::vector<double> lam = circulant_eigenvalues(N, H, dt);
        for (int j = 0; j < M_; ++j)
            scale_[j] = std::sqrt(std::max(lam[j], 0.0) / M_);
        plan_ = fftw_plan_dft_1d(
            M_,
            reinterpret_cast<fftw_complex*>(w_.data()),
            reinterpret_cast<fftw_complex*>(out_.data()),
            FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    ~FbmSampler() { fftw_destroy_plan(plan_); }
    FbmSampler(const FbmSampler&) = delete;
    FbmSampler& operator=(const FbmSampler&) = delete;

    // Writes one path into path[0..N-1] (the real part of one transform).
    // w[j] = sqrt(λ[j] / M) * (a + ib)  →  Cov(Re(IFFT(w))) = Toeplitz(γ)
    void sample(std::mt19937& rng, std::normal_distribution<double>& norm,
                double nu, std::vector<double>& path)
    {
        transform(rng, norm);
        cumsum(nu, path, /*imag=*/false);
    }

    // Writes two independent paths from ONE transform: the real and imaginary parts
    // each have the fGn covariance, and Cov(Re, Im) = 0 because λ_j = λ_{M-j}.
    // Halves the Gaussian draws per path, which dominate the per-path cost.
    void sample_pair(std::mt19937& rng, std::normal_distribution<double>& norm,
                     double nu, std::vector<double>& path_re, std::vector<double>& path_im)
    {
        transform(rng, norm);
        cumsum(nu, path_re, /*imag=*/false);
        cumsum(nu, path_im, /*imag=*/true);
    }

private:
    void transform(std::mt19937& rng, std::normal_distribution<double>& norm) {
        for (int j = 0; j < M_; ++j) {
            double a = norm(rng);
            double b = norm(rng);
            w_[j] = scale_[j] * std::complex<double>(a, b);
        }
        fftw_execute(plan_);
    }

    // out_[i] for i=0..N-1 are fGn increments (FFTW unnormalized IFFT);
    // their cumulative sum is the fBM path
    void cumsum(double nu, std::vector<double>& path, bool imag) const {
        double acc = 0.0;
        for (int i = 0; i < N_; ++i) {
            acc += imag ? out_[i].imag() : out_[i].real();
            path[i] = nu * acc;
        }
    }

    int N_, M_;
    std::vector<double> scale_;
    std::vector<std::complex<double>> w_, out_;
    fftw_plan plan_;
};

// Construction vs MC timing breakdown
struct FFTTimed { double price, t_construct, t_mc; };

inline FFTTimed price_timed(int N, int M_paths, unsigned seed = 42) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;
    double dt = T / N;

    auto t0 = Clock::now();
    FbmSampler sampler(N, H, dt);  // eigenvalues + IFFT plan
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();

    auto rng = make_rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> log_vol_a(N), log_vol_b(N);
    double payoff_sum = 0.0;
    auto add_path = [&](const std::vector<double>& log_vol) {
        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt, sigma0), K);
    };

    t0 = Clock::now();
    int m = 0;
    for (; m + 1 < M_paths; m += 2) {      // two fBM paths per inverse FFT
        sampler.sample_pair(rng, norm, nu, log_vol_a, log_vol_b);
        add_path(log_vol_a);
        add_path(log_vol_b);
    }
    if (m < M_paths) {                       // odd M: one last single path
        sampler.sample(rng, norm, nu, log_vol_a);
        add_path(log_vol_a);
    }
    double t_mc = std::chrono::duration<double>(Clock::now() - t0).count();
    return { std::exp(-r * T) * payoff_sum / M_paths, t_construct, t_mc };
}

inline double price(int N, int M_paths, unsigned seed = 42) {
    return price_timed(N, M_paths, seed).price;
}

// Same sampler (two paths per inverse FFT), priced with the conditional geometric control
// variate (common/control_variate.hpp).
inline CVResult price_cv(int N, int M_paths, unsigned seed = 42) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;
    double dt = T / N;
    auto t0 = Clock::now();
    FbmSampler sampler(N, H, dt);
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();

    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> spare(N);
    bool have_spare = false;
    auto next_path = [&](std::mt19937& rng, std::vector<double>& log_vol) {
        if (have_spare) {
            log_vol.swap(spare);
            have_spare = false;
        } else {
            sampler.sample_pair(rng, norm, nu, log_vol, spare);
            have_spare = true;
        }
    };
    CVResult res = mc_control_variate(next_path, N, M_paths, seed);
    res.t_construct = t_construct;
    return res;
}

// Batched worker (common/batched_mc.hpp): ceil(B/2) inverse FFTs per block in one
// fftw_plan_many_dft call; the real and imaginary part of each give two paths, as in
// FbmSampler::sample_pair. Construct on one thread at a time (FFTW planning is not
// thread-safe); log_vol() may then run concurrently on different workers.
class BatchWorker {
public:
    BatchWorker(const std::vector<double>& scale, int batch)
        : N_(static_cast<int>(scale.size()) / 2), M_(static_cast<int>(scale.size())),
          howmany_((std::max(1, batch) + 1) / 2), scale_(scale)
    {
        const size_t len = static_cast<size_t>(M_) * howmany_;
        in_ = fftw_alloc_complex(len);
        out_ = fftw_alloc_complex(len);
        plan_ = fftw_plan_many_dft(1, &M_, howmany_, in_, nullptr, 1, M_,
                                   out_, nullptr, 1, M_, FFTW_BACKWARD, FFTW_ESTIMATE);
    }
    ~BatchWorker() {
        fftw_destroy_plan(plan_);
        fftw_free(in_);
        fftw_free(out_);
    }
    BatchWorker(const BatchWorker&) = delete;
    BatchWorker& operator=(const BatchWorker&) = delete;

    template <class Src>
    void log_vol(Src& src, int n, Eigen::MatrixXd& LV) {
        const int pairs = (n + 1) / 2;
        for (int t = 0; t < pairs; ++t) {
            fftw_complex* w = in_ + static_cast<size_t>(t) * M_;
            src.fill(&w[0][0], 2 * M_);  // (a_j, b_j) interleaved
            for (int j = 0; j < M_; ++j) {
                w[j][0] *= scale_[j];
                w[j][1] *= scale_[j];
            }
        }
        fftw_execute(plan_);  // transforms past `pairs` (last block only) are discarded
        for (int t = 0; t < pairs; ++t) {
            const fftw_complex* x = out_ + static_cast<size_t>(t) * M_;
            for (int part = 0; part < 2 && 2 * t + part < n; ++part) {
                double* path = LV.col(2 * t + part).data();
                double acc = 0.0;
                for (int j = 0; j < N_; ++j) {
                    acc += x[j][part];
                    path[j] = params::nu * acc;
                }
            }
        }
    }

private:
    int N_, M_, howmany_;
    const std::vector<double>& scale_;
    fftw_complex *in_, *out_;
    fftw_plan plan_;
};

// sqrt(lambda_j / 2N): the per-frequency scaling of FbmSampler
inline std::vector<double> circulant_scale(int N, double H, double dt) {
    std::vector<double> lam = circulant_eigenvalues(N, H, dt), scale(2 * N);
    for (int j = 0; j < 2 * N; ++j) scale[j] = std::sqrt(std::max(lam[j], 0.0) / (2 * N));
    return scale;
}

// Batched FFT pricer. cfg.batch counts paths; batch = 2 is one transform per block, the
// unbatched pairing of price_timed().
inline batched::Result price_batched(int N, int M_paths, const batched::Config& cfg = {},
                                     unsigned seed = 42) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();
    const std::vector<double> scale = circulant_scale(N, params::H, params::T / N);
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();
    batched::Result res = batched::run(N, M_paths, seed, cfg, [&] {
        return std::make_unique<BatchWorker>(scale, cfg.batch);
    });
    res.t_construct = t_construct;
    return res;
}

} // namespace fft_pricer
