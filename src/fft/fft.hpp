#pragma once
// Block 3: Circulant Embedding + FFTW path generator  (Davies-Harte / Wood-Chan)
// O(N log N) per path.
//
// Key insight: fBM itself is non-stationary, so its covariance matrix is NOT Toeplitz.
// However, fBM *increments* (fractional Gaussian noise, fGn) ARE stationary, so their
// covariance IS Toeplitz and embeds into a PSD circulant for H ∈ (0,1).
//
// Algorithm:
//   1. Compute fGn autocovariance γ(k) = (dt^{2H}/2)*((k+1)^{2H} - 2k^{2H} + (k-1)^{2H})
//   2. Embed into 2N circulant: c = [γ(0)..γ(N-1), 0, γ(N-1)..γ(1)]
//   3. FFT(c) → eigenvalues λ (all ≥ 0 for H ≤ 0.5; proved by Wood & Chan 1994)
//   4. Per path: w[j] = sqrt(λ[j]/M) * (a+ib), x = Re(IFFT(w)), first N = fGn increments
//   5. log_vol = cumsum(x[0..N-1])  → fBM path; then simulate GBM prices

#include <fftw3.h>
#include <chrono>
#include <complex>
#include <vector>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include "common/params.hpp"
#include "common/asian_payoff.hpp"
#include "common/rng.hpp"

namespace fft_pricer {
namespace {

// fGn autocovariance at lag k (Var = dt^{2H}, negative for H<0.5 at k≥1)
static inline double fgn_cov(int k, double H, double dt) {
    double h2 = 2.0 * H;
    if (k == 0) return std::pow(dt, h2);
    double km1 = (k == 1) ? 0.0 : std::pow(k - 1.0, h2);
    return 0.5 * std::pow(dt, h2) * (std::pow(k + 1.0, h2) + km1 - 2.0 * std::pow(k, h2));
}

double price(int N, int M_paths, unsigned seed = 42) {
    using namespace params;
    double dt = T / N;
    int M = 2 * N;

    // ── Step 1: Build circulant first row from fGn autocovariance ───────────
    // Using complex arrays for full c2c transform (avoids Hermitian bookkeeping)
    std::vector<std::complex<double>> c_emb(M, 0.0);
    for (int j = 0; j < N; ++j)
        c_emb[j] = fgn_cov(j, H, dt);
    for (int j = 1; j < N; ++j)
        c_emb[M - j] = c_emb[j];  // symmetric reflection; c_emb[N] stays 0

    // ── Step 2: Forward FFT → eigenvalues λ (all real for symmetric input) ──
    std::vector<std::complex<double>> lam(M);
    {
        fftw_plan p = fftw_plan_dft_1d(
            M,
            reinterpret_cast<fftw_complex*>(c_emb.data()),
            reinterpret_cast<fftw_complex*>(lam.data()),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
    }

    // All imaginary parts should be ≈ 0 (real symmetric input)
    for (int k = 0; k < M; ++k)
        if (lam[k].real() < -1e-8)
            throw std::runtime_error("fGn circulant embedding not PSD (unexpected for H<0.5)");

    // ── Step 3: Pre-allocate FFTW synthesis buffer (reused each path) ────────
    std::vector<std::complex<double>> w_buf(M), out_buf(M);
    fftw_plan plan_inv = fftw_plan_dft_1d(
        M,
        reinterpret_cast<fftw_complex*>(w_buf.data()),
        reinterpret_cast<fftw_complex*>(out_buf.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);

    auto rng = make_rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    double payoff_sum = 0.0;

    for (int m = 0; m < M_paths; ++m) {
        // ── Step 4: Generate one fGn path via IFFT ───────────────────────
        // w[j] = sqrt(λ[j].real() / M) * (a + ib)  →  Cov(Re(IFFT(w))) = Toeplitz(γ)
        for (int j = 0; j < M; ++j) {
            double s = std::sqrt(std::max(lam[j].real(), 0.0) / M);
            w_buf[j] = s * std::complex<double>(norm(rng), norm(rng));
        }
        fftw_execute(plan_inv);
        // out_buf[j].real() for j=0..N-1 are fGn increments (FFTW unnormalized IFFT)

        // ── Step 5: Cumsum → fBM log-vol path ────────────────────────────
        std::vector<double> log_vol(N);
        double acc = 0.0;
        for (int i = 0; i < N; ++i) {
            acc += out_buf[i].real();
            log_vol[i] = nu * acc;
        }

        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
    }

    fftw_destroy_plan(plan_inv);
    return std::exp(-r * T) * payoff_sum / M_paths;
}

// Construction vs MC timing breakdown
struct FFTTimed { double price, t_construct, t_mc; };

FFTTimed price_timed(int N, int M_paths, unsigned seed = 42) {
    using namespace params;
    using Clock = std::chrono::high_resolution_clock;
    double dt = T / N;
    int M = 2 * N;

    auto t0 = Clock::now();
    // ── Construction: build eigenvalues + create IFFT plan ───────────────────
    std::vector<std::complex<double>> c_emb(M, 0.0);
    for (int j = 0; j < N; ++j)
        c_emb[j] = fgn_cov(j, H, dt);
    for (int j = 1; j < N; ++j)
        c_emb[M - j] = c_emb[j];

    std::vector<std::complex<double>> lam(M);
    {
        fftw_plan p = fftw_plan_dft_1d(
            M,
            reinterpret_cast<fftw_complex*>(c_emb.data()),
            reinterpret_cast<fftw_complex*>(lam.data()),
            FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(p);
        fftw_destroy_plan(p);
    }
    for (int k = 0; k < M; ++k)
        if (lam[k].real() < -1e-8)
            throw std::runtime_error("fGn circulant embedding not PSD");

    std::vector<std::complex<double>> w_buf(M), out_buf(M);
    fftw_plan plan_inv = fftw_plan_dft_1d(
        M,
        reinterpret_cast<fftw_complex*>(w_buf.data()),
        reinterpret_cast<fftw_complex*>(out_buf.data()),
        FFTW_BACKWARD, FFTW_ESTIMATE);
    double t_construct = std::chrono::duration<double>(Clock::now() - t0).count();

    // ── MC loop ───────────────────────────────────────────────────────────────
    auto rng = make_rng(seed);
    std::normal_distribution<double> norm(0.0, 1.0);
    double payoff_sum = 0.0;

    t0 = Clock::now();
    for (int m = 0; m < M_paths; ++m) {
        for (int j = 0; j < M; ++j) {
            double s = std::sqrt(std::max(lam[j].real(), 0.0) / M);
            w_buf[j] = s * std::complex<double>(norm(rng), norm(rng));
        }
        fftw_execute(plan_inv);
        std::vector<double> log_vol(N);
        double acc = 0.0;
        for (int i = 0; i < N; ++i) {
            acc += out_buf[i].real();
            log_vol[i] = nu * acc;
        }
        auto inno = randn(N, rng);
        payoff_sum += asian_call_payoff(log_vol_to_prices(log_vol, inno, S0, r, dt), K);
    }
    double t_mc = std::chrono::duration<double>(Clock::now() - t0).count();

    fftw_destroy_plan(plan_inv);
    return { std::exp(-r * T) * payoff_sum / M_paths, t_construct, t_mc };
}

} // anonymous namespace
} // namespace fft_pricer
