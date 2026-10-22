// Correctness tests for the three fBM samplers. Run with `ctest --test-dir build`
// or `./build/test_samplers`. Exits non-zero if any check fails.
//
// Each test checks a property that a past or plausible bug would break:
//   - Cholesky: L L^T reproduces C
//   - FFT: every circulant eigenvalue matches the closed form; the smallest is
//     lambda_0 = dt^{2H} (N^{2H} - (N-1)^{2H}) (a halved gamma(0) breaks this)
//   - FFT sampler: the sample covariance of simulated paths matches C
//   - rSVD: error within a few percent of the optimal (Eckart-Young) truncation,
//     including k close to N
//   - Prices: the two exact samplers agree within Monte Carlo error
//   - Batched pricers: fast generator matches N(0,1) and its reference stream; batched
//     paths have covariance C; price independent of thread count; fixed-seed regression
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <vector>
#include <Eigen/Dense>

#include "cholesky/cholesky.hpp"
#include "fft/fft.hpp"
#include "rsvd/lowrank.hpp"
#include "common/covariance.hpp"

static int failures = 0;

static void check(bool ok, const char* name, double value, double limit) {
    std::printf("[%s] %-58s value=%.3e  limit=%.3e\n", ok ? "PASS" : "FAIL", name, value, limit);
    if (!ok) ++failures;
}

static void test_cholesky_factor() {
    const int N = 252;
    const double H = 0.10, T = 1.0;
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    Eigen::MatrixXd F = cholesky::fbm_cholesky_factor(N, H, T);
    Eigen::MatrixXd L = F.triangularView<Eigen::Lower>();
    double err = (L * L.transpose() - C).cwiseAbs().maxCoeff() / C.cwiseAbs().maxCoeff();
    check(err < 1e-12, "cholesky: max|LL^T - C| / max|C|  (N=252)", err, 1e-12);
}

static void test_fft_eigenvalues() {
    for (int N : {64, 252, 1000}) {
        for (double H : {0.05, 0.10, 0.30, 0.50}) {
            double dt = 1.0 / N;
            std::vector<double> lam = fft_pricer::circulant_eigenvalues(N, H, dt);
            double lam_min = *std::min_element(lam.begin(), lam.end());
            double closed = std::pow(dt, 2 * H) * (std::pow(N, 2 * H) - std::pow(N - 1.0, 2 * H));
            double rel = std::abs(lam_min - closed) / closed;
            char name[96];
            std::snprintf(name, sizeof name, "fft: min eigenvalue = closed form  (N=%d, H=%.2f)", N, H);
            check(lam_min > 0 && rel < 1e-8, name, rel, 1e-8);
        }
    }
}

static void test_fft_sampler_covariance() {
    const int N = 16, M = 200000;
    const double H = 0.10, dt = 1.0 / N;
    fft_pricer::FbmSampler sampler(N, H, dt);
    std::mt19937 rng(123);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> path(N);
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(N, N);
    for (int m = 0; m < M; ++m) {
        sampler.sample(rng, norm, 1.0, path);
        Eigen::Map<Eigen::VectorXd> x(path.data(), N);
        S.noalias() += x * x.transpose();
    }
    S /= M;
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, 1.0);
    // Standard error of a covariance entry is <= sqrt(2/M) * max|C|; allow 6 of them
    double err = (S - C).cwiseAbs().maxCoeff();
    double tol = 6.0 * std::sqrt(2.0 / M) * C.cwiseAbs().maxCoeff();
    check(err < tol, "fft sampler: max|sample cov - C|  (N=16, M=200k)", err, tol);
}

static void test_fft_sampler_pair_independent() {
    // Re and Im of one transform must each be fGn paths and be mutually uncorrelated
    const int N = 16, M = 200000;
    const double H = 0.10, dt = 1.0 / N;
    fft_pricer::FbmSampler sampler(N, H, dt);
    std::mt19937 rng(321);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> a(N), b(N);
    Eigen::MatrixXd Sbb = Eigen::MatrixXd::Zero(N, N), Sab = Eigen::MatrixXd::Zero(N, N);
    for (int m = 0; m < M; ++m) {
        sampler.sample_pair(rng, norm, 1.0, a, b);
        Eigen::Map<Eigen::VectorXd> x(a.data(), N), y(b.data(), N);
        Sbb.noalias() += y * y.transpose();
        Sab.noalias() += x * y.transpose();
    }
    Sbb /= M;
    Sab /= M;
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, 1.0);
    double tol = 6.0 * std::sqrt(2.0 / M) * C.cwiseAbs().maxCoeff();
    double err_im = (Sbb - C).cwiseAbs().maxCoeff();
    double err_x = Sab.cwiseAbs().maxCoeff();
    check(err_im < tol, "fft sampler: imaginary-part path covariance matches C", err_im, tol);
    check(err_x < tol, "fft sampler: Re/Im paths uncorrelated", err_x, tol);
}

static void test_rsvd_near_optimal() {
    const int N = 200;
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, 0.10, 1.0);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(C);
    Eigen::VectorXd w = eig.eigenvalues().reverse();  // descending
    for (int k : {4, 16, 64, 198}) {
        Eigen::MatrixXd Lk = lowrank::lowrank_factor(C, k, 42);
        double err = (C - Lk * Lk.transpose()).norm();
        double opt = std::sqrt(w.tail(N - k).squaredNorm());
        double ratio = err / opt;
        char name[96];
        std::snprintf(name, sizeof name, "rsvd: Frobenius error / Eckart-Young optimum  (k=%d)", k);
        check(ratio < 1.05, name, ratio, 1.05);
    }
}

static void test_exact_samplers_agree() {
    // With sigma0 = 0.2, nu = 0.52 the payoff std is ~9.5 (stable across batches);
    // bound it by 12 and allow 4 standard errors of the difference
    const int N = 64, M = 200000;
    double p_chol = cholesky::price(N, M, /*seed=*/11);
    double p_fft = fft_pricer::price(N, M, /*seed=*/12);
    double tol = 4.0 * std::sqrt(2.0) * 12.0 / std::sqrt(M);
    check(std::abs(p_chol - p_fft) < tol, "prices: |Cholesky - FFT|  (N=64, M=200k each)",
          std::abs(p_chol - p_fft), tol);
}

static void test_asian_sample_matches_plain_payoff() {
    // asian_sample() must reproduce the arithmetic payoff of the plain pricers
    const int N = 252;
    const double dt = 1.0 / N;
    std::mt19937 rng(7);
    std::normal_distribution<double> norm(0.0, 1.0);
    double worst = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> lv(N), Z(N);
        double acc = 0.0;
        for (int i = 0; i < N; ++i) { acc += 0.05 * norm(rng); lv[i] = acc; Z[i] = norm(rng); }
        double plain = asian_call_payoff(log_vol_to_prices(lv, Z, 100.0, 0.0, dt, 0.2), 100.0);
        double via = asian_sample(lv, Z, 100.0, 0.0, dt, 0.2, 100.0).arith;
        worst = std::max(worst, std::abs(plain - via) / std::max(1.0, plain));
    }
    check(worst < 1e-12, "cv: asian_sample arithmetic payoff = plain payoff", worst, 1e-12);
}

static void test_geometric_conditional_expectation() {
    // For one fixed volatility path, the closed-form E[(G-K)^+ | sigma] must match the
    // Monte Carlo average of (G-K)^+ over independent price shocks
    const int N = 64, M = 400000;
    const double dt = 1.0 / N;
    fft_pricer::FbmSampler sampler(N, 0.10, dt);
    std::mt19937 rng(99);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::vector<double> lv(N), Z(N);
    sampler.sample(rng, norm, 0.52, lv);
    double closed = 0.0, sum = 0.0, sum2 = 0.0;
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < N; ++i) Z[i] = norm(rng);
        AsianSample a = asian_sample(lv, Z, 100.0, 0.0, dt, 0.2, 100.0);
        closed = a.geom_mean;
        sum += a.geom;
        sum2 += a.geom * a.geom;
    }
    double mean = sum / M, se = std::sqrt((sum2 / M - mean * mean) / M);
    double err = std::abs(mean - closed);
    check(err < 4.0 * se, "cv: E[(G-K)^+|sigma] closed form = MC average (fixed path)", err, 4.0 * se);
}

static void test_cv_pricers_agree() {
    const int N = 64, M = 100000;
    CVResult a = cholesky::price_cv(N, M, /*seed=*/21);
    CVResult b = fft_pricer::price_cv(N, M, /*seed=*/22);
    double diff = std::abs(a.price - b.price);
    double tol = 4.0 * std::sqrt(a.se * a.se + b.se * b.se);
    check(diff < tol, "cv: |Cholesky - FFT| control-variate prices  (N=64, M=100k)", diff, tol);
    double vr_a = (a.se_plain / a.se) * (a.se_plain / a.se);
    double vr_b = (b.se_plain / b.se) * (b.se_plain / b.se);
    check(vr_a > 10.0, "cv: variance reduction, Cholesky", vr_a, 10.0);
    check(vr_b > 10.0, "cv: variance reduction, FFT", vr_b, 10.0);
    double same_run = std::abs(b.price - b.price_plain);
    double tol_same = 4.0 * b.se_plain;
    check(same_run < tol_same, "cv: CV and plain estimates from the same paths agree", same_run, tol_same);
}

static void test_lowrank_variance_correction() {
    const int N = 64, k = 8, M = 100000;
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, 0.10, 1.0);
    Eigen::MatrixXd Lk = lowrank::lowrank_factor(C, k, 42);
    Eigen::VectorXd d = lowrank::residual_sd(C, Lk);
    Eigen::VectorXd marg = Lk.rowwise().squaredNorm() + d.cwiseAbs2();
    double err = ((marg - C.diagonal()).cwiseAbs().array() / C.diagonal().array()).maxCoeff();
    check(err < 1e-12, "low-rank+diag: marginal variances = diag(C)  (k=8)", err, 1e-12);

    // Priced with the control variate so the comparison is precise
    CVResult exact = fft_pricer::price_cv(N, M, /*seed=*/31);
    CVResult corr = lowrank::price_cv(N, M, k, /*seed=*/32, /*corrected=*/true);
    CVResult plain = lowrank::price_cv(N, M, k, /*seed=*/33, /*corrected=*/false);
    double tol = 4.0 * std::sqrt(exact.se * exact.se + corr.se * corr.se);
    double dc = std::abs(corr.price - exact.price);
    check(dc < tol, "low-rank+diag: price = exact price  (k=8, N=64)", dc, tol);
    // Negative control: the uncorrected sampler's bias must be detectable at this precision
    double dp = std::abs(plain.price - exact.price);
    check(dp > tol, "low-rank (uncorrected): bias detected  (k=8, N=64)", dp, tol);
}

// ── Batched pricers and the fast generator (common/batched_mc.hpp, fast_rng.hpp) ──

static void test_xoshiro_reference_outputs() {
    // From state {1, 2, 3, 4}: the first two outputs follow by hand from the reference
    // algorithm (5 * 2^23 + 1, then 7 * 2^23 + 96 + 7); the rest pin the sequence
    fastrng::Xoshiro256pp g(1, 2, 3, 4);
    const uint64_t expect[4] = { 41943041ULL, 58720359ULL, 3588806011781223ULL, 3591011842654386ULL };
    int bad = 0;
    for (uint64_t e : expect) bad += g() != e;
    check(bad == 0, "fast rng: xoshiro256++ reference outputs", bad, 0);
}

static void test_ziggurat_distribution() {
    // 10^7 draws: moments and the CDF at -3..3 within 4 standard errors, and the mass
    // beyond the base strip R (drawn by the separate tail sampler)
    const int n = 10000000;
    fastrng::FastNormal z(2024, 0);
    double s1 = 0, s2 = 0, s4 = 0;
    long below[7] = {0}, tail = 0;
    for (int j = 0; j < n; ++j) {
        double x = z();
        s1 += x; s2 += x * x; s4 += x * x * x * x;
        for (int c = 0; c < 7; ++c) below[c] += x < c - 3;
        tail += std::abs(x) > fastrng::Ziggurat::R;
    }
    double worst = 0.0;  // largest deviation in standard errors
    worst = std::max(worst, std::abs(s1 / n) / std::sqrt(1.0 / n));
    worst = std::max(worst, std::abs(s2 / n - 1.0) / std::sqrt(2.0 / n));
    worst = std::max(worst, std::abs(s4 / n - 3.0) / std::sqrt(96.0 / n));
    for (int c = 0; c < 7; ++c) {
        double p = 0.5 * std::erfc(-(c - 3) / std::sqrt(2.0));
        if (p > 0 && p < 1) worst = std::max(worst, std::abs(double(below[c]) / n - p) / std::sqrt(p * (1 - p) / n));
    }
    double p_tail = std::erfc(fastrng::Ziggurat::R / std::sqrt(2.0));
    worst = std::max(worst, std::abs(double(tail) / n - p_tail) / std::sqrt(p_tail / n));
    check(worst < 4.0, "fast rng: ziggurat moments, CDF and tail (n=1e7, in SEs)", worst, 4.0);
}

static void test_batched_payoff_matches_plain() {
    const int N = 252;
    const double dt = 1.0 / N;
    std::mt19937 rng(17);
    std::normal_distribution<double> norm(0.0, 1.0);
    double worst = 0.0;
    for (int trial = 0; trial < 200; ++trial) {
        std::vector<double> lv(N), Z(N);
        double acc = 0.0;
        for (int j = 0; j < N; ++j) { acc += 0.05 * norm(rng); lv[j] = acc; Z[j] = norm(rng); }
        double plain = asian_call_payoff(log_vol_to_prices(lv, Z, 100.0, 0.0, dt, 0.2), 100.0);
        double fast = batched::asian_payoff(lv.data(), Z.data(), N, 100.0, 0.0, dt, 0.2, 100.0);
        worst = std::max(worst, std::abs(plain - fast));
    }
    check(worst == 0.0, "batched: allocation-free payoff = plain payoff (bitwise)", worst, 0.0);
}

// Sample covariance of nu * W^H paths from a batched worker, divided by nu^2, against C.
// Blocks of 63 paths exercise the partial last pair of the FFT worker.
template <class Worker>
static double batched_cov_error(Worker& worker, int N, int M, const Eigen::MatrixXd& C) {
    const int B = 63;
    Eigen::MatrixXd LV(N, B), S = Eigen::MatrixXd::Zero(N, N);
    int done = 0;
    for (uint32_t b = 0; done < M; ++b) {
        const int n = std::min(B, M - done);
        fastrng::FastNormal src(5, b);
        worker.log_vol(src, n, LV);
        S.noalias() += LV.leftCols(n) * LV.leftCols(n).transpose();
        done += n;
    }
    S /= M * params::nu * params::nu;
    return (S - C).cwiseAbs().maxCoeff();
}

static void test_batched_sampler_covariance() {
    const int N = 16, M = 200000;
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, params::H, params::T);
    double tol = 6.0 * std::sqrt(2.0 / M) * C.cwiseAbs().maxCoeff();

    Eigen::MatrixXd F = cholesky::fbm_cholesky_factor(N, params::H, params::T);
    cholesky::BatchWorker wc(F, 63);
    double ec = batched_cov_error(wc, N, M, C);
    check(ec < tol, "batched cholesky: max|sample cov - C|  (N=16, M=200k)", ec, tol);

    std::vector<double> scale = fft_pricer::circulant_scale(N, params::H, params::T / N);
    fft_pricer::BatchWorker wf(scale, 63);
    double ef = batched_cov_error(wf, N, M, C);
    check(ef < tol, "batched fft: max|sample cov - C|  (N=16, M=200k, odd blocks)", ef, tol);

    // Low-rank at full rank reproduces C itself
    Eigen::MatrixXd Lk = lowrank::lowrank_factor(C, N, 42);
    lowrank::BatchWorker wl(Lk, 63);
    double el = batched_cov_error(wl, N, M, C);
    check(el < tol, "batched low-rank (k=N): max|sample cov - C|  (N=16, M=200k)", el, tol);
}

static void test_batched_thread_invariance() {
    // Blocks own their streams and are reduced in order: any thread count, same price
    const int N = 128, M = 5000;
    batched::Config one{64, 1, batched::Rng::Fast}, four{64, 4, batched::Rng::Fast};
    int bad = 0;
    bad += cholesky::price_batched(N, M, one).price != cholesky::price_batched(N, M, four).price;
    bad += fft_pricer::price_batched(N, M, one).price != fft_pricer::price_batched(N, M, four).price;
    bad += lowrank::price_batched(N, M, 16, one).price != lowrank::price_batched(N, M, 16, four).price;
    batched::Config one_std{1, 1, batched::Rng::Std}, three_std{1, 3, batched::Rng::Std};
    bad += cholesky::price_batched(N, 999, one_std).price != cholesky::price_batched(N, 999, three_std).price;
    check(bad == 0, "batched: price bitwise identical for 1 and several threads", bad, 0);
}

static void test_batched_prices_agree() {
    // Batched exact samplers (both generators) against the plain pricer, N=64
    const int N = 64, M = 200000;
    double ref = cholesky::price(N, M, /*seed=*/41);
    double tol = 4.0 * std::sqrt(2.0) * 12.0 / std::sqrt(M);  // payoff std < 12, as above
    double worst = 0.0;
    for (auto rng : {batched::Rng::Std, batched::Rng::Fast}) {
        batched::Config cfg{64, 2, rng};
        worst = std::max(worst, std::abs(cholesky::price_batched(N, M, cfg, 42).price - ref));
        worst = std::max(worst, std::abs(fft_pricer::price_batched(N, M, cfg, 43).price - ref));
    }
    check(worst < tol, "batched: Cholesky/FFT prices (std + fast rng) = plain price", worst, tol);

    // The batched low-rank sampler has the same truncation bias as the plain one
    double lr_plain = lowrank::price(N, M, 8, /*seed=*/44);
    double lr_batched = lowrank::price_batched(N, M, 8, {64, 2, batched::Rng::Fast}, 45).price;
    check(std::abs(lr_plain - lr_batched) < tol, "batched: low-rank price (k=8) = plain low-rank price",
          std::abs(lr_plain - lr_batched), tol);
}

static void test_batched_fixed_seed_regression() {
    // Pins the fast-generator stream end to end (generator, ziggurat tables, block seeding,
    // FFT and Cholesky workers). Tolerance allows last-bit libm/FFTW differences across
    // platforms; any change to the stream moves the price by ~1e-2.
    batched::Config cfg{64, 1, batched::Rng::Fast};
    double f = fft_pricer::price_batched(64, 10000, cfg, 42).price;
    double c = cholesky::price_batched(64, 10000, cfg, 42).price;
    double err = std::max(std::abs(f - 5.4529229435598605), std::abs(c - 5.3844322364811728));
    check(err < 1e-9, "batched: fixed-seed prices unchanged (FFT, Cholesky; N=64)", err, 1e-9);
}

// Runs one test; an exception (e.g. a non-PSD embedding) counts as a failure
static void run(const char* name, void (*test)()) {
    try {
        test();
    } catch (const std::exception& e) {
        std::printf("[FAIL] %-58s threw: %s\n", name, e.what());
        ++failures;
    }
}

int main() {
    run("cholesky factor", test_cholesky_factor);
    run("fft eigenvalues", test_fft_eigenvalues);
    run("fft sampler covariance", test_fft_sampler_covariance);
    run("fft sampler pair independent", test_fft_sampler_pair_independent);
    run("rsvd near optimal", test_rsvd_near_optimal);
    run("exact samplers agree", test_exact_samplers_agree);
    run("asian_sample matches plain payoff", test_asian_sample_matches_plain_payoff);
    run("geometric conditional expectation", test_geometric_conditional_expectation);
    run("control-variate pricers agree", test_cv_pricers_agree);
    run("low-rank variance correction", test_lowrank_variance_correction);
    run("xoshiro reference outputs", test_xoshiro_reference_outputs);
    run("ziggurat distribution", test_ziggurat_distribution);
    run("batched payoff matches plain", test_batched_payoff_matches_plain);
    run("batched sampler covariance", test_batched_sampler_covariance);
    run("batched thread invariance", test_batched_thread_invariance);
    run("batched prices agree", test_batched_prices_agree);
    run("batched fixed-seed regression", test_batched_fixed_seed_regression);
    std::printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
