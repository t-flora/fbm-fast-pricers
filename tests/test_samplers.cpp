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
#include <cmath>
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
    std::printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
