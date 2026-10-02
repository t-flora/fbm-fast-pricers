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
    std::printf("\n%s: %d failure(s)\n", failures ? "FAILED" : "OK", failures);
    return failures ? 1 : 0;
}
