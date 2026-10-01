#pragma once
// Randomized SVD  (Algorithm 4.4, Halko, Martinsson & Tropp, 2011)
// Returns U (m×k), S (k), Vt (k×n) such that A ≈ U * S.asDiagonal() * Vt
#include <Eigen/Dense>
#include <random>

struct RSVD {
    Eigen::MatrixXd U;
    Eigen::VectorXd S;
    Eigen::MatrixXd Vt;
};

// A: m×n matrix to decompose
// k: target rank
// p: oversampling (default 5)
// q: power iterations (default 2)
inline RSVD rsvd(const Eigen::MatrixXd& A, int k, int p = 5, int q = 2,
                 unsigned seed = 42)
{
    int n = A.cols();
    int l = k + p;

    // Stage A: form a sketch
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Omega(n, l);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < l; ++i)
            Omega(j, i) = dist(rng);

    // Thin orthonormal basis for the range of Y (m×l)
    auto orth = [](const Eigen::MatrixXd& Y) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
        return Eigen::MatrixXd(qr.householderQ() * Eigen::MatrixXd::Identity(Y.rows(), Y.cols()));
    };

    Eigen::MatrixXd Q = orth(A * Omega);

    // Subspace iteration for slowly-decaying spectra. Re-orthonormalizing after
    // each product keeps small singular directions from being lost to round-off:
    // without it, Y = (A A^T)^q A Omega scales by (s_1/s_l)^{2q+1} ~ 1e15 at H=0.1, k=128.
    for (int iter = 0; iter < q; ++iter) {
        Eigen::MatrixXd W = orth(A.transpose() * Q);
        Q = orth(A * W);
    }

    // Stage B: project and SVD on small matrix
    Eigen::MatrixXd B = Q.transpose() * A;   // l×n
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    RSVD result;
    result.U  = Q * svd.matrixU().leftCols(k);
    result.S  = svd.singularValues().head(k);
    result.Vt = svd.matrixV().leftCols(k).transpose();
    return result;
}
