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
    int m = A.rows(), n = A.cols();
    int l = k + p;

    // Stage A: form a sketch
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    Eigen::MatrixXd Omega(n, l);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < l; ++i)
            Omega(j, i) = dist(rng);

    Eigen::MatrixXd Y = A * Omega;

    // Power iteration for better accuracy with slowly-decaying spectra
    for (int iter = 0; iter < q; ++iter) {
        Y = A * (A.transpose() * Y);
    }

    // QR decomposition of Y
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(Y);
    Eigen::MatrixXd Q = qr.householderQ() * Eigen::MatrixXd::Identity(m, l);

    // Stage B: project and SVD on small matrix
    Eigen::MatrixXd B = Q.transpose() * A;   // l×n
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    RSVD result;
    result.U  = Q * svd.matrixU().leftCols(k);
    result.S  = svd.singularValues().head(k);
    result.Vt = svd.matrixV().leftCols(k).transpose();
    return result;
}
