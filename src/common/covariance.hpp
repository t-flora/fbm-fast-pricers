#pragma once
#include <cmath>
#include <Eigen/Dense>

// fBM covariance kernel: C(t,s) = 0.5*(|t|^{2H} + |s|^{2H} - |t-s|^{2H})
inline double fbm_cov(double t, double s, double H) {
    return 0.5 * (std::pow(std::abs(t), 2*H)
                + std::pow(std::abs(s), 2*H)
                - std::pow(std::abs(t - s), 2*H));
}

// N×N fBM covariance matrix on grid {dt, 2dt, ..., T}
inline Eigen::MatrixXd build_fbm_cov_matrix(int N, double H, double T) {
    double dt = T / N;
    Eigen::MatrixXd C(N, N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            C(i, j) = fbm_cov((i+1)*dt, (j+1)*dt, H);
    return C;
}

// First row of the Toeplitz fBM covariance (used by FFT circulant embedding)
inline Eigen::VectorXd fbm_cov_first_row(int N, double H, double T) {
    double dt = T / N;
    Eigen::VectorXd row(N);
    for (int j = 0; j < N; ++j)
        row(j) = fbm_cov(dt, (j+1)*dt, H);
    return row;
}
