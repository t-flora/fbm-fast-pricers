#pragma once
#include <random>
#include <vector>

// Returns a vector of N i.i.d. standard normal samples.
inline std::vector<double> randn(int N, std::mt19937& rng) {
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> out(N);
    for (auto& v : out) v = dist(rng);
    return out;
}

// Seeded RNG factory.
inline std::mt19937 make_rng(unsigned seed = 42) {
    return std::mt19937(seed);
}
