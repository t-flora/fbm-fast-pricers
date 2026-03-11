#include <iostream>
#include <chrono>
#include "hmatrix/hmatrix.hpp"
#include "common/params.hpp"

int main() {
    using namespace params;
    for (int N : {N_SMALL, N_MEDIUM, N_LARGE}) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double p = hmatrix::price(N, M_PATHS);
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::cout << "rSVD  N=" << N << "  price=" << p
                  << "  time=" << elapsed << "s\n";
    }
}
