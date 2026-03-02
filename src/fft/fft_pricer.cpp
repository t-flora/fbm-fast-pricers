#include <iostream>
#include <chrono>
#include "fft/fft.hpp"
#include "common/params.hpp"

int main() {
    using namespace params;
    for (int N : {N_SMALL, N_MEDIUM, N_LARGE}) {
        auto t0 = std::chrono::high_resolution_clock::now();
        double p = fft_pricer::price(N, M_PATHS);
        double elapsed = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();
        std::cout << "FFT  N=" << N << "  price=" << p
                  << "  time=" << elapsed << "s\n";
    }
}
