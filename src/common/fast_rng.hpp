#pragma once
// Fast Gaussian generator: xoshiro256++ bits + a 256-layer ziggurat normal sampler.
//
// std::mt19937 with std::normal_distribution costs ~16 ns per draw on the M2 (the polar
// method needs a log and a sqrt per pair, and rejects 21% of candidate pairs). The
// ziggurat returns x = (random 52-bit integer) * w[layer] after one table comparison in
// ~99% of draws, so a draw is one 64-bit generator step plus a multiply.
//
// Used only by the batched pricers (common/batched_mc.hpp). The plain price() functions
// keep std::mt19937, so the main benchmark and the report's prices are unchanged.
//
// References: Blackman & Vigna (2021) for xoshiro256++; Marsaglia & Tsang (2000) for
// the ziggurat, with the 64-bit layout (8 bits layer, 1 bit sign, 52 bits magnitude)
// used by numpy's Generator.standard_normal so layer and value bits never overlap.
#include <cmath>
#include <cstdint>

namespace fastrng {

// SplitMix64 step: expands a 64-bit seed into well-mixed state words
inline uint64_t splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// xoshiro256++ (period 2^256 - 1). Satisfies UniformRandomBitGenerator.
class Xoshiro256pp {
public:
    using result_type = uint64_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return ~result_type(0); }

    explicit Xoshiro256pp(uint64_t seed) {
        for (auto& w : s_) w = splitmix64(seed);
    }
    Xoshiro256pp(uint64_t s0, uint64_t s1, uint64_t s2, uint64_t s3) : s_{s0, s1, s2, s3} {}

    // Independent stream for (seed, stream): the key is injective for 32-bit seeds
    // and streams, and SplitMix64 maps neighbouring keys to unrelated states
    static Xoshiro256pp stream(uint32_t seed, uint32_t stream) {
        return Xoshiro256pp((static_cast<uint64_t>(seed) << 32) | stream);
    }

    result_type operator()() {
        const uint64_t result = rotl(s_[0] + s_[3], 23) + s_[0];
        const uint64_t t = s_[1] << 17;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = rotl(s_[3], 45);
        return result;
    }

private:
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t s_[4];
};

// Uniform double in [0, 1) from the top 53 bits
template <class G>
inline double uniform01(G& g) {
    return static_cast<double>(g() >> 11) * 0x1.0p-53;
}

// Ziggurat for the standard normal: 255 rectangles of equal area V stacked on a base
// strip (rectangle [0, R] x [0, f(R)] plus the tail x > R, also area V), f(x) = e^{-x^2/2}.
// Layer i >= 1 spans [0, x_i]; x_255 = R and x_0 = 0. A point in layer i with
// |x| < x_{i-1} lies under f for sure (the k_ test); the rest go to a wedge or tail test.
class Ziggurat {
public:
    static constexpr int LAYERS = 256;
    static constexpr double R = 3.6541528853610087963519472518;  // x_255

    Ziggurat() {
        const double m = 0x1.0p52;  // magnitudes are 52-bit integers
        // Common area of every layer, including the base strip with its tail
        const double V = R * f(R) + std::sqrt(M_PI / 2) * std::erfc(R / std::sqrt(2.0));
        const double q = V / f(R);  // width of the base strip if it were a rectangle
        k_[0] = static_cast<uint64_t>(R / q * m);
        k_[1] = 0;  // the top layer [0, x_1] has x_0 = 0: always test the wedge
        w_[0] = q / m;
        w_[LAYERS - 1] = R / m;
        f_[0] = 1.0;
        f_[LAYERS - 1] = f(R);
        double x = R, x_prev = R;
        for (int i = LAYERS - 2; i >= 1; --i) {
            x = std::sqrt(-2.0 * std::log(V / x + f(x)));  // x_i from x_{i+1}: equal areas
            k_[i + 1] = static_cast<uint64_t>(x / x_prev * m);
            x_prev = x;
            f_[i] = f(x);
            w_[i] = x / m;
        }
    }

    template <class G>
    double operator()(G& g) const {
        for (;;) {
            uint64_t bits = g();
            const int layer = static_cast<int>(bits & 0xff);
            bits >>= 8;
            const bool negative = bits & 1;
            const uint64_t mag = (bits >> 1) & 0x000fffffffffffffULL;
            double x = static_cast<double>(mag) * w_[layer];
            if (mag < k_[layer]) return negative ? -x : x;  // ~99% of draws end here
            if (layer == 0) {
                // Tail x > R (Marsaglia 1964): exponential proposals, accepted with
                // probability e^{-(R + e)^2/2} / e^{-R^2/2 - R e}
                for (;;) {
                    double e = -std::log1p(-uniform01(g)) / R;
                    double y = -std::log1p(-uniform01(g));
                    if (y + y > e * e) return negative ? -(R + e) : R + e;
                }
            }
            // Wedge between f(x_layer) and f(x_{layer-1}): accept if under the curve
            if (f_[layer] + uniform01(g) * (f_[layer - 1] - f_[layer]) < f(x))
                return negative ? -x : x;
        }
    }

    // Tables are built once; the sampler itself is stateless and thread-safe
    static const Ziggurat& instance() {
        static const Ziggurat z;
        return z;
    }

private:
    static double f(double x) { return std::exp(-0.5 * x * x); }
    uint64_t k_[LAYERS];
    double w_[LAYERS], f_[LAYERS];
};

// Standard normal draws from xoshiro256++ via the ziggurat
class FastNormal {
public:
    FastNormal(uint32_t seed, uint32_t stream)
        : g_(Xoshiro256pp::stream(seed, stream)), z_(Ziggurat::instance()) {}
    double operator()() { return z_(g_); }
    void fill(double* out, int n) {
        for (int j = 0; j < n; ++j) out[j] = z_(g_);
    }

private:
    Xoshiro256pp g_;
    const Ziggurat& z_;
};

}  // namespace fastrng
