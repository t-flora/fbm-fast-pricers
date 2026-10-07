#pragma once
// Timing helpers shared by the benchmark executables.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

using Clock = std::chrono::high_resolution_clock;
inline double elapsed_s(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Runs f() `reps` times. Every repeat uses the same seed, so the price is identical and
// only the time varies. Returns the run with the median time and the interquartile range
// of the times (linear interpolation between order statistics). `time` maps a result to
// the seconds that are ranked.
template <class R>
struct Repeated {
    R median;
    double q1, q3;
};

template <class F, class TimeOf>
auto repeat_timed(int reps, F f, TimeOf time) -> Repeated<decltype(f())> {
    std::vector<decltype(f())> runs;
    for (int k = 0; k < reps; ++k) runs.push_back(f());
    std::sort(runs.begin(), runs.end(), [&](const auto& a, const auto& b) { return time(a) < time(b); });
    auto quantile = [&](double q) {
        double pos = q * (reps - 1);
        int lo = static_cast<int>(pos);
        int hi = std::min(lo + 1, reps - 1);
        return time(runs[lo]) + (pos - lo) * (time(runs[hi]) - time(runs[lo]));
    };
    return { runs[reps / 2], quantile(0.25), quantile(0.75) };
}

// True if `flag` (e.g. "--quick") is among the command-line arguments
inline bool has_flag(int argc, char** argv, const char* flag) {
    for (int k = 1; k < argc; ++k)
        if (std::strcmp(argv[k], flag) == 0) return true;
    return false;
}
