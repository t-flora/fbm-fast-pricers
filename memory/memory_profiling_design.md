# Memory Profiling Design — TODO #13

## Hardware Reference
- **Chip**: Apple M2 (unified memory architecture)
- **RAM**: 24 GB
- **L1d**: 64 KB
- **L2**: 4 MB (performance cores)
- **L3**: 16 MB
- **Rated memory bandwidth**: 100 GB/s (Apple spec sheet)
- **Measured**: `sudo powermetrics --samplers memory_power -n 1` during a benchmark run

## Why bandwidth explains Cholesky timing
Cholesky at N=1000, M=10k: the 8 MB lower-triangular L matrix must be re-read for each
of 10k paths → 80 GB memory traffic. At 100 GB/s → ~0.8 s lower bound.
Observed α ≈ 1.5 (O(M·N²) dominated, memory-bandwidth limited, not O(N³) factorization).

## Four metrics to record per method

| # | Metric | Formula | Notes |
|---|--------|---------|-------|
| 1 | `theoretical_peak_mb` | Cholesky: N²·8/1e6; FFT: N·8/1e6; H-matrix: N²·8/1e6 (held) or N·k·8/1e6 (freed) | Analytical |
| 2 | `bytes_per_path_kb` | peak_bytes / M / 1024 | Runtime |
| 3 | `est_bandwidth_GBs` | Cholesky only: N²·8·M / wall_time_s / 1e9 | Proxy for memory-bound check |
| 4 | `cache_pressure` | peak_bytes / L3_size (16 MB = 16e6 bytes) | > 1 → L3 spill |

## Two H-matrix variants

**`price_timed` (existing)**: C (N×N) stays allocated for full function scope.
Peak = O(N²). MC sees O(N·k) ops but L3 may still hold C.

**`price_freed_timed` (new)**: C is scoped out before the MC loop:
```cpp
Eigen::MatrixXd Lk;
double t_construct;
{
    auto t0 = Clock::now();
    Eigen::MatrixXd C = build_fbm_cov_matrix(N, H, T);
    RSVD decomp = rsvd(C, rank_k, ...);
    Lk = decomp.U * decomp.S.cwiseSqrt().asDiagonal();
    t_construct = elapsed_s(t0);
}  // C destroyed here — peak_rss can drop before MC
auto t1 = Clock::now();
// ... MC loop uses only Lk (N×k, much smaller) ...
```

## C++ implementation plan

- **Where**: `benchmarks/benchmark.cpp` — add memory measurement after each `price_timed()` call
- **RSS API (macOS)**:
  ```cpp
  #include <mach/mach.h>
  size_t current_rss_bytes() {
      mach_task_basic_info info;
      mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
      task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                (task_info_t)&info, &count);
      return info.resident_size;
  }
  ```
  `getrusage` `ru_maxrss` on macOS reports bytes (not KB like Linux).
- **CSV columns to add**: `peak_rss_mb`, `theoretical_peak_mb`, `cache_pressure`
- **Cholesky only**: add `est_bandwidth_GBs` column

## Python implementation plan

- **Script**: `data/profile_memory.py`
- **API**: `tracemalloc` for heap peak; also measure wall time
- **Sweep**: N ∈ {64, 128, 256, 512, 1024} × M ∈ {100, 1000, 10000}
- **Comparison**: measured bytes vs theoretical O(M·N) for paths matrix
- **Output**: `plots/memory_profile.png`
  - Panel (a): theoretical bytes vs N log-log with O(N²) / O(N·k) reference lines
  - Panel (b): Python heap bytes vs M×N

## Leak prevention checklist
- FFT: `fftw_destroy_plan(plan_fwd)` and `fftw_destroy_plan(plan_inv)` before return ✅ (in price_timed)
- H-matrix `price_freed_timed`: C destroyed by block scope — Lk only remains
- All Eigen matrices: stack/RAII, no manual new/delete needed
- `tracemalloc` in Python: stop + clear in finally block
