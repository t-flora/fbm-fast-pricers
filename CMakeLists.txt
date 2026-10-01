cmake_minimum_required(VERSION 3.16)
project(RoughVolPricer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # lets clangd find Eigen/FFTW headers

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-O3 -march=native -Wall -Wextra -Wno-unused-parameter)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
link_directories(${FFTW3_LIBRARY_DIRS})

# All targets include src/ so headers are found as "common/params.hpp" etc.
include_directories(src)

# ── Cholesky pricer ──────────────────────────────────────────────────────────
add_executable(cholesky_pricer src/cholesky/cholesky_pricer.cpp)
target_link_libraries(cholesky_pricer PRIVATE Eigen3::Eigen)

# ── FFT pricer ───────────────────────────────────────────────────────────────
add_executable(fft_pricer src/fft/fft_pricer.cpp)
target_include_directories(fft_pricer PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(fft_pricer PRIVATE ${FFTW3_LIBRARIES})

# ── rSVD pricer ──────────────────────────────────────────────────────────────
add_executable(rsvd_pricer src/rsvd/rsvd_pricer.cpp)
target_link_libraries(rsvd_pricer PRIVATE Eigen3::Eigen)

# ── Benchmark (includes all three headers → needs all deps) ──────────────────
add_executable(benchmark benchmarks/benchmark.cpp)
target_include_directories(benchmark PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(benchmark PRIVATE Eigen3::Eigen ${FFTW3_LIBRARIES})

# ── Tests (ctest --test-dir build) ───────────────────────────────────────────
enable_testing()
add_executable(test_samplers tests/test_samplers.cpp)
target_include_directories(test_samplers PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(test_samplers PRIVATE Eigen3::Eigen ${FFTW3_LIBRARIES})
add_test(NAME samplers COMMAND test_samplers)
