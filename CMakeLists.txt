cmake_minimum_required(VERSION 3.16)
project(RoughVolPricer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ── H-matrix pricer ──────────────────────────────────────────────────────────
add_executable(hmatrix_pricer src/hmatrix/hmatrix_pricer.cpp)
target_link_libraries(hmatrix_pricer PRIVATE Eigen3::Eigen)

# ── Benchmark (includes all three headers → needs all deps) ──────────────────
add_executable(benchmark benchmarks/benchmark.cpp)
target_include_directories(benchmark PRIVATE ${FFTW3_INCLUDE_DIRS})
target_link_libraries(benchmark PRIVATE Eigen3::Eigen ${FFTW3_LIBRARIES})
