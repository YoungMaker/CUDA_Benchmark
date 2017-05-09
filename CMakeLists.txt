cmake_minimum_required(VERSION 3.7)
project(CUDA_Benchmark C CXX)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES Benchmark.cpp cpu_bmark.h)

add_executable(CUDA_Benchmark ${SOURCE_FILES})
