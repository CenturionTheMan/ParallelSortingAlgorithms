cmake_minimum_required(VERSION 3.18)
project(benchmarking_tool LANGUAGES C CXX CUDA)

cmake_policy(SET CMP0104 NEW)
set(CMAKE_CUDA_ARCHITECTURES 60 61 75)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

file(
    GLOB SOURCES 
    "./benchmarking_tool/src/*.cpp"
    "./benchmarking_tool/src/*.cu"
    "./benchmarking_tool/headers/*.h"
    "./benchmarking_tool/headers/*.cuh"
)

add_executable(benchmarking_tool ${SOURCES})

target_include_directories(benchmarking_tool PRIVATE benchmarking_tool/headers)
set_target_properties(benchmarking_tool PROPERTIES CUDA_SEPARABLE_COMPILATION ON)