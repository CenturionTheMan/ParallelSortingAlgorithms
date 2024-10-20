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

include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(benchmarking_tool ${SOURCES} "./benchmarking_tool/main.cpp")

target_include_directories(benchmarking_tool PRIVATE benchmarking_tool/headers)
set_target_properties(benchmarking_tool PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

enable_testing()

file(
    GLOB TESTS 
    "./test/*.cpp"
    "./test/unit/*.cpp"
    "./test/integration/*.cpp"
)

add_executable(hello_test ${TESTS} ${SOURCES})
target_link_libraries(hello_test GTest::gtest_main)
target_include_directories(hello_test PRIVATE benchmarking_tool/headers test)

include(GoogleTest)
gtest_discover_tests(hello_test)