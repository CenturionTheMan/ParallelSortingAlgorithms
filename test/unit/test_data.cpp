#include <gtest/gtest.h>

#include "data.h"


TEST(divideResultsByScalar, divideByInt) {
    data::results_t results(10);
    results.cpu_bitonic_time_seconds = 5;
    results.gpu_bitonic_time_seconds = 2;
    results.cpu_odd_even_time_seconds = 3;
    results.gpu_odd_even_time_seconds = 4;

    results /= 2;

    EXPECT_EQ(results.cpu_bitonic_time_seconds, 2.5);
    EXPECT_EQ(results.gpu_bitonic_time_seconds, 1);
    EXPECT_EQ(results.cpu_odd_even_time_seconds, 1.5);
    EXPECT_EQ(results.gpu_odd_even_time_seconds, 2);
}


TEST(addResults, addTwoResultsWithAllMeasurements) {
    data::results_t results(10);
    results.cpu_bitonic_time_seconds = 5;
    results.gpu_bitonic_time_seconds = 2;
    results.cpu_odd_even_time_seconds = 3;
    results.gpu_odd_even_time_seconds = 4;

    results += results;

    EXPECT_EQ(results.cpu_bitonic_time_seconds, 10);
    EXPECT_EQ(results.gpu_bitonic_time_seconds, 4);
    EXPECT_EQ(results.cpu_odd_even_time_seconds, 6);
    EXPECT_EQ(results.gpu_odd_even_time_seconds, 8);
}


TEST(addResults, addTwoResultsWithPartialMeasurements) {
    data::results_t results_1(10);
    results_1.cpu_bitonic_time_seconds = 5;
    results_1.gpu_bitonic_time_seconds = 2;
    results_1.cpu_odd_even_time_seconds = 3;
    results_1.gpu_odd_even_time_seconds = 4;
    data::results_t results_2(10);
    results_2.cpu_bitonic_time_seconds = 5;
    results_2.gpu_bitonic_time_seconds = data::MEASUREMENT_NOT_PERFORMED;
    results_2.cpu_odd_even_time_seconds = 3;
    results_2.gpu_odd_even_time_seconds = data::MEASUREMENT_NOT_PERFORMED;

    results_1 += results_2;

    EXPECT_EQ(results_1.cpu_bitonic_time_seconds, 10);
    EXPECT_EQ(results_1.gpu_bitonic_time_seconds, 2);
    EXPECT_EQ(results_1.cpu_odd_even_time_seconds, 6);
    EXPECT_EQ(results_1.gpu_odd_even_time_seconds, 4);
}