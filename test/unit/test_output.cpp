#include <gtest/gtest.h>

#include "test_configuration.h"
#include "config.h"
#include "output.h"


TEST(saveAndPrintErrorOutput, whenCpuImplementationInvalidThenPrintErrorInfo) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::instance_t solution({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::solution_validation_data_t error_data(data::CPU_SOLUTION_ERROR, instance, solution);
    error_data.repetition = 5;
    testing::internal::CaptureStdout();

    output::saveAndPrintErrorOutput(error_data, "algorithm_1");

    EXPECT_EQ(testing::internal::GetCapturedStdout(), testdata::PRINT_FOR_CPU_INVALID_SOLUTION);
}


TEST(saveAndPrintErrorOutput, whenGpuImplementationInvalidThenPrintErrorInfo) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::instance_t solution({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::solution_validation_data_t error_data(data::GPU_SOLUTION_ERROR, instance, solution);
    error_data.repetition = 7;
    testing::internal::CaptureStdout();

    output::saveAndPrintErrorOutput(error_data, "algorithm_1");

    EXPECT_EQ(testing::internal::GetCapturedStdout(), testdata::PRINT_FOR_GPU_INVALID_SOLUTION);
}


TEST(saveAndPrintErrorOutput, whenBothImplementationsInvalidThenPrintErrorInfo) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::instance_t solution({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::solution_validation_data_t error_data(data::SOLUTION_INVALID, instance, solution);
    error_data.repetition = 2137;
    testing::internal::CaptureStdout();

    output::saveAndPrintErrorOutput(error_data, "algorithm_1");

    EXPECT_EQ(testing::internal::GetCapturedStdout(), testdata::PRINT_FOR_BOTH_INVALID_SOLUTIONS);
}


TEST(printNotification, printSomeString) {
    testing::internal::CaptureStdout();

    output::printNotification("Super duper message!");
    
    EXPECT_EQ(testing::internal::GetCapturedStdout(), ">>> Super duper message!\n");
}


TEST(printConfigurationOutput, whenValidConfigLoadedThenPrintIt) {
    config::configuration_t configuration(true, false, true, true);
    configuration.loaded_instances.emplace(data::instance_t({4, 6, 8}, 15));
    configuration.loaded_instances.emplace(data::instance_t({3, 1, 0}, 3));
    configuration.loaded_instances.emplace(data::instance_t({7, 1, 1}, 2));
    configuration.loaded_instances.emplace(data::instance_t({9, 3, 5}, 7));
    testing::internal::CaptureStdout();

    output::printConfigurationOutput(configuration);

    EXPECT_EQ(testing::internal::GetCapturedStdout(), testdata::PRINT_FOR_VALID_CONFIGURATION);
}


TEST(ResultsOutputStream__open, whenStreamNotYetOpenedThenOpenIt) {
    output::ResultsOutputStream& stream = output::ResultsOutputStream::getStream();
    
    stream.open();

    EXPECT_EQ(stream.isOpen(), true);
    
    std::remove("results.csv");
}


TEST(ResultsOutputStream__open, whenStreamAlreadyOpenedThenThrowLogicError) {
    output::ResultsOutputStream& stream = output::ResultsOutputStream::getStream();
    stream.open();
    
    EXPECT_THROW(stream.open(), std::logic_error);
}


TEST(ResultsOutputStream__open, whenStreamNotYetOpenedThenPrintTableHeader) {
    output::ResultsOutputStream& stream = output::ResultsOutputStream::getStream();
    testing::internal::CaptureStdout();
    
    stream.open();

    EXPECT_EQ(testing::internal::GetCapturedStdout(), testdata::TABLE_HEADER);

    stream.close();
    std::remove("results.csv");
}


TEST(ResultsOutputStream__printAverageResult, whenAllResultsPresentThenPrintTableRow) {
    output::ResultsOutputStream& stream = output::ResultsOutputStream::getStream();
    data::results_t results(10);
    results.cpu_bitonic_time_seconds = 1.5;
    results.gpu_bitonic_time_seconds = 2.4;
    results.cpu_odd_even_time_seconds = 3.6;
    results.gpu_odd_even_time_seconds = 4.5;
    testing::internal::CaptureStdout();

    stream.open();
    stream.printAverageResult(results);
    stream.close();

    EXPECT_EQ(testing::internal::GetCapturedStdout(), testdata::TABLE_WITH_ALL_RESULTS);
    
    std::remove("results.csv");
}


TEST(ResultsOutputStream__printAverageResult, whenSomeResultsPresentThenPrintTableRow) {
    output::ResultsOutputStream& stream = output::ResultsOutputStream::getStream();
    data::results_t results(10);
    results.cpu_bitonic_time_seconds = 1.5;
    results.gpu_bitonic_time_seconds = data::MEASUREMENT_NOT_PERFORMED;
    results.cpu_odd_even_time_seconds = 3.6;
    results.gpu_odd_even_time_seconds = data::MEASUREMENT_NOT_PERFORMED;
    testing::internal::CaptureStdout();

    stream.open();
    stream.printAverageResult(results);
    stream.close();

    EXPECT_EQ(testing::internal::GetCapturedStdout(), testdata::TABLE_WITH_PARTIAL_RESULTS);
    
    std::remove("results.csv");
}