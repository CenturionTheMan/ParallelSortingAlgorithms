#include <gtest/gtest.h>

#include "test_configuration.h"
#include "output.h"


TEST(saveAndPrintErrorOutput, whenCpuImplementationInvalidThenCpuDumpErrorToFile) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::instance_t solution({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::solution_validation_data_t error_data(data::CPU_SOLUTION_ERROR, instance, solution);
    error_data.repetition = 5;

    output::saveAndPrintErrorOutput(error_data, "algorithm_1");

    EXPECT_EQ(testconf::readFileContents("error.log"), testdata::ERROR_FILE_FOR_CPU_INVALID_SOLUTION);

    std::remove("error.log");
}


TEST(saveAndPrintErrorOutput, whenGpuImplementationInvalidThenGpuDumpErrorToFile) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::instance_t solution({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::solution_validation_data_t error_data(data::GPU_SOLUTION_ERROR, instance, solution);
    error_data.repetition = 7;

    output::saveAndPrintErrorOutput(error_data, "algorithm_1");

    EXPECT_EQ(testconf::readFileContents("error.log"), testdata::ERROR_FILE_FOR_GPU_INVALID_SOLUTION);

    std::remove("error.log");
}


TEST(saveAndPrintErrorOutput, whenBothImplementationsInvalidThenDumpErrorToFile) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::instance_t solution({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    data::solution_validation_data_t error_data(data::SOLUTION_INVALID, instance, solution);
    error_data.repetition = 2137;

    output::saveAndPrintErrorOutput(error_data, "algorithm_1");

    EXPECT_EQ(testconf::readFileContents("error.log"), testdata::ERROR_FILE_FOR_BOTH_INVALID_SOLUTIONS);

    std::remove("error.log");
}


TEST(ResultsOutputStream__dumpResult, whenAllAlgorithmsAndImplementationsEnabledThenSaveLineToResultsFile) {
    output::ResultsOutputStream& output = output::ResultsOutputStream::getStream();
    data::results_t results(10);
    results.cpu_bitonic_time_seconds = 1.5;
    results.gpu_bitonic_time_seconds = 2.4;
    results.cpu_odd_even_time_seconds = 3.6;
    results.gpu_odd_even_time_seconds = 4.5;

    output.open();
    output.dumpResult(results);
    output.close();

    EXPECT_EQ(testconf::readFileContents("results.csv"), testdata::RESULTS_CSV_ALL_ENABLED);

    std::remove("results.csv");
}


TEST(ResultsOutputStream__dumpResult, whenAllAlgorithmsAndSomeImplementationsDisabledThenSaveLineToResultsFile) {
    output::ResultsOutputStream& output = output::ResultsOutputStream::getStream();
    data::results_t results(10);
    results.cpu_bitonic_time_seconds = 1.5;
    results.gpu_bitonic_time_seconds = data::MEASUREMENT_NOT_PERFORMED;
    results.cpu_odd_even_time_seconds = 3.6;
    results.gpu_odd_even_time_seconds = data::MEASUREMENT_NOT_PERFORMED;

    output.open();
    output.dumpResult(results);
    output.close();

    EXPECT_EQ(testconf::readFileContents("results.csv"), testdata::RESULTS_CSV_SOME_DISABLED);

    std::remove("results.csv");
}


TEST(ResultsOutputStream__open, whenStreamNotYetOpenedThenDumpHeaderToResultFile) {
    output::ResultsOutputStream& stream = output::ResultsOutputStream::getStream();
    
    stream.open();
    stream.close();

    EXPECT_EQ(testconf::readFileContents("results.csv"), testdata::RESULT_FILE_HEADER + "\n");
    
    std::remove("results.csv");
}