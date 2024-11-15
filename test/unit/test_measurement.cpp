#include <algorithm>

#include <gtest/gtest.h>

#include "measurement.h"


TEST(measure_implementation_for_instance, whenSolutionValidThenReturnValidSolutionCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    double measurement_result;
    auto implementation = [](std::vector<int>& array) { 
        return std::sort(array.begin(), array.end());
    };

    data::solution_validation_data_t verification_data = measurement::measure_implementation_for_instance(
        instance, implementation, &measurement_result, true
    );

    EXPECT_EQ(verification_data.validation_code, data::SOLUTION_VALID);
}


TEST(measure_implementation_for_instance, whenSolutionInvalidThenReturnValidSolutionCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    double measurement_result;
    auto implementation = [](std::vector<int>& array) {};

    data::solution_validation_data_t verification_data = measurement::measure_implementation_for_instance(
        instance, implementation, &measurement_result, true
    );

    EXPECT_EQ(verification_data.validation_code, data::SOLUTION_INVALID);
}


TEST(measure_implementation_for_instance, whenFunctionCalledThenSaveTimeMeasurement) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    double measurement_result = -2137;
    auto implementation = [](std::vector<int>& array) { 
        return std::sort(array.begin(), array.end());
    };

    measurement::measure_implementation_for_instance(
        instance, implementation, &measurement_result, true
    );

    EXPECT_TRUE(measurement_result > 0.0);
}


TEST(measure_implementation_for_instance, whenFunctionFinishedThenOriginalInstanceRemainsUnchanged) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    double measurement_result;
    auto implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };

    measurement::measure_implementation_for_instance(
        instance, implementation, &measurement_result, true
    );

    EXPECT_EQ(instance.sequence, std::vector<int>({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}));
}


TEST(measure_algorithm_for_instance, whenBothSolutionsValidAndBothImplementationsEnabledThenInstanceRemainsUnchanged) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, true, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };
    auto gpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };

    measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(instance.sequence, std::vector<int>({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}));
}


TEST(measure_algorithm_for_instance, whenBothSolutionsValidAndBothImplementationsEnabledThenSaveBothResults) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, true, true, true);
    double cpu_measurement_result = -2137, gpu_measurement_result = -2137;
    auto cpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };
    auto gpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };

    measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_TRUE(cpu_measurement_result > 0);
    EXPECT_TRUE(gpu_measurement_result > 0);
}


TEST(measure_algorithm_for_instance, whenBothSolutionsValidAndOnlyCpuImplementationEnabledThenSaveCpuResultOnly) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(false, true, true, true);
    double cpu_measurement_result = -2137, gpu_measurement_result = -2137;
    auto cpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };
    auto gpu_implementation = [](std::vector<int>& array) {};

    measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_TRUE(cpu_measurement_result > 0);
    EXPECT_EQ(gpu_measurement_result, data::MEASUREMENT_NOT_PERFORMED);
}


TEST(measure_algorithm_for_instance, whenBothSolutionsValidAndOnlyGpuImplementationEnabledThenSaveGpuResultOnly) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, false, true, true);
    double cpu_measurement_result = -2137, gpu_measurement_result = -2137;
    auto cpu_implementation = [](std::vector<int>& array) {};
    auto gpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };

    measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(cpu_measurement_result, data::MEASUREMENT_NOT_PERFORMED);
    EXPECT_TRUE(gpu_measurement_result > 0);
}


TEST(measure_algorithm_for_instance, whenBothSolutionsValidAndNoImplementationEnabledThenSaveNoResults) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(false, false, true, true);
    double cpu_measurement_result = -2137, gpu_measurement_result = -2137;
    auto cpu_implementation = [](std::vector<int>& array) {};
    auto gpu_implementation = [](std::vector<int>& array) {};

    measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(cpu_measurement_result, data::MEASUREMENT_NOT_PERFORMED);
    EXPECT_EQ(gpu_measurement_result, data::MEASUREMENT_NOT_PERFORMED);
}


TEST(measure_algorithm_for_instance, whenBothSolutionsValidAndBothImplementationsEnabledThenReturnValidSolutionCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, true, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };
    auto gpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };

    data::solution_validation_data_t verification_data = measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(verification_data.validation_code, data::SOLUTION_VALID);
}


TEST(measure_algorithm_for_instance, whenOnlyGpuSolutionValidAndBothImplementationsEnabledThenReturnCpuInvalidCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, true, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {};
    auto gpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };

    data::solution_validation_data_t verification_data = measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(verification_data.validation_code, data::CPU_SOLUTION_ERROR);
}


TEST(measure_algorithm_for_instance, whenOnlyCpuSolutionValidAndBothImplementationsEnabledThenReturnGpuInvalidCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, true, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };
    auto gpu_implementation = [](std::vector<int>& array) {};

    data::solution_validation_data_t verification_data = measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(verification_data.validation_code, data::GPU_SOLUTION_ERROR);
}


TEST(measure_algorithm_for_instance, whenNoValidAndBothImplementationsEnabledThenReturnInvalidSolutionCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, true, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {};
    auto gpu_implementation = [](std::vector<int>& array) {};

    data::solution_validation_data_t verification_data = measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(verification_data.validation_code, data::SOLUTION_INVALID);
}


TEST(measure_algorithm_for_instance, whenBothImplementationsDisabledThenReturnValidSolutionCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(false, false, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {};
    auto gpu_implementation = [](std::vector<int>& array) {};

    data::solution_validation_data_t verification_data = measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(verification_data.validation_code, data::SOLUTION_VALID);
}


TEST(measure_algorithm_for_instance, whenCpuDisabledAndGpuSolutionValidThenReturnValidSolutionCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(true, false, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {};
    auto gpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };

    data::solution_validation_data_t verification_data = measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(verification_data.validation_code, data::SOLUTION_VALID);
}


TEST(measure_algorithm_for_instance, whenGpuDisabledAndCpuSolutionValidThenReturnValidSolutionCode) {
    data::instance_t instance({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 15);
    config::configuration_t current_config(false, true, true, true);
    double cpu_measurement_result, gpu_measurement_result;
    auto cpu_implementation = [](std::vector<int>& array) {
        return std::sort(array.begin(), array.end());
    };
    auto gpu_implementation = [](std::vector<int>& array) {};

    data::solution_validation_data_t verification_data = measurement::measure_algorithm_for_instance(
        instance,
        cpu_implementation,
        gpu_implementation,
        current_config,
        &cpu_measurement_result,
        &gpu_measurement_result
    );

    EXPECT_EQ(verification_data.validation_code, data::SOLUTION_VALID);
}