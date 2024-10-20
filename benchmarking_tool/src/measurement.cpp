#include "measurement.h"

#include "verification.h"
#include "output.h"

measurement::Timer::Timer(double *result_address)
{
    m_result_address = result_address;
    m_start = std::chrono::high_resolution_clock::now();
}

measurement::Timer::~Timer()
{
    m_end = std::chrono::high_resolution_clock::now();
    m_result = m_end - m_start;
    *m_result_address = m_result.count();
}

data::solution_validation_data_t measurement::measure_implementation_for_instance(
    const data::instance_t& instance, 
    std::function<void(std::vector<int>&)> implementation, 
    double *result_location
) {
    data::instance_t solved_instance = instance;

    {
        Timer start(result_location);
        implementation(solved_instance.sequence);
    }

    return data::solution_validation_data_t(
        data::SOLUTION_VALID ? verification::solution_is_valid(solved_instance) : data::SOLUTION_INVALID,
        instance,
        solved_instance
    );
}

data::solution_validation_data_t measurement::measure_algorithm_for_instance(
    const data::instance_t& instance,
    std::function<void(std::vector<int>&)> cpu_implementation, 
    std::function<void(std::vector<int>&)> gpu_implementation, 
    const config::configuration_t &current_configuration, 
    double *cpu_result_location, 
    double *gpu_result_location
) {
    if (current_configuration.measure_cpu) {
        data::solution_validation_data_t verification_data = measure_implementation_for_instance(
            instance, cpu_implementation, cpu_result_location
        );
        if (verification_data.validation_code == data::SOLUTION_INVALID)
            return verification_data;
    }
    if (current_configuration.measure_gpu) {
        data::solution_validation_data_t verification_data = measure_implementation_for_instance(
            instance, gpu_implementation, gpu_result_location
        );
        if (verification_data.validation_code == data::SOLUTION_INVALID)
            return verification_data;
    }
    return data::solution_validation_data_t(data::SOLUTION_INVALID, instance, instance);
}
