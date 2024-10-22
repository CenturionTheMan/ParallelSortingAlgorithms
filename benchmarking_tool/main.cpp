#include <functional>
#include <iostream>

#include "bitonic_sort.cuh"
#include "config.h"
#include "measurement.h"
#include "odd_even_sort.cuh"
#include "output.h"


int main()
{
    config::configuration_t current_configuration = config::loadConfiguration();
    output::printConfigurationOutput(current_configuration);
    output::ResultsOutputStream& result_output = output::ResultsOutputStream::getStream();
    result_output.open();

    while (!current_configuration.loaded_instances.empty())
    {
        data::instance_t instance = current_configuration.loaded_instances.top();
        current_configuration.loaded_instances.pop();

        data::results_t average_result(instance.sequence.size());

        for (int repetition = 0; repetition < instance.repetitions; repetition++) {
            data::results_t result(instance.sequence.size());

            if (current_configuration.measure_bitonic) {
                data::solution_validation_data_t validation_data = measurement::measure_algorithm_for_instance(
                    instance,
                    sorting::CpuBitonicSort,
                    sorting::GpuBitonicSort,
                    current_configuration, 
                    &result.cpu_bitonic_time_seconds,
                    &result.gpu_bitonic_time_seconds
                );

                if (validation_data.validation_code != data::SOLUTION_VALID) {
                    validation_data.repetition = repetition;
                    output::saveAndPrintErrorOutput(validation_data, "Bitonic Sort");
                    output::waitForReturn();
                    return 100 + validation_data.validation_code;
                }
            }
            if (current_configuration.measure_odd_even) {
                data::solution_validation_data_t validation_data = measurement::measure_algorithm_for_instance(
                    instance,
                    sorting::CpuOddEvenSort, 
                    sorting::GpuOddEvenSort,
                    current_configuration, 
                    &result.cpu_odd_even_time_seconds,
                    &result.gpu_odd_even_time_seconds
                );
                
                if (validation_data.validation_code != data::SOLUTION_VALID) {
                    validation_data.repetition = repetition;
                    output::saveAndPrintErrorOutput(validation_data, "Odd-Even Sort");
                    output::waitForReturn();
                    return 200 + validation_data.validation_code;
                }
            }
            result_output.dumpResult(result);
            
            average_result += result;
        }

        average_result /= instance.repetitions;
        result_output.printAverageResult(average_result);
    }

    result_output.close();
    printf("\n");
    output::printNotification("BENCHMARK COMPLETE!");
    output::waitForReturn();
    return 0;
}