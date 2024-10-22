#ifndef CONFIG_H
#define CONFIG_H

#include "data.h"

/**
 *  @brief Contains functionalities related to obtaining and storing current benchmarking tool's configuration.
 */
namespace config {
    /**
     * @brief Stores current configuration of the benchmarking tool.
     * 
     *  @param loaded_instances Loaded instances to be measured sorted from smallest to biggest in size.
     *  @param measure_gpu Tells if the GPU implementations must mu measured.
     *  @param measure_cpu Tells if the CPU implementations must mu measured.
     *  @param measure_bitonic Tells if Bitonic Sort implementations must be measured.
     *  @param measure_odd_even Tells if Odd-Even Sort implementations must be measured.
     */
    struct configuration_t
    {
        std::priority_queue<data::instance_t, std::vector<data::instance_t>> loaded_instances;
        bool measure_gpu;
        bool measure_cpu;
        bool measure_bitonic;
        bool measure_odd_even;

        configuration_t(
            bool measure_gpu = true,
            bool measure_cpu = true,
            bool measure_odd_even = true,
            bool measure_bitonic = true
        ):
            measure_gpu(measure_gpu),
            measure_cpu(measure_cpu),
            measure_odd_even(measure_odd_even),
            measure_bitonic(measure_bitonic){}
    };


    /**
     * @brief Loads tool's configuration from `configuration.ini` file.
     */
    config::configuration_t loadConfiguration();
}

#endif