#ifndef MEASUREMENT_H
#define MEASUREMENT_H


#include <chrono>
#include <functional>

#include "config.h"
#include "data.h"


namespace measurement {
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> timestamp_t;


    /**
     *  @brief Measures given implementation of given algorithm.
     *  
     *  @param instance Instance to be solved.
     *  @param implementation Given implementation.
     *  @param result_location Memory location of measurement result.
     *  
     *  @return `solution_validation_data_t` with `data::SOLUTION_VALID` verification code when solution is valid
     *  or with `data::SOLUTION_INVALID` verification code when solution is invalid.
     */
    data::solution_validation_data_t measure_implementation_for_instance(
        const data::instance_t& instance, std::function<void(std::vector<int>&)> implementation, double* result_location
    );


    /**
     *  @brief Measures all enabled implementations for given algorithm.
     *  
     *  @param instance Instance to be solved.
     *  @param cpu_implementation CPU implementation of given algorithm
     *  @param gpu_implementation GPU implementation of given algorithm
     *  @param current_configuration Tool's current configuration.
     *  @param cpu_result_location Memory location of measurement result for CPU implementation.
     *  @param gpu_result_location Memory location of measurement result for GPU implementation.
     *  
     *  @return - `solution_validation_data_t` with `data:SOLUTION_VALID` verification code when all implementations
     *  gave valid solutions
     *  @return - `solution_validation_data_t` with `data::GPU_SOLUTION_ERROR` when GPU implementation gave invalid 
     *  solution (in such case measurement results are undefined!)
     *  @return - `solution_validation_data_t` with `data::CPU_SOLUTION_ERROR` when GPU implementation gave invalid 
     *  solution (in such case measurement results are undefined!)
     *  @return - `solution_validation_data_t` with `data::SOLUTION_INVALID` when both implementations gave invalid 
     *  solutions (in such case measurement results are undefined!)
     */
    data::solution_validation_data_t measure_algorithm_for_instance(
        const data::instance_t& instance,
        std::function<void(std::vector<int>&)> cpu_implementation,
        std::function<void(std::vector<int>&)> gpu_implementation,
        const config::configuration_t& current_configuration,
        double* cpu_result_location,
        double* gpu_result_location
    );

    /**
     *  @brief Takes care of time measurements. Measures how many seconds it has been alive and saves it in given
     *  memory location.
     */
    class Timer
    {
    public:
        /**
         * @brief Creates and start the timer.
         * 
         * @param result_address Memory location where measurement result would be saved.
         */
        Timer(double* result_address);

        /**
         *  @brief Destroys and stops timer. Saves result to previously specified memory location.
         */
        ~Timer();
    private:
        measurement::timestamp_t m_start;
        measurement::timestamp_t m_end;
        std::chrono::duration<double> m_result;
        double* m_result_address;
    };
}

#endif