#ifndef DATA_H
#define DATA_H


#include <iostream>
#include <list>
#include <queue>


/**
 * @brief Contains all data structures, constants and custom types that are used within the tool.
 */
namespace data{
    const double MEASUREMENT_NOT_PERFORMED = 0;
    const char SOLUTION_VALID = 0;
    const char GPU_SOLUTION_ERROR = 1;
    const char CPU_SOLUTION_ERROR = 2;
    const char SOLUTION_INVALID = 3;


    /**
     *  @brief Instance of the sorting problem. It could be both solved or not.
     *  
     *  @param sequence Array of `int` that makes up the instance.
     *  @param repetitions Number of repetitions assigned to this instance.
     */
    struct instance_t
    {
        std::vector<int> sequence;
        int repetitions;

        instance_t(std::list<int> sequence, int repetitions);
        
        // overriden for the sake of priority queue
        friend bool operator<(const data::instance_t less, const data::instance_t more) { 
            return less.sequence.size() > more.sequence.size(); 
        }

        friend bool operator==(const data::instance_t left, const data::instance_t right) { 
            return left.sequence.size() == right.sequence.size(); 
        }
    };
    


    /**
     *  @brief Execution times and standard deviation (in seconds) for each implementation and algorithm that were 
     *  measured in a particular repetition and for particular instance size. All times are set to 
     *  `MEASUREMENT_NOT_PERFORMED` by default.
     */
    struct results_t
    {
        int instance_size;
        double cpu_bitonic_time_seconds = MEASUREMENT_NOT_PERFORMED;
        double cpu_bitonic_std_deviation = MEASUREMENT_NOT_PERFORMED;
        double gpu_bitonic_time_seconds = MEASUREMENT_NOT_PERFORMED;
        double gpu_bitonic_std_deviation = MEASUREMENT_NOT_PERFORMED;
        double cpu_odd_even_time_seconds = MEASUREMENT_NOT_PERFORMED;
        double cpu_odd_even_std_deviation = MEASUREMENT_NOT_PERFORMED;
        double gpu_odd_even_time_seconds = MEASUREMENT_NOT_PERFORMED;
        double gpu_odd_even_std_deviation = MEASUREMENT_NOT_PERFORMED;

        results_t(int instance_size):
            instance_size(instance_size){}

        results_t& operator+=(const results_t& other);
        results_t& operator/=(int n);

        /**
         * @brief Inserts standard deviation into the `average_results` structure.
         * 
         * @param results_from_repetitions Results from all repetitions.
         * @param instance_size Instance size.
         * 
         * @return Average results for all repetitions along with standard derivation.
         */
        static results_t calculateMeanAndStandardDeviation(
            std::list<data::results_t>& results_from_repetitions, int instance_size
        );
    };


    /**
     *  @brief Stores data about results of solution verification process.
     *  
     *  @param validation_code Code that indicates if and where and invalid solution was reached.  
     *  @param repetition Repetition number.
     *  @param instance Instance.
     *  @param solution Solution.
     */
    struct solution_validation_data_t
    {
        char validation_code;
        int repetition = -1;
        instance_t instance;
        instance_t solution;

        solution_validation_data_t(char validation_code, const instance_t& instance, const instance_t& solution):
            validation_code(validation_code),
            instance(instance),
            solution(solution){}
    };
}


#endif