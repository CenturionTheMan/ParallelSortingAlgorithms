#ifndef OUTPUT_H
#define OUTPUT_H


#include <exception>
#include <fstream>
#include <iostream>

#include "config.h"
#include "data.h"


/*
 *  Contains all functionalities related to generating benchmarking tool's output. There are 2 ways of
 *  creating output:
 *  
 *  1. Printing data to the command line.
 *  2. Saving (dumping) data to file.
 */
namespace output {
    /**
     *  @brief Saves information about invalid solution to the `error.log` file and prints information about it.
     *  
     *  @param error Error data.
     *  @param algorithm_name Name of an algorithm that gave the invalid solution.
     */
    void saveAndPrintErrorOutput(const data::solution_validation_data_t& error, std::string algorithm_name);


    /**
     *  @brief Prints current tool's configuration.
     * 
     *  @param current_configuration Tool's current configuration.
     */
    void printConfigurationOutput(const config::configuration_t& current_configuration);
        
        
    /**
     * @brief Halts execution until user presses ENTER.
     */
    void waitForReturn();


    /**
     *  @brief Handles everything related to outputting time measurements results. There could only be one instance of
     *  `ResultsOutputStream`.
     */
    class ResultsOutputStream {
    public:
        /**
         *  @brief Gets instance of `ResultsOutput`.
         */
        static output::ResultsOutputStream& getStream();

        /**
         *  @brief Saves time measurement results to the next row in `results.csv` file. Works only when output is
         *  opened.
         *  
         *  @param results Time measurement results for a particular repetition and instance size.
         *  
         *  @throws `std::logic_error` – when output is closed
         */
        void dumpResult(const data::results_t& results);

        /**
         *  @brief Prints a table entry with average time measurement results for the particular instance size. Works
         *  only when output is opened.
         *  
         *  @param average_results Average time measurement results for a particular instance size.
         *  
         *  @throws `std::logic_error` – when output is closed
         */
        void printAverageResult(const data::results_t& average_results);

        /**
         *  @brief Opens output file and prints table header. Sets stream's state to "opened". This can't be called
         *  multiple times in a row.
         * 
         *  @throws `std::logic_error` – when output is opened
         */
        void open();

        /**
         * @brief Closes output file and prints end of table. Sets stream's state to "closed".
         */
        void close();

        bool isOpen() const { return opened; }
        
        ResultsOutputStream(ResultsOutputStream &other) = delete;
        void operator=(const ResultsOutputStream &) = delete;

    protected:
        ResultsOutputStream();
        ~ResultsOutputStream();
    
    private:
        bool opened = false;
        std::fstream results_file;
    };


    /**
     * @brief Prints a notification with given message.
     */
    void printNotification(std::string message);
}

#endif