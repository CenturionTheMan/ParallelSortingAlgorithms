#include <fstream>
#include <string>
#include <sstream>

#include "output.h"


const char* TABLE_SEPARATOR = "#===============#=========================#=========================#==========================#==========================#\n";


output::ResultsOutputStream &output::ResultsOutputStream::getStream()
{
    static output::ResultsOutputStream s_instance;
    return s_instance;
}

void output::ResultsOutputStream::dumpResult(const data::results_t &results)
{
    auto resultToString = [](double result) {
        return (result == data::MEASUREMENT_NOT_PERFORMED ? "" : std::to_string(result));
    };

    results_file << results.instance_size << ";";
    results_file << resultToString(results.cpu_bitonic_time_seconds) << ";";
    results_file << resultToString(results.gpu_bitonic_time_seconds) << ";";
    results_file << resultToString(results.cpu_odd_even_time_seconds) << ";";
    results_file << resultToString(results.gpu_odd_even_time_seconds) << "\n";
}

void output::ResultsOutputStream::printAverageResult(const data::results_t &average_results)
{
    auto resultToString = [](double result, double std_derivation) {
        char string_result[24];
        if (result != data::MEASUREMENT_NOT_PERFORMED)
            sprintf(string_result, "%.2e (%.2e) s", result, std_derivation);
        else
            sprintf(string_result, "");
        return std::string(string_result);
    };

    printf(
        "| %13d | %23s | %23s | %24s | %24s |\n",
        average_results.instance_size,
        resultToString(average_results.cpu_bitonic_time_seconds, average_results.cpu_bitonic_std_deviation).c_str(),
        resultToString(average_results.gpu_bitonic_time_seconds, average_results.gpu_bitonic_std_deviation).c_str(),
        resultToString(average_results.cpu_odd_even_time_seconds, average_results.cpu_odd_even_std_deviation).c_str(),
        resultToString(average_results.gpu_odd_even_time_seconds, average_results.gpu_odd_even_std_deviation).c_str()
    );
}

void output::ResultsOutputStream::open()
{
    if (opened)
        throw std::logic_error("Results output stream is already opened!");
    opened = true;

    output::printNotification("STARTING BENCHMARK...");
    printf("\n");
    printf("%s", TABLE_SEPARATOR);
    printf("%s", "| Instance size |       CPU Bitonic       |       GPU Bitonic       |       CPU Odd-Even       |       GPU Odd-Even       |\n");
    printf("%s", TABLE_SEPARATOR);

    results_file.open("results.csv", std::ios::out);
    if (!results_file.good())
        throw std::runtime_error("Something went wrong while creating \"results.csv\" file!");
    results_file << "instance size;mean bitonic sort (CPU);mean bitonic sort (GPU);mean odd-even sort (CPU);mean odd-event sort (GPU)\n";
}

void output::ResultsOutputStream::close()
{
    if (!opened)
        throw std::logic_error("Results output stream hasn't been opened yet!");
    opened = false;
    results_file.close();

    printf("%s", TABLE_SEPARATOR);
}

void output::waitForReturn()
{
    output::printNotification("Press ENTER to continue...");
    getchar();
}

output::ResultsOutputStream::ResultsOutputStream(): opened(false) {}

output::ResultsOutputStream::~ResultsOutputStream() {}

void output::saveAndPrintErrorOutput(const data::solution_validation_data_t& error, std::string algorithm_name)
{
    std::string implementation_name;
    switch (error.validation_code)
    {
    case data::CPU_SOLUTION_ERROR:
        implementation_name = "The CPU";
        break;
    case data::GPU_SOLUTION_ERROR:
        implementation_name = "The GPU";
        break;
    case data::SOLUTION_INVALID:
        implementation_name = "Both implementations of";
        break;
    default:
        throw std::invalid_argument("Invalid validation code: " + std::to_string((int)error.validation_code));
    }
    std::string error_message = 
        implementation_name + " " + algorithm_name + " has given an invalid solution for instance size " + 
        std::to_string(error.instance.sequence.size()) + " in repetition " + std::to_string(error.repetition) + ".";
    std::string instance_data = "[Instance]:";
    for (int element : error.instance.sequence)
        instance_data += " " + std::to_string(element);
    std::string solution_data = "[Solution]:";
    for (int element : error.solution.sequence)
        solution_data += " " + std::to_string(element);

    printNotification("BENCHMARK TERMINATED!");
    printNotification(error_message);
    printNotification("Please check the \"error.log\" file.");

    std::fstream error_log("error.log", std::ios::out);
    if (!error_log.good())
        throw std::runtime_error("Something went wrong while dumping errors to \"error.log\"!");
    error_log << ">>> " << error_message << "\n";
    error_log << instance_data << "\n";
    error_log << solution_data;
    error_log.close();
}

void output::printConfigurationOutput(const config::configuration_t &current_configuration)
{
    const char* ODD_EVEN_SETUP_LABEL = "Odd-Even Sort measurement";
    const char* CPU_SETUP_LABEL = "CPU measurement";
    const char* GPU_SETUP_LABEL = "GPU measurement";
    const char* BITONIC_SETUP_LABEL = "Bitonic Sort measurement";
    const char* INSTANCES_SETUP_LABEL = "Defined instances";
    const char* VERIFY_LABEL = "Verify";

    auto printConfigRow = [](const char* label = "", const char* value = "") {
        printf("%-32s%s\n", label, value);
    };

    auto setupValueToString = [](const bool& setup_value) {
        return (setup_value ? "ON" : "OFF");
    };

    output::printNotification("CONFIGURATION LOADED");
    printf("\n");
    printConfigRow(CPU_SETUP_LABEL, setupValueToString(current_configuration.measure_cpu));
    printConfigRow(GPU_SETUP_LABEL, setupValueToString(current_configuration.measure_gpu));
    printConfigRow(BITONIC_SETUP_LABEL, setupValueToString(current_configuration.measure_bitonic));
    printConfigRow(ODD_EVEN_SETUP_LABEL, setupValueToString(current_configuration.measure_odd_even));
    printConfigRow(VERIFY_LABEL, setupValueToString(current_configuration.verify_results));
    printf("\n");
    printConfigRow(INSTANCES_SETUP_LABEL, std::to_string(current_configuration.loaded_instances.size()).c_str());
    printf("\n");
}

void output::printNotification(std::string message)
{
    std::cout<<">>> "<<message<<"\n";
}
