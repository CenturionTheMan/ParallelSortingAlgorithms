#include <fstream>
#include <sstream>
#include <regex>
#include <random>

#include "config.h"
#include "output.h"


data::instance_t __loadPredefinedInstanceFromRegexMatch(std::smatch match) {
    std::stringstream sequence_parser(match[2]);
    std::list<int> sequence;
    std::string element;

    while (!sequence_parser.eof()){
        sequence_parser>>element;
        sequence.emplace_back(std::stoi(element));
    }

    return data::instance_t(sequence, std::stoi(match[1]));
}


data::instance_t __generateRandomInstance(int size, int repetitions) {
    std::mt19937 random_engine;
    std::uniform_int_distribution<int> distribution(INT_MIN, INT_MAX);
    std::list<int> sequence;

    for (int i = 0; i < size; i++)
        sequence.emplace_back(distribution(random_engine));
    
    return data::instance_t(sequence, repetitions);
}


void __checkRequiredOptionAndThrowErrorIfItIsNotLoaded(
    std::pair<std::regex, bool> required_option, std::string option_key
) {
    if (required_option.second)
        return;
    output::printNotification("BENCHMARK TERMINATED!");
    output::printNotification("ERROR: Configuration file misses \"" + option_key + "\"!");
    throw std::runtime_error("Configuration line misses \"" + option_key + "\"!");
}


config::configuration_t config::loadConfiguration()
{
    std::ifstream configuration_file("configuration.ini");
    if (!configuration_file.good()){
        output::printNotification("BENCHMARK TERMINATED!");
        output::printNotification("ERROR: Configuration file is invalid or doesn't exist!");
        throw std::runtime_error("Configuration file not exist!");
    }
    config::configuration_t configuration;
    std::string line;

    std::pair<std::regex, bool> measure_gpu("measure_gpu\\s*=(0|1)\\s*", false);
    std::pair<std::regex, bool> measure_cpu("measure_cpu\\s*=(0|1)\\s*", false);
    std::pair<std::regex, bool> measure_bitonic("measure_bitonic\\s*=\\s*(0|1)\\s*", false);
    std::pair<std::regex, bool> measure_odd_even("measure_odd_even\\s*=\\s*(0|1)\\s*", false);
    std::regex PREDEFINED_INSTANCE_LINE("predefined_instance\\s*=\\s*(\\d+)\\s+((?:-?\\d+\\s*)+)");
    std::regex RANDOM_INSTANCE_LINE("random_instance\\s*=\\s*(\\d+)\\s+(\\d+)");

    while (!configuration_file.eof()) {
        std::getline(configuration_file, line);

        std::smatch match;
        if (std::regex_search(line, match, measure_gpu.first)){
            configuration.measure_gpu = std::stoi(match[1]);
            measure_gpu.second = true;
        }
        else if (std::regex_search(line, match, measure_cpu.first)){
            configuration.measure_cpu = std::stoi(match[1]);
            measure_cpu.second = true;
        }
        else if (std::regex_search(line, match, measure_bitonic.first)){
            configuration.measure_bitonic = std::stoi(match[1]);
            measure_bitonic.second = true;
        }
        else if (std::regex_search(line, match, measure_odd_even.first)){
            configuration.measure_odd_even = std::stoi(match[1]);
            measure_odd_even.second = true;
        }
        else if (std::regex_match(line, match, PREDEFINED_INSTANCE_LINE))
            configuration.loaded_instances.emplace(__loadPredefinedInstanceFromRegexMatch(match));
        else if (std::regex_match(line, match, RANDOM_INSTANCE_LINE))
            configuration.loaded_instances.emplace(__generateRandomInstance(std::stoi(match[1]), std::stoi(match[2])));
    }
    configuration_file.close();

    __checkRequiredOptionAndThrowErrorIfItIsNotLoaded(measure_bitonic, "measure_bitonic");
    __checkRequiredOptionAndThrowErrorIfItIsNotLoaded(measure_odd_even, "measure_odd_even");
    __checkRequiredOptionAndThrowErrorIfItIsNotLoaded(measure_cpu, "measure_cpu");
    __checkRequiredOptionAndThrowErrorIfItIsNotLoaded(measure_gpu, "measure_gpu");

    return configuration;
}