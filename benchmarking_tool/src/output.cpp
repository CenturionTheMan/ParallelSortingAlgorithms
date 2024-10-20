#include "output.h"

output::ResultsOutputStream &output::ResultsOutputStream::getStream()
{
    throw std::logic_error("Not yet implemented!");
}

void output::ResultsOutputStream::dumpResult(const data::results_t &results)
{
    throw std::logic_error("Not yet implemented!");
}

void output::ResultsOutputStream::printAverageResult(const data::results_t &average_results)
{
    throw std::logic_error("Not yet implemented!");
}

void output::ResultsOutputStream::open()
{
    throw std::logic_error("Not yet implemented!");
}

void output::ResultsOutputStream::close()
{
}

output::ResultsOutputStream::ResultsOutputStream()
{
    throw std::logic_error("Not yet implemented!");
}

output::ResultsOutputStream::~ResultsOutputStream()
{
}

void output::saveAndPrintErrorOutput(const data::solution_validation_data_t& error)
{
    throw std::logic_error("Not yet implemented!");
}

void output::printConfigurationOutput(const config::configuration_t &current_configuration)
{
    throw std::logic_error("Not yet implemented!");
}

void output::print_notification(std::string message)
{
    std::cout<<">>> "<<message<<"\n";
}
