#include "data.h"

data::results_t &data::results_t::operator+=(const data::results_t &other)
{
    this->cpu_bitonic_time_seconds += other.cpu_bitonic_time_seconds;
    this->gpu_bitonic_time_seconds += other.gpu_bitonic_time_seconds;
    this->cpu_odd_even_time_seconds += other.cpu_odd_even_time_seconds;
    this->gpu_odd_even_time_seconds += other.gpu_odd_even_time_seconds;
    return *this;
}

data::results_t &data::results_t::operator/=(int n)
{
    this->cpu_bitonic_time_seconds /= n;
    this->gpu_bitonic_time_seconds /= n;
    this->cpu_odd_even_time_seconds /= n;
    this->gpu_odd_even_time_seconds /= n;
    return *this;
}

data::results_t data::results_t::calculateMeanAndStandardDeviation(std::list<data::results_t> &results_from_repetitions, int instance_size)
{
    data::results_t average_result(instance_size);
    for (data::results_t results_for_repetition : results_from_repetitions)
        average_result += results_for_repetition;
    average_result /= results_from_repetitions.size();

    for (data::results_t results_from_repetition : results_from_repetitions) {
        average_result.cpu_bitonic_std_deviation += pow(
            results_from_repetition.cpu_bitonic_time_seconds - average_result.cpu_bitonic_time_seconds, 2
        );
        average_result.gpu_bitonic_std_deviation += pow(
            results_from_repetition.gpu_bitonic_time_seconds - average_result.gpu_bitonic_time_seconds, 2
        );
        average_result.cpu_odd_even_std_deviation += pow(
            results_from_repetition.cpu_odd_even_time_seconds - average_result.cpu_odd_even_time_seconds, 2
        );
        average_result.gpu_odd_even_std_deviation += pow(
            results_from_repetition.gpu_odd_even_time_seconds - average_result.gpu_odd_even_time_seconds, 2
        );
    }
    average_result.gpu_bitonic_std_deviation = sqrt(average_result.gpu_bitonic_std_deviation / results_from_repetitions.size());
    average_result.cpu_bitonic_std_deviation = sqrt(average_result.cpu_bitonic_std_deviation / results_from_repetitions.size());
    average_result.cpu_odd_even_std_deviation = sqrt(average_result.cpu_odd_even_std_deviation / results_from_repetitions.size());
    average_result.gpu_odd_even_std_deviation = sqrt(average_result.gpu_odd_even_std_deviation / results_from_repetitions.size());

    return average_result;
}

data::instance_t::instance_t(std::list<int> sequence, int repetitions): repetitions(repetitions)  {
    this->sequence.resize(sequence.size());
    int index = 0;
    for (int element : sequence) {
        this->sequence[index] = element;
        index++;
    }
}
