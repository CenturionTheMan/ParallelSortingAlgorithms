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
    throw std::logic_error("Not yet implemented!");
}

data::instance_t::instance_t(std::list<int> sequence, int repetitions): repetitions(repetitions)  {
    this->sequence.resize(sequence.size());
    int index = 0;
    for (int element : sequence) {
        this->sequence[index] = element;
        index++;
    }
}
