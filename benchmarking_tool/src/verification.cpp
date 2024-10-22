#include "verification.h"

bool verification::solution_is_valid(data::instance_t solution)
{
    auto greater = solution.sequence.begin(); greater++;
    auto lesser = solution.sequence.begin();
    while (greater != solution.sequence.end()){
        if (*greater < *lesser)
            return false;
        greater++;
        lesser++;
    }
    return true;
}