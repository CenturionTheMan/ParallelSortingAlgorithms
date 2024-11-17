#ifndef VERIFICATION_H
#define VERIFICATION_H

#include "data.h"

/**
 * @brief Utilities for verifying if solutions of sorting problem were correct.
 */
namespace verification {
    /**
     * @brief Checks if solution is valid.
     * 
     * @param solution Solution to be checked.
     * @return `true` if solution is valid.
     * @return `false` if solution is not valid.
     */
    bool solution_is_valid(data::instance_t solution, bool enable);
}

#endif