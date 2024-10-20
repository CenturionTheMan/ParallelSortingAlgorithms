#include <gtest/gtest.h>

#include "verification.h"


TEST(solution_is_valid, givenSolutionIsValid) {
    data::instance_t solution({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, 10);

    EXPECT_EQ(verification::solution_is_valid(solution), true);
}


TEST(solution_is_valid, givenSolutionIsNotValid) {
    data::instance_t solution({8, 9, 3, 2, 10, 5, 6, 7, 4, 1}, 10);

    EXPECT_EQ(verification::solution_is_valid(solution), false);
}