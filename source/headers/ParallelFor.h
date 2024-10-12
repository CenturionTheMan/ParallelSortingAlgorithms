#pragma once

#include <thread>
#include <functional>
#include "ThreadPool.h"


class ParallelFor
{
private:
    std::unique_ptr<ThreadPool> pool;

public:
    ParallelFor(std::unique_ptr<ThreadPool> pool);
    ~ParallelFor();

    void Run(int from, int to, int step, std::function<void(int)> func);
    void RunAndWait(int from, int to, int step, std::function<void(int)> func);

    void RunDoubleDimension(int fromI, int toI, int stepI, int fromJ, int toJ, int stepJ, std::function<void(int, int)> func);
    void RunDoubleDimensionAndWait(int fromI, int toI, int stepI, int fromJ, int toJ, int stepJ, std::function<void(int, int)> func);
};






