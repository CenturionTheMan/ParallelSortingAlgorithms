#pragma once

#include <thread>
#include <functional>
#include "ThreadPool.h"


class ParallelFor
{
private:
    std::unique_ptr<ThreadPool> pool;

public:
    ParallelFor(unsigned int threadsAmount = std::thread::hardware_concurrency() - 1);
    ParallelFor(std::unique_ptr<ThreadPool> pool);
    ~ParallelFor();

    void Run(int from, int to, std::function<void(int)> func);
    void RunAndWait(int from, int to, std::function<void(int)> func);

    void RunDoubleDimension(int fromX, int toX, int fromY, int toY, std::function<void(int, int)> func);
    void RunDoubleDimensionAndWait(int fromX, int toX, int fromY, int toY, std::function<void(int, int)> func);
};






