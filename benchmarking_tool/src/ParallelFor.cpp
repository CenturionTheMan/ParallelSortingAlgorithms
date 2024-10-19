#include "ParallelFor.h"

ParallelFor::ParallelFor(unsigned int threadsAmount)
{
    ParallelFor::pool = std::make_unique<ThreadPool>(threadsAmount);
}

ParallelFor::ParallelFor(std::unique_ptr<ThreadPool> pool)
{
    ParallelFor::pool = std::move(pool);
}

ParallelFor::~ParallelFor()
{

}

void ParallelFor::Run(int from, int to, int step, std::function<void(int)> func)
{
    for (int i = from; i < to; i += step)
    {
        ParallelFor::pool->AddTask([func, i]() {
            func(i);
            });
    }
}

void ParallelFor::RunAndWait(int from, int to, int step, std::function<void(int)> func)
{
    ParallelFor::Run(from, to, step, func);
    ParallelFor::pool->WaitForAllAndStop();
}

void ParallelFor::RunDoubleDimension(int fromI, int toI, int stepI, int fromJ, int toJ, int stepJ, std::function<void(int, int)> func)
{
    for (int i = fromI; i < toI; i++)
    {
        for (int j = fromJ; j < toJ; j++)
        {
            ParallelFor::pool->AddTask([func, i, j]() {
                func(i, j);
                });
        }
    }
}

void ParallelFor::RunDoubleDimensionAndWait(int fromI, int toI, int stepI, int fromJ, int toJ, int stepJ, std::function<void(int, int)> func)
{
    ParallelFor::RunDoubleDimension(fromI, toI, stepI, fromJ, toJ, stepJ, func);
    ParallelFor::pool->WaitForAllAndStop();
}