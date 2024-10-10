#include "./../headers/ParallelFor.h"

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

void ParallelFor::Run(int from, int to, std::function<void(int)> func)
{
    for (int i = from; i < to; i++)
    {
        ParallelFor::pool->AddTask([func, i]() {
            func(i);
        });
    }
}

void ParallelFor::RunAndWait(int from, int to, std::function<void(int)> func)
{
    ParallelFor::Run(from, to, func);
    ParallelFor::pool->WaitForAllAndStop();
}

void ParallelFor::RunDoubleDimension(int fromI, int toI, int fromJ, int toJ, std::function<void(int, int)> func)
{
    for (int i = fromI; i < toI; i++)
    {
        for (int j = fromJ; j < toJ; j++)
        {
            ParallelFor::pool->AddTask([func, i, j](){
                func(i, j);
            });
        }
    }
}

void ParallelFor::RunDoubleDimensionAndWait(int fromI, int toI, int fromJ, int toJ, std::function<void(int, int)> func)
{
    ParallelFor::RunDoubleDimension(fromI, toI, fromJ, toJ, func);
    ParallelFor::pool->WaitForAllAndStop();
}