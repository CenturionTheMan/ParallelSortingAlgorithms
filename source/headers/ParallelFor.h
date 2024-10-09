#include <thread>

namespace Thread
{
    const int THREADS_AMOUNT = 2;
    auto threads = new std::thread[THREADS_AMOUNT];
    bool isInitialized = false;


    void ParallelFor(int amount, void (*func)(int* index))
    {
        for (int i = 0; i < amount; i++)
        {
            
        }
        
    }

    std::thread* GetFreeThread()
    {
        for (int i = 0; i < THREADS_AMOUNT; i++)
        {
            if (!threads[i].joinable())
            {
                return &threads[i];
            }
        }
    }


} // namespace Thread
