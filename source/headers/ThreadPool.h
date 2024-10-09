#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool
{
private:
    
    std::vector<std::thread> threads;
    std::queue<std::function<void()>> tasks;
    std::mutex tasksQueueMutex;
    std::condition_variable condition;

    bool stopRequested = false;

public:
    ThreadPool(int threadsAmount = std::thread::hardware_concurrency());
    ~ThreadPool();
};

ThreadPool::ThreadPool(int threadsAmount = std::thread::hardware_concurrency())
{
    //for each thread
    for (int i = 0; i < threadsAmount; i++)
    {
        //create a thread
        auto t = std::thread([this]() {
            //run
            while (true)
            {
                //lock stuff
                std::unique_lock<std::mutex> lock(tasksQueueMutex);

                //check id tasks available (additional check for stop request)
                condition.wait(lock, [this]() 
                { 
                    return stopRequested || !tasks.empty(); 
                });

                //if stop requested - STOP
                if (stopRequested) return;

                //get task from queue
                std::function<void()> task = tasks.front();

                //remove task from queue
                tasks.pop();

                //unlock "lock", for other threads
                lock.unlock();

                //run task
                task();
            }
        });

        if(t.joinable())
            t.detach();

        //add thread to vector
        threads.emplace_back(t);
    }
}

ThreadPool::~ThreadPool()
{
    //lock stuff
    std::unique_lock<std::mutex> lock(tasksQueueMutex);

    //set stop request
    stopRequested = true;

    //notify all threads
    condition.notify_all();

    //unlock "lock", for other threads
    lock.unlock();

    //join all threads
    for (auto &t : threads)
    {
        if(t.joinable())
            t.join();
    }
}
