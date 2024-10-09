#include "./../headers/ThreadPool.h"

ThreadPool::ThreadPool(int threadsAmount)
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
                std::unique_lock<std::mutex> lock(ThreadPool::tasksQueueMutex);

                //check id tasks available (additional check for stop request)
                ThreadPool::condition.wait(lock, [this]() 
                { 
                    return !ThreadPool::tasks.empty() || ThreadPool::forceStop || (ThreadPool::stopWhenEmpty && ThreadPool::tasks.empty()); 
                });

                //check if should stop
                if (ThreadPool::forceStop) return;
                if(ThreadPool::stopWhenEmpty && ThreadPool::tasks.empty()) return;

                //get task from queue
                std::function<void()> task = ThreadPool::tasks.front();

                //remove task from queue
                ThreadPool::tasks.pop();

                //unlock "lock", for other threads
                lock.unlock();

                //run task
                task();
            }
        });

        //add thread to vector
        ThreadPool::threads.emplace_back(
            move(t) //some cpp magic to add vector to collection
        );
    }
}

ThreadPool::~ThreadPool()
{
    //lock stuff
    std::unique_lock<std::mutex> lock(ThreadPool::tasksQueueMutex);

    //set stop request
    ThreadPool::forceStop = true;

    //notify all threads
    ThreadPool::condition.notify_all();

    //unlock "lock", for other threads
    lock.unlock();

    //wait for all threads to finish
    for (auto &t : ThreadPool::threads)
    {
        if(t.joinable())
            t.join();
    }
}

void ThreadPool::AddTask(std::function<void()> task)
{
    std::unique_lock<std::mutex> lock(ThreadPool::tasksQueueMutex);
    ThreadPool::tasks.push(task);
    ThreadPool::condition.notify_one();
}

void ThreadPool::StopAll()
{
    std::unique_lock<std::mutex> lock(ThreadPool::tasksQueueMutex);
    ThreadPool::forceStop = true;
    ThreadPool::condition.notify_all();
    lock.unlock();
}

void ThreadPool::WaitForAllAndStop()
{
    std::unique_lock<std::mutex> lock(ThreadPool::tasksQueueMutex);
    ThreadPool::stopWhenEmpty = true;
    ThreadPool::condition.notify_all();
    lock.unlock();

    for (auto &t : ThreadPool::threads)
    {
        if(t.joinable())
            t.join();
    }
}





