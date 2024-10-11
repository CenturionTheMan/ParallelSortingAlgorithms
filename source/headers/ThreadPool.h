#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool
{
private:
    /// @brief threads
    std::vector<std::thread> threads;

    /// @brief tasks to do
    std::queue<std::function<void()>> tasks;

    /// @brief mutex for tasks
    std::mutex tasksQueueMutex;

    /// @brief condition variable for tasks
    std::condition_variable condition;

    /// @brief flags for force stop tasks
    bool forceStop = false;

    /// @brief flag for stop when empty
    bool stopWhenEmpty = false;

public:
    /// @brief ctor
    /// @param threadsAmount amount of threads to create
    ThreadPool(unsigned int threadsAmount = std::thread::hardware_concurrency() - 1);

    /// @brief dtor
    ~ThreadPool();

    /// @brief add task to queue
    /// @param task task to add
    void AddTask(std::function<void()> task);

    /// @brief stop all threads (as soon as possible)
    void StopAll();

    /// @brief will freeze current thread until all tasks are done, will stop all threads
    void WaitForAllAndStop();
};


