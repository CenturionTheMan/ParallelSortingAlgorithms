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
    // threads
    std::vector<std::thread> threads;

    // tasks to do
    std::queue<std::function<void()>> tasks;

    // mutex for tasks
    std::mutex tasksQueueMutex;

    // condition variable for tasks
    std::condition_variable condition;

    // flags for force stop tasks
    bool forceStop = false;

    // flag for stop when empty
    bool stopWhenEmpty = false;

public:
    /// <summary>
    /// ctor
    /// </summary>
    /// <param name="threadsAmount">threadsAmount amount of threads to create</param>
    ThreadPool(unsigned int threadsAmount = std::thread::hardware_concurrency() - 1);

    /// dtor
    ~ThreadPool();

    /// <summary>
    /// add task to queue
    /// </summary>
    /// <param name="task">task task to add</param>
    void AddTask(std::function<void()> task);

    /// <summary>
    /// stop all threads (as soon as possible)
    /// </summary>
    void StopAll();

    /// <summary>
    /// will freeze current thread until all tasks are done, will stop all threads
    /// </summary>
    void WaitForAllAndStop();
};


