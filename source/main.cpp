#include "headers/ThreadPool.h"
#include <iostream>
#include <chrono>
#include <atomic>

int main(int argc, char const *argv[])
{
    // testing thread pool ...
    // It is a lot slower than normal for loop :(, but will work better for more complex tasks (hope so ...)

    int arrSize = 100000;

    ThreadPool pool(10);

    int count = 0;
    int count2 = 0;

    auto start1 = std::chrono::high_resolution_clock::now();    
    for (int i = 0; i < arrSize; i++)
    {
        pool.AddTask([&count, i]() {
            count+=i;    
        });
    }
    auto end1 = std::chrono::high_resolution_clock::now();
    pool.WaitForAllAndStop();
    
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < arrSize; i++)
    {
        count2+=i;
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    std::cout << "Anti-optimization: " << count << " | " << count2 << std::endl; 

    std::cout << "Time parallel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << std::endl;
    std::cout << "Time normal:   " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << std::endl;

    return 0;
}
