#include <iostream>
#include <chrono>
#include <atomic>
#include "headers/ThreadPool.h"
#include "headers/ParallelFor.h"

void TestThreadPool();
void TestParallelFor();

const int arrSize = 100000;


int main(int argc, char const *argv[])
{
    // TestThreadPool();
    TestParallelFor();
    return 0;
}

void TestParallelFor()
{
    int arr1[arrSize];
    int arr2[arrSize];
    for (int i = 0; i < arrSize; i++)
    {
        arr1[i] = i;
        arr2[i] = arrSize - i;
    }
    
    int arrRes1[arrSize];
    int arrRes2[arrSize];

    ParallelFor pf(10);

    auto start1 = std::chrono::high_resolution_clock::now();    
    pf.RunAndWait(0, arrSize, 1, [&arrRes1, &arr1, &arr2](int i) {
        arrRes1[i] = arr1[i] + arr2[i];
    });
    auto end1 = std::chrono::high_resolution_clock::now();
    
    auto start2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < arrSize; i++)
    {
        arrRes2[i] = arr1[i] + arr2[i];
    }
    auto end2 = std::chrono::high_resolution_clock::now();

    std::cout << "Anti-optimization: " << arrRes1[2] << " | " << arrRes1[3] << std::endl; 

    std::cout << "Time parallel: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << std::endl;
    std::cout << "Time normal:   " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << std::endl;
}

void TestThreadPool()
{
    // testing thread pool ...
    // It is a lot slower than normal for loop :(, but will work better for more complex tasks (hope so ...)

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
}