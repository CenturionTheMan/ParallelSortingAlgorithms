#include <iostream>
#include <chrono>
#include <atomic>
#include <random>
#include "headers/ThreadPool.h"
#include "headers/ParallelFor.h"
#include "headers/OddEvenSort.h"

void TestThreadPool();
void TestParallelFor();

const int arrSize = 10000;


int main(int argc, char const *argv[])
{
    std::vector<int> arr(arrSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-arrSize, arrSize);

    for (int i = 0; i < arrSize; i++)
    {
        arr[i] = dis(gen);
    }

    auto start = std::chrono::high_resolution_clock::now();
    MultiThreaded::OddEvenSort(arr);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "OddEvenSort: " << elapsed.count() << "s" << std::endl;

    for (int i = 0; i < arrSize; i++)
    {
        std::cout << arr[i] << ", ";
    }

    return 0;
}