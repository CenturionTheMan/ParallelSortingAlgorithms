#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "headers/ThreadedBitonicSort.h"

const int arrSize = 2;


int main(int argc, char const *argv[])
{
    std::vector<int> arr(arrSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 999);

    for (int i = 0; i < arrSize; i++)
    {
        arr[i] = dis(gen);
    }

    ThreadedBitonicSort sorter(std::make_unique<ThreadPool>(2));

    auto start = std::chrono::high_resolution_clock::now();
    sorter.Sort(arr);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << "s" << std::endl;
    for (int i = 0; i < arr.size(); i++)
    {
        printf("%d, ", arr[i]);
    }
    

    return 0;
}