#include <iostream>
#include <chrono>
#include <atomic>
#include <random>
#include "ThreadPool.h"
#include "ParallelFor.h"
#include "OddEvenSortCuda.cuh"

const int arrSize = 10000;

int main(int argc, char const* argv[])
{
    std::vector<int> arrCuda(arrSize);
    std::vector<int> arrThreaded(arrSize);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-arrSize, arrSize);

    for (int i = 0; i < arrSize; i++)
    {
        int tmp = dis(gen);
        arrCuda[i] = tmp;
        arrThreaded[i] = tmp;
    }


    auto start = std::chrono::high_resolution_clock::now();
    Cuda::OddEvenSort(arrCuda);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "OddEvenSort CUDA: " << elapsed.count() << "s" << std::endl;
    for (int i = 0; i < arrSize - 1; i++)
        if (arrCuda[i] > arrCuda[i + 1])
			std::cout << "Error CUDA: " << i << std::endl;

	start = std::chrono::high_resolution_clock::now();
	MultiThreaded::OddEvenSort(arrThreaded);
	end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed2 = end - start;
	std::cout << "OddEvenSort MultiThreaded: " << elapsed2.count() << "s" << std::endl;
    for (int i = 0; i < arrSize - 1; i++)
        if (arrThreaded[i] > arrThreaded[i + 1])
            std::cout << "Error Threaded: " << i << std::endl;

    return 0;
}