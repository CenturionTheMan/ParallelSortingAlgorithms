#pragma once
#include <thread>
#include <vector>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

namespace MultiThreaded
{   
    void OddEvenSortCuda(std::vector<int> &arr);
    void OddEvenSort(std::vector<int> &arr);
}


namespace CudaMethods
{
    __global__ void Odd(int* arr, int length);
    __global__ void Even(int* arr, int length);
} // namespace CudaMethods
