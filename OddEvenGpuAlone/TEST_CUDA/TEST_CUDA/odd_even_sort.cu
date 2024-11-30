#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

__global__ void OddEven(int* arr, int length, int phase) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + phase; //get global index
    if (index >= length - 1) return; //check if index is out of bounds

    int current = arr[index];
    int next = arr[index + 1];

    if (current > next)
    {
        arr[index] = next;
        arr[index + 1] = current;
    }
}

int RoundUpToMultiple(float num, int multiple)
{
    return std::ceil(num / (float)multiple) * multiple;
}

void CalculateThreadsBlocksAmount(int& threads, int& blocks, int length)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    const int threadsAmountMin = 32;
    const int blocksPerMultiMax = deviceProp.maxBlocksPerMultiProcessor;
    const int multiMax = deviceProp.multiProcessorCount;

    if (length > multiMax * blocksPerMultiMax * deviceProp.maxThreadsPerBlock)
    {
        throw std::runtime_error("Array is too big");
    }

    blocks = multiMax * blocksPerMultiMax;
    threads = length / (float)blocks < threadsAmountMin ? threadsAmountMin : RoundUpToMultiple(length / (float)blocks, threadsAmountMin);
}

void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
    int* deviceArr;
    cudaMalloc(&deviceArr, arr.size() * sizeof(int));
    cudaMemcpy(deviceArr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    int blocks, threads;
    CalculateThreadsBlocksAmount(threads, blocks, arr.size());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < arr.size(); i++)
    {
        OddEven << <blocks, threads, 0, stream >> > (deviceArr, arr.size(), i % 2);
    }
    cudaMemcpy(arr.data(), deviceArr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceArr);
    cudaStreamDestroy(stream);
}