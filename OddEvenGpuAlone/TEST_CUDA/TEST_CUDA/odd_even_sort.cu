#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

__global__ void Even(int* arr, int length) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x); //get global index
    if (index >= length - 1) return; //check if index is out of bounds

    int current = arr[index];
    int next = arr[index + 1];

    if (current > next)
    {
        arr[index] = next;
        arr[index + 1] = current;
    }
}

__global__ void Odd(int* arr, int length) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1; //get global index
    if (index >= length - 1) return; //check if index is out of bounds

    int current = arr[index];
    int next = arr[index + 1];

    if (current > next)
    {
        arr[index] = next;
        arr[index + 1] = current;
    }
}


void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
    int half = arr.size() / 2;
    int* deviceArr;
    cudaMalloc(&deviceArr, arr.size() * sizeof(int));
    cudaMemcpy(deviceArr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 128;

    int blocks = (int)ceil(half / (double)threads);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < half; i++)
    {
        Even << <blocks, threads, 0, stream >> > (deviceArr, arr.size());
        Odd << <blocks, threads, 0, stream >> > (deviceArr, arr.size());
    }
    cudaMemcpy(arr.data(), deviceArr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceArr); //free memory
    cudaStreamDestroy(stream);
}