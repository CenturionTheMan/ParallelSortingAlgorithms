#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

__global__ void OddEven(int* arr, int length, int phase) {
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIndex = 2 * globalId + phase;

	if (globalIndex < length)
	{
		int curr = arr[globalIndex];
		int next = arr[globalIndex + 1];
		if (curr > next)
		{
			arr[globalIndex] = next;
			arr[globalIndex + 1] = curr;
		}
	}
}


void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
    int* gpuArr;
    cudaMalloc(&gpuArr, arr.size() * sizeof(int));
    cudaMemcpy(gpuArr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    int threads = 512;

    int blocks = (int)ceil(arr.size() / 2 / (double)threads);

    for (int i = 0; i < arr.size(); i++)
    {
        OddEven << <blocks, threads >> > (gpuArr, arr.size(), i%2);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(arr.data(), gpuArr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(gpuArr);
}