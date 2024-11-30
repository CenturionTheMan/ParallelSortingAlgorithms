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


void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
    int* deviceArr; //arr copy for gpu
    cudaMalloc(&deviceArr, arr.size() * sizeof(int)); //allocate memory for d_arr
    cudaMemcpy(deviceArr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice); //copy

	int dymThreadAmount = std::ceill(arr.size() / 32.0) * 32;
	int threads = dymThreadAmount > 1024 ? 1024 : dymThreadAmount;

    int blocks = (int)ceil(arr.size() / 2 / (double)threads);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < arr.size(); i++)
    {
        OddEven << <blocks, threads, 0, stream >> > (deviceArr, arr.size(), i%2);
    }
    cudaMemcpy(arr.data(), deviceArr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost); //copy back

    cudaFree(deviceArr); //free memory
    cudaStreamDestroy(stream);
}
