#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

__global__ void OddEven(int* arr, int length, int phase, int threadsAmount) {
    extern __shared__ int sharedMem[];

    int threadIndex = threadIdx.x;
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (globalIndex + 1 >= length || (blockIdx.x != 0 && threadIndex == 0 && globalIndex % 2 != phase))
	{
		return;
	}
    
    sharedMem[threadIndex] = arr[globalIndex];

    bool isBlockEdge = globalIndex % 2 == phase && (globalIndex + 2 >= length || threadIndex + 1 == threadsAmount);

	if (isBlockEdge)
	{
		sharedMem[threadIndex + 1] = arr[globalIndex + 1];
	}
	__syncthreads();

    if (globalIndex % 2 == phase)
    {
        int current = sharedMem[threadIndex];
        int next = sharedMem[threadIndex + 1];
        if (current > next)
        {
            sharedMem[threadIndex] = next;
            sharedMem[threadIndex + 1] = current;
        }
    }
    __syncthreads();


    arr[globalIndex] = sharedMem[threadIndex];
	if (isBlockEdge)
	{
		arr[globalIndex + 1] = sharedMem[threadIndex + 1];
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
    
    const int threadsAmountMin = deviceProp.warpSize;
    const int blocksPerMultiMax = deviceProp.maxBlocksPerMultiProcessor;
	const int multiMax = deviceProp.multiProcessorCount;
    const int maxThresPerBlock = deviceProp.maxThreadsPerBlock;

    blocks = multiMax * blocksPerMultiMax;
    threads = length / (float)blocks < threadsAmountMin ? threadsAmountMin : RoundUpToMultiple(length / (float)blocks, threadsAmountMin);

    if (threads > maxThresPerBlock)
    {
        threads = maxThresPerBlock;
        blocks = std::ceill(length / (float)threads);
    }
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

	int sharedMemorySize = (threads+1) * sizeof(int);

    for (int i = 0; i < arr.size(); i++)
    {
        OddEven << <blocks, threads, sharedMemorySize, stream >> > (deviceArr, arr.size(), i%2, threads);
    }
    cudaMemcpy(arr.data(), deviceArr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceArr);
    cudaStreamDestroy(stream);
}
