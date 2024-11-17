#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

__global__ void Even(int* arr, int length) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x); //get global index
    if (index >= length - 1) return; //check if index is out of bounds

    //compare and swap
    if (arr[index] > arr[index + 1])
    {
        int tmp = arr[index];
        arr[index] = arr[index + 1];
        arr[index + 1] = tmp;
    }
}

__global__ void Odd(int* arr, int length) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1; //get global index
    if (index >= length - 1) return; //check if index is out of bounds

    if (arr[index] > arr[index + 1])
    {
        int tmp = arr[index];
        arr[index] = arr[index + 1];
        arr[index + 1] = tmp;
    }
}

void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
    int half = arr.size() / 2; //get half size of the array
    int* deviceArr; //arr copy for gpu
    cudaMalloc(&deviceArr, arr.size() * sizeof(int)); //allocate memory for d_arr
    cudaMemcpy(deviceArr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice); //copy

    int threads = 128; //threads per block (should be multiple of 32)

    //number of blocks. 
    //half of array size is used because the odd and even idexes are handled at the same time
    //this calculation guarantees that number of threads is enough to handle all elements
    int blocks = (int)ceil(half / (double)threads);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    //half iterations because we handle even and odd indexes at the same time
    for (int i = 0; i < half; i++)
    {
        Even << <blocks, threads, 0, stream >> > (deviceArr, arr.size()); //handle even
        Odd << <blocks, threads, 0, stream >> > (deviceArr, arr.size()); //handle odd
    }
    cudaMemcpy(arr.data(), deviceArr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost); //copy back

    cudaFree(deviceArr); //free memory
    cudaStreamDestroy(stream);
}
