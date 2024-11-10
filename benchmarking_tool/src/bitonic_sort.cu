#include <iostream>
#include <algorithm>
#include "bitonic_sort.cuh"


__global__ void bitonicSortKernel(int *arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        const int ik = i & k;
        const bool compare = arr[i] > arr[ixj];
        if (((ik) == 0 && compare) || ((ik) != 0 && !compare)) {
            int temp = arr[i];
            arr[i] = arr[ixj];
            arr[ixj] = temp;
        }
    }
}


void sorting::GpuBitonicSort(std::vector<int>& arr) {
    int *d_arr;
    size_t size = arr.size() * sizeof(int);

    cudaMalloc(&d_arr, size);
    cudaMemcpy(d_arr, arr.data(), size, cudaMemcpyHostToDevice);

    const int threadAmount = 512;

    dim3 blocks((arr.size() + threadAmount - 1) / threadAmount);
    dim3 threads(threadAmount);

    int j, k;
    for (k = 2; k <= arr.size(); k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            bitonicSortKernel<<<blocks, threads>>>(d_arr, j, k);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr.data(), d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

}

void sorting::CpuBitonicSort(std::vector<int> &arr)
{
    std::sort(arr.begin(), arr.end());
}
