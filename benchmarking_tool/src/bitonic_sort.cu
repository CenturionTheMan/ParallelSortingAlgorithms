#include <iostream>
#include <algorithm>
#include "bitonic_sort.cuh"


__global__ void bitonicSortKernel(int *arr, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int ixj = i ^ j;

    if (ixj > i) {
        if (((i & k) == 0 && arr[i] > arr[ixj]) || ((i & k) != 0 && arr[i] < arr[ixj])) {
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

    dim3 blocks((arr.size() + 512 - 1) / 512);
    dim3 threads(512);

    int j, k;
    for (k = 2; k <= arr.size(); k <<= 1) {
        for (j = k >> 1; j > 0; j = j >> 1) {
            bitonicSortKernel<<<blocks, threads>>>(d_arr, j, k);
            cudaDeviceSynchronize();
        }
    // dim3 blocks((arr.size() / 2 + 512 - 1) / 512);
    // dim3 threads(512);
    }

    cudaMemcpy(arr.data(), d_arr, size, cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

}

void sorting::CpuBitonicSort(std::vector<int> &arr)
{
    std::sort(arr.begin(), arr.end());
}
