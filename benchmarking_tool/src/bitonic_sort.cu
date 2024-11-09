#include <cmath>
#include <functional>
#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

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
    // std::sort(arr.begin(), arr.end());
    sorting::bitonicSort(arr);
}

static inline void compare(std::vector<int>& arr, const int startPoint, const int k, const int direction) {
    for (int i = startPoint; i < startPoint + k; i++) {
        if (direction == (arr[i] > arr[i+k])) {
            std::swap(arr[i],arr[i+k]);
        }
    }
}

void bitonicSeqMerge(std::vector<int>& a, int start, int BseqSize, int direction) {
    if (BseqSize > 1) {
        int k = BseqSize/2;
        compare(a, start, k, direction);
        bitonicSeqMerge(a, start, k, direction);
        bitonicSeqMerge(a, start+k, k, direction);
    }
}

void bitonicSortrec(std::vector<int>& a, int start, int BseqSize, int direction) {
    if (BseqSize > 1) {
        int k = BseqSize / 2;
        bitonicSortrec(a, start, k, 1);
        bitonicSortrec(a, start+k, k, 0);
        bitonicSeqMerge(a, start, BseqSize, direction);
    }
}

void bitonicSortrecA(std::vector<int>& a, int start, int BseqSize, int direction, int level) {
    if (BseqSize > 1) {
        int k = BseqSize / 2;
        if (level > 0) {
            std::thread aT (std::bind(bitonicSortrecA, std::ref(a), start, k, 1, level - 1));
            std::thread bT (std::bind(bitonicSortrecA, std::ref(a), start+k, k, 0, level - 1));

            aT.join();
            bT.join();
        }
        else {
            bitonicSortrecA(a, start, k, 1, level - 1);
            bitonicSortrecA(a, start+k, k, 0, level - 1);
        }
        bitonicSeqMerge(a, start, BseqSize, direction);
    }
}

//https://github.com/shivaylamba/Hacktoberfest/blob/master/Bitonic_Sorting.cpp
void sorting::bitonicSort(std::vector<int>& arr) {
    bool mt = true;

    if (mt) {
        int level = 3;

        int k = arr.size() / 2;
        int start = 0;

        std::thread a (std::bind(bitonicSortrecA, std::ref(arr), start, k, 1, level));
        std::thread b (std::bind(bitonicSortrecA, std::ref(arr), start+k, k, 0, level));

        a.join();
        b.join();

        bitonicSeqMerge(arr, start, arr.size(), 1);
    }
    else {
        bitonicSortrec(arr, 0, arr.size(), 1);
    }
}
