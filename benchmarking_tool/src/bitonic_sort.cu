#include <cmath>
#include <functional>
#include <iostream>
#include <algorithm>
#include <thread>
#include <vector>

#include "bitonic_sort.cuh"


__global__ void comparePair(int *array, int pair_distance, int sequence_size, int array_size) {
    unsigned int first = threadIdx.x + blockDim.x * blockIdx.x;
    if (first >= array_size)
        return;
    unsigned int second = first ^ pair_distance;

    if (second <= first)
        return;
    const int descending = first & sequence_size;
    const bool first_greater = array[first] > array[second];
    if ((!descending && first_greater) || (descending && !first_greater)) {
        int temp = array[first];
        array[first] = array[second];
        array[second] = temp;
    }
}


void sorting::GpuBitonicSort(std::vector<int>& arr) {
    int *device_array;
    size_t size = arr.size() * sizeof(int);

    cudaMalloc(&device_array, size);
    cudaMemcpy(device_array, arr.data(), size, cudaMemcpyHostToDevice);

    const int THREADS_PER_BLOCK = 512;

    dim3 blocks((arr.size() + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    dim3 threads(THREADS_PER_BLOCK);

    int pair_distance, sequence_size;
    for (sequence_size = 2; sequence_size <= arr.size(); sequence_size <<= 1) {
        for (pair_distance = sequence_size >> 1; pair_distance > 0; pair_distance = pair_distance >> 1) {
            comparePair<<<blocks, threads>>>(device_array, pair_distance, sequence_size, arr.size());
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(arr.data(), device_array, size, cudaMemcpyDeviceToHost);
    cudaFree(device_array);
}

static inline void comparePairs(std::vector<int>& arr, const int start, const int pair_distance, const int sort_direction) {
    for (int i = start; i < start + pair_distance; i++) {
        if (sort_direction == (arr[i] > arr[i+pair_distance])) {
            std::swap(arr[i],arr[i+pair_distance]);
        }
    }
}

void bitonicMerge(std::vector<int>& a, int start, int sequence_size, int sort_direction) {
    if (sequence_size == 1)
        return;

    int split_point = sequence_size/2;
    comparePairs(a, start, split_point, sort_direction);
    bitonicMerge(a, start, split_point, sort_direction);
    bitonicMerge(a, start+split_point, split_point, sort_direction);
}

void bitonicSortRecurrence(std::vector<int>& a, int start, int sequence_size, int sort_direction) {
    if (sequence_size > 1) {
        int k = sequence_size / 2;
        bitonicSortRecurrence(a, start, k, 1);
        bitonicSortRecurrence(a, start+k, k, 0);
        bitonicMerge(a, start, sequence_size, sort_direction);
    }
}

void bitonicSortThreaded(
    std::vector<int>& array, int start, int sequence_size, int sort_direction, int divisions_left
) {
    if (sequence_size > 1) {
        int split_point = sequence_size / 2;
        if (divisions_left > 0) {
            std::thread aT (
                std::bind(bitonicSortThreaded, std::ref(array), start, split_point, 1, divisions_left - 1)
            );
            std::thread bT (
                std::bind(bitonicSortThreaded, std::ref(array), start+split_point, split_point, 0, divisions_left - 1)
            );

            aT.join();
            bT.join();
        }
        else {
            bitonicSortRecurrence(array, start, split_point, 1);
            bitonicSortRecurrence(array, start+split_point, split_point, 0);
        }
        bitonicMerge(array, start, sequence_size, sort_direction);
    }
}

void sorting::CpuBitonicSort(std::vector<int> &arr)
{
    int divisions_left = std::min(std::log2(std::thread::hardware_concurrency()) + 1, std::log2(arr.size() - 1));
    bitonicSortThreaded(arr, 0, arr.size(), 1, divisions_left);
}