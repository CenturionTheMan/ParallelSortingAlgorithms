#include <iostream>
#include <algorithm>

#include "bitonic_sort.cuh"


void sorting::GpuBitonicSort(std::vector<int> &arr)
{
    std::sort(arr.begin(), arr.end());
}

void sorting::CpuBitonicSort(std::vector<int> &arr)
{
    std::sort(arr.begin(), arr.end());
}
