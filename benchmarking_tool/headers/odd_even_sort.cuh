#ifndef ODD_EVEN_SORT_H
#define ODD_EVEN_SORT_H

#include <vector>
#include <math.h>
#include <thread>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace sorting
{
	void GpuOddEvenSort(std::vector<int>& arr);

	void CpuOddEvenSort(std::vector<int>& arr);

	void oldSort(std::vector<int>& arr);
	void newSortJoin(std::vector<int>& arr);

	__global__ void Odd(int* arr, int length);
	__global__ void Even(int* arr, int length);
}

#endif