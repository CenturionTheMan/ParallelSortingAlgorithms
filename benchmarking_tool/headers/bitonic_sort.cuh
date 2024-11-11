#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <vector>
#include "cuda_runtime.h"

namespace sorting
{
	void GpuBitonicSort(std::vector<int>& arr);

	void CpuBitonicSort(std::vector<int>& arr);

	void bitonicSort(std::vector<int>& arr);
}

#endif