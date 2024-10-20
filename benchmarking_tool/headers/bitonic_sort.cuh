#ifndef BITONIC_SORT_H
#define BITONIC_SORT_H

#include <vector>

namespace sorting
{
	void GpuBitonicSort(std::vector<int>& arr);

	void CpuBitonicSort(std::vector<int>& arr);
}

#endif