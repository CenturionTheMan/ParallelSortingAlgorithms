#ifndef ODD_EVEN_SORT_H
#define ODD_EVEN_SORT_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>
#include <thread>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

namespace sorting
{
	void GpuOddEvenSort(std::vector<int>& arr);
}

#endif