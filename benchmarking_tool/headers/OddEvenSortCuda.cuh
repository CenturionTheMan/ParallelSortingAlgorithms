#pragma once
#include <vector>
#include <math.h>
#include <thread>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace MultiThreaded
{
	/// <summary>
	/// OddEvenSort algorithm implementation using multiple threads
	/// </summary>
	/// <param name="arr"></param>
	void OddEvenSort(std::vector<int>& arr);
}

namespace Cuda
{
	__global__ void Odd(int* arr, int length);
	__global__ void Even(int* arr, int length);

	/// <summary>
	/// OddEvenSort algorithm implementation using CUDA
	/// </summary>
	/// <param name="arr"></param>
	void OddEvenSort(std::vector<int>& arr);
}