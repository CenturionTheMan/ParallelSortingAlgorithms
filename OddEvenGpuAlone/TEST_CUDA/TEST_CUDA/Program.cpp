#include "odd_even_sort.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>

std::random_device rd;
std::mt19937 gen(rd());

std::vector<int> generate_random_vector(int size)
{
	std::vector<int> vec(size);
	
	std::uniform_int_distribution<> dis(1, size*2);

	for (int i = 0; i < size; i++)
	{
		vec[i] = dis(gen);
	}

	return vec;
}


bool CheckValidity(std::vector<int>& vecOriginal, const std::vector<int>& vec, int itearation)
{
	std::sort(vecOriginal.begin(), vecOriginal.end());

	for (int i = 0; i < vec.size(); i++)
	{
		if (vec[i] != vecOriginal[i])
		{
			std::cout << "[WRONG] (" << itearation << " | " << i << "): ";
			std::cout << vecOriginal[i] << ", " << vec[i] << "\n";

			std::cout << "[Result]:   ";
			for (int j = 0; j < vec.size(); j++)
			{
				std::cout << vec[j] << ", ";
			}

			std::cout << "\n[Expected]: ";
			for (int j = 0; j < vecOriginal.size(); j++)
			{
				std::cout << vecOriginal[j] << ", ";
			}

			return false;
		}
	}

	std::cout << "[OK] (" << itearation << ")\n";
}

//bool CheckValidity(const std::vector<int>& vec, int originalLen, int itearation)
//{
//	if (originalLen != vec.size())
//	{
//		std::cout << "[WRONG] (" << itearation << "): ";
//		std::cout << "Size is not the same\n";
//		return false;
//	}
//
//	bool isSorted = true;
//	for (int i = 0; i < vec.size() - 1; i++)
//	{
//		if (vec[i] > vec[i + 1])
//		{
//			std::cout << "[WRONG] (" << itearation << " | " << i << "): ";
//			std::cout << vec[i-1] << ", " << vec[i] << ", " << vec[i+1] << "\n";
//
//			isSorted = false;
//		}
//	}
//
//	if (isSorted)
//	{
//		std::cout << "[OK] (" << itearation << ")\n";
//	}
//
//	return isSorted;
//}

int main()
{
	std::cout << ">> Start\n";
	const int size = 100000;
	const int rep = 10;
	double sumTime = 0.0;
	for (int i = 0; i < rep; i++)
	{
		std::vector<int> vec = generate_random_vector(size);
		std::vector<int> vecClone = vec;

		int originalLen = vec.size();

		auto start = std::chrono::high_resolution_clock::now();
		sorting::GpuOddEvenSort(vec);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		sumTime += elapsed_seconds.count();

		std::cout << "Time (" << i << "): " << elapsed_seconds.count() << "s\n";
		

		if (!CheckValidity(vecClone, vec, i))
		{
			return 1;
		}
	
		std::cout << "----------------\n";
	}
	std::cout << ">> Average time: " << sumTime / rep << "s\n";

	//std::cin.get();
	
    return 0;
}

