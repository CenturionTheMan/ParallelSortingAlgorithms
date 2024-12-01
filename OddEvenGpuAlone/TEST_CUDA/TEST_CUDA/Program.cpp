#include "odd_even_sort.cuh"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

std::random_device rd;
std::mt19937 gen(rd());

std::vector<int> generate_random_vector(int size)
{
	std::vector<int> vec(size);
	
	std::uniform_int_distribution<> dis(-size, size);

	for (int i = 0; i < size; i++)
	{
		vec[i] = dis(gen);
	}

	return vec;
}

bool CheckValidity(const std::vector<int>& vec, int originalLen, int itearation)
{
	if (originalLen != vec.size())
	{
		std::cout << "[WRONG] (" << itearation << "): ";
		std::cout << "Size is not the same\n";
		return false;
	}

	bool isSorted = true;
	for (int i = 0; i < vec.size() - 1; i++)
	{
		if (vec[i] > vec[i + 1])
		{
			std::cout << "[WRONG] (" << itearation << "): ";
			std::cout << vec[i] << "\n";

			isSorted = false;
		}
	}

	if (isSorted)
	{
		std::cout << "[OK] (" << itearation << ")\n";
	}

	return isSorted;
}

int main()
{
	std::cout << ">> Start\n";
	const int size = 100000;
	const int rep = 20;
	double sumTime = 0.0;
	for (int i = 0; i < rep; i++)
	{
		std::vector<int> vec = generate_random_vector(size);
		
		/*for (int j = 0; j < vec.size(); j++)
		{
			std::cout << vec[j] << ", ";
		}
		std::cout << "\n";*/

		int originalLen = vec.size();

		auto start = std::chrono::high_resolution_clock::now();
		sorting::GpuOddEvenSort(vec);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		sumTime += elapsed_seconds.count();

		std::cout << "Time (" << i << "): " << elapsed_seconds.count() << "s\n";
		
		/*for (int j = 0; j < vec.size(); j++)
		{
			std::cout << vec[j] << ", ";
		}
		std::cout << "\n";*/

		if (!CheckValidity(vec, originalLen, i))
		{
			return 1;
		}
	
		std::cout << "----------------\n";
	}
	std::cout << ">> Average time: " << sumTime / rep << "s\n";

	//std::cin.get();
	
    return 0;
}

