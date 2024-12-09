#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

__global__ void sorting::Even(int* array, int array_length) {
	int pair_index = 2 * (blockIdx.x * blockDim.x + threadIdx.x); //get global index
	if (pair_index >= array_length - 1) return; //check if index is out of bounds

    //compare and swap
	if (array[pair_index] > array[pair_index + 1]) 
    { 
        int tmp = array[pair_index];
        array[pair_index] = array[pair_index + 1];
        array[pair_index + 1] = tmp;
    }
}

__global__ void sorting::Odd(int* array, int array_length) {
    int pair_index = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1; //get global index
    if (pair_index >= array_length - 1) return; //check if index is out of bounds

    if (array[pair_index] > array[pair_index + 1]) 
    {
        int tmp = array[pair_index];
        array[pair_index] = array[pair_index + 1];
        array[pair_index + 1] = tmp;
    }
}

void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
	int array_half_size = arr.size() / 2; //get half size of the array
    int* device_array; //arr copy for gpu
	cudaMalloc(&device_array, arr.size() * sizeof(int)); //allocate memory for d_arr
    cudaMemcpy(device_array, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice); //copy

	const int THREADS = 32; //threads per block (should be multiple of 32)

    //number of blocks. 
    //half of array size is used because the odd and even idexes are handled at the same time
	//this calculation guarantees that number of threads is enough to handle all elements
	int blocks = (int)ceil(array_half_size / (double)THREADS); 

	//half iterations because we handle even and odd indexes at the same time
    for (int i = 0; i < array_half_size; i++)
    {
        sorting::Even<<<blocks, THREADS>>> (device_array, arr.size()); //handle even
        sorting::Odd<<<blocks, THREADS>>>(device_array, arr.size()); //handle odd
		cudaDeviceSynchronize(); //wait for all threads to finish
    }
	cudaMemcpy(arr.data(), device_array, arr.size() * sizeof(int), cudaMemcpyDeviceToHost); //copy back

	cudaFree(device_array); //free memory
}

inline void runPhasesOnArrayChunk(
    std::vector<int>& arr, const int start_point, const int end_point, std::mutex& is_sorted_mutex, bool& is_sorted
) {
    const int ODD_END = std::min(end_point + 1, int(arr.size() - 1));

    while (true) {
        bool needs_to_change_is_sorted = false;
        bool swap_performed = false;

        for (int i = start_point + 1; i < ODD_END; i += 2) {
            // Odd
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);
                if (!swap_performed) {
                    std::unique_lock<std::mutex> lock(is_sorted_mutex, std::defer_lock);
                    if (lock.try_lock()) {
                        is_sorted = false;
                        swap_performed = true;
                        needs_to_change_is_sorted = false;
                    }
                    else {
                        needs_to_change_is_sorted = true;
                    }
                }
            }
            // Even
            if (arr[i - 1] > arr[i]) {
                std::swap(arr[i - 1], arr[i]);
                if (!swap_performed) {
                    std::unique_lock<std::mutex> lock(is_sorted_mutex, std::defer_lock);
                    if (lock.try_lock()) {
                        is_sorted = false;
                        swap_performed = true;
                        needs_to_change_is_sorted = false;
                    }
                    else {
                        needs_to_change_is_sorted = true;
                    }
                }
            }
        }

        if (arr[ODD_END - 1] > arr[ODD_END]) {
            std::swap(arr[ODD_END - 1], arr[ODD_END]);
            if (!swap_performed) {
                std::unique_lock<std::mutex> lock(is_sorted_mutex, std::defer_lock);
                if (lock.try_lock()) {
                    is_sorted = false;
                    swap_performed = true;
                    needs_to_change_is_sorted = false;
                }
                else {
                    needs_to_change_is_sorted = true;
                }
            }
        }

        if (needs_to_change_is_sorted) {
            std::unique_lock<std::mutex> lock(is_sorted_mutex);
            is_sorted = false;
            swap_performed = true;
        }

        if (!swap_performed)
            break;
    }
}

void sorting::CpuOddEvenSort(std::vector<int>& arr)
{
    const int THREADS_COUNT = std::min(int(std::thread::hardware_concurrency()), std::max(1, int(std::log2(arr.size())) - 5));
    const int ELEMENTS_PER_THREAD = arr.size() / THREADS_COUNT;
    std::vector<std::thread> threads(THREADS_COUNT);

    bool isSorted = false;
    std::mutex isSortedMutex;
    
    int start;
    while (!isSorted) {
        isSorted = true;
        start = 0;

        for (int i = 0; i < THREADS_COUNT - 1; i++) {
            threads[i] = std::thread(std::bind(
                runPhasesOnArrayChunk,
                std::ref(arr), 
                start, 
                start + ELEMENTS_PER_THREAD, 
                std::ref(isSortedMutex), 
                std::ref(isSorted)
            ));
            start += ELEMENTS_PER_THREAD;
        }
        threads[THREADS_COUNT - 1] = std::thread(std::bind(
            runPhasesOnArrayChunk, std::ref(arr), start, arr.size(), std::ref(isSortedMutex), std::ref(isSorted)
        ));

        for (std::thread& t : threads)
            t.join();
    }
}