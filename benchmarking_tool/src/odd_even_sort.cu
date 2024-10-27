#include "odd_even_sort.cuh"
#include <atomic>
#include <thread>

__global__ void sorting::Even(int* arr, int length) {
	int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x); //get global index
	if (index >= length - 1) return; //check if index is out of bounds

    //compare and swap
	if (arr[index] > arr[index + 1]) 
    { 
        int tmp = arr[index];
        arr[index] = arr[index + 1];
        arr[index + 1] = tmp;
    }
}

__global__ void sorting::Odd(int* arr, int length) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1; //get global index
    if (index >= length - 1) return; //check if index is out of bounds

    if (arr[index] > arr[index + 1]) 
    {
        int tmp = arr[index];
        arr[index] = arr[index + 1];
        arr[index + 1] = tmp;
    }
}

void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
	int half = arr.size() / 2; //get half size of the array
    int* d_arr; //arr copy for gpu
	cudaMalloc(&d_arr, arr.size() * sizeof(int)); //allocate memory for d_arr
    cudaMemcpy(d_arr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice); //copy

	int threads = 256; //threads per block (should be multiple of 32)

    //number of blocks. 
    //half of array size is used because the odd and even idexes are handled at the same time
	//this calculation guarantees that number of threads is enough to handle all elements
	int blocks = (int)ceil(half / (double)threads); 

	//half iterations because we handle even and odd indexes at the same time
    for (int i = 0; i < half; i++)
    {
        sorting::Even <<<blocks, threads >>> (d_arr, arr.size()); //handle even
        sorting::Odd <<<blocks, threads>>>(d_arr, arr.size()); //handle odd
		cudaDeviceSynchronize(); //wait for all threads to finish
    }
	cudaMemcpy(arr.data(), d_arr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost); //copy back

	cudaFree(d_arr); //free memory
}


void sorting::CpuOddEvenSort(std::vector<int>& arr)
{
    // sorting::newSortJoin(arr);
    
    sorting::oldSort(arr);
}

void sorting::oldSort(std::vector<int>& arr) {
    bool sorted = false;
    while (!sorted)
    {
        sorted = true;

        std::thread t1([&arr, &sorted] {
            for (int i = 0; i < arr.size() - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    std::swap(arr[i], arr[i + 1]);
                    sorted = false;
                }
            }
            });

        std::thread t2([&arr, &sorted] {
            for (int i = 1; i < arr.size() - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    std::swap(arr[i], arr[i + 1]);
                    sorted = false;
                }
            }
            });

        if (t1.joinable()) t1.join();
        if (t2.joinable()) t2.join();
    }
}

void compare(std::vector<int>& arr, std::atomic_bool &sorted, int startPoint, int endPoint) {
    for (int i = startPoint; i <= endPoint; i += 2) {
        if (arr[i] > arr[i + 1]) {
            std::swap(arr[i], arr[i + 1]);
            sorted = false;
        }
    }
}

void sorting::newSortJoin(std::vector<int>& arr) {
    std::atomic_bool sorted = ATOMIC_VAR_INIT(false);
    bool odd = false;
    int splits = 2;
    int evenComparisons = arr.size() / 2;
    int evenStep = 2 * evenComparisons / splits;
    int oddComparisons = (arr.size() - 1) / 2;
    int oddStep = 2 * oddComparisons / splits;
    std::thread threads[splits];

    while (!sorted) {
        sorted = true;
        int step = odd ? oddStep : evenStep;
        int startPoint = int(odd);
        int endPoint = step == 2 ? 1 + int(odd) : step - 2 * int(!odd);

        for (int i = 0; i < splits; i++) {
            threads[i] = std::thread(compare, std::ref(arr), std::ref(sorted), startPoint, endPoint);

            startPoint = ++endPoint;
            endPoint += step == 2 ? 1 : (odd ? step - 1 : step - 2);
            if (i + 2 == splits) {
                while (arr.size() - endPoint > 2) {
                    endPoint += 2;
                }
            }
        }

        for (std::thread& t : threads) {
            if (t.joinable()) t.join();
        }

        odd = !odd;
    }
}
