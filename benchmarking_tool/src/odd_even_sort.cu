#include "odd_even_sort.cuh"
#include <chrono>
#include <functional>
#include <thread>
#include <vector>

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
    sorting::newSortJoin(arr);
    
    // sorting::oldSort(arr);
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

void compare(std::vector<int>& arr, int startPoint, int endPoint) {
    for (int i = startPoint; i <= endPoint; i += 2) {
        if (arr[i] > arr[i + 1]) {
            std::swap(arr[i], arr[i + 1]);
        }
    }
}

struct vecPoints {
    int start;
    int end;
};

void sorting::newSortJoin(std::vector<int>& arr) {
    // Create threads
    int splits = 10;
    int sleepTime = 10;
    int arrSize = arr.size();
    sorting::ThreadPool p (splits);
    std::vector<vecPoints> evenPoints;
    std::vector<vecPoints> oddPoints;
    bool sorted = false;
    bool odd = false;
    int evenComparisons = arrSize / 2;
    int oddComparisons = (arrSize - 1) / 2;
{

    int evenStep = std::max(2 * evenComparisons / splits, 2);
    int oddStep = std::max(2 * oddComparisons / splits, 2);

    for (int oddInt = 0; oddInt < 2; oddInt++) {
        int step = oddInt ? oddStep : evenStep;
        int startPoint = int(oddInt);
        int endPoint = step == 2 ? 1 + int(oddInt) : step - 2 * int(!oddInt);

        int jobs = std::min(splits, odd ? oddComparisons : evenComparisons);
        odd ? oddPoints.reserve(jobs) : evenPoints.reserve(jobs);
        for (int i = 0; i < jobs; i++) {
            oddInt ? oddPoints.push_back({startPoint, endPoint}) : evenPoints.push_back({startPoint, endPoint});

            startPoint = ++endPoint;
            endPoint += step == 2 ? 1 : (oddInt ? step - 1 : step - 2);
            if (i + 2 == splits) {
                while (arrSize - endPoint > 2) {
                    endPoint += 2;
                }
            }
        }
    }
}

    vecPoints range = {0, 0};
    int loops = 0;
    int jobs = std::min(splits, odd ? oddComparisons : evenComparisons);
    while (!sorted) {
        sorted = true;

        // Assign them 'splits' jobs
        jobs = std::min(splits, odd ? oddComparisons : evenComparisons);
        for (int i = 0; i < jobs; i++) {
            range = odd ? oddPoints[i] : evenPoints[i];
            p.doJob(std::bind (compare, std::ref(arr), range.start, range.end));
        }

        odd = !odd;
        loops = 0;
        while (!p.queueRefill()) {
            std::this_thread::sleep_for(std::chrono::microseconds(sleepTime));
            loops++;
        }

        if (loops > 2) {
            sleepTime++;
        }
        else if (loops == 0) {
            sleepTime = std::max(sleepTime - 1, 1);
        }

        for (int i = 0; i < arrSize - 1; i++) {
            if (arr[i] > arr[i+1]) {
                sorted = false;
                break;
            }
        }
    }
}