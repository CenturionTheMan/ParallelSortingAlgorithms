#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
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
    // sorting::newSortJoin(arr);
    sorting::sortMT(arr);
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

static inline void compare(std::vector<int>& arr, const int startPoint, const int endPoint) {
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
    const int arrSize = arr.size();
    int localThreads = 1;
    while (localThreads * 2 < std::thread::hardware_concurrency()) {
        localThreads *= 2;
    }
    const int evenComparisons = arrSize / 2;
    const int oddComparisons = (arrSize - 1) / 2;
    const int splits = std::min(localThreads, std::max(1, int (evenComparisons / 1024)));
    sorting::ThreadPool p(splits);
    const int jobs[2] = {std::min(splits, evenComparisons), std::min(splits, oddComparisons)};
    std::vector<vecPoints> evenPoints;
    std::vector<vecPoints> oddPoints;
    oddPoints.reserve(jobs[0]);
    evenPoints.reserve(jobs[1]);
    bool sorted = false;
    bool odd = false;

{
    const int evenStep = std::max(2 * evenComparisons / splits, 2);
    const int oddStep = std::max(2 * oddComparisons / splits, 2);

    for (int oddInt = 0; oddInt < 2; oddInt++) {
        int step = oddInt ? oddStep : evenStep;
        int startPoint = int(oddInt);
        int endPoint = step == 2 ? 1 + int(oddInt) : step - 2 * int(!oddInt);

        for (int i = 0; i < jobs[oddInt]; i++) {
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

    int place = 0;
    while (!sorted) {
        // Assign them 'splits' jobs
        for (int j = 0; j < 2; j++) {
            std::vector<vecPoints>& currentPoints = odd ? oddPoints : evenPoints;
            for (int i = 0; i < jobs[odd]; i++) {
                if (currentPoints[i].end < place) continue;
                p.doJob(std::bind (compare, std::ref(arr), currentPoints[i].start, currentPoints[i].end));
            }

            odd = !odd;
            while (!p.queueEmpty()) {
                continue;
            }
        }
        
        sorted = true;
        for (int i = 0; i < arrSize - 1; i++) {
            if (arr[i] > arr[i+1]) {
                sorted = false;
                place = i - 1;
                break;
            }
        }
    }
}

inline void compareMT(std::vector<int>& arr, const int startPoint, const int endPoint, std::mutex& m, bool& sorted) {
    const int oddEnd = std::min(endPoint + 1, int(arr.size() - 1));

    while (true) {
        bool needsLock = false;
        bool swapped = false;

        // Check endings first
        {
            std::unique_lock<std::mutex> lock(m, std::defer_lock);

            lock.lock();
            if (arr[oddEnd - 1] > arr[oddEnd]) {
                std::swap(arr[oddEnd - 1], arr[oddEnd]);

                sorted = false;
                swapped = true;
            }
            if (arr[startPoint] > arr[startPoint + 1]) {
                std::swap(arr[startPoint], arr[startPoint + 1]);
                sorted = false;
                swapped = true;
            }
            if (arr[startPoint + 1] > arr[startPoint + 2]) {
                std::swap(arr[startPoint + 1], arr[startPoint + 2]);
                sorted = false;
                swapped = true;
            }
            lock.unlock();
        }

        for (int i = startPoint + 1; i < oddEnd; i += 2) {
            // Odd
            if (arr[i] > arr[i + 1]) {
                std::swap(arr[i], arr[i + 1]);

                if (!swapped) {
                    std::unique_lock<std::mutex> lock(m, std::defer_lock);
                    
                    if (lock.try_lock()) {
                        sorted = false;
                        swapped = true;
                        needsLock = false;
                    }
                    else {
                        needsLock = true;
                    }
                }
            }
            // Even
            if (arr[i - 1] > arr[i]) {
                std::swap(arr[i - 1], arr[i]);
 
                if (!swapped) {
                    std::unique_lock<std::mutex> lock(m, std::defer_lock);
                    
                    if (lock.try_lock()) {
                        sorted = false;
                        swapped = true;
                        needsLock = false;
                    }
                    else {
                        needsLock = true;
                    }
                }
            }
        }

        if (needsLock) {
            std::unique_lock<std::mutex> lock(m);

            sorted = false;
            swapped = true;
        }

        if (!swapped) break;
    }
}

void sorting::sortMT(std::vector<int>& arr) {
    const int threadsCount = std::min(int(std::thread::hardware_concurrency()), std::max(1, int(std::log2(arr.size())) - 5));
    bool sorted = false;
    const int k = arr.size() / threadsCount;
    int start = 0;
    std::mutex m;
    std::vector<std::thread> group(threadsCount);
    while (!sorted) {
        sorted = true;
        start = 0;

        for (int i = 0; i < threadsCount - 1; i++) {
            group[i] = std::thread(std::bind(compareMT, std::ref(arr), start, start + k, std::ref(m), std::ref(sorted)));
            start += k;
        }
        group[threadsCount - 1] = std::thread(std::bind(compareMT, std::ref(arr), start, arr.size(), std::ref(m), std::ref(sorted)));

        for (std::thread& t : group)
            t.join();
    }
}