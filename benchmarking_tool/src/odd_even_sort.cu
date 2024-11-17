#include "odd_even_sort.cuh"
#include <cmath>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

__global__ void Even(int* arr, int length) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= length - 1) return;

	int current = arr[index];
	int next = arr[index + 1];
    
	if (current > next)
	{
		arr[index] = next;
		arr[index + 1] = current;
	}
}

__global__ void Odd(int* arr, int length) {
    int index = 2 * (blockIdx.x * blockDim.x + threadIdx.x) + 1;
    if (index >= length - 1) return;

    int current = arr[index];
    int next = arr[index + 1];

    if (current > next)
    {
        arr[index] = next;
        arr[index + 1] = current;
    }
}


void sorting::GpuOddEvenSort(std::vector<int>& arr)
{
    int half = arr.size() / 2;
    int* deviceArr;
    cudaMalloc(&deviceArr, arr.size() * sizeof(int));
    cudaMemcpy(deviceArr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);

    const int threads = 128;

    int blocks = (int)ceil(half / (double)threads);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < half; i++)
    {
        Even << <blocks, threads, 0, stream >> > (deviceArr, arr.size());
        Odd << <blocks, threads, 0, stream >> > (deviceArr, arr.size());
    }
    cudaMemcpy(arr.data(), deviceArr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(deviceArr);
    cudaStreamDestroy(stream);
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
    const int oddEnd = std::min(endPoint + 1, int(arr.size()));

    while (true) {
        bool needsLock = false;
        bool swapped = false;

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
    const int threadsCount = 10;
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