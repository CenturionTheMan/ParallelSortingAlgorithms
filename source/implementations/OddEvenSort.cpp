#include "./../headers/OddEvenSort.h"

void MultiThreaded::OddEvenSortCuda(std::vector<int> &arr)
{
    int  half = arr.size() / 2;
    int *d_arr;
    cudaMalloc(&d_arr, arr.size() * sizeof(int));
    cudaMemcpy(d_arr, arr.data(), arr.size() * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < half; i++)
    {
        CudaMethods::Even<<<1, half>>>(d_arr, arr.size());
        CudaMethods::Odd<<<1, half>>>(d_arr, arr.size());
        cudaDeviceSynchronize(); //?? must this be here?
    }
    cudaMemcpy(arr.data(), d_arr, arr.size() * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
}

__global__ void CudaMethods::Even(int *arr, int length) {
    int index = threadIdx.x * 2;
    if(index >= length) return;

    if (arr[index] > arr[index + 1]) {
        int tmp = arr[index];
        arr[index] = arr[index+ 1];
        arr[index + 1] = tmp;
    }
}

__global__ void CudaMethods::Odd(int* arr, int length) {
    int index = threadIdx.x * 2 + 1;
    if(index >= length) return;

    if (arr[index] > arr[index + 1]) {
        int tmp = arr[index];
        arr[index] = arr[index + 1];
        arr[index + 1] = tmp;
    }
}


void MultiThreaded::OddEvenSort(std::vector<int> &arr)
{
    bool sorted = false;
    while (!sorted)
    {
        sorted = true;
        
        std::thread t1([&arr, &sorted]{
            for (int i = 0; i < arr.size() - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    std::swap(arr[i], arr[i + 1]);
                    sorted = false;
                }
            }
        });

        std::thread t2([&arr, &sorted]{
            for (int i = 1; i < arr.size() - 1; i += 2)
            {
                if (arr[i] > arr[i + 1])
                {
                    std::swap(arr[i], arr[i + 1]);
                    sorted = false;
                }
            }
        });

        if(t1.joinable()) t1.join();
        if(t2.joinable()) t2.join();
    }
}
