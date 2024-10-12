#include "./../headers/OddEvenSort.h"


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
