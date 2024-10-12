#pragma once

#include <vector>
#include "ThreadPool.h"
#include "ParallelFor.h"

class ThreadedBitonicSort
{
private:
    std::unique_ptr<ParallelFor> parallelFor;

    void BitonicSort(std::vector<int> arr, int low, int count, bool ascending); 
    void BitonicMerge(std::vector<int> arr, int low, int count, bool ascending);
    void CompAndSwap(std::vector<int> arr, int i, int j, bool ascending);

public:
    ThreadedBitonicSort(std::unique_ptr<ThreadPool> threadPool);
    ~ThreadedBitonicSort();

    void Sort(std::vector<int> arr, bool isAscending = true);
};

ThreadedBitonicSort::ThreadedBitonicSort(std::unique_ptr<ThreadPool> threadPool)
{
    ThreadedBitonicSort::parallelFor = std::make_unique<ParallelFor>(std::move(threadPool));
}

ThreadedBitonicSort::~ThreadedBitonicSort()
{
}

void ThreadedBitonicSort::Sort(std::vector<int> arr, bool isAscending)
{
    ThreadedBitonicSort::BitonicSort(arr, 0, arr.size(), isAscending);
}

void ThreadedBitonicSort::BitonicSort(std::vector<int> arr, int low, int count, bool ascending)
{
    if (count > 1)
    {
        int k = count / 2;

        parallelFor->RunAndWait(low, low + k, 1, [&](int i) {
            BitonicSort(arr, i, k, true);
        });
        parallelFor->RunAndWait(low + k, low + count, 1, [&](int i) {
            BitonicSort(arr, i, k, false);
        });
        BitonicMerge(arr, low, count, ascending);
    }
}

void ThreadedBitonicSort::BitonicMerge(std::vector<int> arr, int low, int count, bool ascending)
{
    if (count > 1)
    {
        int k = count / 2;
        parallelFor->Run(low, low + k, 1, [&](int i) {
            ThreadedBitonicSort::CompAndSwap(arr, i, i + k, ascending);
        });
        parallelFor->RunAndWait(low, low + k, 1, [&](int i) {
            BitonicMerge(arr, i, k, ascending);
        });
        parallelFor->RunAndWait(low + k, low + count, 1, [&](int i) {
            BitonicMerge(arr, i, k, ascending);
        });
    }
}

void ThreadedBitonicSort::CompAndSwap(std::vector<int> arr, int i, int j, bool ascending)
{
    if (ascending == (arr[i] > arr[j]))
    {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}