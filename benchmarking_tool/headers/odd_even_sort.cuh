#ifndef ODD_EVEN_SORT_H
#define ODD_EVEN_SORT_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <vector>
#include <thread>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

namespace sorting
{
	void GpuOddEvenSort(std::vector<int>& arr);

	void CpuOddEvenSort(std::vector<int>& arr);

	void oldSort(std::vector<int>& arr);
	void newSortJoin(std::vector<int>& arr);
	void overSortJoin(std::vector<int>& arr);

	// https://stackoverflow.com/questions/26516683/reusing-thread-in-loop-c
	class ThreadPool
	{
		public:

		ThreadPool (int threads) : shutdown_ (false)
		{
			// Create the specified number of threads
			threads_.reserve (threads);
			for (int i = 0; i < threads; ++i)
				threads_.emplace_back(std::bind(&ThreadPool::threadEntry, this, i));

			// std::cout << "Threads size: " << (sizeof(std::vector<std::thread>) + threads_.size() * sizeof(std::thread))<< "B\n";
		}

		~ThreadPool ()
		{
			// std::cout << "Jobs size: " << prevSizeB << "kB\n";

			{
				// Unblock any threads and tell them to stop
				std::unique_lock<std::mutex> l(lock_);

				shutdown_ = true;
				condVar_.notify_all();
			}

			// Wait for all threads to stop
			for (auto& thread : threads_)
				thread.join();
		}

		void doJob(std::function<void(void)> func)
		{
			// Place a job on the queu and unblock a thread
			std::unique_lock<std::mutex> l(lock_);

			jobs_.emplace(std::move(func));
			// int size = (sizeof(std::vector<std::function<void()>>) + jobs_.size() * sizeof(std::function<void()>)) / 1024;
			// if (size > prevSizeB) {
			// 	prevSizeB = size;
			// }
			condVar_.notify_one();
		}
		
		bool queueRefill() {
			return jobs_.size() <= threads_.size();
		}

		bool queueEmpty() {
			return !jobs_.size();
		}

		int jobsSize() {
			return jobs_.size();
		}

		int tDone() {
			return threadsWorking;
		}
		bool threadsDone() {
			std::unique_lock <std::mutex> l(lock_);

			return threadsWorking == threads_.size();
		}
		void resetDoneThreads() {
			std::unique_lock <std::mutex> l(lock_);

			threadsWorking = 0;
		}

		protected:

		void threadEntry(int i)
		{
			std::function<void(void)> job;

			while (1)
			{
				{
					std::unique_lock <std::mutex> l(lock_);

					while (!shutdown_ && jobs_.empty())
						condVar_.wait (l);

					if (jobs_.empty())
					{
						// No jobs to do and we are shutting down
						return;
					}

					job = std::move(jobs_.front());
					jobs_.pop();
				}

				// Do the job without holding any locks
    			// std::chrono::time_point<std::chrono::high_resolution_clock> now = std::chrono::high_resolution_clock::now();

				{
					std::unique_lock <std::mutex> l(lock_);
					threadsWorking++;
				}
				job();
				{
					std::unique_lock <std::mutex> l(lock_);
					threadsWorking--;
				}
				// {
				// 	std::unique_lock <std::mutex> l(lock_);
				// 	threadsDoneCount++;
				// // 	std::cout << "Thread job: " << (std::chrono::high_resolution_clock::now() - now).count() << "\n";
				// }
			}

		}

		std::mutex lock_;
		std::condition_variable condVar_;
		bool shutdown_;
		std::queue<std::function<void(void)>> jobs_;
		std::vector<std::thread> threads_;
		int threadsWorking = 0;
		int prevSizeB = 0;
	};

	__global__ void Odd(int* arr, int length);
	__global__ void Even(int* arr, int length);
}

#endif