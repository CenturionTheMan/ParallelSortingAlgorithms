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

	void sortMT(std::vector<int>& arr);

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
		}

		~ThreadPool ()
		{
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
				job();
			}

		}

		std::mutex lock_;
		std::condition_variable condVar_;
		bool shutdown_;
		std::queue<std::function<void(void)>> jobs_;
		std::vector<std::thread> threads_;
	};
}

#endif