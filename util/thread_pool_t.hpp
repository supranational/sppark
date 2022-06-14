// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __THREAD_POOL_T_HPP__
#define __THREAD_POOL_T_HPP__

#if __cplusplus < 201103L && !(defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
# error C++11 or later is required.
#endif

#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <deque>
#include <functional>
#ifdef _GNU_SOURCE
# include <sched.h>
#endif

class thread_pool_t {
private:
    std::vector<std::thread> threads;

    std::mutex mtx;                     // Inter-thread synchronization
    std::condition_variable cvar;
    std::atomic<bool> done;

    typedef std::function<void()> job_t;
    std::deque<job_t> fifo;

public:
    thread_pool_t(unsigned int num_threads = 0) : done(false)
    {
        if (num_threads == 0) {
            num_threads = std::thread::hardware_concurrency();
#ifdef _GNU_SOURCE
            cpu_set_t set;
            if (sched_getaffinity(0, sizeof(set), &set) == 0) {
                size_t i, n;
                for (n = 0, i = num_threads; i--;)
                    n += CPU_ISSET(i, &set);
                if (n != 0)
                    num_threads = n;
            }
#endif
        }

        threads.reserve(num_threads);

        for (unsigned int i = 0; i < num_threads; i++)
            threads.push_back(std::thread([this]() { while (execute()); }));
    }

    virtual ~thread_pool_t()
    {
        done = true;
        cvar.notify_all();
        for (auto& tid : threads)
            tid.join();
    }

    size_t size() { return threads.size(); }

    template<class Workable> void spawn(Workable work)
    {
        std::unique_lock<std::mutex> lock(mtx);
        fifo.emplace_back(job_t(work));
        cvar.notify_one();  // wake up a worker thread
    }

private:
    bool execute()
    {
        job_t work;
        {
            std::unique_lock<std::mutex> lock(mtx);

            while (!done && fifo.empty())
                cvar.wait(lock);

            if (done && fifo.empty())
                return false;

            work = fifo.front(); fifo.pop_front();
        }
        work();

        return true;
    }

public:
    // call work(size_t idx) with idx=[0..num_items) in parallel, e.g.
    // pool.par_map(20, [&](size_t i) { std::cout << i << std::endl; });
    template<class Workable>
    void par_map(size_t num_items, Workable work, size_t max_workers = 0)
    {
        size_t num_workers = std::min(size(), num_items);
        if (max_workers > 0)
            num_workers = std::min(num_workers, max_workers);

        std::atomic<size_t> counter(0);
        std::atomic<size_t> done(num_workers);
        std::mutex b_mtx;
        std::condition_variable barrier;

        while (num_workers--) {
            spawn([&, num_items]() {
                size_t idx;
                while ((idx = counter.fetch_add(1, std::memory_order_relaxed))
                            < num_items)
                    work(idx);
                if (--done == 0) {
                    std::unique_lock<std::mutex> lock(b_mtx);
                    barrier.notify_one();
                }
            });
        }

        std::unique_lock<std::mutex> lock(b_mtx);
        barrier.wait(lock, [&] { return done == 0; });
    }
};

template<class T> class channel_t {
private:
    std::deque<T> fifo;
    std::mutex mtx;
    std::condition_variable cvar;

public:
    void send(const T& msg)
    {
        std::unique_lock<std::mutex> lock(mtx);
        fifo.push_back(msg);
        cvar.notify_one();
    }

    T recv()
    {
        std::unique_lock<std::mutex> lock(mtx);
        cvar.wait(lock, [&] { return !fifo.empty(); });
        auto msg = fifo.front(); fifo.pop_front();
        return msg;
    }
};

template<typename T> class counter_t {
    struct inner {
        std::atomic<T> val;
        std::atomic<size_t> ref_cnt;
        inline inner(T v) { val = v, ref_cnt = 1; };
    };
    inner *ptr;
public:
    counter_t(T v=0) { ptr = new inner(v); }
    counter_t(const counter_t& r)
    {   (ptr = r.ptr)->ref_cnt.fetch_add(1, std::memory_order_relaxed);   }
    ~counter_t()
    {
        if (ptr->ref_cnt.fetch_sub(1, std::memory_order_seq_cst) == 1)
            delete ptr;
    }
    size_t ref_cnt() const  { return ptr->ref_cnt; }
    T operator++(int) const { return ptr->val.fetch_add(1, std::memory_order_relaxed); }
    T operator++() const    { return ptr->val++ + 1; }
    T operator--(int) const { return ptr->val.fetch_sub(1, std::memory_order_relaxed); }
    T operator--() const    { return ptr->val-- - 1; }
};
#endif  // __THREAD_POOL_T_HPP__
