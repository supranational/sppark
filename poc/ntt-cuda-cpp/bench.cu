// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cstring>

#ifdef FEATURE_BLS12_377
# include <ff/bls12-377.hpp>
#endif

#include <ntt/ntt.cuh>

#ifndef __CUDA_ARCH__

#include <util/thread_pool_t.hpp>
#include <util/cuda_timer_t.cuh>

const uint32_t min_lg_domain_size = 1u;
const uint32_t max_lg_domain_size = 24u;

#define BENCHMARK_ITERATIONS 10

int main()
{
    thread_pool_t pool;

    const gpu_t& gpu = select_gpu();

    index_t max_domain_size = (index_t)1 << max_lg_domain_size;

    std::vector<fr_t> inout(max_domain_size);

    pool.par_map(max_domain_size, [&](index_t i) {
#ifndef __CUDA_ARCH__
        inout[i] = fr_t((uint64_t)i);
#endif
    });

    cuda_timer_t timer;

    for (uint32_t lg_domain_size = min_lg_domain_size;
         lg_domain_size <= max_lg_domain_size;
         lg_domain_size++)
    {
        std::cout << "Benchmarking NTT for 2^" << lg_domain_size << std::endl;

        float total = 0;
        for (size_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
            timer.start(gpu);

            NTT::Base(gpu, &inout[0], lg_domain_size,
                      NTT::InputOutputOrder::NR,
                      NTT::Direction::forward,
                      NTT::Type::standard);

            total += timer.get_elapsed(gpu);
        }
        std::cout << "  NTT : " << std::fixed << std::setprecision(3)
                  << total / BENCHMARK_ITERATIONS << " milliseconds"
                  << std::endl;

        total = 0;
        for (size_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
            timer.start(gpu);

            NTT::Base(gpu, &inout[0], lg_domain_size,
                      NTT::InputOutputOrder::RN,
                      NTT::Direction::inverse,
                      NTT::Type::standard);

            total += timer.get_elapsed(gpu);
        }

        std::cout << "  iNTT: " << std::fixed << std::setprecision(3)
                  << total / BENCHMARK_ITERATIONS << " milliseconds"
                  << std::endl;
    }

    return 0;
}

#endif
