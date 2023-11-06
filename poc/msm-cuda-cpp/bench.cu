// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cstring>

#ifdef FEATURE_BLS12_377
# include <ff/bls12-377.hpp>
#endif

#include <ec/affine_t.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef Affine_t<fp_t> affine_t;
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef fr_t scalar_t;

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <msm/pippenger.cuh>

#include "util.cpp"
#include <util/thread_pool_t.hpp>
#include <util/cuda_timer_t.cuh>

const uint32_t min_lg_npoints = 16u;
const uint32_t max_lg_npoints = 23u;

#define BENCHMARK_ITERATIONS 5

int main()
{
    thread_pool_t pool;

    const gpu_t& gpu = select_gpu();

    size_t max_npoints = (size_t)1 << max_lg_npoints;

    std::vector<affine_t> points (max_npoints);
    std::vector<fr_t>     scalars(max_npoints);

    pool.par_map(max_npoints, [&](size_t i) {
#ifndef __CUDA_ARCH__
    fr_t fr(i + (size_t)9923);
    points[i] = generate_g1_point(fr);

    scalars[i] = fr_t(i * 195292 + i);
    // the scalar is already in montgomery form, but we do another conversion
    // as a hack to get a more uniform distribution
    scalars[i].to(); scalars[i].to();
#endif
    });

    cuda_timer_t timer;
    point_t out;

    for (uint32_t lg_npoints = min_lg_npoints; lg_npoints <= max_lg_npoints; lg_npoints++) {
        size_t npoints = (size_t)1 << lg_npoints;

        std::cout << "Benchmarking MSM for 2^" << lg_npoints << std::endl;

        float total = 0;
        for (size_t i = 0; i < BENCHMARK_ITERATIONS; i++) {
            timer.start(gpu);
            mult_pippenger<bucket_t>(&out, &points[0], npoints, &scalars[0]);
            total += timer.get_elapsed(gpu);
        }
        std::cout << "  MSM: " << std::fixed << std::setprecision(3)
                  << total / BENCHMARK_ITERATIONS << " milliseconds"
                  << std::endl;
    }

    return 0;
}
