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

const size_t npoints = (size_t)1 << 16;

void test_msm()
{
    thread_pool_t pool;

    const gpu_t& gpu = select_gpu();

    std::vector<affine_t> in_affine (npoints);
    std::vector<fr_t>     in_scalars(npoints);

    pool.par_map(npoints, [&](size_t i) {
#ifndef __CUDA_ARCH__
    fr_t fr(i + (size_t)9923);
    in_affine[i] = generate_g1_point(fr);
    in_scalars[i] = fr_t(i * 10 + 1);
#endif
    });

    point_t out_gpu;
    mult_pippenger<bucket_t>(&out_gpu, &in_affine[0], npoints, &in_scalars[0]);
    point_t out_naive = msm_naive(in_affine, in_scalars);

    affine_t res_gpu   = out_gpu;
    affine_t res_naive = out_naive;

#ifndef __CUDA_ARCH__
    assert(res_gpu == res_naive);
/*
    X: 0x00be8cef6bd0df42da3da1975afa844fafa5957a5b0d3f54aea2c0e468d1cc440f7b44004f23291e209156dbe480694c
    Y: 0x001ca7377dda5fef936ab7f2b1a462606c2861233b072b26f7cc9eb10af02e22ae57423e122216d1fe156a2f40d09d4f

    std::cout << res_gpu << std::endl;
*/
#endif
}

int main()
{
    test_msm();

    return 0;
}
