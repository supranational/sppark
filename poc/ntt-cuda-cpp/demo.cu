// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cstring>

#ifdef FEATURE_BLS12_377
# include <ff/bls12-377.hpp>
#endif

#include <ntt/ntt.cuh>
#include "util.cpp"

#ifndef __CUDA_ARCH__

const uint32_t lg_domain_size = 13u;

// A demo which checks functional correctness via polynomial multiplication
int main()
{
    thread_pool_t pool;

    const gpu_t& gpu = select_gpu();

    index_t domain_size = (index_t)1 << lg_domain_size;

    std::vector<fr_t> in_polynomial1(2 * domain_size),
                      in_polynomial2(2 * domain_size);
    std::vector<fr_t> out_polynomial_gpu(2 * domain_size);

    pool.par_map(domain_size, [&](index_t i) {
#ifndef __CUDA_ARCH__
        in_polynomial1[i] = fr_t((uint64_t)i * 19823);
        in_polynomial2[i] = fr_t((uint64_t)i * 22 + 10230102);

        in_polynomial1[(uint64_t)i + domain_size].zero();
        in_polynomial2[(uint64_t)i + domain_size].zero();
#endif
    });

    dev_ptr_t<fr_t> d_in_polynomial1(2 * domain_size),
                    d_in_polynomial2(2 * domain_size);
    dev_ptr_t<fr_t> d_out_polynomial(2 * domain_size);

    gpu.HtoD(&d_in_polynomial1[0], &in_polynomial1[0], 2 * domain_size);
    gpu.HtoD(&d_in_polynomial2[0], &in_polynomial2[0], 2 * domain_size);
    NTT::Base_dev_ptr(gpu, &d_in_polynomial1[0], lg_domain_size + 1,
                      NTT::InputOutputOrder::NR, NTT::Direction::forward,
                      NTT::Type::standard);
    NTT::Base_dev_ptr(gpu, &d_in_polynomial2[0], lg_domain_size + 1,
                      NTT::InputOutputOrder::NR, NTT::Direction::forward,
                      NTT::Type::standard);
    elementwise_multiplication<<<2 * domain_size / 1024, 1024, 0, gpu>>>(
        &d_out_polynomial[0], &d_in_polynomial1[0], &d_in_polynomial2[0]
    );
    CUDA_OK(cudaGetLastError());
    NTT::Base_dev_ptr(gpu, &d_out_polynomial[0], lg_domain_size + 1,
                      NTT::InputOutputOrder::RN, NTT::Direction::inverse,
                      NTT::Type::standard);
    gpu.DtoH(&out_polynomial_gpu[0], &d_out_polynomial[0], 2 * domain_size);
    gpu.sync();

    std::vector<fr_t> out_polynomial_naive(2 * domain_size);
    naive_polynomial_multiplication(out_polynomial_naive, in_polynomial1,
                                    in_polynomial2, pool);

    for (index_t i = 0; i < 2 * domain_size; i++) {
        assert(out_polynomial_gpu[i] == out_polynomial_naive[i]);
    }

    return 0;
}

#endif
