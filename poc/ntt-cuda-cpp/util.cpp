// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <util/thread_pool_t.hpp>

__launch_bounds__(1024) __global__
void elementwise_multiplication(fr_t* out, const fr_t* in1, const fr_t* in2)
{
    index_t idx = (index_t)blockIdx.x * blockDim.x + threadIdx.x;

    out[idx] = in1[idx] * in2[idx];
}

void naive_polynomial_multiplication(std::vector<fr_t>& out,
                                     const std::vector<fr_t>& in1,
                                     const std::vector<fr_t>& in2,
                                     thread_pool_t& pool)
{
    std::memset(&out[0], 0, out.size() * sizeof(fr_t));

    for (size_t i = 0; i < out.size() / 2; i++) {
        pool.par_map(out.size() / 2, [&](size_t j) {
            out[i + j] += in1[i] * in2[j];
        });
    }
}
