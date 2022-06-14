// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifdef __NVCC__
#include <cstdint>

namespace device {
    static __device__ __constant__ const uint32_t Vesta_P[8] = {
        0x00000001, 0x8c46eb21, 0x0994a8dd, 0x224698fc,
        0x00000000, 0x00000000, 0x00000000, 0x40000000
    };
    static __device__ __constant__ const uint32_t Vesta_RR[8] = { /* (1<<512)%P */
        0x0000000f, 0xfc9678ff, 0x891a16e3, 0x67bb433d,
        0x04ccf590, 0x7fae2310, 0x7ccfdaa9, 0x096d41af
    };
    static __device__ __constant__ const uint32_t Vesta_one[8] = { /* (1<<256)%P */
        0xfffffffd, 0x5b2b3e9c, 0xe3420567, 0x992c350b,
        0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff
    };

    static __device__ __constant__ const uint32_t Pallas_P[8] = {
        0x00000001, 0x992d30ed, 0x094cf91b, 0x224698fc,
        0x00000000, 0x00000000, 0x00000000, 0x40000000
    };
    static __device__ __constant__ const uint32_t Pallas_RR[8] = { /* (1<<512)%P */
        0x0000000f, 0x8c78ecb3, 0x8b0de0e7, 0xd7d30dbd,
        0xc3c95d18, 0x7797a99b, 0x7b9cb714, 0x096d41af
    };
    static __device__ __constant__ const uint32_t Pallas_one[8] = { /* (1<<256)%P */
        0xfffffffd, 0x34786d38, 0xe41914ad, 0x992c350b,
        0xffffffff, 0xffffffff, 0xffffffff, 0x3fffffff
    };
    static __device__ __constant__ /*const*/ uint32_t Pasta_M0 = 0xffffffff;
}

# ifdef __CUDA_ARCH__   // device-side field types
# include "mont_t.cuh"
typedef mont_t<255, device::Vesta_P, device::Pasta_M0,
                    device::Vesta_RR, device::Vesta_one> vesta_t;
typedef mont_t<255, device::Pallas_P, device::Pasta_M0,
                    device::Pallas_RR, device::Pallas_one> pallas_t;
# endif
#endif

#ifndef __CUDA_ARCH__   // host-side field types
# include <pasta_t.hpp>
#endif
