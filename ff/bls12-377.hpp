// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifdef __NVCC__
#include <cstdint>

namespace device {
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)
    static __device__ __constant__ const uint32_t BLS12_377_P[12] = {
        TO_CUDA_T(0x8508c00000000001), TO_CUDA_T(0x170b5d4430000000),
        TO_CUDA_T(0x1ef3622fba094800), TO_CUDA_T(0x1a22d9f300f5138f),
        TO_CUDA_T(0xc63b05c06ca1493b), TO_CUDA_T(0x01ae3a4617c510ea)
    };
    static __device__ __constant__ const uint32_t BLS12_377_RR[12] = { /* (1<<768)%P */
        TO_CUDA_T(0xb786686c9400cd22), TO_CUDA_T(0x0329fcaab00431b1),
        TO_CUDA_T(0x22a5f11162d6b46d), TO_CUDA_T(0xbfdf7d03827dc3ac),
        TO_CUDA_T(0x837e92f041790bf9), TO_CUDA_T(0x006dfccb1e914b88)
    };
    static __device__ __constant__ const uint32_t BLS12_377_one[12] = { /* (1<<384)%P */
        TO_CUDA_T(0x02cdffffffffff68), TO_CUDA_T(0x51409f837fffffb1),
        TO_CUDA_T(0x9f7db3a98a7d3ff2), TO_CUDA_T(0x7b4e97b76e7c6305),
        TO_CUDA_T(0x4cf495bf803c84e8), TO_CUDA_T(0x008d6661e2fdf49a)
    };
    static __device__ __constant__ /*const*/ uint32_t BLS12_377_M0 = 0xffffffff;

    static __device__ __constant__ const uint32_t BLS12_377_r[8] = {
        TO_CUDA_T(0x0a11800000000001), TO_CUDA_T(0x59aa76fed0000001),
        TO_CUDA_T(0x60b44d1e5c37b001), TO_CUDA_T(0x12ab655e9a2ca556)
    };
    static __device__ __constant__ const uint32_t BLS12_377_rRR[8] = { /* (1<<512)%P */
        TO_CUDA_T(0x25d577bab861857b), TO_CUDA_T(0xcc2c27b58860591f),
        TO_CUDA_T(0xa7cc008fe5dc8593), TO_CUDA_T(0x011fdae7eff1c939)
    };
    static __device__ __constant__ const uint32_t BLS12_377_rone[8] = { /* (1<<256)%P */
        TO_CUDA_T(0x7d1c7ffffffffff3), TO_CUDA_T(0x7257f50f6ffffff2),
        TO_CUDA_T(0x16d81575512c0fee), TO_CUDA_T(0x0d4bda322bbb9a9d)
    };
    static __device__ __constant__ /*const*/ uint32_t BLS12_377_m0 = 0xffffffff;
}
# ifdef __CUDA_ARCH__   // device-side field types
# include "mont_t.cuh"
typedef mont_t<377, device::BLS12_377_P, device::BLS12_377_M0,
                    device::BLS12_377_RR, device::BLS12_377_one> fp_t;
typedef mont_t<253, device::BLS12_377_r, device::BLS12_377_m0,
                    device::BLS12_377_rRR, device::BLS12_377_rone> fr_t;
# endif
#endif

#ifndef __CUDA_ARCH__   // host-side field types
# include <blst_t.hpp>

static const vec384 BLS12_377_P = {
    TO_LIMB_T(0x8508c00000000001), TO_LIMB_T(0x170b5d4430000000),
    TO_LIMB_T(0x1ef3622fba094800), TO_LIMB_T(0x1a22d9f300f5138f),
    TO_LIMB_T(0xc63b05c06ca1493b), TO_LIMB_T(0x01ae3a4617c510ea)
};
static const vec384 BLS12_377_RR = {    /* (1<<768)%P */
    TO_LIMB_T(0xb786686c9400cd22), TO_LIMB_T(0x0329fcaab00431b1),
    TO_LIMB_T(0x22a5f11162d6b46d), TO_LIMB_T(0xbfdf7d03827dc3ac),
    TO_LIMB_T(0x837e92f041790bf9), TO_LIMB_T(0x006dfccb1e914b88)
};
static const vec384 BLS12_377_ONE = {   /* (1<<384)%P */
    TO_LIMB_T(0x02cdffffffffff68), TO_LIMB_T(0x51409f837fffffb1),
    TO_LIMB_T(0x9f7db3a98a7d3ff2), TO_LIMB_T(0x7b4e97b76e7c6305),
    TO_LIMB_T(0x4cf495bf803c84e8), TO_LIMB_T(0x008d6661e2fdf49a)
};
typedef blst_384_t<BLS12_377_P, 0x8508bfffffffffffu,
                   BLS12_377_RR, BLS12_377_ONE> fp_t;

static const vec256 BLS12_377_r = { 
    TO_LIMB_T(0x0a11800000000001), TO_LIMB_T(0x59aa76fed0000001),
    TO_LIMB_T(0x60b44d1e5c37b001), TO_LIMB_T(0x12ab655e9a2ca556)
};
static const vec256 BLS12_377_rRR = {   /* (1<<512)%r */
    TO_LIMB_T(0x25d577bab861857b), TO_LIMB_T(0xcc2c27b58860591f),
    TO_LIMB_T(0xa7cc008fe5dc8593), TO_LIMB_T(0x011fdae7eff1c939)
};
static const vec256 BLS12_377_rONE = {  /* (1<<256)%r */
    TO_LIMB_T(0x7d1c7ffffffffff3), TO_LIMB_T(0x7257f50f6ffffff2),
    TO_LIMB_T(0x16d81575512c0fee), TO_LIMB_T(0x0d4bda322bbb9a9d)
};
typedef blst_256_t<BLS12_377_r, 0xa117fffffffffffu,
                   BLS12_377_rRR, BLS12_377_rONE> fr_t;
#endif
