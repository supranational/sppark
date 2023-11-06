// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_ALT_BN128_HPP__
#define __SPPARK_FF_ALT_BN128_HPP__

#ifdef __NVCC__
#include <cstdint>

namespace device {
#define TO_CUDA_T(limb64) (uint32_t)(limb64), (uint32_t)(limb64>>32)
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_P[8] = {
        TO_CUDA_T(0x3c208c16d87cfd47), TO_CUDA_T(0x97816a916871ca8d),
        TO_CUDA_T(0xb85045b68181585d), TO_CUDA_T(0x30644e72e131a029)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_RR[8] = { /* (1<<512)%P */
        TO_CUDA_T(0xf32cfc5b538afa89), TO_CUDA_T(0xb5e71911d44501fb),
        TO_CUDA_T(0x47ab1eff0a417ff6), TO_CUDA_T(0x06d89f71cab8351f),
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_one[8] = { /* (1<<256)%P */
        TO_CUDA_T(0xd35d438dc58f0d9d), TO_CUDA_T(0x0a78eb28f5c70b3d),
        TO_CUDA_T(0x666ea36f7879462c), TO_CUDA_T(0x0e0a77c19a07df2f)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_Px4[8] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0xf082305b61f3f51c), TO_CUDA_T(0x5e05aa45a1c72a34),
        TO_CUDA_T(0xe14116da06056176), TO_CUDA_T(0xc19139cb84c680a6)
    };
    static __device__ __constant__ const uint32_t ALT_BN128_M0 = 0xe4866389;

    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_r[8] = {
        TO_CUDA_T(0x43e1f593f0000001), TO_CUDA_T(0x2833e84879b97091),
        TO_CUDA_T(0xb85045b68181585d), TO_CUDA_T(0x30644e72e131a029)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rRR[8] = { /* (1<<512)%P */
        TO_CUDA_T(0x1bb8e645ae216da7), TO_CUDA_T(0x53fe3ab1e35c59e3),
        TO_CUDA_T(0x8c49833d53bb8085), TO_CUDA_T(0x0216d0b17f4e44a5)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rone[8] = { /* (1<<256)%P */
        TO_CUDA_T(0xac96341c4ffffffb), TO_CUDA_T(0x36fc76959f60cd29),
        TO_CUDA_T(0x666ea36f7879462e), TO_CUDA_T(0x0e0a77c19a07df2f)
    };
    static __device__ __constant__ __align__(16) const uint32_t ALT_BN128_rx4[8] = { /* left-aligned value of the modulus */
        TO_CUDA_T(0x0f87d64fc0000004), TO_CUDA_T(0xa0cfa121e6e5c245),
        TO_CUDA_T(0xe14116da06056174), TO_CUDA_T(0xc19139cb84c680a6)
    };
    static __device__ __constant__ const uint32_t ALT_BN128_m0 = 0xefffffff;
}
# ifdef __CUDA_ARCH__   // device-side field types
# include "mont_t.cuh"
typedef mont_t<254, device::ALT_BN128_P, device::ALT_BN128_M0,
                    device::ALT_BN128_RR, device::ALT_BN128_one,
                    device::ALT_BN128_Px4> fp_mont;
struct fp_t : public fp_mont {
    using mem_t = fp_t;
    __device__ __forceinline__ fp_t() {}
    __device__ __forceinline__ fp_t(const fp_mont& a) : fp_mont(a) {}
};
typedef mont_t<254, device::ALT_BN128_r, device::ALT_BN128_m0,
                    device::ALT_BN128_rRR, device::ALT_BN128_rone,
                    device::ALT_BN128_rx4> fr_mont;
struct fr_t : public fr_mont {
    using mem_t = fr_t;
    __device__ __forceinline__ fr_t() {}
    __device__ __forceinline__ fr_t(const fr_mont& a) : fr_mont(a) {}
};
# endif
#endif

#ifndef __CUDA_ARCH__   // host-side field types
# include <blst_t.hpp>

# if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsubobject-linkage"
# endif

static const vec256 ALT_BN128_P = {
    TO_LIMB_T(0x3c208c16d87cfd47), TO_LIMB_T(0x97816a916871ca8d),
    TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
};
static const vec256 ALT_BN128_RR = {    /* (1<<512)%P */
    TO_LIMB_T(0xf32cfc5b538afa89), TO_LIMB_T(0xb5e71911d44501fb),
    TO_LIMB_T(0x47ab1eff0a417ff6), TO_LIMB_T(0x06d89f71cab8351f),
};
static const vec256 ALT_BN128_ONE = {   /* (1<<256)%P */
    TO_LIMB_T(0xd35d438dc58f0d9d), TO_LIMB_T(0x0a78eb28f5c70b3d),
    TO_LIMB_T(0x666ea36f7879462c), TO_LIMB_T(0x0e0a77c19a07df2f)
};
typedef blst_256_t<254, ALT_BN128_P, 0x87d20782e4866389u,
                        ALT_BN128_RR, ALT_BN128_ONE> fp_mont;
struct fp_t : public fp_mont {
    using mem_t = fp_t;
    inline fp_t() {}
    inline fp_t(const fp_mont& a) : fp_mont(a) {}
};

static const vec256 ALT_BN128_r = {
    TO_LIMB_T(0x43e1f593f0000001), TO_LIMB_T(0x2833e84879b97091),
    TO_LIMB_T(0xb85045b68181585d), TO_LIMB_T(0x30644e72e131a029)
};
static const vec256 ALT_BN128_rRR = {   /* (1<<512)%r */
    TO_LIMB_T(0x1bb8e645ae216da7), TO_LIMB_T(0x53fe3ab1e35c59e3),
    TO_LIMB_T(0x8c49833d53bb8085), TO_LIMB_T(0x0216d0b17f4e44a5)
};
static const vec256 ALT_BN128_rONE = {  /* (1<<256)%r */
    TO_LIMB_T(0xac96341c4ffffffb), TO_LIMB_T(0x36fc76959f60cd29),
    TO_LIMB_T(0x666ea36f7879462e), TO_LIMB_T(0x0e0a77c19a07df2f)
};
typedef blst_256_t<254, ALT_BN128_r, 0xc2e1f593efffffffu,
                        ALT_BN128_rRR, ALT_BN128_rONE> fr_mont;
struct fr_t : public fr_mont {
    using mem_t = fr_t;
    inline fr_t() {}
    inline fr_t(const fr_mont& a) : fr_mont(a) {}
};

static const vec256 ALT_BN128_G1_GEN_X = {
    TO_LIMB_T(0xd35d438dc58f0d9d), TO_LIMB_T(0x0a78eb28f5c70b3d),
    TO_LIMB_T(0x666ea36f7879462c), TO_LIMB_T(0x0e0a77c19a07df2f)
};

static const vec256 ALT_BN128_G1_GEN_Y = {
    TO_LIMB_T(0xa6ba871b8b1e1b3a), TO_LIMB_T(0x14f1d651eb8e167b),
    TO_LIMB_T(0xccdd46def0f28c58), TO_LIMB_T(0x1c14ef83340fbe5e)
};

# if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
# endif
#endif
#endif
