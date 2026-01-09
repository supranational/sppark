// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_CURVE25519_HPP__
#define __SPPARK_FF_CURVE25519_HPP__

#include <cstdint>
#if defined(__CUDACC__) || defined(__HIPCC__)

namespace device {
#define TO_CUDA_T(limb64) \
    (uint32_t)(static_cast<uint64_t>(limb64)), \
    (uint32_t)(static_cast<uint64_t>(limb64) >> 32)
    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_P[8] = {
        TO_CUDA_T(0xffffffffffffffed), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0x7fffffffffffffff)
    };
    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_RR[8] = {
        TO_CUDA_T(0x00000000000005a4), TO_CUDA_T(0x0000000000000000),
        TO_CUDA_T(0x0000000000000000), TO_CUDA_T(0x0000000000000000)
    };
    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_one[8] = {
        TO_CUDA_T(0x0000000000000026), TO_CUDA_T(0x0000000000000000),
        TO_CUDA_T(0x0000000000000000), TO_CUDA_T(0x0000000000000000)
    };
    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_Px2[8] = {
        TO_CUDA_T(0xffffffffffffffda), TO_CUDA_T(0xffffffffffffffff),
        TO_CUDA_T(0xffffffffffffffff), TO_CUDA_T(0xffffffffffffffff)
    };
    static __device__ __constant__ /*const*/ uint32_t CURVE25519_M0 = 0x286bca1bU;

    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_r[8] = {
        TO_CUDA_T(0x5812631a5cf5d3ed), TO_CUDA_T(0x14def9dea2f79cd6),
        TO_CUDA_T(0x0000000000000000), TO_CUDA_T(0x1000000000000000)
    };
    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_rRR[8] = {
        TO_CUDA_T(0xa40611e3449c0f01), TO_CUDA_T(0xd00e1ba768859347),
        TO_CUDA_T(0xceec73d217f5be65), TO_CUDA_T(0x0399411b7c309a3d)
    };
    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_rone[8] = {
        TO_CUDA_T(0xd6ec31748d98951d), TO_CUDA_T(0xc6ef5bf4737dcf70),
        TO_CUDA_T(0xfffffffffffffffe), TO_CUDA_T(0x0fffffffffffffff)
    };
    static __device__ __constant__ __align__(16) const uint32_t CURVE25519_rx16[8] = {
        TO_CUDA_T(0x812631a5cf5d3ed0), TO_CUDA_T(0x4def9dea2f79cd65),
        TO_CUDA_T(0x0000000000000001), TO_CUDA_T(0x0000000000000000)
    };
    static __device__ __constant__ /*const*/ uint32_t CURVE25519_m0 = 0x12547e1bU;
}

# if defined(__CUDA_ARCH__) || defined(__HIPCC__)   // device-side field types
#  if defined(__CUDA_ARCH__)
#   include "mont_t.cuh"
#  elif defined(__HIPCC__)
#   include "mont_t.hip"
typedef uint64_t vec256[4];
#  endif

namespace curve25519 {

typedef mont_t<255, device::CURVE25519_P, device::CURVE25519_M0,
                    device::CURVE25519_RR, device::CURVE25519_one,
                    device::CURVE25519_Px2> fp_mont;
struct fp_t : public fp_mont {
    using mem_t = fp_t;
    __device__ __forceinline__ fp_t() {}
    __device__ __forceinline__ fp_t(const fp_mont& a) : fp_mont(a) {}
    template<typename... Ts> constexpr fp_t(Ts... a) : fp_mont{a...} {}
#  ifdef __HIPCC__
    __host__   __forceinline__ fp_t(vec256 a)         : fp_mont(a) {}
#  endif
};

// Montgomery-encoded short Weierstrass coefficient a (little-endian limbs).
static __device__ __constant__ fp_t CURVE25519_A4 = fp_t(
    0xd90ff0fcU, 0x5555529aU, 0x55555555U, 0x55555555U,
    0x55555555U, 0x55555555U, 0x55555555U, 0x55555555U);

typedef mont_t<252, device::CURVE25519_r, device::CURVE25519_m0,
                    device::CURVE25519_rRR, device::CURVE25519_rone,
                    device::CURVE25519_rx16> fr_mont;
struct fr_t : public fr_mont {
    using mem_t = fr_t;
    __device__ __forceinline__ fr_t() {}
    __device__ __forceinline__ fr_t(const fr_mont& a) : fr_mont(a) {}
    template<typename... Ts> constexpr fr_t(Ts... a) : fr_mont{a...} {}
#  ifdef __HIPCC__
    __host__   __forceinline__ fr_t(vec256 a)         : fr_mont(a) {}
#  endif
};

} // namespace curve25519

# endif
#endif

#if !defined(__CUDA_ARCH__) && !defined(__HIPCC__)  // host-side field types
# include <blst_t.hpp>

# if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsubobject-linkage"
# endif

namespace curve25519 {

static const vec256 CURVE25519_P = {
    TO_LIMB_T(0xffffffffffffffed), TO_LIMB_T(0xffffffffffffffff),
    TO_LIMB_T(0xffffffffffffffff), TO_LIMB_T(0x7fffffffffffffff)
};
static const vec256 CURVE25519_RR = {   /* (1<<512)%P */
    TO_LIMB_T(0x00000000000005a4), TO_LIMB_T(0x0000000000000000),
    TO_LIMB_T(0x0000000000000000), TO_LIMB_T(0x0000000000000000)
};
static const vec256 CURVE25519_ONE = {  /* (1<<256)%P */
    TO_LIMB_T(0x0000000000000026), TO_LIMB_T(0x0000000000000000),
    TO_LIMB_T(0x0000000000000000), TO_LIMB_T(0x0000000000000000)
};
typedef blst_256_t<255, CURVE25519_P, 0x86bca1af286bca1bu,
                        CURVE25519_RR, CURVE25519_ONE> fp_mont;
struct fp_t : public fp_mont {
    using mem_t = fp_t;
    inline fp_t() {}
    inline fp_t(const fp_mont& a) : fp_mont(a) {}
    template<typename... Ts>
    constexpr fp_t(Ts... a) : fp_mont{a...} {}
};

// Montgomery-encoded short Weierstrass coefficient a.
static const fp_t CURVE25519_A4 = fp_t(
    0x5555529ad90ff0fcULL, 0x5555555555555555ULL,
    0x5555555555555555ULL, 0x5555555555555555ULL);

static const vec256 CURVE25519_r = {
    TO_LIMB_T(0x5812631a5cf5d3ed), TO_LIMB_T(0x14def9dea2f79cd6),
    TO_LIMB_T(0x0000000000000000), TO_LIMB_T(0x1000000000000000)
};
static const vec256 CURVE25519_rRR = {  /* (1<<512)%r */
    TO_LIMB_T(0xa40611e3449c0f01), TO_LIMB_T(0xd00e1ba768859347),
    TO_LIMB_T(0xceec73d217f5be65), TO_LIMB_T(0x0399411b7c309a3d)
};
static const vec256 CURVE25519_rONE = { /* (1<<256)%r */
    TO_LIMB_T(0xd6ec31748d98951d), TO_LIMB_T(0xc6ef5bf4737dcf70),
    TO_LIMB_T(0xfffffffffffffffe), TO_LIMB_T(0x0fffffffffffffff)
};
typedef blst_256_t<252, CURVE25519_r, 0xd2b51da312547e1bu,
                        CURVE25519_rRR, CURVE25519_rONE> fr_mont;
struct fr_t : public fr_mont {
    using mem_t = fr_t;
    inline fr_t() {}
    inline fr_t(const fr_mont& a) : fr_mont(a) {}
    template<typename... Ts>
    constexpr fr_t(Ts... a) : fr_mont{a...} {}
};

} // namespace curve25519

# if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
# endif
#endif

#ifdef FEATURE_CURVE25519
using namespace curve25519;
#endif

#endif
