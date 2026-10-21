// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#if defined(__HIPCC__)
# include <hip/hip_runtime.h>
// The MSM driver runs its post-kernel point accumulation on the CPU, which needs
// host-callable field arithmetic. Select the host (blst) field in the HIP host
// pass and the device mont_t field only in the device pass (mirroring CUDA's
// __CUDA_ARCH__ split). NTT, which never does host field math, keeps the device
// field in both passes (upstream default).
# define SPPARK_HIP_HOST_FIELD
#else
# include <cuda.h>
#endif

// The G2 (fp2 extension field) MSM relies on the Montgomery modular inverse
// (vt_inverse_mod_x) provided only by the CUDA mont_t.cuh device field; the
// ROCm/HIP mont_t.hip does not implement it yet, so G2 MSM is not built on
// HIP. G1 MSM is fully supported. See the project notes for the deferred G2
// work.
#if !defined(__HIPCC__)
# define SPPARK_MSM_FP2
#endif

// bn254 (alt_bn128) G1 MSM is not yet supported on the ROCm/HIP backend: the
// kernel hangs the GPU for this curve. Refuse to build it rather than emit a
// binary that wedges the device. bls12_381 and bls12_377 G1 MSM are supported.
// See the project notes for the deferred bn254-on-ROCm investigation.
#if defined(__HIPCC__) && defined(FEATURE_BN254)
# error "bn254 G1 MSM is not yet supported on the ROCm/HIP backend"
#endif

#if defined(FEATURE_BLS12_381)
# if defined(SPPARK_MSM_FP2)
#  include <ff/bls12-381-fp2.hpp>
# else
#  include <ff/bls12-381.hpp>
# endif
#elif defined(FEATURE_BLS12_377)
# if defined(SPPARK_MSM_FP2)
#  include <ff/bls12-377-fp2.hpp>
# else
#  include <ff/bls12-377.hpp>
# endif
#elif defined(FEATURE_BN254)
# if defined(SPPARK_MSM_FP2)
#  include <ff/alt_bn128-fp2.hpp>
# else
#  include <ff/alt_bn128.hpp>
# endif
#else
# error "no FEATURE"
#endif

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_inf_t affine_t;
typedef fr_t scalar_t;

#define SPPARK_DONT_INSTANTIATE_TEMPLATES
#include <msm/pippenger.cuh>

extern "C"
RustError::by_value mult_pippenger_inf(point_t* out, const affine_t points[],
                                       size_t npoints, const scalar_t scalars[],
                                       size_t ffi_affine_sz)
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false, ffi_affine_sz);
}

#if defined(SPPARK_MSM_FP2) && \
    (defined(FEATURE_BLS12_381) || defined(FEATURE_BLS12_377) || defined(FEATURE_BN254))
typedef jacobian_t<fp2_t> point_fp2_t;
typedef xyzz_t<fp2_t> bucket_fp2_t;
typedef bucket_fp2_t::affine_inf_t affine_fp2_t;

extern "C"
RustError::by_value mult_pippenger_fp2_inf(point_fp2_t* out, const affine_fp2_t points[],
                                           size_t npoints, const scalar_t scalars[],
                                           size_t ffi_affine_sz)
{
    return mult_pippenger<bucket_fp2_t>(out, points, npoints, scalars, false, ffi_affine_sz);
}
#endif
