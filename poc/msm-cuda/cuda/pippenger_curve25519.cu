// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>

#include <vector>

#include <ff/curve25519.hpp>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>

using namespace curve25519;

typedef jacobian_t<fp_t, fp_t, &CURVE25519_A4> point_t;
typedef xyzz_t<fp_t, fp_t, &CURVE25519_A4> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

struct curve25519_affine_bytes {
    unsigned char x[32];
    unsigned char y[32];
};

struct curve25519_scalar_bytes {
    unsigned char s[32];
};

struct curve25519_affine_device {
    uint32_t x[8];
    uint32_t y[8];
};

struct curve25519_scalar_device {
    uint32_t s[8];
};

#ifndef __CUDA_ARCH__
extern "C"
RustError mult_pippenger_curve25519(point_t* out, const affine_t points[], size_t npoints,
                                    const scalar_t scalars[])
{
    return mult_pippenger<bucket_t>(out, points, npoints, scalars, false);
}
#endif

#ifndef __CUDA_ARCH__
static void limbs64_to_u32(uint32_t out[8], const uint64_t in[4]) {
    for (size_t i = 0; i < 4; ++i) {
        out[i * 2] = static_cast<uint32_t>(in[i]);
        out[i * 2 + 1] = static_cast<uint32_t>(in[i] >> 32);
    }
}

static void bytes_to_limbs64(uint64_t out[4], const unsigned char bytes[32]) {
    for (size_t i = 0; i < 4; ++i) {
        uint64_t limb = 0;
        for (size_t j = 0; j < 8; ++j) {
            limb |= static_cast<uint64_t>(bytes[i * 8 + j]) << (8 * j);
        }
        out[i] = limb;
    }
}

static bool bytes_are_zero(const unsigned char bytes[32]) {
    unsigned char acc = 0;
    for (size_t i = 0; i < 32; ++i) {
        acc |= bytes[i];
    }
    return acc == 0;
}

static void limbs32_to_u64(uint64_t out[4], const uint32_t in[8]) {
    for (size_t i = 0; i < 4; ++i) {
        out[i] = static_cast<uint64_t>(in[i * 2])
               | (static_cast<uint64_t>(in[i * 2 + 1]) << 32);
    }
}

static curve25519_affine_device affine_device_from_bytes(
    const curve25519_affine_bytes& in) {
    curve25519_affine_device out{};
    if (bytes_are_zero(in.x) && bytes_are_zero(in.y)) {
        return out;
    }

    fp_t x;
    fp_t y;
    x.to(in.x, sizeof(in.x), true);
    y.to(in.y, sizeof(in.y), true);

    uint64_t x_limbs[4];
    uint64_t y_limbs[4];
    x.store(reinterpret_cast<limb_t*>(x_limbs));
    y.store(reinterpret_cast<limb_t*>(y_limbs));
    limbs64_to_u32(out.x, x_limbs);
    limbs64_to_u32(out.y, y_limbs);
    return out;
}

static void affine_bytes_from_device(curve25519_affine_bytes* out,
                                     const curve25519_affine_device& in) {
    uint64_t x_limbs[4];
    uint64_t y_limbs[4];
    limbs32_to_u64(x_limbs, in.x);
    limbs32_to_u64(y_limbs, in.y);

    fp_t x(x_limbs[0], x_limbs[1], x_limbs[2], x_limbs[3]);
    fp_t y(y_limbs[0], y_limbs[1], y_limbs[2], y_limbs[3]);

    fp_t::pow_t xb;
    fp_t::pow_t yb;
    x.to_scalar(xb);
    y.to_scalar(yb);
    for (size_t i = 0; i < sizeof(out->x); ++i) {
        out->x[i] = xb[i];
        out->y[i] = yb[i];
    }
}

static scalar_t scalar_from_bytes_le(const unsigned char bytes[32]) {
    if (bytes_are_zero(bytes)) {
        return scalar_t(0ULL, 0ULL, 0ULL, 0ULL);
    }
    uint64_t limbs[4];
    bytes_to_limbs64(limbs, bytes);
    return scalar_t(limbs[0], limbs[1], limbs[2], limbs[3]);
}

extern "C"
RustError mult_pippenger_curve25519_bytes_affine(
    curve25519_affine_bytes* out,
    const curve25519_affine_bytes* points,
    size_t npoints,
    const curve25519_scalar_bytes* scalars)
{
    std::vector<affine_t> points_aff(npoints);
    std::vector<scalar_t> scalars_fr(npoints);

    for (size_t i = 0; i < npoints; ++i) {
        fp_t x;
        fp_t y;
        x.to(points[i].x, sizeof(points[i].x), true);
        y.to(points[i].y, sizeof(points[i].y), true);
        points_aff[i] = affine_t{x, y};

        scalars_fr[i] = scalar_from_bytes_le(scalars[i].s);
    }

    point_t acc;
    RustError err = mult_pippenger<bucket_t>(
        &acc,
        points_aff.data(),
        npoints,
        scalars_fr.data(),
        false);
    if (err.code != 0) {
        return err;
    }

    if (acc.is_inf()) {
        for (size_t i = 0; i < 32; ++i) {
            out->x[i] = 0;
            out->y[i] = 0;
        }
        return err;
    }

    affine_t aff = acc;
    // Reinterpret to access coordinates for serialization.
    struct affine_raw {
        fp_t X;
        fp_t Y;
    };
    const affine_raw* raw = reinterpret_cast<const affine_raw*>(&aff);
    fp_t x = raw->X;
    fp_t y = raw->Y;
    fp_t::pow_t xb;
    fp_t::pow_t yb;
    x.to_scalar(xb);
    y.to_scalar(yb);
    for (size_t i = 0; i < sizeof(out->x); ++i) {
        out->x[i] = xb[i];
        out->y[i] = yb[i];
    }
    return err;
}

__global__ void curve25519_add_affine_kernel(curve25519_affine_device* out,
                                             const curve25519_affine_device* in);

extern "C"
RustError curve25519_add_affine_bytes(
    curve25519_affine_bytes* out,
    const curve25519_affine_bytes* points,
    size_t npoints)
{
    if (npoints < 2) {
        return RustError{1};
    }

    curve25519_affine_device host_points[2];
    host_points[0] = affine_device_from_bytes(points[0]);
    host_points[1] = affine_device_from_bytes(points[1]);

    curve25519_affine_device* d_in = nullptr;
    curve25519_affine_device* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_in, sizeof(host_points));
    if (err != cudaSuccess) {
        return RustError{err};
    }
    err = cudaMalloc(&d_out, sizeof(curve25519_affine_device));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        return RustError{err};
    }

    err = cudaMemcpy(d_in, host_points, sizeof(host_points), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return RustError{err};
    }

    curve25519_add_affine_kernel<<<1, 32>>>(d_out, d_in);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return RustError{err};
    }

    curve25519_affine_device host_out{};
    err = cudaMemcpy(&host_out, d_out, sizeof(host_out), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    if (err != cudaSuccess) {
        return RustError{err};
    }

    affine_bytes_from_device(out, host_out);
    return RustError{cudaSuccess};
}

__global__ void curve25519_echo_affine_kernel(curve25519_affine_device* out,
                                              const curve25519_affine_device* in);

extern "C"
RustError curve25519_echo_affine_bytes(
    curve25519_affine_bytes* out,
    const curve25519_affine_bytes* points,
    size_t npoints)
{
    if (npoints < 1) {
        return RustError{1};
    }

    curve25519_affine_device host_point = affine_device_from_bytes(points[0]);
    curve25519_affine_device* d_in = nullptr;
    curve25519_affine_device* d_out = nullptr;
    cudaError_t err = cudaMalloc(&d_in, sizeof(host_point));
    if (err != cudaSuccess) {
        return RustError{err};
    }
    err = cudaMalloc(&d_out, sizeof(host_point));
    if (err != cudaSuccess) {
        cudaFree(d_in);
        return RustError{err};
    }

    err = cudaMemcpy(d_in, &host_point, sizeof(host_point), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return RustError{err};
    }

    curve25519_echo_affine_kernel<<<1, 1>>>(d_out, d_in);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        cudaFree(d_in);
        cudaFree(d_out);
        return RustError{err};
    }

    curve25519_affine_device host_out{};
    err = cudaMemcpy(&host_out, d_out, sizeof(host_out), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    if (err != cudaSuccess) {
        return RustError{err};
    }

    affine_bytes_from_device(out, host_out);
    return RustError{cudaSuccess};
}

extern "C"
RustError curve25519_dump_digits(
    uint32_t* out_digits,
    size_t npoints,
    const curve25519_scalar_bytes* scalars,
    uint32_t wbits,
    uint32_t nwins)
{
    if (npoints == 0) {
        return RustError{1};
    }

    std::vector<scalar_t> scalars_fr(npoints);
    for (size_t i = 0; i < npoints; ++i) {
        scalars_fr[i] = scalar_from_bytes_le(scalars[i].s);
    }

    scalar_t* d_scalars = nullptr;
    uint32_t* d_digits = nullptr;
    cudaError_t err = cudaMalloc(&d_scalars, sizeof(scalar_t) * npoints);
    if (err != cudaSuccess) {
        return RustError{err};
    }
    err = cudaMalloc(&d_digits, sizeof(uint32_t) * npoints * nwins);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        return RustError{err};
    }

    err = cudaMemcpy(d_scalars, scalars_fr.data(),
                     sizeof(scalar_t) * npoints, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        cudaFree(d_digits);
        return RustError{err};
    }

    vec2d_t<uint32_t> digits(d_digits, static_cast<uint32_t>(npoints));
    breakdown<<<1, 1024, sizeof(scalar_t) * 1024>>>(
        digits,
        d_scalars,
        npoints,
        nwins,
        wbits,
        false);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        cudaFree(d_digits);
        return RustError{err};
    }

    err = cudaMemcpy(out_digits, d_digits,
                     sizeof(uint32_t) * npoints * nwins,
                     cudaMemcpyDeviceToHost);
    cudaFree(d_scalars);
    cudaFree(d_digits);
    if (err != cudaSuccess) {
        return RustError{err};
    }

    return RustError{cudaSuccess};
}

extern "C"
RustError curve25519_dump_sorted_digits(
    uint32_t* out_digits,
    uint32_t* out_hist,
    size_t npoints,
    const curve25519_scalar_bytes* scalars,
    uint32_t wbits,
    uint32_t nwins)
{
    if (npoints == 0 || wbits == 0 || nwins == 0) {
        return RustError{1};
    }

    std::vector<scalar_t> scalars_fr(npoints);
    for (size_t i = 0; i < npoints; ++i) {
        scalars_fr[i] = scalar_from_bytes_le(scalars[i].s);
    }

    scalar_t* d_scalars = nullptr;
    uint32_t* d_digits = nullptr;
    uint2* d_temps = nullptr;
    uint32_t* d_hist = nullptr;
    const uint32_t sort_bits = wbits - 1;
    const uint32_t row_sz = 1U << sort_bits;
    const size_t digits_sz = sizeof(uint32_t) * npoints * nwins;
    const size_t hist_sz = sizeof(uint32_t) * row_sz * nwins;
    cudaError_t err = cudaMalloc(&d_scalars, sizeof(scalar_t) * npoints);
    if (err != cudaSuccess) {
        return RustError{err};
    }
    err = cudaMalloc(&d_digits, digits_sz);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        return RustError{err};
    }
    err = cudaMalloc(&d_temps, sizeof(uint2) * npoints);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        cudaFree(d_digits);
        return RustError{err};
    }
    err = cudaMalloc(&d_hist, hist_sz);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        cudaFree(d_digits);
        cudaFree(d_temps);
        return RustError{err};
    }

    err = cudaMemcpy(d_scalars, scalars_fr.data(),
                     sizeof(scalar_t) * npoints, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        cudaFree(d_digits);
        cudaFree(d_temps);
        cudaFree(d_hist);
        return RustError{err};
    }

    vec2d_t<uint32_t> digits(d_digits, static_cast<uint32_t>(npoints));
    breakdown<<<1, 1024, sizeof(scalar_t) * 1024>>>(
        digits,
        d_scalars,
        npoints,
        nwins,
        wbits,
        false);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        cudaFree(d_digits);
        cudaFree(d_temps);
        cudaFree(d_hist);
        return RustError{err};
    }

    const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
    vec2d_t<uint2> temps(d_temps, static_cast<uint32_t>(npoints));
    vec2d_t<uint32_t> hist(d_hist, row_sz);
    for (uint32_t win = 0; win < nwins; ++win) {
        uint32_t top = scalar_t::bit_length() - wbits * win;
        uint32_t lsbits = (top < wbits) ? (top - 1) : (wbits - 1);
        sort<<<dim3(1, 1), SORT_BLOCKDIM, shared_sz>>>(
            digits,
            npoints,
            win,
            temps,
            hist,
            sort_bits,
            lsbits,
            0u);
        err = cudaGetLastError();
        if (err == cudaSuccess) {
            err = cudaDeviceSynchronize();
        }
        if (err != cudaSuccess) {
            cudaFree(d_scalars);
            cudaFree(d_digits);
            cudaFree(d_temps);
            cudaFree(d_hist);
            return RustError{err};
        }
    }

    err = cudaMemcpy(out_digits, d_digits, digits_sz, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_scalars);
        cudaFree(d_digits);
        cudaFree(d_temps);
        cudaFree(d_hist);
        return RustError{err};
    }

    err = cudaMemcpy(out_hist, d_hist, hist_sz, cudaMemcpyDeviceToHost);
    cudaFree(d_scalars);
    cudaFree(d_digits);
    cudaFree(d_temps);
    cudaFree(d_hist);
    if (err != cudaSuccess) {
        return RustError{err};
    }

    return RustError{cudaSuccess};
}
#endif

__global__ void curve25519_add_affine_kernel(curve25519_affine_device* out,
                                             const curve25519_affine_device* in) {
#if defined(__CUDA_ARCH__)
    fp_t x1, y1, x2, y2;
    for (size_t i = 0; i < 8; ++i) {
        x1[i] = in[0].x[i];
        y1[i] = in[0].y[i];
        x2[i] = in[1].x[i];
        y2[i] = in[1].y[i];
    }

    fp_t one = fp_t::one();
    fp_t lambda = (y2 - y1) * (one / (x2 - x1));
    fp_t x3 = (lambda ^ 2) - x1 - x2;
    fp_t y3 = (lambda * (x1 - x3)) - y1;

    if (threadIdx.x == 0) {
        x3.store(out->x);
        y3.store(out->y);
    }
#else
    (void)out;
    (void)in;
#endif
}

__global__ void curve25519_echo_affine_kernel(curve25519_affine_device* out,
                                              const curve25519_affine_device* in) {
#if defined(__CUDA_ARCH__)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = in[0];
    }
#else
    (void)out;
    (void)in;
#endif
}
