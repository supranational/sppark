// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

// Note: this file must be compiled as C++ to include SPPARK C++ headers.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include <util/rusterror.h>
#include <cuda_runtime.h>

#include <ff/curve25519.hpp>
#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <msm/pippenger.hpp>

using namespace curve25519;

typedef jacobian_t<fp_t, fp_t, &CURVE25519_A4> point_t;
typedef xyzz_t<fp_t, fp_t, &CURVE25519_A4> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

extern "C" RustError mult_pippenger_curve25519(point_t* out,
                                               const affine_t points[],
                                               size_t npoints,
                                               const scalar_t scalars[]);
extern "C" RustError mult_pippenger_curve25519_bytes_affine(
    struct curve25519_affine_bytes* out,
    const struct curve25519_affine_bytes* points,
    size_t npoints,
    const struct curve25519_scalar_bytes* scalars);
extern "C" RustError curve25519_add_affine_bytes(
    struct curve25519_affine_bytes* out,
    const struct curve25519_affine_bytes* points,
    size_t npoints);
extern "C" RustError curve25519_echo_affine_bytes(
    struct curve25519_affine_bytes* out,
    const struct curve25519_affine_bytes* points,
    size_t npoints);
extern "C" RustError curve25519_dump_digits(
    uint32_t* out_digits,
    size_t npoints,
    const struct curve25519_scalar_bytes* scalars,
    uint32_t wbits,
    uint32_t nwins);
extern "C" RustError curve25519_dump_sorted_digits(
    uint32_t* out_digits,
    uint32_t* out_hist,
    size_t npoints,
    const struct curve25519_scalar_bytes* scalars,
    uint32_t wbits,
    uint32_t nwins);

struct curve25519_affine_bytes {
    unsigned char x[32];
    unsigned char y[32];
};

struct curve25519_scalar_bytes {
    unsigned char s[32];
};

static bool scalar_bit_is_set(const scalar_t& scalar, size_t bit) {
    limb_t limbs[sizeof(scalar_t) / sizeof(limb_t)];
    scalar.store(limbs);
    const size_t limb_bits = 8 * sizeof(limb_t);
    const size_t limb_idx = bit / limb_bits;
    const size_t limb_off = bit % limb_bits;
    return (limbs[limb_idx] >> limb_off) & 1;
}

static point_t scalar_mul(const affine_t& point, const scalar_t& scalar) {
    point_t acc;
    point_t base;

    acc.inf();
    base = point;

    for (size_t i = 0; i < scalar_t::bit_length(); ++i) {
        if (scalar_bit_is_set(scalar, i)) {
            acc.add(base);
        }
        base.dbl();
    }

    return acc;
}

static point_t msm_cpu(const affine_t* points, const scalar_t* scalars, size_t npoints) {
    point_t acc;
    acc.inf();

    for (size_t i = 0; i < npoints; ++i) {
        point_t term = scalar_mul(points[i], scalars[i]);
        acc.add(term);
    }

    return acc;
}

static point_t msm_cpu_pippenger(const affine_t* points, const scalar_t* scalars, size_t npoints) {
    point_t out;
    out.inf();
    mult_pippenger<bucket_t, point_t, scalar_t, affine_t>(
        out, points, npoints, scalars, false, nullptr);
    return out;
}

// Forward declarations for helpers used in the early consistency test.
static affine_t curve25519_basepoint();
static scalar_t scalar_from_u64(uint64_t v);
static void dump_point_mismatch(const point_t& out, const point_t& expected, const char* ctx);

static affine_t affine_double_cpu(const affine_t& p) {
    if (p.is_inf()) {
        return p;
    }

    struct affine_raw {
        fp_t X;
        fp_t Y;
    };
    const affine_raw* raw = reinterpret_cast<const affine_raw*>(&p);
    fp_t x = raw->X;
    fp_t y = raw->Y;

    fp_t two_y = y + y;
    if (two_y.is_zero()) {
        return affine_t{fp_t(), fp_t()};
    }

    fp_t lambda = x^2;
    lambda += lambda + lambda;
    lambda += CURVE25519_A4;
    lambda *= fp_t::one() / two_y;

    fp_t x3 = (lambda ^ 2) - x - x;
    fp_t y3 = (lambda * (x - x3)) - y;
    return affine_t{x3, y3};
}

static void test_xyzz_matches_jacobian() {
    affine_t base = curve25519_basepoint();
    affine_t p1 = base;
    affine_t p2 = scalar_mul(base, scalar_from_u64(2));
    affine_t p3 = scalar_mul(base, scalar_from_u64(3));

    bucket_t b12 = p1;
    b12.add(p2);
    affine_t b12_aff = b12;
    point_t b12_point = b12_aff;

    point_t expected12 = point_t(p1);
    expected12.add(p2);
    if (!(b12_point == expected12)) {
        dump_point_mismatch(b12_point, expected12, "xyzz add 1+2");
        exit(1);
    }

    bucket_t b33 = p3;
    b33.add(p3);
    affine_t b33_aff = b33;
    point_t b33_point = b33_aff;

    affine_t aff33 = affine_double_cpu(p3);
    point_t aff33_point = aff33;

    point_t expected33 = point_t(p3);
    expected33.add(p3);
    if (!(b33_point == aff33_point)) {
        dump_point_mismatch(b33_point, aff33_point, "xyzz dbl 3+3 (affine)");
        exit(1);
    }
    if (!(expected33 == aff33_point)) {
        dump_point_mismatch(expected33, aff33_point, "jacobian dbl 3+3 (affine)");
        exit(1);
    }

}

static affine_t curve25519_basepoint() {
    // Short Weierstrass coordinates for the Edwards basepoint.
    fp_t x(0xaaaaaaaaaaad245aULL, 0xaaaaaaaaaaaaaaaaULL,
           0xaaaaaaaaaaaaaaaaULL, 0x2aaaaaaaaaaaaaaaULL);
    fp_t y(0xd6163a5d81312c14ULL, 0x6dc2b28192839e4dULL,
           0x1fe122d388b72eb3ULL, 0x5f51e65e475f794bULL);
    x.to();
    y.to();
    return affine_t{x, y};
}

static void check_ok(RustError err, const char* where) {
    if (err.code != 0) {
        const int code = err.code;
        const char* msg = nullptr;
        if (code < 0) {
            msg = cudaGetErrorString(static_cast<cudaError_t>(-code));
        } else {
            msg = cudaGetErrorString(static_cast<cudaError_t>(code));
        }
        fprintf(stderr, "%s failed: code=%d (%s)\n", where, err.code, msg ? msg : "unknown");
        exit(1);
    }
}

static void require(bool ok, const char* msg) {
    if (!ok) {
        fprintf(stderr, "%s\n", msg);
        exit(1);
    }
}

static void print_hex(const char* label, const unsigned char* data, size_t len) {
    fprintf(stderr, "%s: 0x", label);
    for (size_t i = 0; i < len; ++i) {
        fprintf(stderr, "%02x", data[i]);
    }
    fprintf(stderr, "\n");
}

static uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static uint32_t booth_encode_cpu(uint32_t wval, uint32_t wmask, uint32_t wbits) {
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0u - wval : wval;
}

static uint32_t get_wval_cpu(const unsigned char bytes[32], uint32_t off) {
    uint32_t byte = (off / 32) * 4;
    uint64_t val = 0;
    for (uint32_t i = 0; i < 8 && (byte + i) < 32; ++i) {
        val |= static_cast<uint64_t>(bytes[byte + i]) << (8 * i);
    }
    return static_cast<uint32_t>(val >> (off % 32));
}

static void build_sorted_digits_cpu(std::vector<uint32_t>& out_digits,
                                    std::vector<uint32_t>& out_hist,
                                    const std::vector<uint32_t>& digits,
                                    size_t npoints,
                                    uint32_t wbits,
                                    uint32_t nwins) {
    const uint32_t sort_bits = wbits - 1;
    const uint32_t row_sz = 1U << sort_bits;
    out_digits.assign(npoints * nwins, 0);
    out_hist.assign(row_sz * nwins, 0);

    const uint32_t key_mask = row_sz - 1;

    for (uint32_t win = 0; win < nwins; ++win) {
        uint32_t top = scalar_t::bit_length() - wbits * win;
        uint32_t lsbits = (top < wbits) ? (top - 1) : (wbits - 1);
        uint32_t lshift = sort_bits - lsbits;

        std::vector<std::pair<uint32_t, uint32_t>> pairs;
        pairs.reserve(npoints);

        for (size_t i = 0; i < npoints; ++i) {
            uint32_t val = digits[win * npoints + i];
            if (val == 0) {
                continue;
            }
            uint32_t key = ((val - 1) << lshift) | (static_cast<uint32_t>(i) & ((1U << lshift) - 1));
            key &= key_mask;
            uint32_t out = (static_cast<uint32_t>(i) & 0x7fffffffU) | (val & 0x80000000U);
            pairs.emplace_back(key, out);
        }

        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        std::vector<uint32_t> counts(row_sz, 0);
        for (const auto& entry : pairs) {
            counts[entry.first] += 1;
        }
        uint32_t acc = 0;
        for (uint32_t k = 0; k < row_sz; ++k) {
            acc += counts[k];
            out_hist[win * row_sz + k] = acc;
        }

        for (size_t idx = 0; idx < pairs.size(); ++idx) {
            out_digits[win * npoints + idx] = pairs[idx].second;
        }
    }
}

static scalar_t scalar_from_rng(uint64_t* state) {
    scalar_t s(splitmix64(state),
               splitmix64(state),
               splitmix64(state),
               splitmix64(state));
    // Reduce into [0, r) so signed recoding matches MSM assumptions.
    s.to();
    s.from();
    return s;
}

static scalar_t scalar_from_u64(uint64_t v) {
    return scalar_t(v, 0ULL, 0ULL, 0ULL);
}

static affine_t affine_infinity() {
    fp_t x;
    fp_t y;
    x.zero();
    y.zero();
    return affine_t{x, y};
}

static scalar_t scalar_from_bit(size_t bit) {
    uint64_t limbs[4] = {0, 0, 0, 0};
    const size_t limb_idx = bit / 64;
    const size_t limb_off = bit % 64;
    if (limb_idx < 4) {
        limbs[limb_idx] = 1ULL << limb_off;
    }
    return scalar_t(limbs[0], limbs[1], limbs[2], limbs[3]);
}

static void affine_to_bytes(curve25519_affine_bytes* out, const affine_t& aff) {
    if (aff.is_inf()) {
        memset(out->x, 0, sizeof(out->x));
        memset(out->y, 0, sizeof(out->y));
        return;
    }

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
    memcpy(out->x, xb, sizeof(out->x));
    memcpy(out->y, yb, sizeof(out->y));
}

static void scalar_to_bytes(curve25519_scalar_bytes* out, const scalar_t& s) {
    limb_t limbs[4];
    s.store(limbs);
    for (size_t i = 0; i < 4; ++i) {
        const uint64_t limb = static_cast<uint64_t>(limbs[i]);
        for (size_t j = 0; j < 8; ++j) {
            out->s[i * 8 + j] = static_cast<unsigned char>((limb >> (8 * j)) & 0xff);
        }
    }
}

static curve25519_scalar_bytes scalar_bytes_from_u64(uint64_t v) {
    curve25519_scalar_bytes out{};
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
        out.s[i] = static_cast<unsigned char>((v >> (8 * i)) & 0xff);
    }
    return out;
}

static affine_t affine_from_bytes(const curve25519_affine_bytes& in) {
    fp_t x;
    fp_t y;
    x.to(in.x, sizeof(in.x), true);
    y.to(in.y, sizeof(in.y), true);
    return affine_t{x, y};
}

static affine_t affine_add_cpu(const affine_t& p1, const affine_t& p2) {
    if (p1.is_inf()) {
        return p2;
    }
    if (p2.is_inf()) {
        return p1;
    }

    struct affine_raw {
        fp_t X;
        fp_t Y;
    };
    const affine_raw* raw1 = reinterpret_cast<const affine_raw*>(&p1);
    const affine_raw* raw2 = reinterpret_cast<const affine_raw*>(&p2);
    fp_t x1 = raw1->X;
    fp_t y1 = raw1->Y;
    fp_t x2 = raw2->X;
    fp_t y2 = raw2->Y;

    if (x1 == x2) {
        return affine_t{fp_t(0), fp_t(0)};
    }

    fp_t lambda = (y2 - y1) * (fp_t::one() / (x2 - x1));
    fp_t x3 = (lambda ^ 2) - x1 - x2;
    fp_t y3 = (lambda * (x1 - x3)) - y1;
    return affine_t{x3, y3};
}

static void dump_point_mismatch(const point_t& out, const point_t& expected, const char* ctx) {
    curve25519_affine_bytes out_bytes{};
    curve25519_affine_bytes exp_bytes{};
    affine_t out_aff = out;
    affine_t exp_aff = expected;
    affine_to_bytes(&out_bytes, out_aff);
    affine_to_bytes(&exp_bytes, exp_aff);

    fprintf(stderr, "curve25519 MSM mismatch (%s)\n", ctx);
    print_hex("out.x", out_bytes.x, sizeof(out_bytes.x));
    print_hex("out.y", out_bytes.y, sizeof(out_bytes.y));
    print_hex("exp.x", exp_bytes.x, sizeof(exp_bytes.x));
    print_hex("exp.y", exp_bytes.y, sizeof(exp_bytes.y));
}

static void test_fixed_two_point() {
    point_t out;
    affine_t points[2];
    scalar_t scalars[2];

    affine_t base = curve25519_basepoint();
    points[0] = base;
    points[1] = base;

    scalars[0] = scalar_from_u64(1);
    scalars[1] = scalar_from_u64(2);

    RustError err = mult_pippenger_curve25519(&out, points, 2, scalars);
    check_ok(err, "mult_pippenger_curve25519 (fixed)");

    point_t expected = msm_cpu(points, scalars, 2);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "fixed");
        exit(1);
    }
}

static void test_simple_scalars() {
    point_t out;
    const size_t npoints = 5;
    affine_t points[npoints];
    scalar_t scalars[npoints];

    affine_t base = curve25519_basepoint();
    for (size_t i = 0; i < npoints; ++i) {
        scalar_t point_scalar = scalar_from_u64(i + 1);
        points[i] = scalar_mul(base, point_scalar);
        scalars[i] = scalar_from_u64(i + 1);
    }

    RustError err = mult_pippenger_curve25519(&out, points, npoints, scalars);
    check_ok(err, "mult_pippenger_curve25519 (simple scalars)");

    point_t expected = msm_cpu(points, scalars, npoints);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "simple scalars");
    }
    point_t expected_pip = msm_cpu_pippenger(points, scalars, npoints);
    if (!(out == expected_pip)) {
        dump_point_mismatch(out, expected_pip, "simple scalars cpu pippenger");
        exit(1);
    }

    std::vector<curve25519_affine_bytes> points_bytes(npoints);
    std::vector<curve25519_scalar_bytes> scalars_bytes(npoints);
    for (size_t i = 0; i < npoints; ++i) {
        affine_to_bytes(&points_bytes[i], points[i]);
        scalars_bytes[i] = scalar_bytes_from_u64(i + 1);
    }

    curve25519_affine_bytes out_bytes{};
    err = mult_pippenger_curve25519_bytes_affine(
        &out_bytes, points_bytes.data(), npoints, scalars_bytes.data());
    check_ok(err, "mult_pippenger_curve25519_bytes_affine (simple scalars)");

    affine_t out_affine = affine_from_bytes(out_bytes);
    point_t out_bytes_point = out_affine;
    if (!(out_bytes_point == expected)) {
        dump_point_mismatch(out_bytes_point, expected, "simple scalars bytes");
        exit(1);
    }
    if (!(out_bytes_point == expected_pip)) {
        dump_point_mismatch(out_bytes_point, expected_pip, "simple scalars bytes cpu pippenger");
        exit(1);
    }

    const uint32_t wbits = 10;
    const uint32_t nwins = (scalar_t::bit_length() - 1) / wbits + 1;
    const uint32_t sort_bits = wbits - 1;
    std::vector<uint32_t> gpu_digits(npoints * nwins, 0);
    err = curve25519_dump_digits(gpu_digits.data(), npoints, scalars_bytes.data(), wbits, nwins);
    check_ok(err, "curve25519_dump_digits");

    const uint32_t wmask = 0xffffffffU >> (31 - wbits);
    std::vector<uint32_t> cpu_digits(npoints * nwins, 0);
    for (size_t i = 0; i < npoints; ++i) {
        const unsigned char* sbytes = scalars_bytes[i].s;
        for (uint32_t win = 1; win < nwins; ++win) {
            uint32_t bit0 = nwins * wbits - 1 - win * wbits;
            uint32_t wval = get_wval_cpu(sbytes, bit0);
            cpu_digits[win * npoints + i] = booth_encode_cpu(wval, wmask, wbits);
        }
        uint32_t wval0 = static_cast<uint32_t>(sbytes[0]) << 1;
        cpu_digits[i] = booth_encode_cpu(wval0, wmask, wbits);
    }

    for (size_t idx = 0; idx < gpu_digits.size(); ++idx) {
        if (gpu_digits[idx] != cpu_digits[idx]) {
            fprintf(stderr, "digit mismatch at idx=%zu gpu=%u cpu=%u\n",
                    idx, gpu_digits[idx], cpu_digits[idx]);
            exit(1);
        }
    }

    std::vector<uint32_t> gpu_sorted_digits(npoints * nwins, 0);
    std::vector<uint32_t> gpu_hist((1U << sort_bits) * nwins, 0);
    err = curve25519_dump_sorted_digits(
        gpu_sorted_digits.data(),
        gpu_hist.data(),
        npoints,
        scalars_bytes.data(),
        wbits,
        nwins);
    check_ok(err, "curve25519_dump_sorted_digits");

    std::vector<uint32_t> cpu_sorted_digits;
    std::vector<uint32_t> cpu_hist;
    build_sorted_digits_cpu(cpu_sorted_digits, cpu_hist, cpu_digits, npoints, wbits, nwins);

    for (size_t idx = 0; idx < gpu_sorted_digits.size(); ++idx) {
        if (gpu_sorted_digits[idx] != cpu_sorted_digits[idx]) {
            fprintf(stderr, "sorted digit mismatch at idx=%zu gpu=%u cpu=%u\n",
                    idx, gpu_sorted_digits[idx], cpu_sorted_digits[idx]);
            exit(1);
        }
    }
    for (size_t idx = 0; idx < gpu_hist.size(); ++idx) {
        if (gpu_hist[idx] != cpu_hist[idx]) {
            fprintf(stderr, "hist mismatch at idx=%zu gpu=%u cpu=%u\n",
                    idx, gpu_hist[idx], cpu_hist[idx]);
            exit(1);
        }
    }

    if (!(out == expected)) {
        exit(1);
    }
}

static void test_zero_scalars() {
    point_t out;
    affine_t points[4];
    scalar_t scalars[4];

    affine_t base = curve25519_basepoint();
    for (size_t i = 0; i < 4; ++i) {
        points[i] = base;
        scalars[i] = scalar_from_u64(0);
    }

    RustError err = mult_pippenger_curve25519(&out, points, 4, scalars);
    check_ok(err, "mult_pippenger_curve25519 (zero scalars)");

    point_t expected = msm_cpu(points, scalars, 4);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "zero scalars");
        exit(1);
    }
    require(out.is_inf(), "curve25519 MSM expected infinity for zero scalars");
}

static void test_permutation_invariance() {
    point_t out_a;
    point_t out_b;
    const size_t npoints = 8;
    std::vector<affine_t> points(npoints);
    std::vector<scalar_t> scalars(npoints);

    affine_t base = curve25519_basepoint();
    for (size_t i = 0; i < npoints; ++i) {
        scalar_t point_scalar = scalar_from_u64(i + 1);
        points[i] = scalar_mul(base, point_scalar);
        scalars[i] = scalar_from_u64(i + 2);
    }

    RustError err = mult_pippenger_curve25519(
        &out_a, points.data(), npoints, scalars.data());
    check_ok(err, "mult_pippenger_curve25519 (perm a)");

    std::vector<affine_t> points_rev(points.rbegin(), points.rend());
    std::vector<scalar_t> scalars_rev(scalars.rbegin(), scalars.rend());
    err = mult_pippenger_curve25519(
        &out_b, points_rev.data(), npoints, scalars_rev.data());
    check_ok(err, "mult_pippenger_curve25519 (perm b)");

    if (!(out_a == out_b)) {
        dump_point_mismatch(out_a, out_b, "permutation");
        exit(1);
    }
}

static void test_random_msm() {
    uint64_t rng = 0x4e554c4c4f505355ULL;
    affine_t base = curve25519_basepoint();

    const size_t sizes[] = {1, 2, 3, 4, 8, 16, 32, 64};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        size_t npoints = sizes[s];
        for (size_t round = 0; round < 4; ++round) {
            std::vector<affine_t> points(npoints);
            std::vector<scalar_t> scalars(npoints);

            for (size_t i = 0; i < npoints; ++i) {
                scalar_t point_scalar = scalar_from_rng(&rng);
                if (point_scalar.is_zero()) {
                    point_scalar = scalar_from_u64(1);
                }
                points[i] = scalar_mul(base, point_scalar);
                scalars[i] = scalar_from_rng(&rng);
            }

            point_t out;
            RustError err = mult_pippenger_curve25519(
                &out, points.data(), npoints, scalars.data());
            check_ok(err, "mult_pippenger_curve25519 (random)");

            point_t expected = msm_cpu(points.data(), scalars.data(), npoints);
            if (!(out == expected)) {
                dump_point_mismatch(out, expected, "random");
                exit(1);
            }
        }
    }
}

static void test_medium_random_msm() {
    uint64_t rng = 0x5f4d3c2b1a0f0e0dULL;
    const size_t npoints = 512;
    std::vector<affine_t> points(npoints);
    std::vector<scalar_t> scalars(npoints);

    affine_t base = curve25519_basepoint();
    for (size_t i = 0; i < npoints; ++i) {
        scalar_t point_scalar = scalar_from_rng(&rng);
        if (point_scalar.is_zero()) {
            point_scalar = scalar_from_u64(1);
        }
        points[i] = scalar_mul(base, point_scalar);
        scalars[i] = scalar_from_rng(&rng);
    }

    point_t out;
    RustError err = mult_pippenger_curve25519(
        &out, points.data(), npoints, scalars.data());
    check_ok(err, "mult_pippenger_curve25519 (medium random)");

    point_t expected = msm_cpu(points.data(), scalars.data(), npoints);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "medium random");
        exit(1);
    }
}

static void test_high_bit_scalars() {
    const size_t npoints = 5;
    affine_t points[npoints];
    scalar_t scalars[npoints];

    affine_t base = curve25519_basepoint();
    for (size_t i = 0; i < npoints; ++i) {
        points[i] = scalar_mul(base, scalar_from_u64(i + 1));
    }

    const size_t top_bit = scalar_t::bit_length() - 1;
    scalars[0] = scalar_from_bit(top_bit);
    scalars[1] = scalar_from_bit(top_bit - 1);
    scalars[2] = scalar_from_bit(top_bit) - scalar_from_u64(1);
    scalars[3] = scalar_from_bit(200);
    scalars[4] = scalar_from_bit(128) + scalar_from_u64(1);

    point_t out;
    RustError err = mult_pippenger_curve25519(&out, points, npoints, scalars);
    check_ok(err, "mult_pippenger_curve25519 (high bit scalars)");

    point_t expected = msm_cpu(points, scalars, npoints);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "high bit scalars");
        exit(1);
    }

    std::vector<curve25519_affine_bytes> points_bytes(npoints);
    std::vector<curve25519_scalar_bytes> scalars_bytes(npoints);
    for (size_t i = 0; i < npoints; ++i) {
        affine_to_bytes(&points_bytes[i], points[i]);
        scalar_to_bytes(&scalars_bytes[i], scalars[i]);
    }

    curve25519_affine_bytes out_bytes{};
    err = mult_pippenger_curve25519_bytes_affine(
        &out_bytes, points_bytes.data(), npoints, scalars_bytes.data());
    check_ok(err, "mult_pippenger_curve25519_bytes_affine (high bit scalars)");

    affine_t out_affine = affine_from_bytes(out_bytes);
    point_t out_bytes_point = out_affine;
    if (!(out_bytes_point == expected)) {
        dump_point_mismatch(out_bytes_point, expected, "high bit scalars bytes");
        exit(1);
    }
}

static void test_infinity_points() {
    const size_t npoints = 4;
    affine_t points[npoints];
    scalar_t scalars[npoints];

    affine_t base = curve25519_basepoint();
    points[0] = affine_infinity();
    points[1] = base;
    points[2] = scalar_mul(base, scalar_from_u64(2));
    points[3] = affine_infinity();

    scalars[0] = scalar_from_u64(5);
    scalars[1] = scalar_from_u64(6);
    scalars[2] = scalar_from_u64(7);
    scalars[3] = scalar_from_u64(8);

    point_t out;
    RustError err = mult_pippenger_curve25519(&out, points, npoints, scalars);
    check_ok(err, "mult_pippenger_curve25519 (infinity points)");

    point_t expected = msm_cpu(points, scalars, npoints);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "infinity points");
        exit(1);
    }
}

static void test_single_point_identity_scalar() {
    point_t out;
    affine_t points[1];
    scalar_t scalars[1];

    affine_t base = curve25519_basepoint();
    points[0] = base;
    scalars[0] = scalar_from_u64(1);

    RustError err = mult_pippenger_curve25519(&out, points, 1, scalars);
    check_ok(err, "mult_pippenger_curve25519 (single point)");

    point_t expected = point_t(base);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "single point");
        exit(1);
    }

    curve25519_affine_bytes points_bytes[1];
    curve25519_scalar_bytes scalars_bytes[1];
    affine_to_bytes(&points_bytes[0], points[0]);
    scalars_bytes[0] = scalar_bytes_from_u64(1);

    curve25519_affine_bytes out_bytes{};
    err = mult_pippenger_curve25519_bytes_affine(
        &out_bytes, points_bytes, 1, scalars_bytes);
    check_ok(err, "mult_pippenger_curve25519_bytes_affine (single point)");

    affine_t out_affine = affine_from_bytes(out_bytes);
    point_t out_bytes_point = out_affine;
    if (!(out_bytes_point == expected)) {
        dump_point_mismatch(out_bytes_point, expected, "single point bytes");
        exit(1);
    }

}

static void test_bytes_api() {
    uint64_t rng = 0x2a2b2c2d2e2f3031ULL;
    const size_t npoints = 16;
    std::vector<affine_t> points(npoints);
    std::vector<scalar_t> scalars(npoints);
    std::vector<curve25519_affine_bytes> points_bytes(npoints);
    std::vector<curve25519_scalar_bytes> scalars_bytes(npoints);

    affine_t base = curve25519_basepoint();
    for (size_t i = 0; i < npoints; ++i) {
        scalar_t point_scalar = scalar_from_rng(&rng);
        if (point_scalar.is_zero()) {
            point_scalar = scalar_from_u64(1);
        }
        points[i] = scalar_mul(base, point_scalar);
        scalars[i] = scalar_from_rng(&rng);
        affine_to_bytes(&points_bytes[i], points[i]);
        scalar_to_bytes(&scalars_bytes[i], scalars[i]);
    }

    curve25519_affine_bytes out_bytes{};
    RustError err = mult_pippenger_curve25519_bytes_affine(
        &out_bytes, points_bytes.data(), npoints, scalars_bytes.data());
    check_ok(err, "mult_pippenger_curve25519_bytes_affine");

    affine_t out_affine = affine_from_bytes(out_bytes);
    point_t out = out_affine;
    point_t expected = msm_cpu(points.data(), scalars.data(), npoints);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "bytes api");
        exit(1);
    }
}

static void test_gpu_add_two_points() {
    affine_t points[2];
    affine_t base = curve25519_basepoint();
    points[0] = base;
    points[1] = scalar_mul(base, scalar_from_u64(2));

    curve25519_affine_bytes points_bytes[2];
    affine_to_bytes(&points_bytes[0], points[0]);
    affine_to_bytes(&points_bytes[1], points[1]);

    curve25519_affine_bytes out_bytes{};
    RustError err = curve25519_add_affine_bytes(&out_bytes, points_bytes, 2);
    check_ok(err, "curve25519_add_affine_bytes");

    point_t expected = point_t(points[0]);
    expected.add(points[1]);
    affine_t expected_affine = affine_add_cpu(points[0], points[1]);
    point_t expected_affine_point = expected_affine;

    affine_t out_affine = affine_from_bytes(out_bytes);
    point_t out = out_affine;
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "gpu add two points");
    }
    if (!(out == expected_affine_point)) {
        dump_point_mismatch(out, expected_affine_point, "gpu add two points affine");
        exit(1);
    }

}

static void test_gpu_echo_affine() {
    curve25519_affine_bytes point_bytes{};
    affine_t base = curve25519_basepoint();
    affine_to_bytes(&point_bytes, base);

    curve25519_affine_bytes out_bytes{};
    RustError err = curve25519_echo_affine_bytes(&out_bytes, &point_bytes, 1);
    check_ok(err, "curve25519_echo_affine_bytes");

    affine_t out_affine = affine_from_bytes(out_bytes);
    point_t out = out_affine;
    point_t expected = point_t(base);
    if (!(out == expected)) {
        dump_point_mismatch(out, expected, "gpu echo affine");
        exit(1);
    }

}

int main(void) {
    test_xyzz_matches_jacobian();
    test_single_point_identity_scalar();
    test_gpu_echo_affine();
    test_gpu_add_two_points();
    test_simple_scalars();
    test_fixed_two_point();
    test_zero_scalars();
    test_permutation_invariance();
    test_random_msm();
    test_medium_random_msm();
    test_high_bit_scalars();
    test_infinity_points();
    test_bytes_api();

    return 0;
}
