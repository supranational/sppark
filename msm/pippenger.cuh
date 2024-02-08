// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_PIPPENGER_CUH__
#define __SPPARK_MSM_PIPPENGER_CUH__

#include <cassert>
#include <cooperative_groups.h>
#include <cuda.h>

#include <util/slice_t.hpp>
#include <util/vec2d_t.hpp>

#include "batch_addition.cuh"
#include "sort.cuh"

#ifndef WARP_SZ
#define WARP_SZ 32
#endif
#ifdef __GNUC__
#define asm __asm__ __volatile__
#else
#define asm asm volatile
#endif

/*
 * Break down |scalars| to signed |wbits|-wide digits.
 */

#ifdef __CUDA_ARCH__
// Transposed scalar_t
template <class scalar_t> class scalar_T {
    uint32_t val[sizeof(scalar_t) / sizeof(uint32_t)][WARP_SZ];

  public:
    __device__ const uint32_t &operator[](size_t i) const { return val[i][0]; }
    __device__ scalar_T &operator()(uint32_t laneid) { return *reinterpret_cast<scalar_T *>(&val[0][laneid]); }
    __device__ scalar_T &operator=(const scalar_t &rhs) {
        for (size_t i = 0; i < sizeof(scalar_t) / sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

template <class scalar_t>
__device__ __forceinline__ static uint32_t get_wval(const scalar_T<scalar_t> &scalar, uint32_t off,
                                                    uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1) {
    uint32_t i = off / 32;
    uint64_t ret = scalar[i];

    if (i < top_i)
        ret |= (uint64_t)scalar[i + 1] << 32;

    return ret >> (off % 32);
}

__device__ __forceinline__ static uint32_t booth_encode(uint32_t wval, uint32_t wmask, uint32_t wbits) {
    uint32_t sign = (wval >> wbits) & 1;
    wval = ((wval + 1) & wmask) >> 1;
    return sign ? 0 - wval : wval;
}
#endif

template <class scalar_t>
__launch_bounds__(1024) __global__ void breakdown(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
                                                  uint32_t nwins, uint32_t wbits, bool mont = true) {
    assert(len <= (1U << 31) && wbits < 32);

#ifdef __CUDA_ARCH__
    extern __shared__ scalar_T<scalar_t> xchange[];
    const uint32_t tid = threadIdx.x;
    const uint32_t tix = threadIdx.x + blockIdx.x * blockDim.x;

    const uint32_t top_i = (scalar_t::nbits + 31) / 32 - 1;
    const uint32_t wmask = 0xffffffffU >> (31 - wbits); // (1U << (wbits+1)) - 1;

    auto &scalar = xchange[tid / WARP_SZ](tid % WARP_SZ);

#pragma unroll 1
    for (uint32_t i = tix; i < (uint32_t)len; i += gridDim.x * blockDim.x) {
        auto s = scalars[i];

#if 0
        s.from();
        if (!mont) s.to();
#else
        if (mont)
            s.from();
#endif

        // clear the most significant bit
        uint32_t msb = s[top_i] >> ((scalar_t::nbits - 1) % 32);
        s.cneg(msb);
        msb <<= 31;

        scalar = s;

#pragma unroll 1
        for (uint32_t bit0 = nwins * wbits - 1, win = nwins; --win;) {
            bit0 -= wbits;
            uint32_t wval = get_wval(scalar, bit0, top_i);
            wval = booth_encode(wval, wmask, wbits);
            if (wval)
                wval ^= msb;
            digits[win][i] = wval;
        }

        uint32_t wval = s[0] << 1;
        wval = booth_encode(wval, wmask, wbits);
        if (wval)
            wval ^= msb;
        digits[0][i] = wval;
    }
#endif
}

#ifndef LARGE_L1_CODE_CACHE
#if __CUDA_ARCH__ - 0 >= 800
#define LARGE_L1_CODE_CACHE 1
#define ACCUMULATE_NTHREADS 384
#else
#define LARGE_L1_CODE_CACHE 0
#define ACCUMULATE_NTHREADS (bucket_t::degree == 1 ? 384 : 256)
#endif
#endif

#ifndef MSM_NTHREADS
#define MSM_NTHREADS 256
#endif
#if MSM_NTHREADS < 32 || (MSM_NTHREADS & (MSM_NTHREADS - 1)) != 0
#error "bad MSM_NTHREADS value"
#endif
#ifndef MSM_NSTREAMS
#define MSM_NSTREAMS 8
#elif MSM_NSTREAMS < 2
#error "invalid MSM_NSTREAMS"
#endif

template <class bucket_t, class affine_h, class bucket_h = class bucket_t::mem_t,
          class affine_t = class bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__
    void accumulate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                    /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits, const vec2d_t<uint32_t> histogram,
                    uint32_t sid = 0) {
    vec2d_t<bucket_h> buckets{buckets_, 1U << --wbits};
    const affine_h *points = points_;

    static __device__ uint32_t streams[MSM_NSTREAMS];
    uint32_t &current = streams[sid % MSM_NSTREAMS];
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t lane_id = laneid / degree;

    uint32_t x, y;
#if 1
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(&current, blockDim.x / degree);
    __syncthreads();
    x = xchg + threadIdx.x / degree;
#else
    x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
    x = __shfl_sync(0xffffffff, x, 0) + lane_id;
#endif

    while (x < (nwins << wbits)) {
        y = x >> wbits;
        x &= (1U << wbits) - 1;
        const uint32_t *h = &histogram[y][x];

        uint32_t idx, len = h[0];

        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}"
            : "=r"(idx)
            : "r"(len), "r"(degree));

        if (lane_id == 0 && x != 0)
            idx = h[-1];

        if ((len -= idx) && !(x == 0 && y == 0)) {
            const uint32_t *digs_ptr = &digits[y][idx];
            uint32_t digit = *digs_ptr++;

            affine_t p = points[digit & 0x7fffffff];
            bucket_t bucket = p;
            bucket.cneg(digit >> 31);

            while (--len) {
                digit = *digs_ptr++;
                p = points[digit & 0x7fffffff];
                if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
                    bucket.add(p, digit >> 31);
                else
                    bucket.uadd(p, digit >> 31);
            }

            buckets[y][x] = bucket;
        } else {
            buckets[y][x].inf();
        }

        x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
        x = __shfl_sync(0xffffffff, x, 0) + lane_id;
    }

    cooperative_groups::this_grid().sync();

    if (threadIdx.x + blockIdx.x == 0)
        current = 0;
}

template <class bucket_t, class bucket_h = class bucket_t::mem_t>
__launch_bounds__(256) __global__ void integrate(bucket_h buckets_[], uint32_t nwins, uint32_t wbits, uint32_t nbits) {
    const uint32_t degree = bucket_t::degree;
    uint32_t Nthrbits = 31 - __clz(blockDim.x / degree);

    assert((blockDim.x & (blockDim.x - 1)) == 0 && wbits - 1 > Nthrbits);

    vec2d_t<bucket_h> buckets{buckets_, 1U << (wbits - 1)};
    extern __shared__ uint4 scratch_[];
    auto *scratch = reinterpret_cast<bucket_h *>(scratch_);
    const uint32_t tid = threadIdx.x / degree;
    const uint32_t bid = blockIdx.x;

    auto *row = &buckets[bid][0];
    uint32_t i = 1U << (wbits - 1 - Nthrbits);
    row += tid * i;

    uint32_t mask = 0;
    if ((bid + 1) * wbits > nbits) {
        uint32_t lsbits = nbits - bid * wbits;
        mask = (1U << (wbits - lsbits)) - 1;
    }

    bucket_t res, acc = row[--i];

    if (i & mask) {
        if (sizeof(res) <= 128)
            res.inf();
        else
            scratch[tid].inf();
    } else {
        if (sizeof(res) <= 128)
            res = acc;
        else
            scratch[tid] = acc;
    }

    bucket_t p;

#pragma unroll 1
    while (i--) {
        p = row[i];

        uint32_t pc = i & mask ? 2 : 0;
#pragma unroll 1
        do {
            if (sizeof(bucket_t) <= 128) {
                p.add(acc);
                if (pc == 1) {
                    res = p;
                } else {
                    acc = p;
                    if (pc == 0)
                        p = res;
                }
            } else {
                if (LARGE_L1_CODE_CACHE && degree == 1)
                    p.add(acc);
                else
                    p.uadd(acc);
                if (pc == 1) {
                    scratch[tid] = p;
                } else {
                    acc = p;
                    if (pc == 0)
                        p = scratch[tid];
                }
            }
        } while (++pc < 2);
    }

    __syncthreads();

    buckets[bid][2 * tid] = p;
    buckets[bid][2 * tid + 1] = acc;
}
#undef asm

#ifndef SPPARK_DONT_INSTANTIATE_TEMPLATES
template __global__ void accumulate<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[], uint32_t nwins,
                                                               uint32_t wbits,
                                                               /*const*/ affine_t::mem_t points_[],
                                                               const vec2d_t<uint32_t> digits,
                                                               const vec2d_t<uint32_t> histogram, uint32_t sid);
template __global__ void batch_addition<bucket_t>(bucket_t::mem_t buckets[], const affine_t::mem_t points[],
                                                  size_t npoints, const uint32_t digits[], const uint32_t &ndigits);
template __global__ void integrate<bucket_t>(bucket_t::mem_t buckets_[], uint32_t nwins, uint32_t wbits,
                                             uint32_t nbits);
template __global__ void breakdown<scalar_t>(vec2d_t<uint32_t> digits, const scalar_t scalars[], size_t len,
                                             uint32_t nwins, uint32_t wbits, bool mont);
#endif

#include <vector>

#include <util/exception.cuh>
#include <util/gpu_t.cuh>
#include <util/rusterror.h>

template <class bucket_t, class point_t, class affine_t, class scalar_t, class affine_h = class affine_t::mem_t,
          class bucket_h = class bucket_t::mem_t>
class msm_t {
    const gpu_t &gpu;

    // main data
    bool owned;
    affine_h *d_points;
    scalar_t *d_scalars;
    uint32_t *d_pidx;
    size_t npoints;
    size_t nscalars;

    // per setup constants
    uint32_t wbits, nwins;
    uint32_t batch;
    uint32_t stride;

    // auxiliary space
    char *d_total_blob;
    bucket_h *d_buckets;
    vec2d_t<uint32_t> d_hist;
    vec2d_t<uint2> d_temps;
    vec2d_t<uint32_t> d_digits;

    template <typename T> using vec_t = slice_t<T>;

    class result_t {
        bucket_t ret[MSM_NTHREADS / bucket_t::degree][2];

      public:
        result_t() {}
        inline operator decltype(ret) & () { return ret; }
        inline const bucket_t *operator[](size_t i) const { return ret[i]; }
    };

    constexpr static int lg2(size_t n) {
        int ret = 0;
        while (n >>= 1)
            ret++;
        return ret;
    }

  public:
    // Initialize the MSM by moving the points to the device
    msm_t(const affine_t points[], size_t npoints, bool owned, int device_id = -1) : gpu(select_gpu(device_id)) {
        // set default values for fields
        this->d_points = nullptr;
        this->d_scalars = nullptr;
        this->d_pidx = nullptr;
        this->npoints = npoints;
        this->owned = owned;

        this->d_total_blob = nullptr;

        d_points = reinterpret_cast<decltype(d_points)>(gpu.Dmalloc(npoints * sizeof(d_points[0])));
        gpu.HtoD(d_points, points, npoints, sizeof(affine_h));
        CUDA_OK(cudaGetLastError());
    }

    msm_t(affine_h *d_points, size_t npoints, int device_id = -1) : gpu(select_gpu(device_id)) {
        // set default values for fields
        this->d_points = d_points;
        this->d_scalars = nullptr;
        this->d_pidx = nullptr;
        this->npoints = npoints;
        this->owned = false;

        this->d_total_blob = nullptr;
    }

    ~msm_t() {
        gpu.sync();
        if (d_total_blob)
            gpu.Dfree(d_total_blob);
        if (d_points && owned)
            gpu.Dfree(d_points);
    }

    affine_h *get_d_points() { return d_points; }
    void set_d_points(affine_h *d_points) {
        assert(!this->owned);
        this->d_points = d_points;
    }

  private:
    void digits(const scalar_t d_scalars[], size_t len, vec2d_t<uint32_t> &d_digits, vec2d_t<uint2> &d_temps, bool mont,
                uint32_t *d_pidx) {
        // Using larger grid size doesn't make 'sort' run faster, actually
        // quite contrary. Arguably because global memory bus gets
        // thrashed... Stepping far outside the sweet spot has significant
        // impact, 30-40% degradation was observed. It's assumed that all
        // GPUs are "balanced" in an approximately the same manner. The
        // coefficient was observed to deliver optimal performance on
        // Turing and Ampere...
        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));

        breakdown<<<2 * grid_size, 1024, sizeof(scalar_t) * 1024, gpu[2]>>>(d_digits, d_scalars, len, nwins, wbits,
                                                                            mont);
        CUDA_OK(cudaGetLastError());

        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
#if 0
        uint32_t win;
        for (win = 0; win < nwins-1; win++) {
            gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, 0u);
        }
        uint32_t top = scalar_t::bit_length() - wbits * win;
        gpu[2].launch_coop(sort, {grid_size, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
#else
        // On the other hand a pair of kernels launched in parallel run
        // ~50% slower but sort twice as much data...
        uint32_t top = scalar_t::bit_length() - wbits * (nwins - 1);
        uint32_t win;
        for (win = 0; win < nwins - 1; win += 2) {
            gpu[2].launch_coop(sort, {{grid_size, 2}, SORT_BLOCKDIM, shared_sz}, d_digits, len, win, d_temps, d_hist,
                               wbits - 1, wbits - 1, win == nwins - 2 ? top - 1 : wbits - 1, d_pidx);
        }
        if (win < nwins) {
            gpu[2].launch_coop(sort, {{grid_size, 1}, SORT_BLOCKDIM, shared_sz}, d_digits, len, win, d_temps, d_hist,
                               wbits - 1, top - 1, 0u, d_pidx);
        }
#endif
    }

  public:
    // Compute various constants (stride length, window size) based on the number of scalars.
    // Also allocate scratch space.
    void setup_scratch(size_t nscalars, uint32_t *pidx) {
        this->nscalars = nscalars;

        uint32_t lg_n = lg2(nscalars + nscalars / 2);

        wbits = 17;
        if (nscalars > 192) {
            wbits = std::min(lg_n, (uint32_t)18);
            if (wbits < 10)
                wbits = 10;
        } else if (nscalars > 0) {
            wbits = 10;
        }
        nwins = (scalar_t::bit_length() - 1) / wbits + 1;

        uint32_t row_sz = 1U << (wbits - 1);

        size_t d_buckets_sz = (nwins * row_sz) + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        d_buckets_sz *= sizeof(d_buckets[0]);
        size_t d_hist_sz = nwins * row_sz * sizeof(uint32_t);

        this->batch = 1 << (std::max(lg_n, wbits) - wbits);
        batch >>= 6;
        batch = batch ? batch : 1;
        this->stride = (nscalars + batch - 1) / batch;
        stride = (stride + WARP_SZ - 1) & ((size_t)0 - WARP_SZ);

        size_t temp_sz = stride * std::max(2 * sizeof(uint2), sizeof(scalar_t));
        size_t digits_sz = nwins * stride * sizeof(uint32_t);
        size_t pidx_sz = pidx ? stride * sizeof(uint32_t) : 0;

        size_t d_blob_sz = d_buckets_sz + d_hist_sz + temp_sz + digits_sz + pidx_sz;

        d_total_blob = reinterpret_cast<char *>(gpu.Dmalloc(d_blob_sz));
        size_t offset = 0;
        d_buckets = reinterpret_cast<decltype(d_buckets)>(&d_total_blob[offset]);
        offset += d_buckets_sz;
        d_hist = vec2d_t<uint32_t>((uint32_t *)&d_total_blob[offset], row_sz);
        offset += d_hist_sz;

        d_temps = vec2d_t<uint2>((uint2 *)&d_total_blob[offset], stride);
        d_scalars = (scalar_t *)&d_total_blob[offset];
        offset += temp_sz;
        d_digits = vec2d_t<uint32_t>((uint32_t *)&d_total_blob[offset], stride);
        offset += digits_sz;
        if (pidx) d_pidx = (uint32_t *)&d_total_blob[offset];
    }

    RustError invoke(point_t &out, const scalar_t *scalars, size_t nscalars, uint32_t pidx[], bool mont = true) {
        // assert(this->nscalars <= nscalars);
        setup_scratch(nscalars, pidx);

        std::vector<result_t> res(nwins);
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

        out.inf();
        point_t p;

        try {
            size_t d_off = 0; // device offset
            size_t h_off = 0; // host offset
            size_t num = stride > nscalars ? nscalars : stride;
            event_t ev;

            gpu[2].HtoD(&d_scalars[0], &scalars[h_off], num);
            if (pidx)
                gpu[2].HtoD(&d_pidx[0], &pidx[h_off], num);

            digits(&d_scalars[0], num, d_digits, d_temps, mont, d_pidx);
            gpu[2].record(ev);

            for (uint32_t i = 0; i < batch; i++) {
                gpu[i & 1].wait(ev);

                batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE, 0, gpu[i & 1]>>>(
                    &d_buckets[nwins << (wbits - 1)], &d_points[pidx ? 0 : d_off], num, &d_digits[0][0], d_hist[0][0]);
                CUDA_OK(cudaGetLastError());

                gpu[i & 1].launch_coop(accumulate<bucket_t, affine_h>, {gpu.sm_count(), 0}, d_buckets, nwins, wbits,
                                       &d_points[pidx ? 0 : d_off], d_digits, d_hist, i & 1);
                CUDA_OK(cudaGetLastError());
                gpu[i & 1].record(ev);

                integrate<bucket_t>
                    <<<nwins, MSM_NTHREADS, sizeof(bucket_t) * MSM_NTHREADS / bucket_t::degree, gpu[i & 1]>>>(
                        d_buckets, nwins, wbits, scalar_t::bit_length());
                CUDA_OK(cudaGetLastError());

                if (i < batch - 1) {
                    d_off += stride;
                    num = d_off + stride <= nscalars ? stride : nscalars - d_off;

                    gpu[2].HtoD(&d_scalars[0], &scalars[d_off], num);
                    if (pidx)
                        gpu[2].HtoD(&d_pidx[0], &pidx[d_off], num);

                    gpu[2].wait(ev);
                    digits(&d_scalars[0], num, d_digits, d_temps, mont, d_pidx);
                    gpu[2].record(ev);
                }

                if (i > 0) {
                    collect(p, res, ones);
                    out.add(p);
                }

                gpu[i & 1].DtoH(ones, d_buckets + (nwins << (wbits - 1)));
                gpu[i & 1].DtoH(res, d_buckets, sizeof(bucket_h) << (wbits - 1));
                gpu[i & 1].sync();
            }
        } catch (const cuda_error &e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        collect(p, res, ones);
        out.add(p);

        return RustError{cudaSuccess};
    }

  private:
    point_t integrate_row(const result_t &row, uint32_t lsbits) {
        const int NTHRBITS = lg2(MSM_NTHREADS / bucket_t::degree);

        assert(wbits - 1 > NTHRBITS);

        size_t i = MSM_NTHREADS / bucket_t::degree - 1;

        if (lsbits - 1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS - (lsbits - 1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask)
                res.inf();
            else
                res = acc;

            while (i--) {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }

            return res;
        }

        point_t res = row[i][0];
        bucket_t acc = row[i][1];

        while (i--) {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits - 1 - NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(row[i][0]);
            if (i)
                acc.add(row[i][1]);
        }

        return res;
    }

    void collect(point_t &out, const std::vector<result_t> &res, const std::vector<bucket_t> &ones) {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins - 1, total = 0;

        grid[0].x = 0;
        grid[0].y = y;
        grid[0].dy = scalar_t::bit_length() - y * wbits;
        total++;

        while (y--) {
            grid[total].x = grid[0].x;
            grid[total].y = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins); /* zeroed */
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0)
                    break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }
};

template <typename T> struct msm_context_t {
    T *d_points;
    size_t npoints;
};

template <typename T> void drop_msm_context_t(msm_context_t<T> &ref) { CUDA_OK(cudaFree(ref.d_points)); }

template <class bucket_t, class point_t, class affine_t, class scalar_t, class affine_h = class affine_t::mem_t,
          class bucket_h = class bucket_t::mem_t>
static RustError mult_pippenger_init(const affine_t points[], size_t npoints, msm_context_t<affine_h> *msm_context) {
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{points, npoints, false};
        msm_context->d_points = msm.get_d_points();
        msm_context->npoints = npoints;
        return RustError{cudaSuccess};
    } catch (const cuda_error &e) {
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

template <class bucket_t, class point_t, class affine_t, class scalar_t>
static RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints, const scalar_t scalars[],
                                size_t nscalars, uint32_t pidx[], bool mont = true) {
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{points, npoints, true};
        // msm.setup_scratch(nscalars);
        return msm.invoke(*out, scalars, nscalars, pidx, mont);
    } catch (const cuda_error &e) {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

template <class bucket_t, class point_t, class affine_t, class scalar_t, class affine_h = class affine_t::mem_t,
          class bucket_h = class bucket_t::mem_t>
static RustError mult_pippenger_with(point_t *out, msm_context_t<affine_h> *msm_context, const scalar_t scalars[],
                                     size_t nscalars, uint32_t pidx[], bool mont = true) {
    try {
        msm_t<bucket_t, point_t, affine_t, scalar_t> msm{msm_context->d_points, msm_context->npoints};
        // msm.setup_scratch(nscalars);
        return msm.invoke(*out, scalars, nscalars, pidx, mont);
    } catch (const cuda_error &e) {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

#endif