// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __PIPPENGER_CUH__
#define __PIPPENGER_CUH__

#include <cuda.h>

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

#ifndef NTHREADS
# define NTHREADS 256
#endif
#if NTHREADS < 32 || (NTHREADS & (NTHREADS-1)) != 0
# error "bad NTHREADS value"
#endif

constexpr static int log2(int n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

static const int NTHRBITS = log2(NTHREADS);

#ifndef NBITS
# define NBITS 255
#endif
#ifndef WBITS
# define WBITS 16
#endif
#define NWINS ((NBITS+WBITS-1)/WBITS)   // ceil(NBITS/WBITS)

#ifndef LARGE_L1_CODE_CACHE
# define LARGE_L1_CODE_CACHE 0
#endif

//
// To be launched as 'pippenger<<<dim3(NWINS, N), NTHREADS>>>(...)'  with
// |npoints| being around N*2**20 and N*NWINS*NTHREADS not exceeding the
// occupancy limit.
//
__global__
void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars, bool mont,
               bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] = nullptr);

#ifdef __CUDA_ARCH__

#include <cooperative_groups.h>

// Transposed scalar_t
class scalar_T {
    uint32_t val[sizeof(scalar_t)/sizeof(uint32_t)][WARP_SZ];

public:
    __device__ uint32_t& operator[](size_t i)              { return val[i][0]; }
    __device__ const uint32_t& operator[](size_t i) const  { return val[i][0]; }
    __device__ scalar_T& operator=(const scalar_t& rhs)
    {
        for (size_t i = 0; i < sizeof(scalar_t)/sizeof(uint32_t); i++)
            val[i][0] = rhs[i];
        return *this;
    }
};

class scalars_T {
    scalar_T* ptr;

public:
    __device__ scalars_T(void* rhs) { ptr = (scalar_T*)rhs; }
    __device__ scalar_T& operator[](size_t i)
    {   return *(scalar_T*)&(&ptr[i/WARP_SZ][0])[i%WARP_SZ];   }
    __device__ const scalar_T& operator[](size_t i) const
    {   return *(const scalar_T*)&(&ptr[i/WARP_SZ][0])[i%WARP_SZ];   }
};

constexpr static __device__ int dlog2(int n)
{   int ret=0; while (n>>=1) ret++; return ret;   }

static __device__ int is_unique(int wval, int dir=0)
{
    extern __shared__ int wvals[];
    const uint32_t tid = threadIdx.x;
    dir &= 1;   // force logical operations on predicates

    wvals[tid] = wval;
    NTHREADS > WARP_SZ ? __syncthreads() : __syncwarp();

    // Straightforward scan appears to be the fastest option for NTHREADS.
    // Bitonic sort complexity, a.k.a. amount of iterations, is [~3x] lower,
    // but each step is [~5x] slower...
    int negatives = 0;
    int uniq = 1;
    #pragma unroll 16
    for (uint32_t i=0; i<NTHREADS; i++) {
        int b = wvals[i];   // compiled as 128-bit [broadcast] loads:-)
        if (((i<tid)^dir) && i!=tid && wval==b)
            uniq = 0;
        negatives += (b < 0);
    }

    return uniq | (int)(NTHREADS-1-negatives)>>31;
    // return value is 1, 0 or -1.
}

#if WBITS==16
template<class scalar_t>
static __device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t ret = d[off/32];
    return (ret >> (off%32)) & ((1<<bits) - 1);
}
#else
template<class scalar_t>
static __device__ int get_wval(const scalar_t& d, uint32_t off, uint32_t bits)
{
    uint32_t top = off + bits - 1;
    uint64_t ret = ((uint64_t)d[top/32] << 32) | d[off/32];

    return (int)(ret >> (off%32)) & ((1<<bits) - 1);
}
#endif

__global__
void pippenger(const affine_t* points, size_t npoints,
               const scalar_t* scalars_, bool mont,
               bucket_t (*buckets)[NWINS][1<<WBITS],
               bucket_t (*ret)[NWINS][NTHREADS][2] /*= nullptr*/)
{
    assert(blockDim.x == NTHREADS);
    assert(gridDim.x == NWINS);
    assert(npoints == (uint32_t)npoints);

    if (gridDim.y > 1) {
        uint32_t delta = ((uint32_t)npoints + gridDim.y - 1) / gridDim.y;
        delta = (delta+WARP_SZ-1) & (0U-WARP_SZ);
        uint32_t off = delta * blockIdx.y;

        points   += off;
        scalars_ += off;
        if (blockIdx.y == gridDim.y-1)
            npoints -= off;
        else
            npoints = delta;
    }

    scalars_T scalars = const_cast<scalar_t*>(scalars_);

    const int NTHRBITS = dlog2(NTHREADS);
    const uint32_t tid = threadIdx.x;
    const uint32_t bid = blockIdx.x;
    const uint32_t bit0 = bid * WBITS;
    bucket_t* row = buckets[blockIdx.y][bid];

    if (mont) {
        #pragma unroll 1
        for (uint32_t i = NTHREADS*bid + tid; i < npoints; i += NTHREADS*NWINS) {
            scalar_t s = scalars_[i];
            s.from();
            scalars[i] = s;
        }
        cooperative_groups::this_grid().sync();
    } else { // if (typeid(scalars) != typeid(scalars_)) {
        #pragma unroll 1
        for (uint32_t i = NTHREADS*bid + tid; i < npoints; i += NTHREADS*NWINS) {
            scalar_t s = scalars_[i];
            __syncwarp();
            scalars[i] = s;
        }
        cooperative_groups::this_grid().sync();
    }

#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(tid<NTHREADS);
#endif
    #pragma unroll 4
    for (uint32_t i = tid; i < 1<<WBITS; i += NTHREADS)
        row[i].inf();

    int wbits = (bit0 > NBITS-WBITS) ? NBITS-bit0 : WBITS;
    int bias  = (tid >> max(wbits+NTHRBITS-WBITS, 0)) << max(wbits, WBITS-NTHRBITS);

    int dir = 1;
    for (uint32_t i = tid; true; ) {
        int wval = -1;
        affine_t point;

        if (i < npoints) {
            wval = get_wval(scalars[i], bit0, wbits);
            wval += wval ? bias : 0;
            point = points[i];
        }

        int uniq = is_unique(wval, dir^=1) | wval==0;
        if (uniq < 0)   // all |wval|-s are negative, all done
            break;

        if (i < npoints && uniq) {
            if (wval) {
                row[wval-1].add(point);
            }
            i += NTHREADS;
        }
    }
    if (NTHREADS > WARP_SZ && sizeof(bucket_t) > 128)
        __syncthreads();

    extern __shared__ bucket_t scratch[];
    uint32_t i = 1<<(WBITS-NTHRBITS);
    row += tid * i;
    bucket_t res, acc = row[--i];
    if (sizeof(res) <= 128)
        res = acc;
    else
        scratch[tid] = acc;
    #pragma unroll 1
    while (i--) {
        bucket_t p = row[i];
        #pragma unroll 1
        for (int pc = 0; pc < 2; pc++) {
            if (sizeof(res) <= 128) {
                acc.add(p);
                p = res;
                res = acc;
            } else {
                acc.add(p);
                p = scratch[tid];
                scratch[tid] = acc;
            }
        }
        acc = p;
    }

    if (ret == nullptr) {
        cooperative_groups::this_grid().sync();
        ret = reinterpret_cast<decltype(ret)>(buckets);
    }

    if (sizeof(res) <= 128)
        ret[blockIdx.y][bid][tid][0] = res;
    else
        ret[blockIdx.y][bid][tid][0] = scratch[tid];
    ret[blockIdx.y][bid][tid][1] = acc;
}

#else

#include <cassert>
#include <vector>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/thread_pool_t.hpp>

static thread_pool_t da_pool;

template<class point_t, class bucket_t>
static point_t integrate_row(const bucket_t row[NTHREADS][2], int wbits = WBITS)
{
    size_t i = NTHREADS-1;
    size_t mask = (1U << max(wbits+NTHRBITS-WBITS, 0)) - 1;

    if (mask == 0) {
        bucket_t res = row[i][0];
        while (i--)
            res.add(row[i][0]);
        return res;
    }

    point_t ret, res = row[i][0];
    bucket_t acc = row[i][1];
    ret.inf();
    while (i--) {
        point_t raise = acc;
        for (size_t j = 0; j < WBITS-NTHRBITS; j++)
            raise.dbl();
        res.add(raise);
        res.add(row[i][0]);
        if (i & mask) {
            acc.add(row[i][1]);
        } else {
            ret.add(res);
            if (i-- == 0)
                break;
            res = row[i][0];
            acc = row[i][1];
        }
    }

    return ret;
}

#if 0
static point_t pippenger_final(const bucket_t ret[NWINS][NTHREADS][2])
{
    size_t i = NWINS-1;
    point_t res = integrate_row(ret[i], NBITS%WBITS ? NBITS%WBITS : WBITS);

    while (i--) {
        for (size_t j = 0; j < WBITS; j++)
            res.dbl();
        res.add(integrate_row(ret[i]));
    }

    return res;
}
#endif

template<typename... Types>
inline void launch_coop(void(*f)(Types...),
                        dim3 gridDim, dim3 blockDim,
                        size_t shared_sz, cudaStream_t stream,
                        Types... args)
{
    void* va_args[sizeof...(args)] = { &args... };
    CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim,
                                        va_args, shared_sz, stream));
}

class stream_t {
    cudaStream_t stream;
public:
    stream_t()  { cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); }
    ~stream_t() { cudaStreamDestroy(stream); }
    inline operator decltype(stream)() { return stream; }
};

template<class bucket_t> class result_t {
    bucket_t ret[NWINS][NTHREADS][2];
public:
    result_t() {}
    inline operator decltype(ret)&() { return ret; }
};

//
// This could be called from distinct threads. For example as
// std::thread([=]() { mult_pippenger(out, points, npoints, scalars); })
// Since we compile with --default-stream=per-thread, the otherwise
// synchronous operations should overlap with other threads [according to
// https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/].
//
template<class bucket_t, class point_t, class affine_t, class scalar_t> static
RustError mult_pippenger(point_t *out, const affine_t points[], size_t npoints,
                                       const scalar_t scalars[], bool mont = true,
                         size_t ffi_affine_sz = sizeof(affine_t))
{
    assert(WBITS > NTHRBITS);

    bucket_t (*d_buckets)[NWINS][1<<WBITS] = nullptr;
    bucket_t (*d_none)[NWINS][NTHREADS][2] = nullptr;
    affine_t *d_points = nullptr;
    scalar_t *d_scalars = nullptr;
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
    stream_t stream;
#else
    cudaStream_t stream = nullptr;
#endif

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess || prop.major < 7)
        return RustError{cudaErrorInvalidDevice};

    size_t N = (prop.multiProcessorCount*256) / (NTHREADS*NWINS);
    size_t delta = ((npoints+N-1)/N+WARP_SZ-1) & (0U-WARP_SZ);
    N = (npoints+delta-1) / delta;
    std::vector<result_t<bucket_t>> res(N);

    try {
        size_t n = (npoints+WARP_SZ-1) & ((size_t)0-WARP_SZ);
        size_t blob_sz = n * sizeof(*d_points);
        blob_sz += n * sizeof(*d_scalars);
        blob_sz += N * sizeof(d_buckets[0]);

        CUDA_OK(cudaMalloc(&d_points, blob_sz));

        d_scalars = reinterpret_cast<decltype(d_scalars)>(d_points + n);
        d_buckets = reinterpret_cast<decltype(d_buckets)>(d_scalars + n);

        if (ffi_affine_sz != sizeof(*d_points))
            CUDA_OK(cudaMemcpy2DAsync(d_points, sizeof(*d_points),
                                      points, ffi_affine_sz,
                                      ffi_affine_sz, npoints,
                                      cudaMemcpyHostToDevice, stream));
        else
            CUDA_OK(cudaMemcpyAsync(d_points, points, npoints*sizeof(*d_points),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaMemcpyAsync(d_scalars, scalars, npoints*sizeof(*d_scalars),
                                cudaMemcpyHostToDevice, stream));

        launch_coop(pippenger, dim3(NWINS, N), NTHREADS,
                               sizeof(bucket_t)*NTHREADS, stream,
                    (const affine_t*)d_points, npoints,
                    (const scalar_t*)d_scalars, mont,
                    d_buckets, d_none);

        CUDA_OK(cudaMemcpyAsync(res[0], d_buckets, N*sizeof(res[0]),
                                cudaMemcpyDeviceToHost, stream));

        void *p = d_points;
        d_points = nullptr;
        CUDA_OK(cudaFree(p));
        CUDA_OK(cudaStreamSynchronize(stream));
    } catch (const cuda_error& e) {
        if (d_points != nullptr)
            cudaFree(d_points);
        cudaStreamSynchronize(stream);
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()}
#endif
    }

    struct tile_t {
        size_t x, y, dy;
        point_t p;
        tile_t() {}
    };
    std::vector<tile_t> grid(NWINS*N);

    size_t y = NWINS-1, total = 0;

    while (total < N) {
        grid[total].x  = total;
        grid[total].y  = y;
        grid[total].dy = NBITS - y*WBITS;
        total++;
    }

    while (y--) {
        for (size_t i = 0; i < N; i++, total++) {
            grid[total].x  = grid[i].x;
            grid[total].y  = y;
            grid[total].dy = WBITS;
        }
    }

    std::vector<std::atomic<size_t>> row_sync(NWINS); /* zeroed */
    counter_t<size_t> counter(0);
    channel_t<size_t> ch;

    auto n_workers = min(da_pool.size(), total);
    while (n_workers--) {
        da_pool.spawn([&, total, N, counter]() {
            for (size_t work; (work = counter++) < total;) {
                auto item = &grid[work];
                auto y = item->y;
                item->p = integrate_row<point_t>((res[item->x])[y], item->dy);
                if (++row_sync[y] == N)
                    ch.send(y);
            }
        });
    }

    out->inf();
    size_t row = 0, ny = NWINS;
    while (ny--) {
        auto y = ch.recv();
        row_sync[y] = -1U;
        while (grid[row].y == y) {
            while (row < total && grid[row].y == y)
                out->add(grid[row++].p);
            if (y == 0)
                break;
            for (size_t i = 0; i < WBITS; i++)
                out->dbl();
            if (row_sync[--y] != -1U)
                break;
        }
    }

    return RustError{cudaSuccess};
}

#endif
#endif
