// Copyright Supranational LLC
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_MSM_SORT_CUH__
#define __SPPARK_MSM_SORT_CUH__

/*
 * Custom sorting, we take in digits and return their indices.
 */

#define SORT_BLOCKDIM 1024
#ifndef DIGIT_BITS
# define DIGIT_BITS 13
#endif
#if DIGIT_BITS < 10 || DIGIT_BITS > 14
# error "impossible DIGIT_BITS"
#endif

__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1);

// Non-cooperative fallback kernels for GPUs/runtimes without cooperative launch
// (cooperativeLaunch == 0, e.g. RDNA4 has no Global Wave Sync and Windows ROCm
// under-reports it on RDNA3). The cooperative `sort` above runs the cross-block
// top-level radix pass as one persistent kernel with grid-wide barriers; these
// split that pass at its barriers into ordinary kernels whose boundaries provide
// the same grid-wide ordering. Launch sort_upper_count, then sort_upper_scatter,
// then sort_lower (same grid/block as the cooperative launch) for the two-level
// case; the single-block case reuses `sort` with gridDim.x == 1.
__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort_upper_count(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1);
__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort_upper_scatter(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1);
__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort_lower(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1);

#ifndef __MSM_SORT_DONT_IMPLEMENT__

#ifndef WARP_SZ
# define WARP_SZ 32
#endif
#ifdef __GNUC__
# define asm __asm__ __volatile__
#else
# define asm asm volatile
#endif

static const uint32_t N_COUNTERS = 1<<DIGIT_BITS;
static const uint32_t N_SUMS = N_COUNTERS / SORT_BLOCKDIM;
extern __shared__ uint32_t counters[/*N_COUNTERS*/];

__device__ __forceinline__
uint32_t pack(uint32_t a, uint32_t mask, uint32_t b)
{
    uint32_t ret;

#if defined(__HIPCC__)
    ret = (a & ~mask) | (mask & b);
#else
    asm("lop3.b32 %0, %1, %2, %3, 0xb8;" // a & ~mask | mask & b
        : "=r"(ret)
        : "r"(a), "r"(mask), "r"(b));
#endif

    return ret;
}

__device__ __forceinline__
uint32_t sum_up(uint32_t sum, const uint32_t limit = WARP_SZ)
{
#if defined(__HIPCC__)
    const uint32_t laneid = threadIdx.x % WARP_SZ;
    #pragma unroll
    for (uint32_t off = 1; off < limit; off <<= 1) {
        uint32_t v = __shfl_up_sync(0xffffffff, sum, off, WARP_SZ);
        if (laneid >= off)
            sum += v;
    }
#else
    #pragma unroll
    for (uint32_t off = 1; off < limit; off <<= 1)
        asm("{ .reg.b32 %v; .reg.pred %did;"
            "  shfl.sync.up.b32 %v|%did, %0, %1, 0, 0xffffffff;"
            "  @%did add.u32 %0, %0, %v;"
            "}" : "+r"(sum) : "r"(off));
#endif

    return sum;
}

__device__ __forceinline__
void zero_counters()
{
#if DIGIT_BITS >= 12
    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS/4; i++)
        ((uint4*)counters)[threadIdx.x + i*SORT_BLOCKDIM] = uint4{0, 0, 0, 0};
#else
    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS; i++)
        counters[threadIdx.x + i*SORT_BLOCKDIM] = 0;
#endif
    __syncthreads();
}

__device__ __forceinline__
void count_digits(const uint32_t src[], uint32_t base, uint32_t len,
                  uint32_t lshift, uint32_t rshift, uint32_t mask)
{
    zero_counters();

    const uint32_t pack_mask = 0xffffffffU << lshift;

    src += base;
    // count occurrences of each non-zero digit
    for (uint32_t i = threadIdx.x; i < len; i += SORT_BLOCKDIM) {
        auto val = src[(size_t)i];
        auto pck = pack(base+i, pack_mask, (val-1) << lshift);
        if (val)
            (void)atomicAdd(&counters[(pck >> rshift) & mask], 1);
    }

    __syncthreads();
}

__device__ __forceinline__
void scatter(uint2 dst[], const uint32_t src[], uint32_t base, uint32_t len,
             uint32_t lshift, uint32_t rshift, uint32_t mask,
             uint32_t pidx[] = nullptr)
{
    const uint32_t pack_mask = 0xffffffffU << lshift;

    src += base;
    #pragma unroll 1    // the subroutine is memory-io-bound, unrolling makes no difference
    for (uint32_t i = threadIdx.x; i < len; i += SORT_BLOCKDIM) {
        auto val = src[(size_t)i];
        auto pck = pack(base+i, pack_mask, (val-1) << lshift);
        if (val) {
            uint32_t idx = atomicSub(&counters[(pck >> rshift) & mask], 1) - 1;
            uint32_t pid = pidx ? pidx[base+i] : base+i;
            dst[idx] = uint2{pck, pack(pid, 0x80000000, val)};
        }
    }
}

// Partition |len| elements across the grid (one contiguous slice per block).
// Pure function of len, gridDim.x, blockIdx.x, so each phase recomputes it
// identically whether the phases run as one cooperative kernel or several.
__device__ __forceinline__
static void upper_partition(uint32_t len, uint32_t& base, uint32_t& slice)
{
    uint32_t grid_div = 31 - __clz(gridDim.x);
    uint32_t grid_rem = (1<<grid_div) - 1;

    slice = len >> grid_div;            // / gridDim.x;
    uint32_t rem = len & grid_rem;      // % gridDim.x;

    if (blockIdx.x < rem)
        base = ++slice * blockIdx.x;
    else
        base = slice * blockIdx.x + rem;
}

// Phase 1 of the cross-block top-level radix pass: count this block's digits and
// publish its per-digit histogram to global memory. All cross-block state lives in
// |histogram|, so a grid-wide barrier (cooperative path) or a kernel boundary
// (fallback path) after this is sufficient before phase 2.
__device__ __forceinline__
static void upper_count_emit(const uint32_t src[], uint32_t len,
                             uint32_t lsbits, uint32_t bits, uint32_t digit,
                             uint32_t histogram[])
{
    uint32_t base, slice;
    upper_partition(len, base, slice);

    const uint32_t mask = (1<<bits) - 1;
    const uint32_t lshift = digit + bits - lsbits;

    count_digits(src, base, slice, lshift, digit, mask);

    // collect counters from SMs in the histogram
    #pragma unroll 1
    for (uint32_t i = threadIdx.x; i < 1<<bits; i += SORT_BLOCKDIM)
        histogram[2 + (i<<digit) + blockIdx.x] = counters[i];
}

// Phase 2: cross-block exclusive prefix sum over the per-block histograms (warp
// shuffles, requires gridDim.x a power of two and <= WARP_SZ), scatter src->dst by
// the resulting per-block offsets, and (block 0) finalize the global digit bases.
// The shared counters[] are recomputed from |histogram| here, so this needs no
// shared state carried over from phase 1.
__device__ __forceinline__
static void upper_scan_scatter(uint2 dst[], const uint32_t src[], uint32_t len,
                               uint32_t lsbits, uint32_t bits, uint32_t digit,
                               uint32_t histogram[])
{
    uint32_t base, slice;
    upper_partition(len, base, slice);

    const uint32_t mask = (1<<bits) - 1;
    const uint32_t lshift = digit + bits - lsbits;

    uint32_t grid_div = 31 - __clz(gridDim.x);
    uint32_t grid_rem = (1<<grid_div) - 1;

    __syncthreads();    // eliminate BRA.DIV?

    const uint32_t warpid = threadIdx.x / WARP_SZ;
    const uint32_t laneid = threadIdx.x % WARP_SZ;
    const uint32_t sub_warpid = laneid >> grid_div; // / gridDim.x;
    const uint32_t sub_laneid = laneid & grid_rem;  // % gridDim.x;
    const uint32_t stride = WARP_SZ >> grid_div;    // / gridDim.x;

    uint2 h = uint2{0, 0};
    uint32_t sum, warp_off = warpid*WARP_SZ*N_SUMS + sub_warpid;

    #pragma unroll 1
    for (uint32_t i = 0; i < WARP_SZ*N_SUMS; i += stride, warp_off += stride) {
        auto* hptr = &histogram[warp_off << digit];

        sum = (warp_off < 1<<bits) ? hptr[2 + sub_laneid] : 0;
        sum = sum_up(sum) + h.x;

        if (sub_laneid == blockIdx.x)
            counters[warp_off] = sum;

#if defined(__HIPCC__)
        {
            uint32_t v = __shfl_up_sync(0xffffffff, sum, 1, WARP_SZ);
            if ((threadIdx.x % WARP_SZ) >= 1)
                h.x = v;
        }
#else
        asm("{ .reg.b32 %v; .reg.pred %did;");
        asm("shfl.sync.up.b32 %v|%did, %0, 1, 0, 0xffffffff;" :: "r"(sum));
        asm("@%did mov.b32 %0, %v;" : "+r"(h.x));
        asm("}");
#endif
        h.y = __shfl_down_sync(0xffffffff, sum, gridDim.x-1) - h.x;

        if (blockIdx.x == 0 && sub_laneid == 0 && warp_off < 1<<bits)
            *(uint2*)hptr = h;

        h.x = __shfl_sync(0xffffffff, sum, WARP_SZ-1, WARP_SZ);
    }

    if (warpid == 0)    // offload some counters to registers
        sum = counters[laneid];

    __syncthreads();

    // carry over most significant prefix sums from each warp
    if (laneid == WARP_SZ-1)
        counters[warpid] = h.x;

    __syncthreads();

    uint32_t carry_sum = laneid ? counters[laneid-1] : 0;

    __syncthreads();

    if (warpid == 0)    // restore offloaded counters
        counters[laneid] = sum;

    __syncthreads();

    carry_sum = sum_up(carry_sum, SORT_BLOCKDIM/WARP_SZ);
    carry_sum = __shfl_sync(0xffffffff, carry_sum, warpid);

    uint32_t lane_off = warpid*WARP_SZ*N_SUMS + laneid;

    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS; i++)
        (void)atomicAdd(&counters[lane_off + i*WARP_SZ], carry_sum);

    __syncthreads();

    scatter(dst, src, base, slice, lshift, digit, mask);

    if (blockIdx.x == 0) {
        #pragma unroll 1
        for (uint32_t i = 0; i < N_SUMS; i++, lane_off += WARP_SZ)
            if (lane_off < 1<<bits)
                (void)atomicAdd(&histogram[lane_off << digit], carry_sum);
    }
}

// Cooperative top-level pass: the two phases with a grid-wide barrier between them,
// and a trailing barrier before the per-block lower passes consume the finalized
// bases. Behavior-identical to the original single-function implementation.
__device__
static void upper_sort(uint2 dst[], const uint32_t src[], uint32_t len,
                       uint32_t lsbits, uint32_t bits, uint32_t digit,
                       uint32_t histogram[])
{
    upper_count_emit(src, len, lsbits, bits, digit, histogram);

#if defined(__HIP_DEVICE_COMPILE__)
    SPPARK_GRID_SYNC();
#elif defined(__CUDA_ARCH__)
    cooperative_groups::this_grid().sync();
#endif

    upper_scan_scatter(dst, src, len, lsbits, bits, digit, histogram);

#if defined(__HIP_DEVICE_COMPILE__)
    SPPARK_GRID_SYNC();
#elif defined(__CUDA_ARCH__)
    cooperative_groups::this_grid().sync();
#endif
    __syncthreads();
}

__device__ __forceinline__
void count_digits(const uint2 src[], uint32_t len, uint32_t mask)
{
    zero_counters();

    // count occurrences of each digit
    for (size_t i = threadIdx.x; i < len; i += SORT_BLOCKDIM)
        (void)atomicAdd(&counters[src[i].x & mask], 1);

    __syncthreads();
}

__device__ __forceinline__
void scatter(uint32_t dst[], const uint2 src[], uint32_t len, uint32_t mask)
{
    #pragma unroll 1    // the subroutine is memory-io-bound, unrolling makes no difference
    for (uint32_t i = threadIdx.x; i < len; i += SORT_BLOCKDIM) {
        auto val = src[(size_t)i];
        uint32_t idx = atomicSub(&counters[val.x & mask], 1) - 1;
        dst[idx] = val.y;
    }
}

__device__ __noinline__
static void lower_sort(uint32_t dst[], const uint2 src[],
                       uint32_t base, uint32_t len, uint32_t histogram[],
                       uint32_t bits = DIGIT_BITS)
{
    const uint32_t mask = (1<<bits) - 1;

    count_digits(src += base, len, mask);

    const uint32_t warpid = threadIdx.x / WARP_SZ;
    const uint32_t laneid = threadIdx.x % WARP_SZ;

    uint32_t prefix_sums[N_SUMS];
    uint32_t lane_off = warpid*WARP_SZ*N_SUMS + laneid;

    // calculate per-warp prefix sums
    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS; i++) {
        uint32_t off = lane_off + i*WARP_SZ;
        uint32_t sum = counters[off];

        sum = sum_up(sum);
        if (i > 0)
            sum += __shfl_sync(0xffffffff, prefix_sums[i-1], WARP_SZ-1);

        prefix_sums[i] = sum;
    }

    // carry over most significant prefix sums from each warp
    if (laneid == WARP_SZ-1)
        counters[warpid*(WARP_SZ*N_SUMS+1)] = prefix_sums[N_SUMS-1];

    __syncthreads();

    uint32_t carry_sum = laneid ? counters[(laneid-1)*(WARP_SZ*N_SUMS+1)] : 0;

    __syncthreads();

    carry_sum = sum_up(carry_sum, SORT_BLOCKDIM/WARP_SZ);
    carry_sum = __shfl_sync(0xffffffff, carry_sum, warpid);
    carry_sum += base;

    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS; i++)
        counters[lane_off + i*WARP_SZ] = prefix_sums[i] += carry_sum;

    // store the prefix sums to histogram[]
    #pragma unroll
    for (uint32_t i = 0; i < N_SUMS; i++, lane_off += WARP_SZ) {
        if (lane_off < 1<<bits)
            histogram[lane_off] = prefix_sums[i];
    }

    __syncthreads();

    scatter(dst, src, len, mask);

    __syncthreads();
}

// Split |wbits| into a top (cross-block) and low (per-block) radix, the same way
// the two-level sort_row path does. Pure function of wbits and gridDim.x, so the
// cooperative path and the fallback kernels derive identical splits.
__device__ __forceinline__
static void upper_bits(uint32_t wbits, uint32_t& top_bits, uint32_t& low_bits)
{
    uint32_t lg_gridDim = 31 - __clz(gridDim.x);

    top_bits = wbits / 2;
    low_bits = wbits - top_bits;

    if (low_bits < lg_gridDim+1) {
        low_bits = lg_gridDim+1;
        top_bits = wbits - low_bits;
    }
}

// Per-block lower passes: each block sorts the |low_bits| sub-buckets the top-level
// pass assigned to it, reading the finalized {base,len} slices from |histogram|.
// Uses only __syncthreads, so it never needs cooperative launch.
__device__ __forceinline__
static void lower_sort_loop(uint32_t inout[], uint2 temp[], uint32_t histogram[],
                            uint32_t top_bits, uint32_t low_bits)
{
    histogram += blockIdx.x<<low_bits;

    #pragma unroll 1
    for (uint32_t i = blockIdx.x; i < 1<<top_bits; i += gridDim.x) {
        uint2 slice = *(uint2*)histogram;
        lower_sort(inout, temp, slice.x, slice.y, histogram, low_bits);
        histogram += gridDim.x<<low_bits;
    }
}

__device__ __forceinline__
void sort_row(uint32_t inout[], size_t len, uint2 temp[],
              uint32_t histogram[], uint32_t wbits, uint32_t lsbits)
{
    assert(len <= (1U<<31) && wbits <= 2*DIGIT_BITS && gridDim.x <= WARP_SZ);

    uint32_t lg_gridDim = 31 - __clz(gridDim.x);

    if (wbits > DIGIT_BITS || (lg_gridDim && wbits > lg_gridDim+1)) {
        uint32_t top_bits, low_bits;
        upper_bits(wbits, top_bits, low_bits);

        upper_sort(temp, inout, len, lsbits, top_bits, low_bits, histogram);

        lower_sort_loop(inout, temp, histogram, top_bits, low_bits);
    } else if (blockIdx.x == 0) {
        counters[0] = 0;
        __syncthreads();

        uint32_t lshift = wbits - lsbits;
        uint32_t pack_mask = 0xffffffffU << lshift;

        #pragma unroll 1
        for (uint32_t i = threadIdx.x; i < (uint32_t)len; i += SORT_BLOCKDIM) {
            auto val = inout[(size_t)i];
            auto pck = pack(i, pack_mask, (val-1) << lshift);
            if (val) {
                auto idx = atomicAdd(&counters[0], 1);
                //uint32_t pid = pidx ? pidx[i] : i;
                temp[idx] = uint2{pck, pack(i, 0x80000000, val)};
            }
        }

        __syncthreads();

        lower_sort(inout, temp, 0, counters[0], histogram, wbits);
    }
}

#if 0
__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort(uint32_t inout[], size_t len, uint2 temp[],
                     uint32_t histogram[], uint32_t wbits, uint32_t lsbits)
{   sort_row(inout, len, temp, histogram, wbits, lsbits);   }
#endif

__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1)
{
    win += blockIdx.y;
    sort_row(inouts[win], len, temps[blockIdx.y], histograms[win],
             wbits, blockIdx.y==0 ? lsbits0 : lsbits1);
}

// Non-cooperative fallback for the two-level (cross-block) sort_row path. The host
// launches these three kernels in order, with the same grid/block as the
// cooperative `sort`, when the device lacks cooperative launch. Each kernel
// boundary supplies the grid-wide ordering the cooperative path got from
// grid.sync. The single-block sort_row path needs no barrier and reuses `sort`
// with gridDim.x == 1, so it has no fallback kernel here.
__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort_upper_count(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1)
{
    win += blockIdx.y;
    uint32_t lsbits = blockIdx.y==0 ? lsbits0 : lsbits1;
    uint32_t top_bits, low_bits;
    upper_bits(wbits, top_bits, low_bits);
    upper_count_emit(inouts[win], len, lsbits, top_bits, low_bits, histograms[win]);
}

__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort_upper_scatter(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1)
{
    win += blockIdx.y;
    uint32_t lsbits = blockIdx.y==0 ? lsbits0 : lsbits1;
    uint32_t top_bits, low_bits;
    upper_bits(wbits, top_bits, low_bits);
    upper_scan_scatter(temps[blockIdx.y], inouts[win], len, lsbits,
                       top_bits, low_bits, histograms[win]);
}

__launch_bounds__(SORT_BLOCKDIM)
__global__ void sort_lower(vec2d_t<uint32_t> inouts, size_t len, uint32_t win,
                     vec2d_t<uint2> temps, vec2d_t<uint32_t> histograms,
                     uint32_t wbits, uint32_t lsbits0, uint32_t lsbits1)
{
    win += blockIdx.y;
    uint32_t top_bits, low_bits;
    upper_bits(wbits, top_bits, low_bits);
    lower_sort_loop(inouts[win], temps[blockIdx.y], histograms[win],
                    top_bits, low_bits);
}

# undef asm
#endif
#endif
