// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

template <int intermediate_mul>
__launch_bounds__(768, 1) __global__
void _GS_NTT(const unsigned int radix, const unsigned int lg_domain_size,
             const unsigned int stage, const unsigned int iterations,
             fr_t* d_inout, const fr_t (*d_partial_twiddles)[WINDOW_SIZE],
             const fr_t* d_radix6_twiddles, const fr_t* d_radixX_twiddles,
             const fr_t* d_intermediate_twiddles,
             const unsigned int intermediate_twiddle_shift,
             const bool is_intt, const fr_t d_domain_size_inverse)
{
#if (__CUDACC_VER_MAJOR__-0) >= 11
    __builtin_assume(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
    __builtin_assume(radix <= lg_domain_size);
    __builtin_assume(stage <= lg_domain_size);
#endif
    const index_t tid = threadIdx.x + blockDim.x * (index_t)blockIdx.x;

    const index_t inp_mask = ((index_t)1 << (stage - 1)) - 1;
    const index_t out_mask = ((index_t)1 << (stage - iterations)) - 1;

    // rearrange |tid|'s bits
    index_t idx0 = (tid & ~inp_mask) * 2;
    idx0 += (tid << (stage - iterations)) & inp_mask;
    idx0 += (tid >> (iterations - 1)) & out_mask;
    index_t idx1 = idx0 + ((index_t)1 << (stage - 1));

    fr_t r0 = d_inout[idx0];
    fr_t r1 = d_inout[idx1];

    for (int s = iterations; --s >= 6;) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;

        fr_t t = d_radixX_twiddles[rank << (radix - (s + 1))];

        t *= (r0 - r1);
        r0 = r0 + r1;
        r1 = t;

        extern __shared__ fr_t shared_exchange[];

        bool pos = rank < laneMask;
#ifdef __CUDA_ARCH__
        t = fr_t::csel(r1, r0, pos);
        __syncthreads();
        shared_exchange[threadIdx.x] = t;
        __syncthreads();
        t = shared_exchange[threadIdx.x ^ laneMask];
        r0 = fr_t::csel(t, r0, !pos);
        r1 = fr_t::csel(t, r1, pos);
#endif
    }

    for (int s = min(iterations, 6); --s >= 1;) {
        unsigned int laneMask = 1 << (s - 1);
        unsigned int thrdMask = (1 << s) - 1;
        unsigned int rank = threadIdx.x & thrdMask;

        fr_t t = d_radix6_twiddles[rank << (6 - (s + 1))];

        t *= (r0 - r1);
        r0 = r0 + r1;
        r1 = t;

        bool pos = rank < laneMask;
#ifdef __CUDA_ARCH__
        t = fr_t::csel(r1, r0, pos);
        t.shfl_bfly(laneMask);
        r0 = fr_t::csel(t, r0, !pos);
        r1 = fr_t::csel(t, r1, pos);
#endif
    }

    {
        fr_t t = r0 - r1;
        r0 = r0 + r1;
        r1 = t;
    }

    if (intermediate_mul == 1) {
        index_t thread_ntt_pos = (tid & inp_mask) >> (iterations - 1);
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
        unsigned int nbits = MAX_LG_DOMAIN_SIZE - (stage - iterations);

        index_t root_idx0 = bit_rev(thread_ntt_idx, nbits) * thread_ntt_pos;
        index_t root_idx1 = root_idx0 + (thread_ntt_pos << (nbits - 1));

        fr_t first_root, second_root;
        get_intermediate_roots(first_root, second_root,
                               root_idx0, root_idx1, d_partial_twiddles);

        r0 *= first_root;
        r1 *= second_root;
    } else if (intermediate_mul == 2) {
        index_t thread_ntt_pos = (tid & inp_mask) >> (iterations - 1);
        unsigned int diff_mask = (1 << (iterations - 1)) - 1;
        unsigned int thread_ntt_idx = (tid & diff_mask) * 2;
        unsigned int nbits = intermediate_twiddle_shift + iterations;

        index_t root_idx0 = bit_rev(thread_ntt_idx, nbits);
        index_t root_idx1 = bit_rev(thread_ntt_idx + 1, nbits);

        fr_t t0 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx0];
        fr_t t1 = d_intermediate_twiddles[(thread_ntt_pos << radix) + root_idx1];

        r0 *= t0;
        r1 *= t1;
    }

    if (is_intt && stage == iterations) {
        r0 *= d_domain_size_inverse;
        r1 *= d_domain_size_inverse;
    }

    // rotate "iterations" bits in indices
    index_t mask = (index_t)((1 << iterations) - 1) << (stage - iterations);
    index_t rotw = idx0 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx0 = (idx0 & ~mask) | (rotw & mask);
    rotw = idx1 & mask;
    rotw = (rotw << 1) | (rotw >> (iterations - 1));
    idx1 = (idx1 & ~mask) | (rotw & mask);

    d_inout[idx0] = r0;
    d_inout[idx1] = r1;
}

#define NTT_ARGUMENTS \
        unsigned int, unsigned int, unsigned int, unsigned int, fr_t*, \
        const fr_t (*)[WINDOW_SIZE], const fr_t*, const fr_t*, const fr_t*, \
        unsigned int, bool, fr_t

template __global__ void _GS_NTT<0>(NTT_ARGUMENTS);
template __global__ void _GS_NTT<1>(NTT_ARGUMENTS);
template __global__ void _GS_NTT<2>(NTT_ARGUMENTS);

#undef NTT_ARGUMENTS

#ifndef __CUDA_ARCH__

class GS_launcher {
    fr_t* d_inout;
    const int lg_domain_size;
    bool is_intt;
    int stage;
    const NTTParameters& ntt_parameters;
    const cudaStream_t& stream;

public:
    GS_launcher(fr_t* d_ptr, int lg_dsz, bool innt,
                const NTTParameters& params, const cudaStream_t& s)
      : d_inout(d_ptr), lg_domain_size(lg_dsz), is_intt(innt), stage(lg_dsz),
        ntt_parameters(params), stream(s)
    {}

    void step(int iterations)
    {
        assert(iterations <= 10);

        const int radix = iterations < 6 ? 6 : iterations;

        index_t num_threads = (index_t)1 << (lg_domain_size - 1);
        index_t block_size = 1 << (radix - 1);
        index_t num_blocks;

        block_size = (num_threads <= block_size) ? num_threads : block_size;
        num_blocks = (num_threads + block_size - 1) / block_size;

        assert(num_blocks == (unsigned int)num_blocks);

        fr_t* d_radixX_twiddles = nullptr;
        fr_t* d_intermediate_twiddles = nullptr;
        int intermediate_twiddle_shift = 0;

        #define NTT_CONFIGURATION \
            num_blocks, block_size, sizeof(fr_t) * block_size, stream

        #define NTT_ARGUMENTS \
            radix, lg_domain_size, stage, iterations, d_inout, \
            ntt_parameters.partial_twiddles, ntt_parameters.radix6_twiddles, \
            d_radixX_twiddles, d_intermediate_twiddles, \
            intermediate_twiddle_shift, \
            is_intt, domain_size_inverse[lg_domain_size]

        switch (radix) {
        case 6:
            switch (stage) {
            case 6:
                _GS_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 12:
                intermediate_twiddle_shift = std::max(12 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix6_twiddles_6;
                _GS_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 18:
                intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix6_twiddles_12;
                _GS_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _GS_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 7:
            d_radixX_twiddles = ntt_parameters.radix7_twiddles;
            switch (stage) {
            case 7:
                _GS_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 14:
                intermediate_twiddle_shift = std::max(14 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix7_twiddles_7;
                _GS_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _GS_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 8:
            d_radixX_twiddles = ntt_parameters.radix8_twiddles;
            switch (stage) {
            case 8:
                _GS_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 16:
                intermediate_twiddle_shift = std::max(16 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix8_twiddles_8;
                _GS_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _GS_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 9:
            d_radixX_twiddles = ntt_parameters.radix9_twiddles;
            switch (stage) {
            case 9:
                _GS_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            case 18:
                intermediate_twiddle_shift = std::max(18 - lg_domain_size, 0);
                d_intermediate_twiddles = ntt_parameters.radix9_twiddles_9;
                _GS_NTT<2><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _GS_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        case 10:
            d_radixX_twiddles = ntt_parameters.radix10_twiddles;
            switch (stage) {
            case 10:
                _GS_NTT<0><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            default:
                _GS_NTT<1><<<NTT_CONFIGURATION>>>(NTT_ARGUMENTS);
                break;
            }
            break;
        default:
            assert(false);
        }

        #undef NTT_CONFIGURATION
        #undef NTT_ARGUMENTS

        CUDA_OK(cudaGetLastError());

        stage -= iterations;
    }
};

void GS_NTT(fr_t* d_inout, const int lg_domain_size, const bool is_intt,
    const NTTParameters& ntt_parameters, const cudaStream_t& stream)
{
    GS_launcher params{d_inout, lg_domain_size, is_intt, ntt_parameters, stream};

    if (lg_domain_size <= 10) {
        params.step(lg_domain_size);
    } else if (lg_domain_size <= 12) {
        params.step(lg_domain_size - 6);
        params.step(6);
    } else if (lg_domain_size <= 18) {
        params.step(lg_domain_size / 2 + lg_domain_size % 2);
        params.step(lg_domain_size / 2);
    } else if (lg_domain_size <= 30) {
        int step = lg_domain_size / 3;
        int rem = lg_domain_size % 3;
        params.step(step + (rem > 0));
        params.step(step + (rem > 1));
        params.step(step);
    } else if (lg_domain_size <= 40) {
        int step = lg_domain_size / 4;
        int rem = lg_domain_size % 4;
        params.step(step + (rem > 0));
        params.step(step + (rem > 1));
        params.step(step + (rem > 2));
        params.step(step);
    } else {
        assert(false);
    }
}

#endif
