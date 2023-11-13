// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

__global__
void generate_partial_twiddles(fr_t (*roots)[WINDOW_SIZE],
                               const fr_t root_of_unity)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    assert(tid < WINDOW_SIZE);
    fr_t root;

    root = root_of_unity^tid;

    roots[0][tid] = root;

    for (int off = 1; off < WINDOW_NUM; off++) {
        for (int i = 0; i < LG_WINDOW_SIZE; i++)
#if defined(__CUDA_ARCH__)
            root.sqr();
#else
            root *= root;
#endif
        roots[off][tid] = root;
    }
}

__global__
void generate_all_twiddles(fr_t* d_radixX_twiddles, const fr_t root6,
                                                    const fr_t root7,
                                                    const fr_t root8,
                                                    const fr_t root9,
                                                    const fr_t root10)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int pow;
    fr_t root_of_unity;

    if (tid < 64) {
        pow = tid;
        root_of_unity = root7;
    } else if (tid < 64 + 128) {
        pow = tid - 64;
        root_of_unity = root8;
    } else if (tid < 64 + 128 + 256) {
        pow = tid - 64 - 128;
        root_of_unity = root9;
    } else if (tid < 64 + 128 + 256 + 512) {
        pow = tid - 64 - 128 - 256;
        root_of_unity = root10;
    } else if (tid < 64 + 128 + 256 + 512 + 32) {
        pow = tid - 64 - 128 - 256 - 512;
        root_of_unity = root6;
    } else {
        assert(false);
    }

    d_radixX_twiddles[tid] = root_of_unity^pow;
}

#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
__launch_bounds__(512) __global__
void generate_radixX_twiddles_X(fr_t* d_radixX_twiddles_X, int n,
                                const fr_t root_of_unity)
{
    if (gridDim.x == 1) {
        d_radixX_twiddles_X[threadIdx.x] = fr_t::one();
        d_radixX_twiddles_X += blockDim.x;

        fr_t root0 = root_of_unity^threadIdx.x;

        d_radixX_twiddles_X[threadIdx.x] = root0;
        d_radixX_twiddles_X += blockDim.x;

        fr_t root1 = root0;

        for (int i = 2; i < n; i++) {
            root1 *= root0;
            d_radixX_twiddles_X[threadIdx.x] = root1;
            d_radixX_twiddles_X += blockDim.x;
        }
    } else {
        fr_t root0 = root_of_unity^(threadIdx.x * gridDim.x);

        unsigned int pow = blockIdx.x * threadIdx.x;
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

        fr_t root1 = root_of_unity^pow;

        d_radixX_twiddles_X[tid] = root1;
        d_radixX_twiddles_X += gridDim.x * blockDim.x;

        for (int i = gridDim.x; i < n; i += gridDim.x) {
            root1 *= root0;
            d_radixX_twiddles_X[tid] = root1;
            d_radixX_twiddles_X += gridDim.x * blockDim.x;
        }
    }
}
#endif
