// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_NTT_PARAMETERS_CUH__
#define __SPPARK_NTT_PARAMETERS_CUH__

// Maximum domain size supported. Can be adjusted at will, but with the
// target field in mind. Most fields handle up to 2^32 elements, BLS12-377
// can handle up to 2^47, alt_bn128 - 2^28...
#ifndef MAX_LG_DOMAIN_SIZE
# if defined(FEATURE_BN254)
#  define MAX_LG_DOMAIN_SIZE 28
# elif defined(FEATURE_BABY_BEAR)
#  define MAX_LG_DOMAIN_SIZE 27
# else
#  define MAX_LG_DOMAIN_SIZE 28 // tested only up to 2^32 for now
# endif
#endif

#if MAX_LG_DOMAIN_SIZE <= 32
typedef unsigned int index_t;
#else
typedef size_t index_t;
#endif

#if defined(FEATURE_BABY_BEAR)
# define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 4) / 5)
#elif defined(FEATURE_GOLDILOCKS)
# if MAX_LG_DOMAIN_SIZE <= 28
#  define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 3) / 4)
# else
#  define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 4) / 5)
# endif
#else // 256-bit fields
# if MAX_LG_DOMAIN_SIZE <= 28
#  define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 1) / 2)
# else
#  define LG_WINDOW_SIZE ((MAX_LG_DOMAIN_SIZE + 2) / 3)
# endif
#endif
#if LG_WINDOW_SIZE < 6
# define LG_WINDOW_SIZE 6
#endif

#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

#if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
# if defined(FEATURE_BLS12_377)
#  include "parameters/bls12_377.h"
# elif defined(FEATURE_BLS12_381)
#  include "parameters/bls12_381.h"
# elif defined(FEATURE_PALLAS)
#  include "parameters/vesta.h"     // Fr for Pallas curve is Vesta
# elif defined(FEATURE_VESTA)
#  include "parameters/pallas.h"    // Fr for Vesta curve is Pallas
# elif defined(FEATURE_BN254)
#  include "parameters/alt_bn128.h"
# elif defined(FEATURE_BABY_BEAR)
#  include "parameters/baby_bear.h"
# elif defined(FEATURE_GOLDILOCKS)
#  include "parameters/goldilocks.h"
# endif
#else
extern const fr_t group_gen, group_gen_inverse;
extern const fr_t forward_roots_of_unity[];
extern const fr_t inverse_roots_of_unity[];
extern const fr_t domain_size_inverse[];
#endif

template<typename T>
__device__ __forceinline__
T bit_rev(T i, unsigned int nbits)
{
    if (sizeof(i) == 4 || nbits <= 32)
        return __brev(i) >> (8*sizeof(unsigned int) - nbits);
    else
        return __brevll(i) >> (8*sizeof(unsigned long long) - nbits);
}

template<typename T>
static __device__ __host__ constexpr uint32_t lg2(T n)
{   uint32_t ret=0; while (n>>=1) ret++; return ret;   }

template<class fr_t>
__device__ __forceinline__
fr_t get_intermediate_root(index_t pow, const fr_t (*roots)[WINDOW_SIZE])
{
    unsigned int off = 0;

    fr_t t, root;

    if (sizeof(fr_t) <= 8) {
        root = fr_t::one();
        bool root_set = false;

        #pragma unroll
        for (unsigned int pow_win, i = 0; i < WINDOW_NUM; i++) {
            if (!root_set && (pow_win = pow % WINDOW_SIZE)) {
                root = roots[i][pow_win];
                root_set = true;
            }
            if (!root_set) {
                pow >>= LG_WINDOW_SIZE;
                off++;
            }
        }
    } else {
        if ((pow % WINDOW_SIZE) == 0) {
            pow >>= LG_WINDOW_SIZE;
            off++;
        }
        root = roots[off][pow % WINDOW_SIZE];
    }

    #pragma unroll 1
    while (pow >>= LG_WINDOW_SIZE)
        root *= (t = roots[++off][pow % WINDOW_SIZE]);

    return root;
}

template<class fr_t>
__device__ __forceinline__
void get_intermediate_roots(fr_t& root0, fr_t& root1,
                            index_t idx0, index_t idx1,
                            const fr_t (*roots)[WINDOW_SIZE])
{
    int win = (WINDOW_NUM - 1) * LG_WINDOW_SIZE;
    int off = (WINDOW_NUM - 1);
    index_t idxo = idx0 | idx1;
    index_t mask = ((index_t)1 << win) - 1;

    root0 = roots[off][idx0 >> win];
    root1 = roots[off][idx1 >> win];
    #pragma unroll 1
    while (off-- && (idxo & mask)) {
        fr_t t;
        win -= LG_WINDOW_SIZE;
        mask >>= LG_WINDOW_SIZE;
        root0 *= (t = roots[off][(idx0 >> win) % WINDOW_SIZE]);
        root1 *= (t = roots[off][(idx1 >> win) % WINDOW_SIZE]);
    }
}

template<class fr_t> __global__
void generate_partial_twiddles(fr_t (*roots)[WINDOW_SIZE],
                               const fr_t root_of_unity)
{
    const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    assert(tid < WINDOW_SIZE);

    fr_t root = root_of_unity^tid;

    roots[0][tid] = root;

    for (int off = 1; off < WINDOW_NUM; off++) {
        for (int i = 0; i < LG_WINDOW_SIZE; i++)
            root.sqr();
        roots[off][tid] = root;
    }
}

template<class fr_t> __launch_bounds__(512) __global__
void generate_inner_twiddles(fr_t* d_inner_twiddles, const fr_t root10)
{
    fr_t root = root10^bit_rev(threadIdx.x, 9);

    d_inner_twiddles[threadIdx.x] = root;
}

template<class fr_t> __launch_bounds__(512) __global__
void generate_stage_twiddles(fr_t* d_radixX_twiddles_X, int n,
                             const fr_t root_of_unity)
{
    unsigned int nbits = 31 - __clz(blockDim.x);
    unsigned int pow_rev = bit_rev(threadIdx.x, nbits);

    if (gridDim.x == 1) {
        d_radixX_twiddles_X[threadIdx.x] = fr_t::one();
        d_radixX_twiddles_X += blockDim.x;

        fr_t root0 = root_of_unity^pow_rev;

        d_radixX_twiddles_X[threadIdx.x] = root0;
        d_radixX_twiddles_X += blockDim.x;

        fr_t root1 = root0;

        for (int i = 2; i < n; i++) {
            root1 *= root0;
            d_radixX_twiddles_X[threadIdx.x] = root1;
            d_radixX_twiddles_X += blockDim.x;
        }
    } else {
        fr_t root0 = root_of_unity^(pow_rev * gridDim.x);
        fr_t root1 = root_of_unity^(pow_rev * blockIdx.x);
        unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

        d_radixX_twiddles_X[tid] = root1;
        d_radixX_twiddles_X += gridDim.x * blockDim.x;

        for (int i = gridDim.x; i < n; i += gridDim.x) {
            root1 *= root0;
            d_radixX_twiddles_X[tid] = root1;
            d_radixX_twiddles_X += gridDim.x * blockDim.x;
        }
    }
}

template<class fr_t> __launch_bounds__(1024) __global__
void generate_plus_one_twiddles(fr_t (*d_plus_one_twiddles)[1024],
                                const fr_t root_of_unity)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    fr_t root = root_of_unity^bit_rev(tid, 10);

    d_plus_one_twiddles[0][tid] = root;

    for (unsigned int off = 1; off < MAX_LG_DOMAIN_SIZE-15; off++) {
        root.sqr();
        d_plus_one_twiddles[off][tid] = root;
    }
}

class NTTParameters {
private:
    const gpu_t& gpu;
    bool inverse;

public:
    fr_t (*partial_twiddles)[WINDOW_SIZE];

    fr_t* inner_twiddles;

    fr_t (*partial_group_gen_powers)[WINDOW_SIZE]; // for LDE

#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
    fr_t* stage6_twiddles, * stage7_twiddles,
        * stage8_twiddles, * stage9_twiddles;

private:
    fr_t* stage_twiddles(int num_blocks, int block_size, const fr_t& root,
                         stream_t& s)
    {
        fr_t* ret = (fr_t*)s.Dmalloc(num_blocks * block_size * sizeof(fr_t));
        generate_stage_twiddles<<<16, block_size, 0, s>>>(ret, num_blocks, root);
        CUDA_OK(cudaGetLastError());
        return ret;
    }
#else
    fr_t (*plus_one_twiddles)[1024];
#endif

public:
    NTTParameters(const bool _inverse, int id)
        : gpu(select_gpu(id)), inverse(_inverse)
    {
        const fr_t* roots = inverse ? inverse_roots_of_unity
                                    : forward_roots_of_unity;

        inner_twiddles = reinterpret_cast<decltype(inner_twiddles)>
                         (gpu[0].Dmalloc(512 * sizeof(fr_t)));

        generate_inner_twiddles<<<1, 512, 0, gpu[0]>>>(inner_twiddles, roots[10]);
        CUDA_OK(cudaGetLastError());

#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
        stage6_twiddles = stage_twiddles(64, 64, roots[12], gpu[1]);
        stage7_twiddles = stage_twiddles(128, 128, roots[14], gpu[2]);
        stage8_twiddles = stage_twiddles(256, 256, roots[16], gpu[0]);
        stage9_twiddles = stage_twiddles(512, 512, roots[18], gpu[1]);
#else
        plus_one_twiddles = reinterpret_cast<decltype(plus_one_twiddles)>
                            (gpu[1].Dmalloc((MAX_LG_DOMAIN_SIZE-15) * 1024 * sizeof(fr_t)));

        generate_plus_one_twiddles<<<1, 1024, 0, gpu[1]>>>(plus_one_twiddles,
                                                           roots[MAX_LG_DOMAIN_SIZE]);
        CUDA_OK(cudaGetLastError());
#endif

        const size_t partial_sz = WINDOW_NUM * WINDOW_SIZE;

        partial_twiddles = reinterpret_cast<decltype(partial_twiddles)>
                           (gpu[2].Dmalloc(2 * partial_sz * sizeof(fr_t)));
        partial_group_gen_powers = &partial_twiddles[WINDOW_NUM];

        generate_partial_twiddles<<<WINDOW_SIZE/64, 64, 0, gpu[2]>>>
            (partial_twiddles, roots[MAX_LG_DOMAIN_SIZE]);
        CUDA_OK(cudaGetLastError());

        generate_partial_twiddles<<<WINDOW_SIZE/64, 64, 0, gpu[2]>>>
            (partial_group_gen_powers, inverse ? group_gen_inverse
                                               : group_gen);
        CUDA_OK(cudaGetLastError());
    }
    NTTParameters(const NTTParameters&) = delete;
    NTTParameters(NTTParameters&&) = default;

    ~NTTParameters()
    {
        int current_id;
        if (cudaGetDevice(&current_id) != cudaSuccess) {
            gpu.select();

            (void)cudaFreeAsync(partial_twiddles, gpu[2]);
#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
            (void)cudaFreeAsync(stage9_twiddles, gpu[1]);
            (void)cudaFreeAsync(stage8_twiddles, gpu[0]);
            (void)cudaFreeAsync(stage7_twiddles, gpu[2]);
            (void)cudaFreeAsync(stage6_twiddles, gpu[1]);
#else
            (void)cudaFreeAsync(plus_one_twiddles, gpu[1]);
#endif
            (void)cudaFreeAsync(inner_twiddles, gpu[0]);

            (void)cudaSetDevice(current_id);
        }
    }

    inline void sync() const    { gpu.sync(); }

private:
    class all_params { friend class NTTParameters;
        std::vector<NTTParameters> forward;
        std::vector<NTTParameters> inverse;

        all_params()
        {
            int current_id;
            (void)cudaGetDevice(&current_id);

            size_t nids = ngpus();
            forward.reserve(nids);
            for (size_t id = 0; id < nids; id++)
                forward.emplace_back(false, id);
            inverse.reserve(nids);
            for (size_t id = 0; id < nids; id++)
                inverse.emplace_back(true, id);
            for (size_t id = 0; id < nids; id++)
                inverse[id].sync();

            (void)cudaSetDevice(current_id);
        }
    };

public:
    static const auto& all(bool inverse = false)
    {
        static all_params params;
        return inverse ? params.inverse : params.forward;
    }
};
#endif /* __SPPARK_NTT_PARAMETERS_CUH__ */
