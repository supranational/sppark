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
#  define MAX_LG_DOMAIN_SIZE 28 // tested only up to 2^31 for now
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

#define WINDOW_SIZE (1 << LG_WINDOW_SIZE)
#define WINDOW_NUM ((MAX_LG_DOMAIN_SIZE + LG_WINDOW_SIZE - 1) / LG_WINDOW_SIZE)

__device__ __constant__ fr_t forward_radix6_twiddles[32];
__device__ __constant__ fr_t inverse_radix6_twiddles[32];

#include "gen_twiddles.cu"

#ifndef __CUDA_ARCH__

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

class NTTParameters {
private:
    stream_t& gpu;
    bool inverse;

public:
    fr_t (*partial_twiddles)[WINDOW_SIZE];

    fr_t* radix6_twiddles, * radix7_twiddles, * radix8_twiddles,
        * radix9_twiddles, * radix10_twiddles;

#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
    fr_t* radix6_twiddles_6, * radix6_twiddles_12, * radix7_twiddles_7,
        * radix8_twiddles_8, * radix9_twiddles_9;
#endif

    fr_t (*partial_group_gen_powers)[WINDOW_SIZE]; // for LDE

#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
private:
    fr_t* twiddles_X(int num_blocks, int block_size, const fr_t& root)
    {
        fr_t* ret = (fr_t*)gpu.Dmalloc(num_blocks * block_size * sizeof(fr_t));
        generate_radixX_twiddles_X<<<16, block_size, 0, gpu>>>(ret, num_blocks, root);
        CUDA_OK(cudaGetLastError());
        return ret;
    }
#endif

public:
    NTTParameters(const bool _inverse, int id)
        : gpu(select_gpu(id)), inverse(_inverse)
    {
        const fr_t* roots = inverse ? inverse_roots_of_unity
                                    : forward_roots_of_unity;

        const size_t blob_sz = 64 + 128 + 256 + 512 + 32;

        CUDA_OK(cudaGetSymbolAddress((void**)&radix6_twiddles,
                                     inverse ? inverse_radix6_twiddles
                                             : forward_radix6_twiddles));
        radix7_twiddles = (fr_t*)gpu.Dmalloc(blob_sz * sizeof(fr_t));
        radix8_twiddles = radix7_twiddles + 64;
        radix9_twiddles = radix8_twiddles + 128;
        radix10_twiddles = radix9_twiddles + 256;

        generate_all_twiddles<<<blob_sz/32, 32, 0, gpu>>>(radix7_twiddles,
                                                          roots[6],
                                                          roots[7],
                                                          roots[8],
                                                          roots[9],
                                                          roots[10]);
        CUDA_OK(cudaGetLastError());

        CUDA_OK(cudaMemcpyAsync(radix6_twiddles, radix10_twiddles + 512,
                                32 * sizeof(fr_t), cudaMemcpyDeviceToDevice,
                                gpu));

#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
        radix6_twiddles_6 = twiddles_X(64, 64, roots[12]);
        radix6_twiddles_12 = twiddles_X(4096, 64, roots[18]);
        radix7_twiddles_7 = twiddles_X(128, 128, roots[14]);
        radix8_twiddles_8 = twiddles_X(256, 256, roots[16]);
        radix9_twiddles_9 = twiddles_X(512, 512, roots[18]);
#endif

        const size_t partial_sz = WINDOW_NUM * WINDOW_SIZE;

        partial_twiddles = reinterpret_cast<decltype(partial_twiddles)>
                           (gpu.Dmalloc(2 * partial_sz * sizeof(fr_t)));
        partial_group_gen_powers = &partial_twiddles[WINDOW_NUM];

        generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
            (partial_twiddles, roots[MAX_LG_DOMAIN_SIZE]);
        CUDA_OK(cudaGetLastError());

        generate_partial_twiddles<<<WINDOW_SIZE/32, 32, 0, gpu>>>
            (partial_group_gen_powers, inverse ? group_gen_inverse
                                               : group_gen);
        CUDA_OK(cudaGetLastError());
    }
    NTTParameters(const NTTParameters&) = delete;

    ~NTTParameters()
    {
        gpu.Dfree(partial_twiddles);

#if !defined(FEATURE_BABY_BEAR) && !defined(FEATURE_GOLDILOCKS)
        gpu.Dfree(radix9_twiddles_9);
        gpu.Dfree(radix8_twiddles_8);
        gpu.Dfree(radix7_twiddles_7);
        gpu.Dfree(radix6_twiddles_12);
        gpu.Dfree(radix6_twiddles_6);
#endif

        gpu.Dfree(radix7_twiddles);
    }

    inline void sync() const    { gpu.sync(); }

private:
    class all_params { friend class NTTParameters;
        std::vector<const NTTParameters*> forward;
        std::vector<const NTTParameters*> inverse;

        all_params()
        {
            int current_id;
            cudaGetDevice(&current_id);

            size_t nids = ngpus();
            for (size_t id = 0; id < nids; id++)
                forward.push_back(new NTTParameters(false, id));
            for (size_t id = 0; id < nids; id++)
                inverse.push_back(new NTTParameters(true, id));
            for (size_t id = 0; id < nids; id++)
                inverse[id]->sync();

            cudaSetDevice(current_id);
        }
        ~all_params()
        {
            for (auto* ptr: forward) delete ptr;
            for (auto* ptr: inverse) delete ptr;
        }
    };

public:
    static const auto& all(bool inverse = false)
    {
        static all_params params;
        return inverse ? params.inverse : params.forward;
    }
};

#endif
#endif /* __SPPARK_NTT_PARAMETERS_CUH__ */
