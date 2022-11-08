// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <iostream>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include "parameters.cuh"
#include "kernels.cu"

#ifndef __CUDA_ARCH__

class NTT {
public:
    enum class InputOutputOrder { NN, NR, RN, RR };
    enum class Direction { forward, inverse };
    enum class Type { standard, coset };
    enum class Algorithm { GS, CT };

protected:
    static void bit_rev(fr_t* d_out, const fr_t* d_inp,
                        uint32_t lg_domain_size, stream_t& stream)
    {
        assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

        size_t domain_size = (size_t)1 << lg_domain_size;

        if (domain_size <= WARP_SZ)
            bit_rev_permutation
                <<<1, domain_size, 0, stream>>>
                (d_out, d_inp, lg_domain_size);
        else if (d_out == d_inp)
            bit_rev_permutation
                <<<domain_size/WARP_SZ, WARP_SZ, 0, stream>>>
                (d_out, d_inp, lg_domain_size);
        else if (domain_size < 1024)
            bit_rev_permutation_aux
                <<<1, domain_size / 8, domain_size * sizeof(fr_t), stream>>>
                (d_out, d_inp, lg_domain_size);
        else
            bit_rev_permutation_aux
                <<<domain_size / 1024, 1024 / 8, 1024 * sizeof(fr_t), stream>>>
                (d_out, d_inp, lg_domain_size);
    }

private:
    static void LDE_powers(fr_t* inout, bool innt, bool bitrev,
                           uint32_t lg_domain_size, uint32_t lg_blowup,
                           stream_t& stream)
    {
        size_t domain_size = (size_t)1 << lg_domain_size;
        const auto gen_powers =
            NTTParameters::all(innt)[stream]->partial_group_gen_powers;

        if (domain_size < WARP_SZ)
            LDE_distribute_powers<<<1, domain_size, 0, stream>>>
                                 (inout, lg_blowup, bitrev, gen_powers);
        else if (domain_size < 512)
            LDE_distribute_powers<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                                 (inout, lg_blowup, bitrev, gen_powers);
        else
            LDE_distribute_powers<<<domain_size / 512, 512, 0, stream>>>
                                 (inout, lg_blowup, bitrev, gen_powers);
    }

protected:
    static void NTT_internal(fr_t* d_inout, uint32_t lg_domain_size,
                             InputOutputOrder order, Direction direction,
                             Type type, stream_t& stream)
    {
        // Pick an NTT algorithm based on the input order and the desired output
        // order of the data. In certain cases, bit reversal can be avoided which
        // results in a considerable performance gain.

        const bool intt = direction == Direction::inverse;
        const auto& ntt_parameters = *NTTParameters::all(intt)[stream];
        bool bitrev;
        Algorithm algorithm;

        switch (order) {
            case InputOutputOrder::NN:
                bit_rev(d_inout, d_inout, lg_domain_size, stream);
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::NR:
                bitrev = false;
                algorithm = Algorithm::GS;
                break;
            case InputOutputOrder::RN:
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::RR:
                bitrev = true;
                algorithm = Algorithm::GS;
                break;
            default:
                assert(false);
        }

        if (!intt && type == Type::coset)
            LDE_powers(d_inout, intt, bitrev, lg_domain_size, 0, stream);

        switch (algorithm) {
            case Algorithm::GS:
                GS_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
            case Algorithm::CT:
                CT_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
        }

        if (intt && type == Type::coset)
            LDE_powers(d_inout, intt, !bitrev, lg_domain_size, 0, stream);

        if (order == InputOutputOrder::RR)
            bit_rev(d_inout, d_inout, lg_domain_size, stream);
    }

public:
    static RustError Base(const gpu_t& gpu, fr_t* inout, uint32_t lg_domain_size,
                          InputOutputOrder order, Direction direction,
                          Type type)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> d_inout(domain_size);
            gpu.HtoD(&d_inout[0], inout, domain_size);

            NTT_internal(&d_inout[0], lg_domain_size, order, direction, type, gpu);

            gpu.DtoH(inout, &d_inout[0], domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    static RustError LDE(const gpu_t& gpu, fr_t* inout,
                         uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size << lg_blowup;
            dev_ptr_t<fr_t> d_ext_domain(ext_domain_size);
            fr_t* d_domain = &d_ext_domain[ext_domain_size - domain_size];

            gpu.HtoD(&d_domain[0], inout, domain_size);

            NTT_internal(&d_domain[0], lg_domain_size,
                         InputOutputOrder::NR, Direction::inverse,
                         Type::standard, gpu);

            const auto gen_powers =
                NTTParameters::all()[gpu.id()]->partial_group_gen_powers;

            LDE_launch(gpu, &d_ext_domain[0], &d_domain[0],
                       gen_powers, lg_domain_size, lg_blowup);

            NTT_internal(&d_ext_domain[0], lg_domain_size + lg_blowup,
                         InputOutputOrder::RN, Direction::forward,
                         Type::standard, gpu);

            gpu.DtoH(inout, &d_ext_domain[0], ext_domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

protected:
    static void LDE_launch(const gpu_t& gpu,
                           fr_t* ext_domain_data, fr_t* domain_data,
                           const fr_t (*gen_powers)[WINDOW_SIZE],
                           uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        assert(lg_domain_size + lg_blowup <= MAX_LG_DOMAIN_SIZE);
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size << lg_blowup;

        // Determine the max power of 2 SM count
        size_t kernel_sms = gpu.sm_count();
        while (kernel_sms & (kernel_sms - 1))
            kernel_sms -= (kernel_sms & (0 - kernel_sms));

        size_t device_max_threads = kernel_sms * 1024;
        uint32_t num_blocks, block_size;

        if (device_max_threads < domain_size) {
            num_blocks = kernel_sms;
            block_size = 1024;
        } else if (domain_size < 1024) {
            num_blocks = 1;
            block_size = domain_size;
        } else {
            num_blocks = domain_size / 1024;
            block_size = 1024;
        }

        gpu.launch_coop(LDE_spread_distribute_powers,
                              num_blocks, block_size, sizeof(fr_t) * block_size,
                        ext_domain_data, domain_data, gen_powers,
                        lg_domain_size, lg_blowup);
    }

public:
    static RustError LDE_aux(const gpu_t& gpu, fr_t* inout,
                             uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        try {

            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size << lg_blowup;
            // The 2nd to last 'domain_size' chunk will hold the original data
            // The last chunk will get the bit reversed iNTT data
            dev_ptr_t<fr_t> d_inout(ext_domain_size + domain_size); // + domain_size for aux buffer
            cudaDeviceSynchronize();
            fr_t* aux_data = &d_inout[ext_domain_size];
            fr_t* domain_data = &d_inout[ext_domain_size - domain_size]; // aligned to the end
            fr_t* ext_domain_data = &d_inout[0];
            gpu.HtoD(domain_data, inout, domain_size);

            NTT_internal(domain_data, lg_domain_size,
                         InputOutputOrder::NR, Direction::inverse,
                         Type::standard, gpu);

            const auto gen_powers =
                NTTParameters::all()[gpu.id()]->partial_group_gen_powers;

            bit_rev(aux_data, domain_data, lg_domain_size, gpu);

            LDE_launch(gpu, ext_domain_data, domain_data, gen_powers,
                       lg_domain_size, lg_blowup);

            // NTT - RN
            NTT_internal(ext_domain_data, lg_domain_size + lg_blowup,
                         InputOutputOrder::RN, Direction::forward,
                         Type::standard, gpu);

            gpu.DtoH(inout, ext_domain_data, domain_size << lg_blowup);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }
};

#endif
