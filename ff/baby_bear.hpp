// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_BABY_BEAR_HPP__
#define __SPPARK_FF_BABY_BEAR_HPP__

#ifdef __CUDACC__   // CUDA device-side field types
# include "mont32_t.cuh"
# define inline __device__ __forceinline__

using bb31_base = mont32_t<31, 0x78000001, 0x77ffffff, 0x45dddde3, 0x0ffffffe>;

struct bb31_t : public bb31_base {
    using mem_t = bb31_t;

    inline bb31_t() {}
    inline bb31_t(const bb31_base& a) : bb31_base(a) {}
    inline bb31_t(const uint32_t *p)  : bb31_base(p) {}
    // this is used in constant declaration, e.g. as bb31_t{11}
    __host__ __device__ constexpr bb31_t(int a)      : bb31_base(a) {}
    __host__ __device__ constexpr bb31_t(uint32_t a) : bb31_base(a) {}

    inline bb31_t reciprocal() const
    {
        bb31_t x11, xff, ret = *this;

        x11 = sqr_n_mul(ret, 4, ret);   // 0b10001
        ret = sqr_n_mul(x11, 1, x11);   // 0b110011
        ret = sqr_n_mul(ret, 1, x11);   // 0b1110111
        xff = sqr_n_mul(ret, 1, x11);   // 0b11111111
        ret = sqr_n_mul(ret, 8, xff);   // 0b111011111111111
        ret = sqr_n_mul(ret, 8, xff);   // 0b11101111111111111111111
        ret = sqr_n_mul(ret, 8, xff);   // 0b1110111111111111111111111111111

        return ret;
    }
    friend inline bb31_t operator/(int one, bb31_t a)
    {   if (one != 1) asm("trap;"); return a.reciprocal();   }
    friend inline bb31_t operator/(bb31_t a, bb31_t b)
    {   return a * b.reciprocal();   }
    inline bb31_t& operator/=(const bb31_t a)
    {   *this *= a.reciprocal(); return *this;   }

    inline bb31_t heptaroot() const
    {
        bb31_t x03, x18, x1b, ret = *this;

        x03 = sqr_n_mul(ret, 1, ret);   // 0b11
        x18 = sqr_n(x03, 3);            // 0b11000
        x1b = x18*x03;                  // 0b11011
        ret = x18*x1b;                  // 0b110011
        ret = sqr_n_mul(ret, 6, x1b);   // 0b110011011011
        ret = sqr_n_mul(ret, 6, x1b);   // 0b110011011011011011
        ret = sqr_n_mul(ret, 6, x1b);   // 0b110011011011011011011011
        ret = sqr_n_mul(ret, 6, x1b);   // 0b110011011011011011011011011011
        ret = sqr_n_mul(ret, 1, *this); // 0b1100110110110110110110110110111

        return ret;
    }
};
# undef inline

typedef bb31_t fr_t;

#endif
#endif
