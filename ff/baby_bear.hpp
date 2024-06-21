// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_BABY_BEAR_HPP__
#define __SPPARK_FF_BABY_BEAR_HPP__

#ifdef __CUDACC__   // CUDA device-side field types
# include <cassert>
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
    {   assert(one == 1); return a.reciprocal();   }
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

class __align__(16) bb31_4_t {
    union { bb31_t c[4]; uint32_t u[4]; };

    static const uint32_t MOD   = 0x78000001;
    static const uint32_t M     = 0x77ffffff;
#ifdef BABY_BEAR_CANONICAL
    static const uint32_t BETA  = 0x37ffffe9;   // (11<<32)%MOD
#else                                           // such as RISC Zero
    static const uint32_t BETA  = 0x40000018;   // (-11<<32)%MOD
#endif

public:
    static const uint32_t degree = 4;
    using mem_t = bb31_4_t;

    inline bb31_t& operator[](size_t i)             { return c[i]; }
    inline const bb31_t& operator[](size_t i) const { return c[i]; }
    inline size_t len() const                       { return 4; }

    inline bb31_4_t()           {}
    inline bb31_4_t(bb31_t a)   { c[0] = a;         u[1] = u[2] = u[3] = 0; }
    // this is used in constant declaration, e.g. as bb31_4_t{1, 2, 3, 4}
    __host__ __device__ __forceinline__ bb31_4_t(int a)
    {   c[0] = bb31_t{a}; u[1] = u[2] = u[3] = 0;   }
    __host__ __device__ __forceinline__ bb31_4_t(int d, int f, int g, int h)
    {   c[0] = bb31_t{d}; c[1] = bb31_t{f}; c[2] = bb31_t{g}; c[3] = bb31_t{h};   }

    // Polynomial multiplication modulo x^4 - BETA
    friend __device__ __noinline__ bb31_4_t operator*(bb31_4_t a, bb31_4_t b)
    {
        bb31_4_t ret;

# ifdef __CUDA_ARCH__
#  ifdef __GNUC__
#   define asm __asm__ __volatile__
#  else
#   define asm asm volatile
#  endif
        // ret[0] = a[0]*b[0] + BETA*(a[1]*b[3] + a[2]*b[2] + a[3]*b[1]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %6;      mul.hi.u32  %hi, %4, %6;\n\t"
            "mad.lo.cc.u32 %lo, %3, %7, %lo; madc.hi.u32 %hi, %3, %7, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %8, %lo; madc.hi.u32 %hi, %2, %8, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %hi, %m, %9, %hi;\n\t"
            //"setp.ge.u32 %p, %hi, %9;\n\t"
            //"@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %lo, %hi, %11;    mul.hi.u32  %hi, %hi, %11;\n\t"
            "mad.lo.cc.u32 %lo, %1, %5, %lo; madc.hi.u32 %hi, %1, %5, %hi;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[0])
                : "r"(a.u[0]), "r"(a.u[1]), "r"(a.u[2]), "r"(a.u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[1] = a[0]*b[1] + a[1]*b[0] + BETA*(a[2]*b[3] + a[3]*b[2]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %7;      mul.hi.u32  %hi, %4, %7;\n\t"
            "mad.lo.cc.u32 %lo, %3, %8, %lo; madc.hi.u32 %hi, %3, %8, %hi;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %hi, %m, %9, %hi;\n\t"
            //"setp.ge.u32 %p, %hi, %9;\n\t"
            //"@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %lo, %hi, %11;    mul.hi.u32  %hi, %hi, %11;\n\t"
            "mad.lo.cc.u32 %lo, %2, %5, %lo; madc.hi.u32 %hi, %2, %5, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %1, %6, %lo; madc.hi.u32 %hi, %1, %6, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[1])
                : "r"(a.u[0]), "r"(a.u[1]), "r"(a.u[2]), "r"(a.u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + BETA*(a[3]*b[3]);
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %8;      mul.hi.u32  %hi, %4, %8;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %hi, %m, %9, %hi;\n\t"
            //"setp.ge.u32 %p, %hi, %9;\n\t"
            //"@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %lo, %hi, %11;    mul.hi.u32  %hi, %hi, %11;\n\t"
            "mad.lo.cc.u32 %lo, %3, %5, %lo; madc.hi.u32 %hi, %3, %5, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %6, %lo; madc.hi.u32 %hi, %2, %6, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %1, %7, %lo; madc.hi.u32 %hi, %1, %7, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[2])
                : "r"(a.u[0]), "r"(a.u[1]), "r"(a.u[2]), "r"(a.u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(MOD), "r"(M), "r"(BETA));

        // ret[3] = a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0];
        asm("{ .reg.b32 %lo, %hi, %m; .reg.pred %p;\n\t"
            "mul.lo.u32    %lo, %4, %5;      mul.hi.u32  %hi, %4, %5;\n\t"
            "mad.lo.cc.u32 %lo, %3, %6, %lo; madc.hi.u32 %hi, %3, %6, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %2, %7, %lo; madc.hi.u32 %hi, %2, %7, %hi;\n\t"
            "mad.lo.cc.u32 %lo, %1, %8, %lo; madc.hi.u32 %hi, %1, %8, %hi;\n\t"
            "setp.ge.u32 %p, %hi, %9;\n\t"
            "@%p sub.u32 %hi, %hi, %9;\n\t"

            "mul.lo.u32    %m, %lo, %10;\n\t"
            "mad.lo.cc.u32 %lo, %m, %9, %lo; madc.hi.u32 %0, %m, %9, %hi;\n\t"
            "setp.ge.u32 %p, %0, %9;\n\t"
            "@%p sub.u32 %0, %0, %9;\n\t"
            "}" : "=r"(ret.u[3])
                : "r"(a.u[0]), "r"(a.u[1]), "r"(a.u[2]), "r"(a.u[3]),
                  "r"(b.u[0]), "r"(b.u[1]), "r"(b.u[2]), "r"(b.u[3]),
                  "r"(MOD), "r"(M), "r"(BETA));
#  undef asm
# else
        union { uint64_t ul; uint32_t u[2]; };

        // ret[0] = a[0]*b[0] + BETA*(a[1]*b[3] + a[2]*b[2] + a[3]*b[1]);
        ul  = a.u[1] * (uint64_t)b.u[3];
        ul += a.u[2] * (uint64_t)b.u[2];
        ul += a.u[3] * (uint64_t)b.u[1];    if (u[1] >= MOD) u[1] -= MOD;
        ul += (u[0] * M) * (uint64_t)MOD;   // if (u[1] >= MOD) u[1] -= MOD;
        ul  = u[1] * (uint64_t)BETA;
        ul += a.u[0] * (uint64_t)b.u[0];
        ul += (u[0] * M) * (uint64_t)MOD;
        ret.u[0] = u[1] >= MOD ?  u[1] - MOD : u[1];

        // ret[1] = a[0]*b[1] + a[1]*b[0] + BETA*(a[2]*b[3] + a[3]*b[2]);
        ul  = a.u[2] * (uint64_t)b.u[3];
        ul += a.u[3] * (uint64_t)b.u[2];
        ul += (u[0] * M) * (uint64_t)MOD;   // if (u[1] >= MOD) u[1] -= MOD;
        ul  = u[1] * (uint64_t)BETA;
        ul += a.u[0] * (uint64_t)b.u[1];
        ul += a.u[1] * (uint64_t)b.u[0];    if (u[1] >= MOD) u[1] -= MOD;
        ul += (u[0] * M) * (uint64_t)MOD;
        ret.u[1] = u[1] >= MOD ?  u[1] - MOD : u[1];

        // ret[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + BETA*(a[3]*b[3]);
        ul  = a.u[3] * (uint64_t)b.u[3];
        ul += (u[0] * M) * (uint64_t)MOD;   // if (u[1] >= MOD) u[1] -= MOD;
        ul  = u[1] * (uint64_t)BETA;
        ul += a.u[0] * (uint64_t)b.u[2];
        ul += a.u[1] * (uint64_t)b.u[1];
        ul += a.u[2] * (uint64_t)b.u[0];    if (u[1] >= MOD) u[1] -= MOD;
        ul += (u[0] * M) * (uint64_t)MOD;
        ret.u[2] = u[1] >= MOD ?  u[1] - MOD : u[1];

        // ret[3] = a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0];
        ul  = a.u[0] * (uint64_t)b.u[3];
        ul += a.u[1] * (uint64_t)b.u[2];
        ul += a.u[2] * (uint64_t)b.u[1];
        ul += a.u[3] * (uint64_t)b.u[0];    if (u[1] >= MOD) u[1] -= MOD;
        ul += (u[0] * M) * (uint64_t)MOD;
        ret.u[3] = u[1] >= MOD ?  u[1] - MOD : u[1];
# endif

        return ret;
    }
    inline bb31_4_t& operator*=(const bb31_4_t& b)
    {   return *this = *this * b;   }

    friend __device__ __noinline__ bb31_4_t operator*(bb31_4_t a, bb31_t b)
    {
        bb31_4_t ret;

        for (size_t i = 0; i < 4; i++)
            ret[i] = a[i] * b;

        return ret;
    }
    friend inline bb31_4_t operator*(bb31_t b, const bb31_4_t& a)
    {   return a * b;   }
    inline bb31_4_t& operator*=(bb31_t b)
    {   return *this = *this * b;   }

    friend inline bb31_4_t operator+(const bb31_4_t& a, const bb31_4_t& b)
    {
        bb31_4_t ret;

        for (size_t i = 0; i < 4; i++)
            ret[i] = a[i] + b[i];

        return ret;
    }
    inline bb31_4_t& operator+=(const bb31_4_t& b)
    {   return *this = *this + b;   }

    friend inline bb31_4_t operator+(const bb31_4_t& a, bb31_t b)
    {
        bb31_4_t ret;

        ret[0] = a[0] + b;
        ret[1] = a[1];
        ret[2] = a[2];
        ret[3] = a[3];

        return ret;
    }
    friend inline bb31_4_t operator+(bb31_t b, const bb31_4_t& a)
    {   return a + b;   }
    inline bb31_4_t& operator+=(bb31_t b)
    {   c[0] += b; return *this;   }

    friend inline bb31_4_t operator-(const bb31_4_t& a, const bb31_4_t& b)
    {
        bb31_4_t ret;

        for (size_t i = 0; i < 4; i++)
            ret[i] = a[i] - b[i];

        return ret;
    }
    inline bb31_4_t& operator-=(const bb31_4_t& b)
    {   return *this = *this - b;   }

    friend inline bb31_4_t operator-(const bb31_4_t& a, bb31_t b)
    {
        bb31_4_t ret;

        ret[0] = a[0] - b;
        ret[1] = a[1];
        ret[2] = a[2];
        ret[3] = a[3];

        return ret;
    }
    friend inline bb31_4_t operator-(bb31_t b, const bb31_4_t& a)
    {
        bb31_4_t ret;

        ret[0] = b - a[0];
        ret[1] = -a[1];
        ret[2] = -a[2];
        ret[3] = -a[3];

        return ret;
    }
    inline bb31_4_t& operator-=(bb31_t b)
    {   c[0] -= b; return *this;   }

    inline bb31_4_t reciprocal() const
    {
        const bb31_t beta{BETA};

        // don't bother with breaking this down, 1/x dominates.
        bb31_t b0 = c[0]*c[0] - beta*(c[1]*bb31_t{u[3]<<1} - c[2]*c[2]);
        bb31_t b2 = c[0]*bb31_t{u[2]<<1} - c[1]*c[1] - beta*(c[3]*c[3]);

        bb31_t inv = 1/(b0*b0 - beta*b2*b2);

        b0 *= inv;
        b2 *= inv;

        bb31_4_t ret;
        bb31_t beta_b2 = beta*b2;
        ret[0] = c[0]*b0 - c[2]*beta_b2;
        ret[1] = c[3]*beta_b2 - c[1]*b0;
        ret[2] = c[2]*b0 - c[0]*b2;
        ret[3] = c[1]*b2 - c[3]*b0;

        return ret;
    }
    friend inline bb31_4_t operator/(int one, const bb31_4_t& a)
    {   assert(one == 1); return a.reciprocal();   }
    friend inline bb31_4_t operator/(const bb31_4_t& a, const bb31_4_t& b)
    {   return a * b.reciprocal();   }
    friend inline bb31_4_t operator/(bb31_t a, const bb31_4_t& b)
    {   return b.reciprocal() * a;   }
    friend inline bb31_4_t operator/(const bb31_4_t& a, bb31_t b)
    {   return a * b.reciprocal();   }
    inline bb31_4_t& operator/=(const bb31_4_t& a)
    {   return *this *= a.reciprocal();   }
    inline bb31_4_t& operator/=(bb31_t a)
    {   return *this *= a.reciprocal();   }

    inline bool is_one() const
    {   return c[0].is_one() & u[1]==0 & u[2]==0 & u[3]==0;   }
    inline bool is_zero() const
    {   return u[0]==0 & u[1]==0 & u[2]==0 & u[3]==0;   }
# undef inline

public:
    friend inline bool operator==(const bb31_4_t& a, const bb31_4_t& b)
    {   return a.u[0]==b.u[0] & a.u[1]==b.u[1] & a.u[2]==b.u[2] & a.u[3]==b.u[3];   }
    friend inline bool operator!=(const bb31_4_t& a, const bb31_4_t& b)
    {   return a.u[0]!=b.u[0] | a.u[1]!=b.u[1] | a.u[2]!=b.u[2] | a.u[3]!=b.u[3];   }

# if defined(_GLIBCXX_IOSTREAM) || defined(_IOSTREAM_) // non-standard
    friend std::ostream& operator<<(std::ostream& os, const bb31_4_t& a)
    {
        os << "[" << a.c[0] << ", " << a.c[1] << ", " << a.c[2] << ", " << a.c[3] << "]";
        return os;
    }
# endif
};

typedef bb31_t fr_t;
typedef bb31_4_t fr4_t;

#endif
#endif
