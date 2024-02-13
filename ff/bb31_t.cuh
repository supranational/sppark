// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_FF_BB31_T_CUH__
#define __SPPARK_FF_BB31_T_CUH__

# include <cstdint>

#ifdef __CUDA_ARCH__
# define inline __device__ __forceinline__
# ifdef __GNUC__
#  define asm __asm__ __volatile__
# else
#  define asm asm volatile
# endif

class bb31_t {
private:
    uint32_t val;

    static const uint32_t M = 0x77ffffffu;
    static const uint32_t RR = 0x45dddde3u;
    static const uint32_t ONE = 0x0ffffffeu;
public:
    using mem_t = bb31_t;
    static const uint32_t degree = 1;
    static const uint32_t nbits = 31;
    static const uint32_t MOD = 0x78000001u;
    static constexpr size_t __device__ bit_length()     { return 31;  }

    inline uint32_t& operator[](size_t i)               { return val; }
    inline uint32_t& operator*()                        { return val; }
    inline const uint32_t& operator[](size_t i) const   { return val; }
    inline uint32_t operator*() const                   { return val; }
    inline size_t len() const                           { return 1;   }

    inline bb31_t() {}
    inline bb31_t(const uint32_t a)         { val = a;  }
    inline bb31_t(const uint32_t *p)        { val = *p; }
    // this is used in constant declaration, e.g. as bb31_t{11}
    inline constexpr bb31_t(int a) : val(((uint64_t)a << 32) % MOD) {}

    inline operator uint32_t() const        { return mul_by_1(); }
    inline void store(uint32_t *p) const    { *p = mul_by_1();   }
    inline bb31_t& operator=(uint32_t b)    { val = b; to(); return *this; }

    inline bb31_t& operator+=(const bb31_t b)
    {
        val += b.val;
        if (val >= MOD) val -= MOD;

        return *this;
    }
    friend inline bb31_t operator+(bb31_t a, const bb31_t b)
    {   return a += b;   }

    inline bb31_t& operator<<=(uint32_t l)
    {
        while (l--) {
            val <<= 1;
            if (val >= MOD) val -= MOD;
        }

        return *this;
    }
    friend inline bb31_t operator<<(bb31_t a, uint32_t l)
    {   return a <<= l;   }

    inline bb31_t& operator>>=(uint32_t r)
    {
        while (r--) {
            val += val&1 ? MOD : 0;
            val >>= 1;
        }

        return *this;
    }
    friend inline bb31_t operator>>(bb31_t a, uint32_t r)
    {   return a >>= r;   }

    inline bb31_t& operator-=(const bb31_t b)
    {
        asm("{");
        asm(".reg.pred %brw;");
        asm("setp.lt.u32 %brw, %0, %1;" :: "r"(val), "r"(b.val));
        asm("sub.u32 %0, %0, %1;"       : "+r"(val) : "r"(b.val));
        asm("@%brw add.u32 %0, %0, %1;" : "+r"(val) : "r"(MOD));
        asm("}");

        return *this;
    }
    friend inline bb31_t operator-(bb31_t a, const bb31_t b)
    {   return a -= b;   }

    inline bb31_t cneg(bool flag)
    {
        asm("{");
        asm(".reg.pred %flag;");
        asm("setp.ne.u32 %flag, %0, 0;" :: "r"(val));
        asm("@%flag setp.ne.u32 %flag, %0, 0;" :: "r"((int)flag));
        asm("@%flag sub.u32 %0, %1, %0;" : "+r"(val) : "r"(MOD));
        asm("}");

        return *this;
    }
    static inline bb31_t cneg(bb31_t a, bool flag)
    {   return a.cneg(flag);   }
    inline bb31_t operator-() const
    {   return cneg(*this, true);   }

    static inline const bb31_t one()    { return bb31_t{ONE}; }
    inline bool is_one() const          { return val == ONE;  }
    inline bool is_zero() const         { return val == 0;    }
    inline void zero()                  { val = 0;            }

    friend inline bb31_t czero(const bb31_t a, int set_z)
    {
        bb31_t ret;

        asm("{");
        asm(".reg.pred %set_z;");
        asm("setp.ne.s32 %set_z, %0, 0;" : : "r"(set_z));
        asm("selp.u32 %0, 0, %1, %set_z;" : "=r"(ret.val) : "r"(a.val));
        asm("}");

        return ret;
    }

    static inline bb31_t csel(const bb31_t a, const bb31_t b, int sel_a)
    {
        bb31_t ret;

        asm("{");
        asm(".reg.pred %sel_a;");
        asm("setp.ne.s32 %sel_a, %0, 0;" :: "r"(sel_a));
        asm("selp.u32 %0, %1, %2, %sel_a;" : "=r"(ret.val) : "r"(a.val), "r"(b.val));
        asm("}");

        return ret;
    }

private:
    static inline void final_sub(uint32_t& val)
    {
        asm("{");
        asm(".reg.pred %p;");
        asm("setp.ge.u32 %p, %0, %1;" :: "r"(val), "r"(MOD));
        asm("@%p sub.u32 %0, %0, %1;" : "+r"(val) : "r"(MOD));
        asm("}");
    }

    inline bb31_t& mul(const bb31_t b)
    {
        uint32_t tmp[2], red;

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(tmp[0]), "=r"(tmp[1])
            : "r"(val), "r"(b.val));
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
            : "+r"(tmp[0]), "=r"(val)
            : "r"(red), "r"(MOD), "r"(tmp[1]));

        final_sub(val);

        return *this;
    }

    inline uint32_t mul_by_1() const
    {
        uint32_t tmp[2], red;

        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(val), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %4; madc.hi.u32 %1, %2, %3, 0;"
            : "=r"(tmp[0]), "=r"(tmp[1])
            : "r"(red), "r"(MOD), "r"(val));
        return tmp[1];
    }

public:
    friend inline bb31_t operator*(bb31_t a, const bb31_t b)
    {   return a.mul(b);   }
    inline bb31_t& operator*=(const bb31_t a)
    {   return mul(a);   }

    // raise to a variable power, variable in respect to threadIdx,
    // but mind the ^ operator's precedence!
    inline bb31_t& operator^=(uint32_t p)
    {
        bb31_t sqr = *this;
        *this = csel(val, ONE, p&1);

        #pragma unroll 1
        while (p >>= 1) {
            sqr.mul(sqr);
            if (p&1)
                mul(sqr);
        }

        return *this;
    }
    friend inline bb31_t operator^(bb31_t a, uint32_t p)
    {   return a ^= p;   }
    inline bb31_t operator()(uint32_t p)
    {   return *this^p;   }

    // raise to a constant power, e.g. x^7, to be unrolled at compile time
    inline bb31_t& operator^=(int p)
    {
        if (p < 2)
            asm("trap;");

        bb31_t sqr = *this;
        if ((p&1) == 0) {
            do {
                sqr.mul(sqr);
                p >>= 1;
            } while ((p&1) == 0);
            *this = sqr;
        }
        for (p >>= 1; p; p >>= 1) {
            sqr.mul(sqr);
            if (p&1)
                mul(sqr);
        }

        return *this;
    }
    friend inline bb31_t operator^(bb31_t a, int p)
    {   return a ^= p;   }
    inline bb31_t operator()(int p)
    {   return *this^p;   }
    friend inline bb31_t sqr(bb31_t a)
    {   return a.sqr();   }
    inline bb31_t& sqr()
    {   return mul(*this);   }

    inline void to()   { mul(RR); }
    inline void from() { val = mul_by_1(); }

    template<size_t T>
    static inline bb31_t dot_product(const bb31_t a[T], const bb31_t b[T])
    {
        uint32_t acc[2];
        size_t i = 1;

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[0]), "=r"(acc[1]) : "r"(*a[0]), "r"(*b[0]));
        if ((T&1) == 0) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[i]));
            i++;
        }
        for (; i < T; i += 2) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[i]));
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i+1]), "r"(*b[i+1]));
            final_sub(acc[1]);
        }

        uint32_t red;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1]) : "r"(red), "r"(MOD));
        final_sub(acc[1]);

        return acc[1];
    }

    template<size_t T>
    static inline bb31_t dot_product(bb31_t a0, bb31_t b0,
                                     const bb31_t a[T-1], const bb31_t *b,
                                     size_t stride_b = 1)
    {
        uint32_t acc[2];
        size_t i = 0;

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(acc[0]), "=r"(acc[1]) : "r"(*a0), "r"(*b0));
        if ((T&1) == 0) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[0]));
            i++, b += stride_b;
        }
        for (; i < T-1; i += 2) {
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i]), "r"(*b[0]));
            b += stride_b;
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
                : "+r"(acc[0]), "+r"(acc[1]) : "r"(*a[i+1]), "r"(*b[0]));
            b += stride_b;
            final_sub(acc[1]);
        }

        uint32_t red;
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(acc[0]), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(acc[0]), "+r"(acc[1]) : "r"(red), "r"(MOD));
        final_sub(acc[1]);

        return acc[1];
    }

private:
    static inline bb31_t sqr_n(bb31_t s, uint32_t n)
    {
#if 0
        #pragma unroll 2
        while (n--)
            s.sqr();
#else   // +20% [for reciprocal()]
        #pragma unroll 2
        while (n--) {
            uint32_t tmp[2], red;

            asm("mul.lo.u32 %0, %2, %2; mul.hi.u32 %1, %2, %2;"
                : "=r"(tmp[0]), "=r"(tmp[1])
                : "r"(s.val));
            asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M));
            asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %4;"
                : "+r"(tmp[0]), "=r"(s.val)
                : "r"(red), "r"(MOD), "r"(tmp[1]));

            if (n&1)
                final_sub(s.val);
        }
#endif
        return s;
    }

    static inline bb31_t sqr_n_mul(bb31_t s, uint32_t n, bb31_t m)
    {
        s = sqr_n(s, n);
        s.mul(m);

        return s;
    }

public:
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
    {   return *this *= a.reciprocal();   }

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

    inline void shfl_bfly(uint32_t laneMask)
    {   val = __shfl_xor_sync(0xFFFFFFFF, val, laneMask);   }
};

#  undef inline
#  undef asm
# endif
#endif /* __SPPARK_FF_BB31_T_CUH__ */
