// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __BB31_T_CUH__
#define __BB31_T_CUH__

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

public:
    static const uint32_t nbits = 31;
    static const uint32_t MOD = 0x78000001u;
    static const uint32_t M = 0x77ffffffu;
    static const uint32_t R = 0x0ffffffeu;
    static const uint32_t RR = 0x45dddde3u;

    inline uint32_t& operator[](size_t i)               { return val; }
    inline const uint32_t& operator[](size_t i) const   { return val; }
    inline size_t len() const                           { return 1;   }

    inline bb31_t() {}
    inline bb31_t(const uint32_t a) { val = a; }
    inline bb31_t(const uint32_t *p) { val = *p; }

    inline operator uint32_t() const
    { auto ret = *this; ret.from(); return ret.val; }
    inline void store(uint32_t *p) const
    { *p = *this; }
    inline void operator=(uint32_t b)
    { val = b; to(); }

    inline bb31_t operator+=(const bb31_t b)
    {
        val += b.val;
        if (val >= MOD) val -= MOD;

        return *this;
    }
    friend inline bb31_t operator+(bb31_t a, const bb31_t b)
    {   return a += b;   }

    inline bb31_t operator<<=(uint32_t l)
    {
        while (l--) {
            val <<= 1;
            if (val >= MOD) val -= MOD;
        }

        return *this;
    }
    friend inline bb31_t operator<<(bb31_t a, uint32_t l)
    {   return a <<= l;   }

    inline bb31_t operator>>=(uint32_t r)
    {
        while (r--) {
            val += val&1 ? MOD : 0;
            val >>= 1;
        }

        return *this;
    }
    friend inline bb31_t operator>>(bb31_t a, uint32_t r)
    {   return a >>= r;   }

    inline bb31_t operator-=(const bb31_t b)
    {
        val -= b.val;
        if (val > MOD) val += MOD;

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
    friend inline bb31_t cneg(bb31_t a, bool flag)
    {   return a.cneg(flag);   }
    inline bb31_t operator-() const
    {   bb31_t ret = *this; return ret.cneg(true);   }

    static inline const bb31_t one()
    {   return bb31_t{R};   }

    inline bool is_zero() const
    {   return val == 0;   }

    inline void zero()
    {   val = 0;   }

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
    inline void mul(const bb31_t b)
    {
        uint32_t tmp[2], red;

        asm("mul.lo.u32 %0, %2, %3; mul.hi.u32 %1, %2, %3;"
            : "=r"(tmp[0]), "=r"(tmp[1])
            : "r"(val), "r"(b.val));
        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(tmp[0]), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.u32 %1, %2, %3, %1;"
            : "+r"(tmp[0]), "+r"(tmp[1])
            : "r"(red), "r"(MOD));

        asm("{");
        asm(".reg.pred %p;");
        asm("sub.u32 %0, %1, %2;" : "=r"(tmp[0]) : "r"(tmp[1]), "r"(MOD));
        asm("setp.ge.u32 %p, %0, %1;" :: "r"(tmp[1]), "r"(MOD));
        asm("selp.u32 %0, %1, %2, %p;" : "=r"(val) : "r"(tmp[0]), "r"(tmp[1]));
        asm("}");
    }

    inline void mul_by_1()
    {
        uint32_t tmp[2], red;

        asm("mul.lo.u32 %0, %1, %2;" : "=r"(red) : "r"(val), "r"(M));
        asm("mad.lo.cc.u32 %0, %2, %3, %4; madc.hi.u32 %1, %2, %3, 0;"
            : "=r"(tmp[0]), "=r"(tmp[1])
            : "r"(red), "r"(MOD), "r"(val));
        val = tmp[1];
    }

public:
    friend inline bb31_t operator*(bb31_t a, const bb31_t b)
    {   a.mul(b); return a;   }

    inline bb31_t operator*=(const bb31_t a)
    {   mul(a); return *this;   }

    // simplified exponentiation, but mind the ^ operator's precedence!
    inline bb31_t operator^=(uint32_t p)
    {
        if (p < 2)
            asm("trap;");

        bb31_t sqr = *this;
        for (; (p&1) == 0; p >>= 1)
            sqr.mul(sqr);
        *this = sqr;
        for (p >>= 1; p; p >>= 1) {
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
    friend inline bb31_t sqr(const bb31_t a)
    {   return a^2u;   }
    inline bb31_t sqr()
    {   return *this ^= 2u;   }

    inline void to() { bb31_t t{RR}; t *= *this; *this = t; }
    inline void from() { mul_by_1(); }
};

#  undef inline
#  undef asm
# endif
#endif /* __BB31_T_CUH__ */
