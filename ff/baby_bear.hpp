// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifdef __NVCC__
# include "bb31_t.cuh"  // device-side field types
# ifndef __CUDA_ARCH__  // host-side stand-in to make CUDA code compile,
#  include <cstdint>    // and provide some debugging support, but
                        // not to produce correct computational result...

#  if defined(__GNUC__) || defined(__clang__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#  endif
class bb31_t {
    uint32_t val;

    static const uint32_t M = 0x77ffffff;
public:
    using mem_t = bb31_t;
    static const uint32_t degree = 1;
    static const uint32_t nbits = 31;
    static const uint32_t MOD = 0x78000001;

    inline bb31_t()                     {}
    inline bb31_t(uint32_t a) : val(a)  {}
    // this is used in constant declaration, e.g. as bb31_t{11}
    inline constexpr bb31_t(int a) : val(((uint64_t)a << 32) % MOD) {}

    static inline const bb31_t one()                { return bb31_t(1); }
    inline bb31_t& operator+=(bb31_t b)             { return *this;     }
    inline bb31_t& operator-=(bb31_t b)             { return *this;     }
    inline bb31_t& operator*=(bb31_t b)             { return *this;     }
    inline bb31_t& operator^=(int b)                { return *this;     }
    inline bb31_t& sqr()                            { return *this;     }
    friend bb31_t operator+(bb31_t a, bb31_t b)     { return a += b;    }
    friend bb31_t operator-(bb31_t a, bb31_t b)     { return a -= b;    }
    friend bb31_t operator*(bb31_t a, bb31_t b)     { return a *= b;    }
    friend bb31_t operator^(bb31_t a, uint32_t b)   { return a ^= b;    }
    inline void zero()                              { val = 0;          }
    inline bool is_zero() const                     { return val==0;    }
    inline operator uint32_t() const
    {   return ((val*M)*(uint64_t)MOD + val) >> 32;  }
    inline void to()    { val = ((uint64_t)val<<32) % MOD;  }
    inline void from()  { val = *this; }
#  if defined(_GLIBCXX_IOSTREAM) || defined(_IOSTREAM_) // non-standard
    friend std::ostream& operator<<(std::ostream& os, const bb31_t& obj)
    {
        auto f = os.flags();
        os << "0x" << std::hex << (uint32_t)obj;
        os.flags(f);
        return os;
    }
#  endif
};
#  if defined(__GNUC__) || defined(__clang__)
#   pragma GCC diagnostic pop
#  endif
# endif
typedef bb31_t fr_t;
#endif
