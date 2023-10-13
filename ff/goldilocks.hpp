// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifdef __NVCC__
# include "gl64_t.cuh"  // device-side field types
# ifndef __CUDA_ARCH__  // host-side stand-in to make CUDA code compile,
#  include <cstdint>    // not to produce correct result...

#  if defined(__GNUC__) || defined(__clang__)
#   pragma GCC diagnostic push
#   pragma GCC diagnostic ignored "-Wunused-parameter"
#  endif
class gl64_t {
    uint64_t val;
public:
    using mem_t = gl64_t;
    static const uint32_t degree = 1;
    static const uint64_t MOD = 0xffffffff00000001U;

    inline gl64_t()                     {}
    inline gl64_t(uint64_t a) : val(a)  {}
    inline operator uint64_t() const    { return val;   }
    static inline const gl64_t one()    { return 1;     }
    inline gl64_t& operator+=(gl64_t b) { return *this; }
    inline gl64_t& operator-=(gl64_t b) { return *this; }
    inline gl64_t& operator*=(gl64_t b) { return *this; }
    inline gl64_t& operator^=(int p)    { return *this; }
    inline gl64_t& sqr()                { return *this; }
    inline void zero()                  { val = 0;      }
};
#  if defined(__GNUC__) || defined(__clang__)
#   pragma GCC diagnostic pop
#  endif
# endif
typedef gl64_t fr_t;
#endif
