// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_EC_AFFINE_T_HPP__
#define __SPPARK_EC_AFFINE_T_HPP__

template<class field_t, class field_h = typename field_t::mem_t, const field_h* = nullptr> class Affine_t;
template<class field_t, class field_h = typename field_t::mem_t, const field_h* = nullptr> class Affine_inf_t;
template<class field_t, const typename field_t::mem_t* = nullptr> class jacobian_t;
template<class field_t, class field_h = typename field_t::mem_t, const field_h* = nullptr> class xyzz_t;

#ifndef __CUDACC__
# undef  __host__
# define __host__
# undef  __device__
# define __device__
# undef  __noinline__
# define __noinline__
#endif

template<class field_t, class field_h, const field_h* a4>
class Affine_t {
    friend class Affine_inf_t<field_t, field_h, a4>;
    friend class jacobian_t<field_t, a4>;
    friend class xyzz_t<field_t, field_h, a4>;

    typedef Affine_inf_t<field_t, field_h, a4> Affine_inf;
    typedef jacobian_t<field_t, a4> jacobian;
    typedef xyzz_t<field_t, field_h, a4> xyzz;

    field_t X, Y;

public:
    Affine_t(const field_t& x, const field_t& y) : X(x), Y(y) {}
    inline __host__ __device__ Affine_t() {}

#ifdef __CUDA_ARCH__
    inline __device__ bool is_inf() const
    {   return (bool)(X.is_zero(Y));   }
#else
    inline __host__   bool is_inf() const
    {   return (bool)((int)X.is_zero() & (int)Y.is_zero());   }
#endif

    inline __host__ Affine_t& operator=(const jacobian& a)
    {
        Y = 1/a.Z;
        X = Y^2;    // 1/Z^2
        Y *= X;     // 1/Z^3
        X *= a.X;   // X/Z^2
        Y *= a.Y;   // Y/Z^3
        return *this;
    }
    inline __host__ Affine_t(const jacobian& a) { *this = a; }

    inline __host__ Affine_t& operator=(const xyzz& a)
    {
        Y = 1/a.ZZZ;
        X = Y * a.ZZ;   // 1/Z
        X = X^2;        // 1/Z^2
        X *= a.X;       // X/Z^2
        Y *= a.Y;       // Y/Z^3
        return *this;
    }
    inline __host__ Affine_t(const xyzz& a)  { *this = a; }

    inline __host__ __device__ operator jacobian() const
    {
        jacobian p;
        p.X = X;
        p.Y = Y;
        p.Z = field_t::one(is_inf());
        return p;
    }

    inline __host__ __device__ operator xyzz() const
    {
        xyzz p;
        p.X = X;
        p.Y = Y;
        p.ZZZ = p.ZZ = field_t::one(is_inf());
        return p;
    }

#ifdef __CUDACC__
    class mem_t {
        field_h X, Y;

    public:
        inline __device__ operator Affine_t() const
        {
            Affine_t p;
            p.X = X;
            p.Y = Y;
            return p;
        }
    };
#else
    using mem_t = Affine_t;
#endif
};

template<class field_t, class field_h, const field_h* a4>
class Affine_inf_t {
    typedef Affine_t<field_t, field_h, a4> Affine;
    typedef jacobian_t<field_t, a4> jacobian;
    typedef xyzz_t<field_t, field_h, a4> xyzz;

    field_t X, Y;
    bool inf;

    inline __host__ __device__ bool is_inf() const
    {   return inf;   }

public:
    inline __host__ __device__ operator Affine() const
    {
        bool inf = is_inf();
        Affine p;
        p.X = czero(X, inf);
        p.Y = czero(Y, inf);
        return p;
    }

    inline __host__ __device__ operator jacobian() const
    {
        jacobian p;
        p.X = X;
        p.Y = Y;
        p.Z = field_t::one(is_inf());
        return p;
    }

    inline __host__ __device__ operator xyzz() const
    {
        xyzz p;
        p.X = X;
        p.Y = Y;
        p.ZZZ = p.ZZ = field_t::one(is_inf());
        return p;
    }

#ifdef __CUDACC__
    class mem_t {
        field_h X, Y;
        int inf[sizeof(field_t)%16 ? 2 : 4];

        inline __device__ bool is_inf() const
        {   return inf[0]&1 != 0;   }

    public:
        inline __device__ operator Affine() const
        {
            bool inf = is_inf();
            Affine p;
            p.X = czero((field_t)X, inf);
            p.Y = czero((field_t)Y, inf);
            return p;
        }

        inline __device__ operator Affine_inf_t() const
        {
            bool inf = is_inf();
            Affine_inf_t p;
            p.X = czero((field_t)X, inf);
            p.Y = czero((field_t)Y, inf);
            p.inf = inf;
            return p;
        }
    };
#else
    using mem_t = Affine_inf_t;
#endif
};
#endif
