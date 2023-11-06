// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <vector>

// for mult function
#include <msm/pippenger.hpp>

template<class point_t, class affine_t>
static void mult(point_t& ret, const affine_t point, const fr_t& fr,
                 size_t top = fr_t::nbits)
{
#ifndef __CUDA_ARCH__
    fr_t::pow_t scalar;
    fr.to_scalar(scalar);

    mult(ret, point, scalar, top);
#endif
}

static point_t msm_naive(const std::vector<affine_t>& points,
                         const std::vector<scalar_t>& scalars)
{
    assert(points.size() == scalars.size());

    point_t ret; ret.inf();
    for (size_t i = 0; i < points.size(); i++) {
        point_t res;
        mult(res, points[i], scalars[i]);
        ret.add(res);
    }

    return ret;
}

static affine_t get_g1_generator() {
    affine_t ret;

#ifndef __CUDA_ARCH__
    ret = affine_t(fp_t(BLS12_377_G1_GEN_X), fp_t(BLS12_377_G1_GEN_Y));
#endif

    return ret;
}

static affine_t generate_g1_point(fr_t fr) {
    affine_t g1_generator = get_g1_generator();
    point_t res;

    mult(res, g1_generator, fr);

    return res;
}
