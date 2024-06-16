// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifdef __CUDACC__
# include "gl64_t.cuh"  // CUDA device-side field types
typedef gl64_t fr_t;
#elif defined(__HIPCC__)
# include "gl64_t.hip"
typedef gl64_t fr_t;
#endif
