// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __EXCEPTION_CUH__
#define __EXCEPTION_CUH__

#include <cstdio>
#include <string>
#include <stdexcept>

class cuda_error : public std::runtime_error {
    cudaError_t _code;
public:
    cuda_error(cudaError_t err, const std::string& reason) : std::runtime_error{reason}
    {   _code = err;   }
    inline cudaError_t code() const
    {   return _code;   }
};

template<typename... Types>
inline std::string fmt(const char* fmt, Types... args)
{
    size_t len = std::snprintf(nullptr, 0, fmt, args...);
    std::string ret(++len, '\0');
    std::snprintf(&ret.front(), len, fmt, args...);
    ret.resize(--len);
    return ret;
}

#define CUDA_OK(expr) do {                                  \
    cudaError_t code = expr;                                \
    if (code != cudaSuccess) {                              \
        auto str = fmt("%s@%d failed: %s", #expr, __LINE__, \
                       cudaGetErrorString(code));           \
        throw cuda_error(code, str);                        \
    }                                                       \
} while(0)

#endif
