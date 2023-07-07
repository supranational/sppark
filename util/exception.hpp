// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_EXCEPTION_HPP__
#define __SPPARK_UTIL_EXCEPTION_HPP__

#include <cstdio>
#include <cstring>
#include <string>
#include <stdexcept>

class sppark_error : public std::runtime_error {
    int _code;
public:
    sppark_error(int err, const std::string& reason) : std::runtime_error{reason}
    {   _code = err;   }
    inline int code() const
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

template<typename... Types>
inline std::string fmt_errno(int errnum, const char* fmt, Types... args)
{
    const size_t ERRLEN = 48;
    size_t len = std::snprintf(nullptr, 0, fmt, args...);
    std::string ret(len + ERRLEN, '\0');
    std::snprintf(&ret[0], len + 1, fmt, args...);
    auto errmsg = &ret[len];
#if defined(_WIN32)
    (void)strerror_s(errmsg, ERRLEN, errnum);
#elif defined(_GNU_SOURCE)
    auto errstr = strerror_r(errnum, errmsg, ERRLEN);
    if (errstr != errmsg)
        strncpy(errmsg, errstr, ERRLEN - 1);
#else
    (void)strerror_r(errnum, errmsg, ERRLEN);
#endif
    ret.resize(len + std::strlen(errmsg));
    return ret;
}

inline std::string fmt_errno(int errnum, const char* msg = "")
{
    const size_t ERRLEN = 48;
    size_t len = std::strlen(msg);
    std::string ret(len + ERRLEN, '\0');
    std::strcpy(&ret[0], msg);
    auto errmsg = &ret[len];
#if defined(_WIN32)
    (void)strerror_s(errmsg, ERRLEN, errnum);
#elif defined(_GNU_SOURCE)
    auto errstr = strerror_r(errnum, errmsg, ERRLEN);
    if (errstr != errmsg)
        strncpy(errmsg, errstr, ERRLEN - 1);
#else
    (void)strerror_r(errnum, errmsg, ERRLEN);
#endif
    ret.resize(len + std::strlen(errmsg));
    return ret;
}

#endif
