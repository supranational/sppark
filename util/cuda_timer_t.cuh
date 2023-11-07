// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_UTIL_CUDA_TIMER_T_CUH__
#define __SPPARK_UTIL_CUDA_TIMER_T_CUH__

#include <iostream>
#include <iomanip>

class cuda_timer_t {
    cudaEvent_t begin, end;
public:
    cuda_timer_t() : begin(nullptr), end(nullptr) {
        CUDA_OK(cudaEventCreate(&begin));
        CUDA_OK(cudaEventCreate(&end));
    }

    ~cuda_timer_t() {
        if (begin) cudaEventDestroy(begin);
        if (end) cudaEventDestroy(end);
    }

    inline void start() {
        CUDA_OK(cudaEventRecord(begin, 0));
    }

    inline void start(cudaStream_t stream) {
        CUDA_OK(cudaEventRecord(begin, stream));
    }

    inline void stop(const std::string str) {
        float elapsed;

        CUDA_OK(cudaEventRecord(end, 0));
        CUDA_OK(cudaEventSynchronize(end));
        CUDA_OK(cudaEventElapsedTime(&elapsed, begin, end));

        std::cout << str << ": " << std::fixed << std::setprecision(3)
                  << elapsed << " ms" << std::endl;
    }

    inline void stop(cudaStream_t stream, const std::string str) {
        float elapsed;

        CUDA_OK(cudaEventRecord(end, stream));
        CUDA_OK(cudaEventSynchronize(end));
        CUDA_OK(cudaEventElapsedTime(&elapsed, begin, end));

        std::cout << str << ": " << std::fixed << std::setprecision(3)
                  << elapsed << " ms" << std::endl;
    }

    inline float get_elapsed() {
        float elapsed;

        CUDA_OK(cudaEventRecord(end, 0));
        CUDA_OK(cudaEventSynchronize(end));
        CUDA_OK(cudaEventElapsedTime(&elapsed, begin, end));

        return elapsed;
    }

    inline float get_elapsed(cudaStream_t stream) {
        float elapsed;

        CUDA_OK(cudaEventRecord(end, stream));
        CUDA_OK(cudaEventSynchronize(end));
        CUDA_OK(cudaEventElapsedTime(&elapsed, begin, end));

        return elapsed;
    }
};

#endif
