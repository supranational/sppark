// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifdef __HIPCC__

#include <hip/hip_runtime.h>

static const auto cudaGetDeviceCount        = hipGetDeviceCount;
static const auto cudaGetDevice             = hipGetDevice;
static const auto cudaSetDevice             = hipSetDevice;
static const auto cudaDeviceSynchronize     = hipDeviceSynchronize;

using cudaDeviceProp                        = hipDeviceProp_t;
static const auto cudaGetDeviceProperties   = hipGetDeviceProperties;
static const auto cudaMemGetInfo            = hipMemGetInfo;

using cudaMemcpyKind                        = hipMemcpyKind;
static const auto cudaMemcpy                = hipMemcpy;
static const auto cudaMemcpyAsync           = hipMemcpyAsync;
static const auto cudaMemcpy2DAsync         = hipMemcpy2DAsync;
#define           cudaMemcpyHostToDevice      hipMemcpyHostToDevice
#define           cudaMemcpyDeviceToHost      hipMemcpyDeviceToHost
#define           cudaMemcpyDeviceToDevice    hipMemcpyDeviceToDevice
static const auto cudaMemsetAsync           = hipMemsetAsync;

using cudaError_t                           = hipError_t;
static const auto cudaGetLastError          = hipGetLastError;
static const auto cudaGetErrorString        = hipGetErrorString;
#define           cudaSuccess                 hipSuccess
#define           cudaErrorNoDevice           hipErrorNoDevice

using cudaEvent_t                           = hipEvent_t;
static const auto cudaEventCreate           = hipEventCreate;
static const auto cudaEventCreateWithFlags  = hipEventCreateWithFlags;
#define           cudaEventDisableTiming      hipEventDisableTiming
static const auto cudaEventRecord           = hipEventRecord;
static const auto cudaEventDestroy          = hipEventDestroy;

using cudaStream_t                          = hipStream_t;
static const auto cudaStreamCreateWithFlags = hipStreamCreateWithFlags;
#define           cudaStreamNonBlocking       hipStreamNonBlocking
static const auto cudaStreamDestroy         = hipStreamDestroy;
static const auto cudaStreamSynchronize     = hipStreamSynchronize;
static const auto cudaStreamWaitEvent       = hipStreamWaitEvent;

using cudaHostFn_t                          = hipHostFn_t;
static const auto cudaLaunchHostFunc        = hipLaunchHostFunc;

template<typename T>
static inline cudaError_t cudaMalloc(T** devPtr, size_t size)
{   return hipMalloc(devPtr, size);   }
static const auto cudaFree                  = hipFree;

template<typename T>
static inline cudaError_t cudaMallocAsync(T** devPtr, size_t size,
                                          cudaStream_t stream)
{   return hipMallocAsync(devPtr, size, stream);   }
static const auto cudaFreeAsync             = hipFreeAsync;

template<typename T>
static inline cudaError_t cudaMallocManaged(T** uniPtr, size_t size)
{   return hipMallocManaged(uniPtr, size);   }

#define cudaHostAllocDefault                  hipHostMallocDefault
#define cudaHostAllocPortable                 hipHostMallocPortable
#define cudaHostAllocaMapped                  hipHostMallocMapped
#define cudaHostAllocWriteCombined            hipHostMallocWriteCombined
template<typename T>
static inline cudaError_t cudaHostAlloc(T** pinnedPtr, size_t size,
                                        unsigned int flags = hipHostMallocDefault)
{   return hipHostMalloc(pinnedPtr, size, flags);   }
template<typename T>
static inline cudaError_t cudaMallocHost(T** pinnedPtr, size_t size)
{   return hipHostMalloc(pinnedPtr, size, hipHostMallocDefault);   }
static const auto cudaFreeHost              = hipHostFree;

static const auto cudaHostGetDevicePointer  = hipHostGetDevicePointer;
static const auto cudaHostGetFlags          = hipHostGetFlags;

template<typename T>
static inline cudaError_t cudaGetSymbolAddress(void** devPtr, const T& symbol)
{   return hipGetSymbolAddress(devPtr, symbol);   }

template<typename T>
static inline cudaError_t
cudaOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
                                   size_t dynamicSMemSize = 0,
                                   int blockSizeLimit = 0)
{   return hipOccupancyMaxPotentialBlockSize(minGridSize, blockSize, func,
                                             dynamicSMemSize, blockSizeLimit);
}

using cudaFuncAttributes                    = hipFuncAttributes;

template<typename T>
static inline cudaError_t
cudaFuncGetAttributes(cudaFuncAttributes* attr, T func)
{   return hipFuncGetAttributes(attr, reinterpret_cast<const void*>(func));   }

template<typename T>
static inline cudaError_t
cudaLaunchCooperativeKernel(const T* func, dim3 gridDim, dim3 blockDim,
                            void** args, size_t sharedMem = 0,
                            cudaStream_t stream = 0)
{   return hipLaunchCooperativeKernel(func, gridDim, blockDim, args, sharedMem,
                                      stream);
}

static inline __device__ void __syncwarp() { asm volatile(""); }

#endif
