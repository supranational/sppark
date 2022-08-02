// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __GPU_T_CUH__
#define __GPU_T_CUH__

#include "thread_pool_t.hpp"
#include "exception.cuh"

#ifndef WARP_SZ
# define WARP_SZ 32
#endif

class stream_t {
    cudaStream_t stream;
public:
    stream_t()  { cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking); }
    ~stream_t() { cudaStreamDestroy(stream); }
    inline operator decltype(stream)() { return stream; }

    template<typename T>
    inline void HtoD(T* dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T))
    {   if (sz == sizeof(T))
            CUDA_OK(cudaMemcpyAsync(dst, src, nelems*sizeof(T),
                                    cudaMemcpyHostToDevice, stream));
        else
            CUDA_OK(cudaMemcpy2DAsync(dst, sizeof(T), src, sz, sz, nelems,
                                      cudaMemcpyHostToDevice, stream));
    }
    template<typename T>
    inline void HtoD(T& dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T))
    {   HtoD(&dst, src, nelems, sz);   }

    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim,
                                                size_t shared_sz,
                            Types... args)
    {   void* va_args[sizeof...(args)] = { &args... };
        CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim,
                                            va_args, shared_sz, stream));
    }

    template<typename T>
    inline void DtoH(T* dst, const void* src, size_t nelems)
    {   CUDA_OK(cudaMemcpyAsync(dst, src, nelems*sizeof(T),
                                cudaMemcpyDeviceToHost, stream));
    }
    template<typename T>
    inline void DtoH(T& dst, const void* src, size_t nelems)
    {   DtoH(&dst, src, nelems);   }

    inline void sync() const
    {   CUDA_OK(cudaStreamSynchronize(stream));   }
};

class gpu_t {
    int device_id;
    cudaDeviceProp prop;
    size_t total_mem;
    mutable stream_t zero, flipflop[2];
    mutable thread_pool_t pool;

    inline static cudaDeviceProp props(int id)
    {   cudaDeviceProp prop;
        CUDA_OK(cudaGetDeviceProperties(&prop, id));
        cudaSetDevice(id);
        return prop;
    }

public:
    gpu_t(int id, const cudaDeviceProp& p) : device_id(id), prop(p)
    {   size_t freeMem;
        CUDA_OK(cudaMemGetInfo(&freeMem, &total_mem));
    }
    gpu_t(int id = 0) : gpu_t(id, props(id)) {}

    inline int id() const               { return device_id; }
    inline int sm_count() const         { return prop.multiProcessorCount; }
    inline void select() const          { cudaSetDevice(device_id); }
    stream_t& operator[](size_t i)      { return flipflop[i&1]; }

    inline size_t ncpus() const         { return pool.size(); }
    template<class Workable>
    inline void spawn(Workable work)    { pool.spawn(work); }

    inline void* Dmalloc(size_t sz)
    {   void *d_ptr;
        CUDA_OK(cudaMalloc(&d_ptr, sz));
        return d_ptr;
    }
    inline void Dfree(void* d_ptr)
    {   cudaFree(d_ptr);   }

    template<typename T>
    inline void HtoD(T* dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T))
    {   zero.HtoD(dst, src, sz, nelems);   }
    template<typename T>
    inline void HtoD(T& dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T))
    {   HtoD(&dst, src, nelems, sz);   }

    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim,
                                                size_t shared_sz,
                            Types... args)
    {   zero.launch_coop((const void*)f, gridDim, blockDim, shared_sz,
                         args...);
    }

    template<typename T>
    inline void DtoH(T* dst, const void* src, size_t nelems)
    {   zero.DtoH(dst, src, nelems);   }
    template<typename T>
    inline void DtoH(T& dst, const void* src, size_t nelems)
    {   DtoH(&dst, src, nelems);   }

    inline void sync() const
    {   zero.sync();   }
};


size_t ngpus();
gpu_t& select_gpu(int id = 0);
extern "C" bool cuda_available();

template<typename T> class gpu_ptr_t {
    struct inner {
        const T* ptr;
        std::atomic<size_t> ref_cnt;
        int device_id;
        inline inner(const T* p, int id) : ptr(p), ref_cnt(1), device_id(id) {};
    };
    inner *ptr;
public:
    gpu_ptr_t(const T* p, int id) { ptr = new inner(p, id); }
    gpu_ptr_t(const gpu_ptr_t& r)
    {   (ptr = r.ptr)->ref_cnt.fetch_add(1, std::memory_order_relaxed);   }
    ~gpu_ptr_t()
    {
        if (ptr->ref_cnt.fetch_sub(1, std::memory_order_seq_cst) == 1) {
            int current_id;
            cudaGetDevice(&current_id);
            if (current_id != ptr->device_id)
                cudaSetDevice(ptr->device_id);
            cudaFree(ptr->ptr);
            if (current_id != ptr->device_id)
                cudaSetDevice(current_id);
            delete ptr;
        }
    }

    gpu_ptr_t& operator=(const gpu_ptr_t& r) = delete;
    gpu_ptr_t& operator=(gpu_ptr_t&& r) noexcept
    {
        if (this != &r) {
            ptr = r.ptr;
            r.ptr = nullptr;
        }
        return *this;
    }

    inline operator T*() const                  { return ptr->ptr; }
    inline operator void*() const               { return (void*)ptr->ptr; }
    inline const T& operator[](size_t i) const  { return ptr->ptr[i]; }
    inline T& operator[](size_t i)              { return ptr->ptr[i]; }
};

// A simple way to pack a pointer and array's size length, but more
// importantly...
template<typename T> class vec_t {
    const T* ptr;
    size_t nelems;
public:
    vec_t(const T* p, size_t n) : ptr(p), nelems(n) {}

    inline operator decltype(ptr)() const       { return ptr; }
    inline operator void*() const               { return (void*)ptr; }
    inline size_t size() const                  { return nelems; }
    inline const T& operator[](size_t i) const  { return ptr[i]; }
};

// ... pin the buffer to physical memory. For example, if a function accepts
// vec_t<T> one can pass pin_t<T>{ptr, nelems} to pin the memory for the
// duration of the call.
template<typename T> class pin_t : public vec_t<T> {
public:
    pin_t(const T* p, size_t n) : vec_t<T>(p, n)
    {   CUDA_OK(cudaHostRegister(*this, n*sizeof(T),
                                 cudaHostRegisterPortable|cudaHostRegisterReadOnly));
    }
    ~pin_t()
    {   cudaHostUnregister(*this);   }
};

// A simple way to allocate a temporary device pointer without having to
// care about freeing it.
template<typename T> class dev_ptr_t {
    T* d_ptr;
public:
    dev_ptr_t(size_t nelems) : d_ptr(nullptr)
    {
        if (nelems) {
            size_t n = (nelems+WARP_SZ-1) & ((size_t)0-WARP_SZ);
            CUDA_OK(cudaMalloc(&d_ptr, n * sizeof(T)));
        }
    }
    ~dev_ptr_t() { if (d_ptr) cudaFree((void*)d_ptr); }

    inline operator const T*() const            { return d_ptr; }
    inline operator T*() const                  { return d_ptr; }
    inline operator void*() const               { return (void*)d_ptr; }
    inline const T& operator[](size_t i) const  { return d_ptr[i]; }
    inline T& operator[](size_t i)              { return d_ptr[i]; }
};

#endif
