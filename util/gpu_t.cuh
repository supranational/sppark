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
    const int gpu_id;
public:
    stream_t(int id) : gpu_id(id)
    {   cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);   }
    ~stream_t()
    {   cudaStreamDestroy(stream);   }
    inline operator decltype(stream)() const    { return stream; }
    inline int id() const                       { return gpu_id; }
    inline operator int() const                 { return gpu_id; }

    inline void* Dmalloc(size_t sz) const
    {   void *d_ptr;
        CUDA_OK(cudaMallocAsync(&d_ptr, sz, stream));
        return d_ptr;
    }
    inline void Dfree(void* d_ptr) const
    {   CUDA_OK(cudaFreeAsync(d_ptr, stream));   }

    template<typename T>
    inline void HtoD(T* dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   if (sz == sizeof(T))
            CUDA_OK(cudaMemcpyAsync(dst, src, nelems*sizeof(T),
                                    cudaMemcpyHostToDevice, stream));
        else
            CUDA_OK(cudaMemcpy2DAsync(dst, sizeof(T), src, sz, sz, nelems,
                                      cudaMemcpyHostToDevice, stream));
    }
    template<typename T>
    inline void HtoD(T& dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   HtoD(&dst, src, nelems, sz);   }
    template<typename T, typename U>
    inline void HtoD(T& dst, const std::vector<U> src,
                     size_t sz = sizeof(T)) const
    {   HtoD(&dst, &src[0], src.size(), sz);   }

    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim,
                                                size_t shared_sz,
                            Types... args) const
    {   void* va_args[sizeof...(args)] = { &args... };
        CUDA_OK(cudaLaunchCooperativeKernel((const void*)f, gridDim, blockDim,
                                            va_args, shared_sz, stream));
    }

    template<typename T>
    inline void DtoH(T* dst, const void* src, size_t nelems) const
    {   CUDA_OK(cudaMemcpyAsync(dst, src, nelems*sizeof(T),
                                cudaMemcpyDeviceToHost, stream));
    }
    template<typename T>
    inline void DtoH(T& dst, const void* src, size_t nelems) const
    {   DtoH(&dst, src, nelems);   }
    template<typename T>
    inline void DtoH(std::vector<T> dst, const void* src) const
    {   DtoH(&dst[0], src, dst.size());   }

    inline void sync() const
    {   CUDA_OK(cudaStreamSynchronize(stream));   }
};

class gpu_t {
public:
    static const size_t FLIP_FLOP = 3;
private:
    int gpu_id, cuda_id;
    cudaDeviceProp prop;
    size_t total_mem;
    mutable stream_t zero = {gpu_id};
    mutable stream_t flipflop[FLIP_FLOP] = {gpu_id, gpu_id, gpu_id};
    mutable thread_pool_t pool;

public:
    gpu_t(int id, int real_id, const cudaDeviceProp& p)
    : gpu_id(id), cuda_id(real_id), prop(p)
    {   size_t freeMem;
        CUDA_OK(cudaMemGetInfo(&freeMem, &total_mem));
    }

    inline int id() const                   { return gpu_id; }
    inline operator int() const             { return gpu_id; }
    inline const auto& props() const        { return prop; }
    inline int sm_count() const             { return prop.multiProcessorCount; }
    inline void select() const              { cudaSetDevice(cuda_id); }
    stream_t& operator[](size_t i) const    { return flipflop[i%FLIP_FLOP]; }
    inline operator stream_t&() const       { return zero; }
    inline operator cudaStream_t() const    { return zero; }

    inline size_t ncpus() const             { return pool.size(); }
    template<class Workable>
    inline void spawn(Workable work) const  { pool.spawn(work); }

    inline void* Dmalloc(size_t sz) const
    {   void *d_ptr = zero.Dmalloc(sz);
        zero.sync();
        return d_ptr;
    }
    inline void Dfree(void* d_ptr) const
    {   zero.Dfree(d_ptr);
        zero.sync();
    }

    template<typename T>
    inline void HtoD(T* dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   zero.HtoD(dst, src, nelems, sz);   }
    template<typename T>
    inline void HtoD(T& dst, const void* src, size_t nelems,
                     size_t sz = sizeof(T)) const
    {   HtoD(&dst, src, nelems, sz);   }
    template<typename T, typename U>
    inline void HtoD(T& dst, const std::vector<U> src,
                     size_t sz = sizeof(T)) const
    {   HtoD(&dst, &src[0], src.size(), sz);   }

    template<typename... Types>
    inline void launch_coop(void(*f)(Types...), dim3 gridDim, dim3 blockDim,
                                                size_t shared_sz,
                            Types... args) const
    {   zero.launch_coop(f, gridDim, blockDim, shared_sz,
                         args...);
    }

    template<typename T>
    inline void DtoH(T* dst, const void* src, size_t nelems) const
    {   zero.DtoH(dst, src, nelems);   }
    template<typename T>
    inline void DtoH(T& dst, const void* src, size_t nelems) const
    {   DtoH(&dst, src, nelems);   }
    template<typename T>
    inline void DtoH(std::vector<T> dst, const void* src) const
    {   DtoH(&dst[0], src, dst.size());   }

    inline void sync() const
    {
        zero.sync();
        for (auto& f : flipflop)
            f.sync();
    }
};

size_t ngpus();
const gpu_t& select_gpu(int id = 0);
const std::vector<const gpu_t*>& all_gpus();
extern "C" bool cuda_available();

template<typename T> class gpu_ptr_t {
    struct inner {
        const T* ptr;
        std::atomic<size_t> ref_cnt;
        int real_id;
        inline inner(const T* p) : ptr(p), ref_cnt(1)
        {   cudaGetDevice(&real_id);   }
    };
    inner *ptr;
public:
    gpu_ptr_t(const T* p)         { ptr = new inner(p); }
    gpu_ptr_t(const gpu_ptr_t& r) { *this = r; }
    ~gpu_ptr_t()
    {
        if (ptr && ptr->ref_cnt.fetch_sub(1, std::memory_order_seq_cst) == 1) {
            int current_id;
            cudaGetDevice(&current_id);
            if (current_id != ptr->real_id)
                cudaSetDevice(ptr->real_id);
            cudaFree((void*)ptr->ptr);
            if (current_id != ptr->real_id)
                cudaSetDevice(current_id);
            delete ptr;
        }
    }

    gpu_ptr_t& operator=(const gpu_ptr_t& r)
    {
        if (this != &r)
            (ptr = r.ptr)->ref_cnt.fetch_add(1, std::memory_order_relaxed);
        return *this;
    }
    gpu_ptr_t& operator=(gpu_ptr_t&& r) noexcept
    {
        if (this != &r) {
            ptr = r.ptr;
            r.ptr = nullptr;
        }
        return *this;
    }

    inline operator const T*() const            { return ptr->ptr; }
    inline operator void*() const               { return (void*)ptr->ptr; }
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
