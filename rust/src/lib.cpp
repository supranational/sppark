#include <cuda_runtime.h>
#include <util/gpu_t.cuh>

extern "C" void drop_gpu_ptr_t(gpu_ptr_t<void>& ref)
{   ref.~gpu_ptr_t();   }

extern "C" void clone_gpu_ptr_t(gpu_ptr_t<void>& ret, const gpu_ptr_t<void>& rhs)
{   ret = rhs;   }
