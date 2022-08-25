#include <cuda_runtime.h>
#include "gpu_t.cuh"

class gpus_t {
    std::vector<const gpu_t*> gpus;
public:
    gpus_t()
    {
        int n;
        if (cudaGetDeviceCount(&n) != cudaSuccess)
            return;
        for (int id = 0; id < n; id++) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, id) == cudaSuccess &&
                prop.major >= 7) {
                cudaSetDevice(id);
                gpus.push_back(new gpu_t(gpus.size(), id, prop));
            }
        }
        cudaSetDevice(0);
    }
    ~gpus_t()
    {   for (auto* ptr: gpus) delete ptr;   }

    static const auto& all()
    {
        static gpus_t all_gpus;
        return all_gpus.gpus;
    }
};

const gpu_t& select_gpu(int id)
{
    auto* gpu = gpus_t::all()[id];
    gpu->select();
    return *gpu;
}

size_t ngpus()
{   return gpus_t::all().size();   }

const std::vector<const gpu_t*>& all_gpus()
{   return gpus_t::all();   }

extern "C" bool cuda_available()
{   return gpus_t::all().size() != 0;   }
