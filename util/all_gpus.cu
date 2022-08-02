#include "gpu_t.cuh"

class gpus_t {
    std::vector<gpu_t*> gpus;
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
                gpus.push_back(new gpu_t(id, prop));
            }
        }
        cudaSetDevice(0);
    }
    ~gpus_t()
    {   for (auto* ptr: gpus) delete ptr;   }

    inline gpu_t* operator[](size_t i) const
    {   return gpus[i];   }
    inline size_t ngpus() const
    {   return gpus.size();   }

    static gpus_t& all()
    {
        static gpus_t all_gpus;
        return all_gpus;
    }
};

gpu_t& select_gpu(int id)
{
    cudaSetDevice(id);
    return *gpus_t::all()[id];
}

size_t ngpus()
{   return gpus_t::all().ngpus();   }

extern "C" bool cuda_available() { return gpus_t::all().ngpus() != 0; }
