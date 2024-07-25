#include <stdio.h>
#include <string.h>
#ifdef _MSC_VER
# define strdup _strdup
#endif

__global__ void kernel()
{
    printf("hello from GPU\n");
}

struct Error {
    int code;
    char *message;
};

extern "C"
#ifdef _WIN32
__declspec(dllexport)
#else
__attribute__((visibility("default")))
#endif
Error cuda_func(void *ptr)
{
    kernel<<<1,1>>>();
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        return {err, strdup(cudaGetErrorString(err))};
    err = cudaDeviceSynchronize();
    return {err, strdup(cudaGetErrorString(err))};
    (void)ptr;
}
