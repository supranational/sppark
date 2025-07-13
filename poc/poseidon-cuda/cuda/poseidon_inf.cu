#include <iostream>
#include <iomanip>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

/// TODO: This include to be done based on a switching parameter passed at compile time
#include "poseidon_fr_r2_c1_t8_p31_a17.cuh"



// Check PoseidonStateType size is as expected
static_assert(
  sizeof(PoseidonStateType) == (PoseidonParametersType::RateSize + PoseidonParametersType::CapacitySize) * sizeof(PoseidonParametersType::FieldType), 
  "PoseidonStateType size is not as expected");

// Check PosidonPramaetersType size is as expected
static_assert(
  sizeof(PoseidonParametersType) == sizeof(PoseidonParametersType::FieldType) *
          (PoseidonParametersType::StateSize*PoseidonParametersType::TotalRounds + 
          PoseidonParametersType::StateSize*PoseidonParametersType::StateSize), 
          "PoseidonParametersType size is not as expected");

/// TODO: Make sure when passing the state/params from Rust that:
/// (1) member data are same types/sizes, 
/// (2) members are at same order, and 
/// (3) the structs have same padding/alignment.
SPPARK_FFI
RustError::by_value cuda_poseidon_permuration(size_t device_id,
                                              void* p_inout_state,
                                              const void* p_params)
{
    // Check if the pointers are null
    if (!p_inout_state || !p_params) {
        std::cerr << "Error: Null pointer passed for state or parameters." << std::endl;
        return RustError{cudaErrorInvalidValue};
    }

    /// Cast provided objects to state/params types
    PoseidonStateType* h_state = reinterpret_cast<PoseidonStateType*>(p_inout_state);
    PoseidonParametersType* h_params = reinterpret_cast<PoseidonParametersType*>(const_cast<void*>(p_params));

    // Get targeted GPU device
    auto& gpu = select_gpu(device_id);

    try {

        // Activate current GPU device
        gpu.select();

        // Allocate device memory for state and parameters
        dev_ptr_t<PoseidonStateType> d_state{1, gpu};
        dev_ptr_t<PoseidonParametersType> d_params{1, gpu};

        // Copy host data to device memory
        gpu.HtoD(&d_state[0], h_state, 1);
        gpu.HtoD(&d_params[0], h_params, 1);

        // Call the CUDA Kernel CUDA_Poseidon_Permutation
        CUDA_Poseidon_Permutation<<<1, PoseidonParametersType::StateSize, 0, gpu>>>(&d_state[0], &d_params[0]);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA_Poseidon_Permutation CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
        }

        // Copy back the result from device to host
        gpu.DtoH(h_state, &d_state[0], 1);

        // Sync the GPU to ensure all operations are complete
        gpu.sync();

    } catch (const cuda_error& e) {
        gpu.sync();
        return RustError{e.code(), e.what()};
    }

    // dummy return with success for now
    return RustError{cudaSuccess};
}