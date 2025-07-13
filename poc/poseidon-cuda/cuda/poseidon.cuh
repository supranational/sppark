#include <cuda.h>

#if defined(FEATURE_BLS12_381)
# include <ff/bls12-381-fp2.hpp>
#elif defined(FEATURE_BLS12_377)
# include <ff/bls12-377-fp2.hpp>
#else
# error "no FEATURE"
#endif

/// @brief Poseidon hash state structure
///
/// This structure holds the state of the Poseidon hash, including the capacity and rate states,
/// as well as the current round index. It is used to manage the internal state during the
/// calculation of the Posiedon permutation.
///
/// @tparam FieldT The field type used in the Poseidon hash, typically a finite field.
/// @tparam RateV The number of elements in the rate state.
/// @tparam CapacityV The number of elements in the capacity state.
template<typename FieldT, uint32_t RateV, uint32_t CapacityV>
struct PoseidonState {

    /// @brief An alias for the field type used in the Poseidon hash
    using FieldType = FieldT;

    // Operator [] to access elements in the state
    __device__ __forceinline__
    FieldType& operator[](uint32_t index) {
        
        // Demultiplex the index to access either capacity or rate state
        if (index < CapacityV) {
            return capacity_state[index];
        } else {
            return rate_state[index - CapacityV];
        }
    }

    /// @brief Capacity state for the Poseidon hash
    FieldType capacity_state[CapacityV];

    /// @brief Rate state for the Poseidon hash
    FieldType rate_state[RateV];
};


template<typename FieldT, uint32_t RateV, uint32_t CapacityV, uint32_t FullRoundsV, uint32_t PartialRoundsV, uint32_t AlphaV>
struct PoseidonParameters {

    /// @brief An alias for the field type used in the Poseidon hash
    using FieldType = FieldT;

    /// @brief The number of elements in the rate state
    static constexpr uint32_t RateSize = RateV;

    /// @brief The number of elements in the capacity state
    static constexpr uint32_t CapacitySize = CapacityV;

    /// @brief The total number of elements in the state (rate + capacity)
    static constexpr uint32_t StateSize = RateSize + CapacitySize;

    /// @brief The number of full rounds in the Poseidon permutation
    static constexpr uint32_t FullRounds = FullRoundsV;

    /// @brief The number of full rounds in the first (or last) part of the Poseidon permutation 
    static constexpr uint32_t FullRoundsOver2 = FullRounds / 2;

    /// @brief The number of partial rounds in the Poseidon permutation
    static constexpr uint32_t PartialRounds = PartialRoundsV;

    /// @brief The total number of rounds in the Poseidon permutation (full + partial)
    static constexpr uint32_t TotalRounds = FullRounds + PartialRounds;

    /// @brief The exponent used in the S-Box function
    static constexpr uint32_t Alpha = AlphaV;

    // Additive round keys
    FieldType ark[TotalRounds][StateSize];

    // Maximally Distance Separating (MDS) matrix
    FieldType mds[StateSize][StateSize];
};


/// TODO: 
/// 1- Here we assume that a block is repsonisble for a single state and 
/// after the sync, the state has all s-box oprations done on it. We need to 
/// limit the launching of this kernel to blocks that are responsible for a single state.
/// i stated it below with __launch_bounds__ but it needes to aligned to standard grid sizes (the +% trick).
///
/// 2- IF there is space lelft in the constant memory, move params to __constant__ memory space. This requires extracting
/// ark and mds as seprate __constant__ since they are less templated (remember: __constant__ cannot be templated) and 
/// will also requilre fixing the field type in the file.
template<typename PoseidonStateT, typename PoseidonParametersT>
__launch_bounds__(PoseidonParametersT::StateSize)
__global__ 
void CUDA_Poseidon_Permutation(PoseidonStateT* state, const PoseidonParametersT* params) {
    // Alias for the field type used in the Poseidon hash
    using FieldType = typename PoseidonParametersT::FieldType;

    // Ensure the thread is within bounds
    if (threadIdx.x >= PoseidonParametersT::StateSize) {
        return;
    }

    // Shared memory for the state and the parameters
    __shared__ PoseidonStateT shared_state;
    __shared__ PoseidonParametersT shared_params;

    // Load the state and parameters into shared memory
    shared_state[threadIdx.x] = (*state)[threadIdx.x];
    for (uint32_t i = 0; i < PoseidonParametersT::TotalRounds; ++i) {
        shared_params.ark[i][threadIdx.x] = params->ark[i][threadIdx.x];
    }
    for (uint32_t i = 0; i < PoseidonParametersT::StateSize; ++i) {
        shared_params.mds[threadIdx.x][i] = params->mds[threadIdx.x][i];
    }

    // Ensure all threads have loaded their data into shared memory
    __syncthreads();

    #pragma unroll
    for (uint32_t round = 0; round < PoseidonParametersT::TotalRounds; ++round) {

        // Apply the round key addition
        shared_state[threadIdx.x] += shared_params.ark[round][threadIdx.x];

        // Apply the S-Box function
        if(round < PoseidonParametersT::FullRoundsOver2  || round >= PoseidonParametersT::TotalRounds - PoseidonParametersT::FullRoundsOver2) {
            shared_state[threadIdx.x] ^= PoseidonParametersT::Alpha;
        } else {
            if (threadIdx.x == 0) {
                shared_state[threadIdx.x] ^= PoseidonParametersT::Alpha;
            }
        }

        // Sync threads to ensure all threads have completed their operations
        __syncthreads();

        // Apply the MDS matrix multiplication corrsponding to the current state element
        FieldType matrix_mult_result = FieldType{};
        matrix_mult_result.zero();
        for (uint32_t i = 0; i < PoseidonParametersT::StateSize; ++i) {
            matrix_mult_result += shared_params.mds[threadIdx.x][i] * shared_state[i];
        }

        // Sync threads to ensure all threads have completed the MDS multiplication
        __syncthreads();

        // Store the result back into the rate state
        shared_state[threadIdx.x] = matrix_mult_result;

        // Sync threads to ensure all threads have completed their operations before starting the next round
        __syncthreads();
    }

    // Write the final shared state back to global memory
    (*state)[threadIdx.x] = shared_state[threadIdx.x];
}
