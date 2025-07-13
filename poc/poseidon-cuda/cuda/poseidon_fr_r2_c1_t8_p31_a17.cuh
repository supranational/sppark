#include "poseidon.cuh"

// Define the types for the values fixed at this header
using PoseidonStateType = PoseidonState<fr_t, 2, 1>;
using PoseidonParametersType = PoseidonParameters<fr_t, 2, 1, 8, 31, 17>;

static_assert(sizeof(fr_t) == 4*sizeof(uint64_t), "For Posidon_fr_r2_c1_f8_p31_a17, fr_t is expected to be 4*64 bits");
