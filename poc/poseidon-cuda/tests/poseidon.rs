use ark_ff::Zero;

use poseidon_cuda::*;

#[cfg(any(feature = "bls12_381", feature = "bls12_377"))]
#[test]
fn poseidon_permutation_fr_r2_c1_t8_p31_a17_correctness() {

    // Zero-initialized state (scalar field)
    let poseidon_state = util::poseidon_initial_state_r2_c1();

    // Assert that the state is zero-initialized
    for f_i in 0..poseidon_state.state.len() {
        let field_element = poseidon_state.state[f_i];
        assert_eq!(field_element.is_zero(), true, "Poseidon state is not zero-initialized at index {}", f_i);
    }

    // Some parameters for the Poseidon permutation
    let params = util::poseidon_params_fr_r2_c1_t8_p31_a17();

    // Assert paramers are consistent
    assert_eq!(params.ark.len(), 39, "ARK length should be 39 (8 partial rounds + 31 full rounds)");
    assert_eq!(params.mds.len(), 3, "MDS length should be 3 (2 rate elements + 1 capacity element)");
    assert_eq!(params.ark[0].len(), 3, "ARK elements should have length 3 (2 rate elements + 1 capacity element)");
    assert_eq!(params.mds[0].len(), 3, "MDS elements should have length 3 (2 rate elements + 1 capacity element)");
    
    // // Run the Poseidon permutation on both CUDA and serial implementations
    let device_id = 0;
    let cuda_result = poseidon_state;
    util::cuda_posidon_fr_r2_c1_t8_p31_a17(device_id, &cuda_result, &params);

    // Run the serial Poseidon permutation
    let mut serial_result = poseidon_state;
    util::poseidon_permuration_fr_r2_c1_t8_p31_a17(&mut serial_result, &params);

    // Check that the results are equal
    assert_eq!(cuda_result.state, serial_result.state);
}
