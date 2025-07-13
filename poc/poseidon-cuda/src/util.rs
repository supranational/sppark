#[cfg(feature = "bls12_377")]
use ark_bls12_377::Fr;
#[cfg(feature = "bls12_381")]
use ark_bls12_381::Fr;

use ark_ff::BigInteger256;
use ark_ff::Zero;
use ark_ff::Field;
use crate::*;


// Define the function ge
pub fn poseidon_params_fr_r2_c1_t8_p31_a17() -> PoseidonParameters<Fr, 3, 39, 17>
{
    // Constants
    const RATE: usize = 2;
    const CAPACITY: usize = 1;
    const FULL_ROUNDS: usize = 8;
    const PARTIAL_ROUNDS: usize = 31;
    const ALPHA: usize = 17;
    const STATE_SIZE: usize = RATE + CAPACITY;
    const TOTAL_ROUNDS: usize = FULL_ROUNDS + PARTIAL_ROUNDS;

    // Empty params object
    let mut params = PoseidonParameters::<Fr, STATE_SIZE, TOTAL_ROUNDS, ALPHA>::default();
    
    // ARC values as collected from SnarkVM tests
    let ark_values: [[[u64; 4]; STATE_SIZE]; TOTAL_ROUNDS] = [
        [[0xfedb02a23316c7c3, 0xd79f6be92f3b4910, 0x09f492a4fe798a26, 0x0307d480bee26c72], [0x230efbfbb6ba836e, 0x1e6dbe9654d09cb3, 0x142aa06b838d3daa, 0x0a54d82d4d532ccb], [0x17e5eee7a7cb00b8, 0x5ecc7865ca1c750f, 0xc4d0f5d4821cee3f, 0x07afe79d13e6a9c3]],
        [[0x9f36b1a0a45d4c11, 0x2d8c95448062102d, 0x5cd1ba4c81d64d4c, 0x0264abad713e9702], [0x473383ed8b9ae98e, 0xd3229bb40e48ef00, 0x7f21b3fa7e0d5cb4, 0x0af1478f1ad439b9], [0x555062e78dce5654, 0x7f5522fba02a341c, 0x4a3fe6d8d35d7159, 0x02f12ace0eb1e5b7]],
        [[0xf706f9980dcddf0b, 0x86bd2ac358fe8ad8, 0x2f68510aba63261a, 0x05db911867289567], [0x7def77bfbc9f6096, 0x156aefb4f76a3ef2, 0xd1d62e2bb167c0d8, 0x0afc97a617b6b930], [0x15116e19213e7c5b, 0xe6e4d0defacd01a9, 0xc256fdbe658dad8d, 0x007462bc7fd33817]],
        [[0x29af84d19e6c8a87, 0x611a7829cd289ed3, 0x00c4ae3f95240cf4, 0x0d966251193b9ea5], [0xf50ff23caa3d3061, 0xeb5b80eb81c62dda, 0x94f4b402f85e34fd, 0x0c973d4b7dd3be72], [0xa4f07e9f014f8f07, 0xaa278005853ce165, 0xece620147e13e516, 0x0cb1b3a15c59763b]],
        [[0xde95adccf868588e, 0xbb896cd4c3bde7eb, 0x658cff0f217b5632, 0x0e4daf81f72f357f], [0x69cceee089810016, 0x7c7c3824e8a11f63, 0xf8fac3c3d4551807, 0x082f07554d97d6cb], [0xd72f9bce8c9d5d0f, 0x45a72ad16857f3b8, 0xc0b74074db6c9d33, 0x086014dfad22fe0b]],
        [[0x99870db2eda38efe, 0x85a74bda2b436d44, 0xeca4e496002c64b2, 0x1072f4b6cac11eec], [0xfc2db53beaaa4363, 0x45c17ef67a1ebac5, 0xfd3e16d2b1423c9c, 0x01c073b3a10f7914], [0xd21d9a6a0c1379ab, 0x87a51144c666894e, 0x00e0624800c8bab6, 0x07c3c353126b6293]],
        [[0xadb2ff35372780b4, 0x94d80244b281d16f, 0x737a45f81b71528e, 0x075b10be40bd1444], [0x81a82857dfa86ff3, 0x286bd38054eb4549, 0x88395c0cddf85d76, 0x0fd37e4eee4816f7], [0x0fd7e604ffe507aa, 0x84a424508f0ff1df, 0x38ca20c7167c346e, 0x0e89f2e8a4b03a79]],
        [[0x1aef77627fe8e0b7, 0x3350a029a36362f7, 0x2d66a30c2bcef215, 0x03deb6f5728e55c1], [0x87da051558d66fea, 0x8db2c6ccb41b61d9, 0x4a2cccee321063bb, 0x0eb02b1f7544742d], [0x9fb12ffde3a5f094, 0xc450290b1a1fb53b, 0x06546a0cc3ae0e33, 0x091462b60c6f206a]],
        [[0x91bf8a4a3174e9c8, 0x88abc3d8549503a4, 0x3dba21682df95a44, 0x0b7cf8d11b9f05cc], [0xc12a1498c0b36b02, 0x3df271f231ca9ce4, 0xa5f99a85affb082d, 0x0e0ffc757467e7ce], [0x559e5f9d17e712d1, 0xc3a85268b6136759, 0x1a5df701bf76bf95, 0x12a0da493ad25f70]],
        [[0x98c97947bbfe6934, 0xcabf08c0bc8875cc, 0xdb6f8ed389ecb832, 0x10bd77a599513f17], [0xd1b0d0dc746e79d3, 0xf7c8454a16e9d83e, 0x4def021a9e0dc63f, 0x0288ee189ab84327], [0x5636b18ca274da6b, 0x1f2a823cdcc08c96, 0x014e8c229eafe933, 0x0f9f078074db7a56]],
        [[0x96a0f0cb984da715, 0x7e626edd7d43c338, 0xa1f5a6709ca7ab8a, 0x06a02d7073a97427], [0x0ee1c11944027745, 0x619d72d76bc5525a, 0x71b5deb32017564d, 0x10f45b0281d4cf8f], [0xbdac583fc7c40e3d, 0xa7f360fdc004dd39, 0xf9ca61add6bb9a17, 0x02121ccc383abdb3]],
        [[0x2dc663a7a99ddc0e, 0x5a5823855cdc444d, 0x5843976dc1d5932b, 0x0db4ef0792ac922c], [0xe9ad19ced7f4aa9f, 0xd849fa8e73a91eb5, 0x064a0d8f13210ca5, 0x03a7a3c2871a152d], [0xed23766606e9241e, 0x79afbfa5f2f7c401, 0x2fd62ed28ce20335, 0x118d84ac780977fe]],
        [[0x267c99f693bdbbce, 0xd56c22406cc95ffc, 0xca80752268c666b7, 0x0b9b7a6610f66cad], [0xb29bce5efeec804e, 0x89e2f7629c7a58b5, 0x3cf86fc9bc8be9be, 0x008584277576cee9], [0x960987601b0f27ef, 0x650741f07afcf501, 0x70b06f1ba5722664, 0x0e9c8a632df1143f]],
        [[0x77d45baad0e049eb, 0x064017981df5e3f7, 0x271a8c9879c7a61e, 0x0184bcef16ba483e], [0x7ed41bcdb3a6b462, 0xe4a0a819c1b45681, 0x3b177b82f94bfac5, 0x0f1d8ea1494ed41f], [0xb8f389d2d93a4c87, 0x1a530f6d408ad178, 0x63cc46958d85cbf1, 0x09c0b6ac4c3cae88]],
        [[0xc3f23e5f9c04e615, 0x48918767bc9cede0, 0x578e6e5a0dabcb6b, 0x112b29b32ac9ff4c], [0xd3bfc3f84faeff62, 0xefb71536ea5e4cf9, 0xa89239640f5605b4, 0x00bc376bf198cf32], [0x1d1dcf7da9b41346, 0x3177719237106374, 0x1ae374341cc80155, 0x12a8110996eb0207]],
        [[0xb7a660f6c85a5169, 0x0ad178d8a9c18a44, 0x303cbf59cdc185cc, 0x09aa067c02a465ca], [0x89b05b4fc3db7e2d, 0xcf0363b9eed17c99, 0x39051ba2d1d1465a, 0x09e2f5c060f92f95], [0x37e9c9bf7c640631, 0xb044c07bb4417f4f, 0xc629016644faa30b, 0x118aa5747b8ff486]],
        [[0x92063c2e46d6b0f2, 0x2a32ec35d1235fdd, 0x279a89dca5ff868f, 0x0749fee5673d3bbd], [0x2b9ba5f4002ba812, 0x00bdec8a20a46734, 0xe51ca39c9aecf43c, 0x114ee3d9b3c2d428], [0xae274e08d760c23e, 0x2d0a577c59ec0ed3, 0x1634866b0ecc0ea6, 0x0df53bf55d591959]],
        [[0x69ea85ca9e6a0fc5, 0x3496f4207ed50b42, 0x15939957b854a414, 0x10f6d0d133190034], [0x835db8e44c300a67, 0xfa8c196942808fce, 0x3fba71a0a636c6d7, 0x07efc6f100f019e9], [0xf2081a295caf8dd8, 0x5def4169c0781a9d, 0xe414a62d1fbf8f38, 0x07bb683e27a08bf7]],
        [[0xcccc9ccaf15ea5d3, 0xa50aaf8314bb3d08, 0x70af9b0abadd6f93, 0x07f4ebde76415f37], [0x1942b8c24df2d3da, 0xf6215145e3789d52, 0x87a4601e3b771f78, 0x1266ebb7f976b99c], [0xc0a8f58e9bdba3d0, 0x1f7927a1526716be, 0xcac4be68cbd3e06d, 0x05953b41852559e3]],
        [[0x089cb4ca0f7ee9bb, 0xf66c70424fb86128, 0xe3c3b6bf5fecab64, 0x03b141a51d548b62], [0x7b909f652a052c82, 0x64157759d3cf0a5a, 0xe7ac372773faf613, 0x05c899af6fc4aca4], [0xaf346ef8805848e8, 0xf619a82237108be0, 0x67b1de042f1ce58c, 0x08808a176f326803]],
        [[0x6f967cc520f89cab, 0xf32e49e537b6306c, 0x2b4aa4fa0ed43720, 0x11ef1391c12a68c8], [0x2743d2b1d37e5007, 0x3f567f6e7ffd1441, 0xd5e65635372364d7, 0x128ff92324eb79c3], [0x7e5d21679807b233, 0x3ea170c0eb7378ea, 0x510aefdf7dada02a, 0x10646def9fc91208]],
        [[0x22ff7aa76fd2b8bb, 0x338aab92b861dd7c, 0x8dcbc99f490f9a8b, 0x085a726d447390b1], [0x39bb73c59d8de75c, 0x1b4223720f157fdf, 0x7aba527498ce7d5e, 0x05abf2cc037390cb], [0x631fb0639e0d4c2f, 0xbe381077cd5b2faf, 0x5b777f2baa5fdf9f, 0x10171d04fc91ce49]],
        [[0x7907e584737e78e3, 0xd5e24b9499956dc2, 0x75dd6a2501d52242, 0x0e71b843c009ade1], [0xabbadd2390898e64, 0x85e396eac1b1bc5d, 0xae6bc70cc86de2c7, 0x0f8739ea9ce9b5ff], [0x17ff452facbcb79a, 0xaa80e36ac9073242, 0x5a56e43e46b0265f, 0x0e45ee2879666b68]],
        [[0x1f89a05c7cd5aa4a, 0xa08423b5ba646fda, 0x21e7fe3ccff21499, 0x062674460dcbdcf8], [0xdd4a02da9f26c881, 0x974aab9cea91387a, 0x1a9e069a173fdd8c, 0x112d0dabccf2680f], [0x96dacc50a9cdec78, 0xa7c11fa4205e5fcd, 0x512b410b20d6af72, 0x0e4a3d428cabf444]],
        [[0x2f7c37375d8333c5, 0x48823e3ce54bc448, 0xd7ac411fc8eae8ba, 0x0970300578bd7df7], [0x298b105bfa0ff885, 0x2381dc97a1c76b09, 0xb1a1d12164bcde83, 0x0c6bb23616cb27d5], [0x347d59f9520ec47f, 0x2489045eb7bf0218, 0x6209b5f53a632ccf, 0x027eab5ff826c4e0]],
        [[0x8fea5fb7e3df4c27, 0xf7a9fa450b30a7db, 0x318daa5e4d5d1e14, 0x0f41ae12b2548a82], [0x8b3a4d3cc36bf3de, 0xe50f2fc86a91b296, 0x1c0930a9e5d8d7f1, 0x0856800b5931666f], [0x2472cf35d5ecf4e0, 0x9d02d32fa52ff1df, 0xb1392eede49bae18, 0x09117502ea450c21]],
        [[0x6cc095a2ae3ebf9c, 0xead1c6f420b84700, 0x2181ab09b0b9fe21, 0x101064073a301b61], [0xab2bb4e6d07a494e, 0x358e9184e1228c8e, 0x7c927cb0ee23fc91, 0x0a8981470e2fce98], [0xcc140411cfd7fd1b, 0x8383af171885ba59, 0x80a5b8c87b4b4f5a, 0x0e3881fdd702cd8e]],
        [[0x07c87b5c806a3273, 0x0a9690be40ed9629, 0x30224670d73966ee, 0x0e1067b4e0998ca3], [0x0feff4814fdd6730, 0xb89befded672338d, 0x970e4b1dedfdba8f, 0x10d352388525c062], [0xd6f088d61168dbfc, 0xf4bacc68a3d1fe39, 0x364947d2929bf4a0, 0x00cd8d99e082e98d]],
        [[0xb76cc17fe5b36fa3, 0xd0fa4905d55bb556, 0x727afbeb6def9994, 0x07e028ba08700a7f], [0x5c24fb59fa2ed194, 0x99f2540a78790d35, 0x20e9831e8e9ac8af, 0x1134c44aa045f42a], [0xbebc93180e11f787, 0x40d006295a5db655, 0xdf7fde87d3436749, 0x02d1efeba08fd0dd]],
        [[0x4d3968aa23037694, 0xda9c7d393a6c3de9, 0x1a7b0f59149eab7c, 0x0b48d886504db803], [0x6e78056c924453a6, 0x7dc7a9b60eb199a2, 0xae3744adc0e8c9b7, 0x087651e3026fc020], [0xf5cff36af9af19fb, 0xd705275653bf42e5, 0x73f3f11c3ad1e5f4, 0x0e7459892743ec6f]],
        [[0xe2dc686abbe6f235, 0x38b27366e86fdf06, 0x858f7c93881bb14b, 0x05ce030cc0767405], [0x3fae03e55e9c87af, 0xe2b35bb276ba36ca, 0x41383d6259da9be7, 0x0503ab481a639c00], [0x67bb1e11409b700e, 0xac2f581e57c66348, 0x861923be666dc7a9, 0x125ac1e78f807d10]],
        [[0xea470f824d83cc66, 0x310f8ea22742f706, 0x35fc984b859a1e2b, 0x0dad626de723f4cf], [0xd0baa2601aff958c, 0x2cb197160057bb78, 0xe68ca5c22b50ed86, 0x0378b337f556065d], [0xd30f21c5843ccc68, 0x1111d6584f87da34, 0xd0e9ac28ad069e67, 0x0e394c529291569a]],
        [[0xf7497cf9bafc8276, 0x4e83760fa08cfe1c, 0x35330a35cb40a046, 0x083f82a8bc4f18ce], [0x268c5fab676316d2, 0x9db66cad65a49e33, 0xd9b8f074dd510138, 0x09340308168da302], [0xdd501cbb3ee98447, 0x9e82281e7945e94d, 0x78f3df49faf74d3d, 0x0acd2420579cd06b]],
        [[0x7a1c758be3ad96f4, 0xa149d9a69d8db8d0, 0xff6074cf1967cfca, 0x01804e7d47f84ecb], [0xa36a7fdeae0d5bdb, 0x5aee8cd39a53a5b2, 0x1879b350ed1102b1, 0x0f93844da88b198d], [0xea9567a73f79e2a2, 0xf8013360d5d86563, 0x3840c72cf69af8df, 0x01c2cb3f42071831]],
        [[0x0b402b6c4937e3b0, 0x504dcdc537714288, 0x42a7e2f14fcc0d9c, 0x0f52ae672d91ce90], [0xbf9c0fa86a26f1e7, 0x0d32a503dbbffcab, 0x971d9ab4bd2e5ed2, 0x08798c8d422951ed], [0xc0b328e7e675df4c, 0x3caa2cf0be07de4a, 0x0c263122c896405f, 0x0d7243501249c5a0]],
        [[0x2ec9b2ed6a4f19fe, 0x817522a808791f50, 0x1fb1125aadf5724f, 0x086ffecd7dbe7a8b], [0x5248c36cebd231e1, 0x2b34c5029375734b, 0xee65b29c8b07c3aa, 0x00201c4b6894c566], [0x5a321b26a3cb8600, 0xe7e3dbd174e1f694, 0x2469e8fa1caeda7d, 0x0361089640502d97]],
        [[0xc9c0c5426811dff1, 0xd23974c651affc33, 0x0ebe152c0203a940, 0x08f506d00151a9bc], [0x4260403beff66f1c, 0xc5d6fb25c9f73d14, 0x409d542f2e87c444, 0x020daee89b362afe], [0xaf71394c692c8f40, 0x0abb7c66bb62b2ae, 0x3e40cfc616281141, 0x05b1dbfbb46ed65e]],
        [[0xf525d1f7a463e477, 0x474abc9975a459d9, 0x974756b69b9c9d2a, 0x094298c826c54a44], [0x4654c49f6c14a747, 0x2e6d3d2321a0e5e1, 0x49cd4fb28e4b1b11, 0x048b739208685782], [0x1a5b9f386d9c152e, 0xfaf211a6c39ab616, 0x6b6b5d2d9b78c153, 0x0a1dabd79d757d50]],
        [[0x2e7871c2057b7ad0, 0xb295a2a1484e1f12, 0x7097b468d8637fea, 0x0419e41bba470996], [0x6a0e737e170d65e1, 0x61d3d7a148b1572f, 0x56f47372b0791992, 0x0914795802449b87], [0x8099ef4618f1bfd1, 0x555bc20168acd86e, 0x5923a3d3fb2b2109, 0x09d156904205a828]],
    ];

    // MDS Values as collected from SnarkVM tests
    let mds_values: [[[u64; 4]; STATE_SIZE]; STATE_SIZE] = [
        [[0x7a3704f0cf07fbbe, 0xd0e7a8fa1f2d8850, 0x4f0e169fcd6d64b4, 0x0d78c589988ab34a], [0x287fc71ba76ae486, 0x268284725bcdfe08, 0x51aabafb237befef, 0x0d31ec43a4c45ccd], [0x22edeecfd74a42de, 0x9b62dff32a8d593c, 0x45d3caefb32ab420, 0x026ed790451b64ac]],
        [[0xbef63a46045f06f0, 0x6aea440e54691cb9, 0xab663c93b94e5450, 0x06fd0da62b974678], [0x7f15a2eb8dc39f24, 0x3c00e863039bfec8, 0x0f9b26685732f9f4, 0x052b75be29e87fdd], [0x5a864724c62ada25, 0x75145aa23d89b1a8, 0xb30bdfd2e94dd375, 0x07061f0420cf7278]],
        [[0xdb973f763cc7643d, 0x435669c9c487af66, 0x5ea93b1af04ff084, 0x050d5d28e6479c0a], [0x425b3aff18edca1d, 0x6859790d752aef47, 0x8728bbd7ec1c2e6c, 0x002fc194d6c13db6], [0x05e02b4d41d87e2e, 0x794370b846cdb63b, 0x76a1d723ecc5f755, 0x124ed072fb2e8c69]],
    ];

    // Initialize the ARK
    for i in 0..TOTAL_ROUNDS {
        for j in 0..STATE_SIZE {
            let limbs = [
                ark_values[i][j][0],
                ark_values[i][j][1],
                ark_values[i][j][2],
                ark_values[i][j][3],
            ];
            let bigint = BigInteger256::new(limbs);
            params.ark[i][j] = Fr::new(bigint);
        }
    }

    // Initialize the MDS
    for i in 0..STATE_SIZE {
        for j in 0..STATE_SIZE {
            let limbs = [
                mds_values[i][j][0],
                mds_values[i][j][1],
                mds_values[i][j][2],
                mds_values[i][j][3],
            ];
            let bigint = BigInteger256::new(limbs);
            params.mds[i][j] = Fr::new(bigint);
        }
    }


    return params;
}


pub fn poseidon_initial_state_r2_c1() -> PoseidonState<Fr, 3>
{
    let poseidon_state = PoseidonState::<Fr, 3>::default();

    // Check that the state is initialized to all zeros
    assert!(poseidon_state.state.iter().all(|&x| x.is_zero()));
    
    return poseidon_state;
}

pub fn cuda_posidon_fr_r2_c1_t8_p31_a17(
    device_id: usize, 
    inout_state: &PoseidonState<Fr, 3>, 
    params: &PoseidonParameters<Fr, 3, 39, 17>)
{
    poseidon_permuration::<Fr, 3, 39, 17>(device_id, &inout_state, &params);
}


pub fn poseidon_permuration_fr_r2_c1_t8_p31_a17(
    state: &mut PoseidonState<Fr, 3>, 
    params: &PoseidonParameters<Fr, 3, 39, 17>)
{

    // Constrants
    const RATE: usize = 2;
    const CAPACITY: usize = 1;
    const FULL_ROUNDS: usize = 8;
    const PARTIAL_ROUNDS: usize = 31;
    const ALPHA: u64 = 17;
    const STATE_SIZE: usize = RATE + CAPACITY;
    const TOTAL_ROUNDS: usize = FULL_ROUNDS + PARTIAL_ROUNDS;
    const FULL_ROUNDS_OVER_TWO: usize = FULL_ROUNDS / 2;

    for round in 0..TOTAL_ROUNDS {

        // Apply the round constants
        for i in 0..STATE_SIZE {
            state.state[i] += params.ark[round][i];
        }

        if round < FULL_ROUNDS_OVER_TWO || round >= TOTAL_ROUNDS - FULL_ROUNDS_OVER_TWO {
            // Apply the S-box to all elements only if we are in the full rounds
            for i in 0..STATE_SIZE {
                state.state[i] = state.state[i].pow([ALPHA as u64]);
            }
        } else {
            // Apply the S-box to the first element only in the partial rounds
            state.state[0] = state.state[0].pow([ALPHA as u64]);
        }

        // Apply the MDS matrix
        let mut new_state = poseidon_initial_state_r2_c1();
        for i in 0..STATE_SIZE {
            for j in 0..STATE_SIZE {
                new_state.state[i] += params.mds[i][j] * state.state[j];
            }
        }
        
        // Update the state
        for i in 0..STATE_SIZE {
            state.state[i] = new_state.state[i];
        }
    }

}