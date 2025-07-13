use ark_ff::PrimeField;

pub mod util;

// Poseidon parameters type
#[repr(C)]
pub struct PoseidonParameters<Field: PrimeField, const STATE_SIZE: usize, const TOTAL_ROUNDS: usize, const ALPHA: usize> {
    /// Additive Round keys. These are added before each MDS matrix application to make it an affine shift.
    /// They are indexed by `ark[round_num][state_element_index]`
    pub ark: [[Field; STATE_SIZE]; TOTAL_ROUNDS],

    /// Maximally Distance Separating Matrix.
    pub mds: [[Field; STATE_SIZE]; STATE_SIZE],
}

impl<Field: PrimeField, const STATE_SIZE: usize, const TOTAL_ROUNDS: usize, const ALPHA: usize> Default
for PoseidonParameters<Field, STATE_SIZE, TOTAL_ROUNDS, ALPHA> {
    fn default() -> Self {
        Self { 
            ark: [[Field::zero(); STATE_SIZE]; TOTAL_ROUNDS], 
            mds: [[Field::zero(); STATE_SIZE]; STATE_SIZE] 
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct PoseidonState<Field: PrimeField, const STATE_SIZE: usize> {
    pub state: [Field; STATE_SIZE],
}

impl<Field: PrimeField, const STATE_SIZE: usize> Default for PoseidonState<Field, STATE_SIZE> {
    fn default() -> Self {
        Self {
            state: [Field::zero(); STATE_SIZE]
        }
    }
}

// External symbol for the cuda_poseidon_permuration
extern "C" {
    fn cuda_poseidon_permuration(
        device_id: usize,
        inout_state: *mut core::ffi::c_void,
        params: *const core::ffi::c_void
    ) -> sppark::Error;
}

// Wrappper to call the ubnsafe function and do the checking of the error
pub fn poseidon_permuration<Field: PrimeField, const STATE_SIZE: usize, const TOTAL_ROUNDS: usize, const ALPHA: usize>(
    device_id: usize,
    inout_state: &PoseidonState<Field, STATE_SIZE>,
    params: &PoseidonParameters<Field, STATE_SIZE, TOTAL_ROUNDS, ALPHA>
) {

    // // Print the MDS matrix from the parameters on the screen as 64-bit libmbs shown as hex values
    // println!("MDS matrix (Rust Side):");
    // for i in 0..STATE_SIZE {
    //     for j in 0..STATE_SIZE {
    //         print!("{} ", params.mds[i][j].into_repr().to_string());
    //     }
    //     println!();
    // }
    // println!("Address of parameters: {:p}", &params);
    // println!("Address of ARK matrix: {:p}", &params.ark[0][0]);
    // println!("Address of MDS matrix: {:p}", &params.mds);
    // println!("Size of MDS by addrss difference: {}", (&params.mds[0] as *const _ as u64 - &params.ark[0][0] as *const _ as u64)/8);
    // println!("Size of MDS by addrss difference: {}", (&params.mds[0] as *const _ as u64 - &params.ark[0][0] as *const _ as u64)/8);

    // println!("First byte in MDS: ");
    // let first_byte = unsafe {
    //     let mds_ptr = params.mds.as_ptr() as *const u8;
    //     *mds_ptr
    // };
    // println!("{}", first_byte);
    // println!("Last byte in MDS: ");
    // let last_byte = unsafe {
    //     let mds_ptr = params.mds.as_ptr() as *const u8;
    //     *mds_ptr.add(std::mem::size_of::<[[Field; STATE_SIZE]; STATE_SIZE]>() - 1)
    // };
    // println!("{}", last_byte);
    

    // println!("First byte in ARK: ");
    // let first_byte_ark = unsafe {
    //     let ark_ptr = params.ark.as_ptr() as *const u8;
    //     *ark_ptr
    // };
    // println!("{}", first_byte_ark);


    // println!("Byte dump of MDS matrix:");
    // let mds_bytes = unsafe {
    //     let mds_ptr = params.mds.as_ptr() as *const u8;
    //     std::slice::from_raw_parts(mds_ptr, std::mem::size_of::<[[Field; STATE_SIZE]; STATE_SIZE]>())
    // };
    // for (i, byte) in mds_bytes.iter().enumerate() {
    //     print!("{:02x} ", byte);
    //     if (i + 1) % 32 == 0 {
    //         println!();
    //     }
    // }
    // if mds_bytes.len() % 16 != 0 {
    //     println!(); // Ensure a new line if the last line is not complete
    // }


    let err = unsafe {
        cuda_poseidon_permuration(
            device_id,
            inout_state as *const _ as *mut core::ffi::c_void,
            params as *const _ as *const core::ffi::c_void
        )
    };
    if err.code != 0 {
        panic!("{}", String::from(err));
    }

}
