#[cfg(feature = "bls12_377")]
use ark_bls12_377::{G1Affine, G2Affine};
#[cfg(feature = "bls12_381")]
use ark_bls12_381::{G1Affine, G2Affine};
#[cfg(feature = "bn254")]
use ark_bn254::G1Affine;
use ark_ff::BigInteger256;

use std::str::FromStr;

use msm_cuda::*;

/// cargo run --release --example msm --features bn254
fn main() {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("16".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();

    let (points, mut scalars) =
        util::generate_points_scalars::<G1Affine>(1usize << npoints_npow);

    let ret = multi_scalar_mult_arkworks(
        &points.as_slice(),
        unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        },
        &(0..1u32 << npoints_npow).rev().collect::<Vec<u32>>(),
    );

    scalars.reverse();
    let ret_rev = multi_scalar_mult_arkworks(
        &points.as_slice(),
        unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        },
        &(0..1u32 << npoints_npow).collect::<Vec<u32>>(),
    );

    assert_eq!(ret, ret_rev);
    println!("success")
}
