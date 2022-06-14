// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
#![allow(unused_imports)]

use criterion::{criterion_group, criterion_main, Criterion};

use blst_msm::multi_scalar_mult_arkworks;
include!("../src/tests.rs");

#[cfg(feature = "bls12_377")]
use ark_bls12_377::G1Affine;
#[cfg(feature = "bls12_381")]
use ark_bls12_381::G1Affine;
#[cfg(feature = "bn254")]
use ark_bn254::{Fr, G1Affine};
use ark_ff::BigInteger256;
use ark_std::{UniformRand, Zero};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use std::str::FromStr;

fn criterion_benchmark(c: &mut Criterion) {
    let bench_npow = std::env::var("BENCH_NPOW").unwrap_or("23".to_string());
    let npoints_npow = i32::from_str(&bench_npow).unwrap();

    const GEN_NPOINTS: usize = 1 << 11;
    let mut rng = ChaCha20Rng::from_entropy();
    let (mut points, mut scalars) =
        crate::tests::create_scalar_bases::<G1Affine>(&mut rng, GEN_NPOINTS);
    points[0] = G1Affine::zero(); // Test infinity
    for _ in 0..npoints_npow - 11 {
        points.append(&mut points.clone());
        scalars.append(&mut scalars.clone());
    }

    assert!(points.len() == 1usize << npoints_npow);

    let mut group = c.benchmark_group("CUDA");
    group.sample_size(20);

    let name = format!("2**{}", npoints_npow);
    group.bench_function(name, |b| {
        b.iter(|| {
            let _ = multi_scalar_mult_arkworks(&points.as_slice(), unsafe {
                std::mem::transmute::<&[_], &[BigInteger256]>(
                    scalars.as_slice(),
                )
            });
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
