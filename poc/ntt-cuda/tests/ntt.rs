// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use ntt_cuda::NTTInputOutputOrder;

const DEFAULT_GPU: usize = 0;

#[cfg(any(feature = "bls12_377", feature = "bls12_381"))]
#[test]
fn test_against_arkworks() {
    #[cfg(feature = "bls12_377")]
    use ark_bls12_377::Fr;
    #[cfg(feature = "bls12_381")]
    use ark_bls12_381::Fr;
    use ark_ff::{PrimeField, UniformRand};
    use ark_poly::{
        domain::DomainCoeff, EvaluationDomain, GeneralEvaluationDomain,
    };
    use ark_std::test_rng;

    fn test_ntt<
        F: PrimeField,
        T: DomainCoeff<F> + UniformRand + core::fmt::Debug + Eq,
        R: ark_std::rand::Rng,
        D: EvaluationDomain<F>,
    >(
        rng: &mut R,
    ) {
        for lg_domain_size in 1..24 {
            let domain_size = 1usize << lg_domain_size;

            let domain = D::new(domain_size).unwrap();

            let mut v = vec![];
            for _ in 0..domain_size {
                v.push(T::rand(rng));
            }

            v.resize(domain.size(), T::zero());
            let mut vtest = v.clone();

            domain.fft_in_place(&mut v);
            ntt_cuda::NTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NN);
            assert!(vtest == v);

            domain.ifft_in_place(&mut v);
            ntt_cuda::iNTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NN);
            assert!(vtest == v);

            ntt_cuda::NTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NR);
            ntt_cuda::iNTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::RN);
            assert!(vtest == v);

            domain.coset_fft_in_place(&mut v);
            ntt_cuda::coset_NTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NN);
            assert!(vtest == v);

            domain.coset_ifft_in_place(&mut v);
            ntt_cuda::coset_iNTT(DEFAULT_GPU, &mut vtest, NTTInputOutputOrder::NN);
            assert!(vtest == v);
        }
    }

    let rng = &mut test_rng();

    test_ntt::<Fr, Fr, _, GeneralEvaluationDomain<Fr>>(rng);
}
