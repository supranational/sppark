// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    #[cfg(feature = "bls12_377")]
    use ark_bls12_377::G1Affine;
    #[cfg(feature = "bls12_381")]
    use ark_bls12_381::G1Affine;
    #[cfg(feature = "bn254")]
    use ark_bn254::{Fr, G1Affine};
    use ark_ec::msm::VariableBaseMSM;
    use ark_ec::{AffineCurve, ProjectiveCurve};
    use ark_ff::BigInteger256;
    use ark_std::{UniformRand, Zero};

    pub fn create_scalar_bases<G: AffineCurve>(
        mut rng: &mut ChaCha20Rng,
        size: usize,
    ) -> (Vec<G>, Vec<G::ScalarField>) {
        let bases = (0..size)
            .map(|_| G::Projective::rand(&mut rng))
            .collect::<Vec<_>>();
        let bases =
            <G::Projective as ProjectiveCurve>::batch_normalization_into_affine(
                &bases,
            );
        let scalars = (0..size)
            .map(|_| G::ScalarField::rand(&mut rng))
            .collect::<Vec<_>>();
        (bases, scalars)
    }

    #[test]
    fn test_msm_correctness() {
        const NPOINTS_POW: usize = 14;
        const GEN_NPOINTS: usize = 1 << 11;
        const NPOINTS: usize = 1 << NPOINTS_POW;

        let mut rng = ChaCha20Rng::from_entropy();
        let (mut points, mut scalars) =
            create_scalar_bases::<G1Affine>(&mut rng, GEN_NPOINTS);
        points[0] = G1Affine::zero();
        for _ in 0..NPOINTS_POW - 11 {
            points.append(&mut points.clone());
            scalars.append(&mut scalars.clone());
        }

        assert!(points.len() == NPOINTS);

        let msm_result =
            multi_scalar_mult_arkworks(points.as_slice(), unsafe {
                std::mem::transmute::<&[_], &[BigInteger256]>(
                    scalars.as_slice(),
                )
            })
            .into_affine();
        println!("MSM res      : {:?}", msm_result);

        let arkworks_result =
            VariableBaseMSM::multi_scalar_mul(points.as_slice(), unsafe {
                std::mem::transmute::<&[_], &[BigInteger256]>(
                    scalars.as_slice(),
                )
            })
            .into_affine();
        println!("Arkworks res : {:?}", arkworks_result);

        assert_eq!(msm_result, arkworks_result);
    }
}
