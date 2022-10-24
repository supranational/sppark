// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::env;

fn feature_check() {
    let fr_s = ["bls12_377", "bls12_381"];
    let fr_s_as_features: Vec<String> = (0..fr_s.len())
        .map(|i| format!("CARGO_FEATURE_{}", fr_s[i].to_uppercase()))
        .collect();

    let mut fr_counter = 0;
    for fr_feature in fr_s_as_features.iter() {
        fr_counter += env::var(&fr_feature).is_ok() as i32;
    }

    match fr_counter {
        0 => panic!("Can't run without a field being specified,\nplease select one with --features=<field>. Available options are\n{:#?}\n", fr_s),
        2.. => panic!("Multiple fields are not supported, please select only one."),
        _ => (),
    };
}

fn main() {
    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        panic!("unsupported compiler");
    }

    feature_check();

    let mut fr = "";
    if cfg!(feature = "bls12_377") {
        fr = "FEATURE_BLS12_377";
    } else if cfg!(feature = "bls12_381") {
        fr = "FEATURE_BLS12_381";
    }

    let mut nvcc = cc::Build::new();
    nvcc.cuda(true);
    nvcc.flag("-arch=sm_70");
    nvcc.flag("-Xcompiler").flag("-Wno-unused-parameter");
    nvcc.flag("-Xcompiler").flag("-Wno-subobject-linkage");
    nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
    nvcc.define(fr, None);
    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        nvcc.include(include);
    }
    if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
        nvcc.include(include);
    }
    nvcc.file("cuda/ntt_api.cu").compile("ntt_cuda");

    println!("cargo:rerun-if-changed=cuda");
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
}
