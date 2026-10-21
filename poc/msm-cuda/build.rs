// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn feature_check() {
    let curves = ["bn254", "bls12_377", "bls12_381"];
    let curves_as_features: Vec<String> = (0..curves.len())
        .map(|i| format!("CARGO_FEATURE_{}", curves[i].to_uppercase()))
        .collect();

    let mut curve_counter = 0;
    for curve_feature in curves_as_features.iter() {
        curve_counter += env::var(&curve_feature).is_ok() as i32;
    }

    match curve_counter {
        0 => panic!("Can't run without a curve being specified, please select one with --features=<curve>. Available options are\n{:#?}\n", curves),
        2.. => panic!("Multiple curves are not supported, please select only one."),
        _ => (),
    };
}

fn main() {
    feature_check();

    let mut curve = "";
    if cfg!(feature = "bn254") {
        curve = "FEATURE_BN254";
    } else if cfg!(feature = "bls12_377") {
        curve = "FEATURE_BLS12_377";
    } else if cfg!(feature = "bls12_381") {
        curve = "FEATURE_BLS12_381";
    }

    // account for cross-compilation [by examining environment variable]
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Set CC environment variable to choose alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    let mut cc = cc::Build::new();

    let c_src_dir = PathBuf::from("src");
    let files = vec![c_src_dir.join("lib.c")];
    let mut cc_opt = None;

    match (cfg!(feature = "portable"), cfg!(feature = "force-adx")) {
        (true, false) => {
            println!("Compiling in portable mode without ISA extensions");
            cc_opt = Some("__BLST_PORTABLE__");
        }
        (false, true) => {
            if target_arch.eq("x86_64") {
                println!("Enabling ADX support via `force-adx` feature");
                cc_opt = Some("__ADX__");
            } else {
                println!("`force-adx` is ignored for non-x86_64 targets");
            }
        }
        (false, false) => {
            #[cfg(target_arch = "x86_64")]
            if target_arch.eq("x86_64") && std::is_x86_feature_detected!("adx")
            {
                println!("Enabling ADX because it was detected on the host");
                cc_opt = Some("__ADX__");
            }
        }
        (true, true) => panic!(
            "Cannot compile with both `portable` and `force-adx` features"
        ),
    }

    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if !cfg!(debug_assertions) {
        cc.opt_level(2);
    }
    if let Some(def) = cc_opt {
        cc.define(def, None);
    }
    if let Some(include) = env::var_os("DEP_BLST_C_SRC") {
        cc.include(include);
    }
    cc.files(&files).compile("msm_cuda");

    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        return;
    }
    // Detect a CUDA (nvcc) or ROCm (hipcc) compiler and compile the GPU MSM
    // accordingly. nvcc is preferred when both are present; set NVCC=off to
    // force the ROCm path. The sppark build dependency auto-detects the same
    // toolchain and exports DEP_SPPARK_TARGET, which sppark::build::ccmd()
    // reads to return the matching cc::Build (CUDA or ROCm).
    println!("cargo:rerun-if-env-changed=NVCC");
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };
    println!("cargo:rerun-if-env-changed=HIPCC");
    let hipcc = match env::var("HIPCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("hipcc"),
    };

    let backend = if nvcc.is_ok() {
        Some("cuda")
    } else if hipcc.is_ok() {
        Some("rocm")
    } else {
        None
    };

    if let Some(backend) = backend {
        // bn254 (alt_bn128) G1 MSM is not yet supported on the ROCm/HIP backend:
        // the kernel hangs the GPU for this curve (see the project notes for the
        // deferred bn254-on-ROCm investigation). Refuse the combination up front
        // so a user never builds a binary that wedges the device. bls12_381 and
        // bls12_377 G1 MSM are fully supported on ROCm.
        if backend == "rocm" && cfg!(feature = "bn254") {
            panic!(
                "the bn254 curve is not yet supported on the ROCm/HIP MSM backend; \
                 use bls12_381 or bls12_377, or the CUDA backend for bn254"
            );
        }
        let mut ccmd: cc::Build = sppark::build::ccmd();
        if backend == "cuda" && cfg!(feature = "quiet") {
            ccmd.flag("-diag-suppress=177"); // bug in the warning system.
        }
        ccmd.define(curve, None);
        if let Some(def) = cc_opt {
            ccmd.define(def, None);
        }
        if backend == "rocm" {
            // The MSM kernels pass CUDA's 32-bit warp mask; suppress ROCm 7's
            // native 64-bit-mask *_sync builtins so the compat polyfills apply.
            ccmd.define("SPPARK_DISABLE_NATIVE_WARP_SYNC", None);
        }
        ccmd.file("cuda/pippenger_inf.cu").compile("blst_cuda_msm");

        println!("cargo:rustc-cfg=feature=\"{}\"", backend);
        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-env-changed=CXXFLAGS");
    }
}
