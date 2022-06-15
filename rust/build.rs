// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let mut base_dir = manifest_dir.join("sppark");
    if !base_dir.exists() {
        // Reach out to .., which is the root of the sppark repo.
        // Use an absolute path to avoid issues with relative paths
        // being treated as strings by `cc` and getting concatenated
        // in ways that reach out of the OUT_DIR.
        base_dir = manifest_dir
            .parent()
            .expect("can't access parent of current directory")
            .into();
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("ec").to_string_lossy()
        );
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("ff").to_string_lossy()
        );
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("msm").to_string_lossy()
        );
        println!(
            "cargo:rerun-if-changed={}",
            base_dir.join("util").to_string_lossy()
        );
    }
    // pass DEP_SPPARK_* variables to dependents
    println!("cargo:ROOT={}", base_dir.to_string_lossy());
}
