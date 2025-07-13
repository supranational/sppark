#!/bin/sh

# This script builds/runs the Poseidon CUDA project.
cargo update
cargo test --release --features=bls12_377