#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
msm_dir="${root_dir}/poc/msm-cuda"
blst_dir="${root_dir}/blst"

if [[ ! -f "${blst_dir}/libblst.a" ]]; then
  (cd "${blst_dir}" && ./build.sh)
fi

cd "${msm_dir}"
nvcc -std=c++17 -D__ADX__ -I../.. -I../../blst/src \
  cuda/pippenger_curve25519.cu tests/curve25519_msm.cpp \
  ../../util/all_gpus.cpp \
  ../../blst/libblst.a -o curve25519_msm
./curve25519_msm
