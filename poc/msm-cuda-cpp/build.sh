#!/bin/bash

set -e
set -x

if [ ! -f "blst/libblst.a" ]; then
    git clone https://github.com/supranational/blst.git
    cd blst
    sh build.sh
    cd ..
fi

SPPARK_MAIN_PATH="../.."

INCLUDES="-I$SPPARK_MAIN_PATH -Iblst/src"
CXXFLAGS="-std=c++14 -D__ADX__ -g -Xcompiler -O2"
NVCCFLAGS="-O2 -arch=sm_80 -gencode arch=compute_70,code=sm_70 -t0"
LIBS="-Lblst -lblst"

nvcc -DFEATURE_BLS12_377 $CXXFLAGS $NVCCFLAGS $INCLUDES $LIBS $SPPARK_MAIN_PATH/util/all_gpus.cpp demo.cu  -o run_demo.out
nvcc -DFEATURE_BLS12_377 $CXXFLAGS $NVCCFLAGS $INCLUDES $LIBS $SPPARK_MAIN_PATH/util/all_gpus.cpp bench.cu -o run_benchmarks.out
