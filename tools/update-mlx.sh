#!/bin/zsh

# See MAINTENANCE.md : Updating `mlx` and `mlx-c`

set -e

if [[ ! -d Source ]]
then
    echo "Please run from the root of the repository, e.g. ./tools/update-mlx.sh"
    exit 1
fi

# copy mlx-c headers to build area
rm -f Source/Cmlx/include/mlx/c/*
cp Source/Cmlx/mlx-c/mlx/c/*.h Source/Cmlx/include/mlx/c

# copy mlx C++ public headers to build area
rm -rf Source/Cxxmlx/include/mlx
mkdir -p Source/Cxxmlx/include/mlx
rsync -a \
    --include='*/' \
    --include='*.h' \
    --include='*.hpp' \
    --exclude='*' \
    Source/Cxxmlx/mlx/mlx/ \
    Source/Cxxmlx/include/mlx/

# copy metal-cpp headers used by public MLX Metal backend headers
for header_dir in Foundation Metal MetalFX QuartzCore
do
    rm -rf "Source/Cxxmlx/include/${header_dir}"
    rsync -a \
        "Source/Cxxmlx/metal-cpp/${header_dir}" \
        Source/Cxxmlx/include/
done

# run the command to do the build-time code generation

mkdir build
cd build
cmake ../Source/Cxxmlx/mlx -DMLX_METAL_JIT=ON -DMACOS_VERSION=14.0

# run the cmake build to generate the source files
cd mlx/backend/metal
make \
    arange \
    binary \
    binary_ops \
    binary_two \
    conv \
    copy \
    fft \
    fp_quantized \
    fp_quantized_nax \
    gather \
    gather_axis \
    gather_front \
    gemm \
    gemm_nax \
    gemv_masked \
    hadamard \
    logsumexp \
    masked_scatter \
    quantized \
    quantized_nax \
    quantized_utils \
    reduce \
    reduce_utils \
    scan \
    scatter \
    scatter_axis \
    softmax \
    sort \
    steel_attention \
    steel_attention_nax \
    steel_conv \
    steel_conv_3d \
    steel_conv_general \
    steel_gemm_fused \
    steel_gemm_fused_nax \
    steel_gemm_gather \
    steel_gemm_gather_nax \
    steel_gemm_masked \
    steel_gemm_segmented \
    steel_gemm_splitk \
    steel_gemm_splitk_nax \
    ternary \
    ternary_ops \
    unary \
    unary_ops \
    utils

cd ../../..
make cpu_compiled_preamble

# run the command to do the build-time code generation for CUDA
cmake \
  -DMLX_SOURCE_ROOT="../Source/Cxxmlx/mlx/mlx/backend/cuda" \
  -DMLX_JIT_SOURCES="device/atomic_ops.cuh:device/binary_ops.cuh:device/cast_op.cuh:device/complex.cuh:device/config.h:device/fp16_math.cuh:device/gather.cuh:device/gather_axis.cuh:device/hadamard.cuh:device/indexing.cuh:device/scatter.cuh:device/scatter_axis.cuh:device/scatter_ops.cuh:device/ternary_ops.cuh:device/unary_ops.cuh:device/utils.cuh" \
  -P "../Source/Cxxmlx/mlx/mlx/backend/cuda/bin2h.cmake"

cd ..

rm -rf Source/Cxxmlx/mlx-generated/metal
rm -rf Source/Cxxmlx/mlx-generated/cuda
rm -f Source/Cxxmlx/mlx-generated/*
mkdir -p Source/Cxxmlx/mlx-generated/cuda
cp build/mlx/backend/metal/jit/* Source/Cxxmlx/mlx-generated
cp build/mlx/backend/cpu/compiled_preamble.cpp Source/Cxxmlx/mlx-generated
cp build/gen/cuda_jit_sources.h Source/Cxxmlx/mlx-generated/cuda

# we don't need the cmake build directory any more
rm -rf build

# remove any absolute paths and make them relative to the package root
for x in Source/Cxxmlx/mlx-generated/*.cpp ; do \
    sed -i .tmp -e "s:`pwd`/::g" $x
done;
rm Source/Cxxmlx/mlx-generated/*.tmp

# Update the headers
./tools/fix-metal-includes.sh

# prepare xcodeproj files
./tools/update-mlx-xcodeproj.sh
