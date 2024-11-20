#!/bin/zsh

# See MAINTENANCE.md : Updating `mlx` and `mlx-c`

set -e

if [[ ! -d Source ]]
then
    echo "Please run from the root of the repository, e.g. ./tools/update-mlx.sh"
    exit 1
fi

# run the command to do the build-time code generation

mkdir build
cd build
cmake ../Source/Cmlx/mlx -DMLX_METAL_JIT=ON -DMACOS_VERSION=14.0

# NOTE:
# until mlx supports overriding the METAL_VERSION you will need to edit
# Source/Cmlx/mlx/mlx/backend/metal/CMakeLists.txt and manually set the METAL_VERSION
# to "3.0"

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
    gather \
    gemm \
    gemv_masked \
    hadamard \
    quantized \
    reduce \
    reduce_utils \
    scan \
    scatter \
    softmax \
    sort \
    steel_conv \
    steel_conv_general \
    steel_gemm_fused \
    steel_gemm_masked \
    steel_gemm_splitk \
    ternary \
    ternary_ops \
    unary \
    unary_ops \
    utils

cd ../common
make cpu_compiled_preamble

cd ../../../..

rm Source/Cmlx/mlx-generated/*
cp build/mlx/backend/metal/jit/* Source/Cmlx/mlx-generated
cp build/mlx/backend/common/compiled_preamble.cpp Source/Cmlx/mlx-generated

# we don't need the cmake build directory any more
rm -rf build

# remove any absolute paths and make them relative to the package root
for x in Source/Cmlx/mlx-generated/*.cpp ; do \
    sed -i .tmp -e "s:`pwd`/::g" $x
done;
rm Source/Cmlx/mlx-generated/*.tmp

# Update the headers
./tools/fix-metal-includes.sh