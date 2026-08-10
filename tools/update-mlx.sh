#!/bin/zsh

# See MAINTENANCE.md : Updating `mlx` and `mlx-c`

set -e

if [[ ! -d Source ]]
then
    echo "Please run from the root of the repository, e.g. ./tools/update-mlx.sh"
    exit 1
fi

mlx_source="$PWD/Source/Cmlx/mlx"

# copy mlx-c headers to build area
rm -f Source/Cmlx/include/mlx/c/*
cp Source/Cmlx/mlx-c/mlx/c/*.h Source/Cmlx/include/mlx/c

# run the command to do the build-time code generation for Metal

mkdir build
cd build
cmake ../Source/Cmlx/mlx -DMLX_METAL_JIT=ON -DMACOS_VERSION=14.0

# run the cmake build to generate the source files
cd mlx/backend/metal

# one target per make_jit_source() in mlx/backend/metal/CMakeLists.txt, which is
# where the set of Metal jit sources is declared
metal_jit_targets=(${(f)"$(tr '\n' ' ' < "$mlx_source/mlx/backend/metal/CMakeLists.txt" \
    | grep -oE 'make_jit_source\( *[a-zA-Z_0-9/]+' \
    | sed -E 's/make_jit_source\( *//' \
    | sed 's|.*/||' \
    | sort -u)"})

if (( ! $#metal_jit_targets ))
then
    echo "Found no make_jit_source() targets in mlx/backend/metal/CMakeLists.txt"
    exit 1
fi

make $metal_jit_targets

cd ../../..
make cpu_compiled_preamble

# run the command to do the build-time code generation for CUDA

cuda_source="$mlx_source/mlx/backend/cuda"

# the same set that the file(GLOB) in mlx/backend/cuda/CMakeLists.txt embeds
cuda_jit_sources=(${cuda_source}/device/*.(h|cuh)(N))
cuda_jit_sources=(${cuda_jit_sources#$cuda_source/})

if (( ! $#cuda_jit_sources ))
then
    echo "Found no jit sources in mlx/backend/cuda/device"
    exit 1
fi

cmake \
  -DMLX_SOURCE_ROOT="$cuda_source" \
  -DMLX_JIT_SOURCES="${(j.:.)cuda_jit_sources}" \
  -P "$cuda_source/bin2h.cmake"

cd ..

rm -rf Source/Cmlx/mlx-generated/metal
rm -rf Source/Cmlx/mlx-generated/cuda
rm -f Source/Cmlx/mlx-generated/*
mkdir -p Source/Cmlx/mlx-generated/cuda
cp build/mlx/backend/metal/jit/* Source/Cmlx/mlx-generated
cp build/mlx/backend/cpu/compiled_preamble.cpp Source/Cmlx/mlx-generated
cp build/gen/cuda_jit_sources.h Source/Cmlx/mlx-generated/cuda

# we don't need the cmake build directory any more
rm -rf build

# remove any absolute paths and make them relative to the package root
for x in Source/Cmlx/mlx-generated/*.cpp ; do \
    sed -i .tmp -e "s:`pwd`/::g" $x
done;
rm Source/Cmlx/mlx-generated/*.tmp

# Update the headers
./tools/fix-metal-includes.sh

# prepare xcodeproj files
./tools/update-mlx-xcodeproj.sh
