#!/bin/bash
set -ex

# [Setup] Install dependencies inside the container.
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        libblas-dev \
        liblapacke-dev \
        libopenblas-dev

elif command -v dnf >/dev/null 2>&1; then
    dnf update -y
    dnf install -y \
        blas-devel \
        lapack-devel \
        openblas-devel \
        make \
        cmake \
        clang \
        ninja-build

else
    echo "No supported package manager found (apt-get, dnf)"
    exit 1
fi

# [CMake] CI Build Sanity Check: Verifies code compilation, not for release.
export CMAKE_ARGS="-DCMAKE_COMPILE_WARNING_AS_ERROR=ON"
export DEBUG=1
export CMAKE_C_COMPILER=/usr/bin/clang
export CMAKE_CXX_COMPILER=/usr/bin/clang++

rm -rf build
mkdir -p build
pushd build
cmake -DMLX_BUILD_METAL=OFF .. -G Ninja
ninja
./example1
./tutorial
popd
