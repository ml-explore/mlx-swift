#!/bin/bash
set -ex

# [Setup] Install dependencies inside the container.
# Note: mlx-c requires a C++20 toolchain with <format> support:
#   - libstdc++ >= 13 (Ubuntu noble, Fedora 39+)
#   - or libc++ >= 17
# Debian bookworm (libstdc++ 12) is *not* sufficient.
if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y \
        build-essential \
        cmake \
        ninja-build \
        libblas-dev \
        liblapacke-dev \
        libopenblas-dev

    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++

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

    # Fedora's default gcc (13+) has <format>; keep it as the compiler so this
    # build keeps matching what has been passing in CI.
    export CC=/usr/bin/gcc
    export CXX=/usr/bin/g++

else
    echo "No supported package manager found (apt-get, dnf)"
    exit 1
fi

# [CMake] CI Build Sanity Check: Verifies code compilation, not for release.
export CMAKE_ARGS="-DCMAKE_COMPILE_WARNING_AS_ERROR=ON"
export DEBUG=1

"$CXX" --version

rm -rf build
mkdir -p build
pushd build
cmake -DMLX_BUILD_METAL=OFF .. -G Ninja
ninja
./example1
./tutorial
popd
