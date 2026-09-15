#!/bin/bash
set -ex

# [Pre-setup] Check required env variables.
: "${TOOLKIT_VERSION:?Error: TOOLKIT_VERSION is not set. Please define it in the GitHub Actions workflow.}"
CUDA_BIN_PATH="/usr/local/${TOOLKIT_VERSION}/bin"
export PATH="$CUDA_BIN_PATH:$PATH"

# [CMake] CI Build Sanity Check: Verifies code compilation, not for release.
export CMAKE_ARGS="-DCMAKE_COMPILE_WARNING_AS_ERROR=ON"
export DEBUG=1
export CMAKE_C_COMPILER=/usr/bin/clang
export CMAKE_CXX_COMPILER=/usr/bin/clang++

rm -rf build
mkdir -p build
pushd build
cmake -DMLX_BUILD_METAL=OFF -DMLX_BUILD_CUDA=ON -DMLX_C_BUILD_EXAMPLES=OFF .. -G Ninja

# Verify the local mlx patch is actually in place: without it the examples below
# crash on exit, which is easy to misread as an unrelated CUDA failure.
git -C _deps/mlx-src apply --reverse --check "$PWD/../cmake/mlx.patch" || {
  echo "error: cmake/mlx.patch is not applied to _deps/mlx-src"
  exit 1
}

ninja
./example1 --device gpu
./tutorial --device gpu
popd
