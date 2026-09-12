cmake_minimum_required(VERSION 3.16)
project(MLXSwift LANGUAGES C CXX Swift)

# ----------------------------- Setup -----------------------------
# note: 1:1 mirror of MLX configs

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_INSTALL_MESSAGE NEVER)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ----------------------------- Configuration -----------------------------
# note: mirrors a subset of MLX options exactly (1:1 mapping)

# note: MLX_BUILD_EXAMPLES is owned by mlx itself (see below), so the Swift
# examples use their own option name to avoid colliding with mlx's C++ examples
option(MLX_SWIFT_BUILD_EXAMPLES "Build Swift examples for mlx-swift" ON)
option(MLX_BUILD_METAL "Build metal backend" ON)
option(MLX_BUILD_CUDA "Build cuda backend" OFF)

# ----------------------------- Lib -----------------------------

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# mlx

# note: patches are applied via cmake/apply-patch.cmake so that they are
# idempotent (the patch step re-runs on every update step) while still failing
# the build loudly if a patch does not apply -- a silently unpatched dependency
# is very hard to diagnose (it typically shows up as a runtime failure).
set(apply_patch ${CMAKE_COMMAND} -DREPO=<SOURCE_DIR>)
set(apply_patch_script -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-patch.cmake)

# local apply of https://github.com/ml-explore/mlx/pull/4480 for CUDA
set(mlx_patch
    ${apply_patch} -DPATCH=${CMAKE_CURRENT_SOURCE_DIR}/cmake/mlx.patch
    ${apply_patch_script})

# note: this must be declared (and named `mlx`) before mlx-c is made available
# so that our pinned/patched version wins over the one mlx-c declares. Because
# we now pull mlx in first, we are also responsible for the option defaults that
# mlx-c would normally set for us.
set(MLX_BUILD_TESTS OFF)
set(MLX_BUILD_EXAMPLES OFF)
set(MLX_BUILD_BENCHMARKS OFF)
set(MLX_BUILD_PYTHON_BINDINGS OFF)

FetchContent_Declare(
  mlx
  GIT_REPOSITORY "https://github.com/ml-explore/mlx.git"
  GIT_TAG "1f8e74e3f12f31365464a6867c6579f0e9b29d85"
  PATCH_COMMAND ${mlx_patch})
FetchContent_MakeAvailable(mlx)

# mlx-c
set(MLX_C_BUILD_EXAMPLES OFF)
FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG "c74db5307cc8ce122f48d97ef951b30578674e7f")
FetchContent_MakeAvailable(mlx-c)

# swift-numerics
set(swift_numerics_patch
    ${apply_patch}
    -DPATCH=${CMAKE_CURRENT_SOURCE_DIR}/cmake/swift-numerics.patch
    ${apply_patch_script})
FetchContent_Declare(
  swift-numerics
  GIT_REPOSITORY "https://github.com/apple/swift-numerics.git"
  GIT_TAG "1.0.2"
  PATCH_COMMAND ${swift_numerics_patch})
FetchContent_MakeAvailable(swift-numerics)

# MLX package
if(MLX_BUILD_CUDA)
  find_package(CUDAToolkit REQUIRED)
  include(${cudnn_frontend_SOURCE_DIR}/cmake/cuDNN.cmake)
  message(STATUS "Patching 'mlxc' target to add public CUDA dependencies.")
  target_link_libraries(
    mlxc
    PUBLIC CUDA::cudart
           CUDA::cublasLt # Promoted from PRIVATE, required by mlx
           CUDA::nvrtc # Promoted from PRIVATE, required by mlx
           CUDA::cuda_driver # Promoted from PRIVATE, required by mlx
           CUDNN::cudnn_all # Promoted from PRIVATE, required by mlx
  )
endif()
file(GLOB MLX-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLX/*.swift)

if(NOT MLX_BUILD_METAL)
  list(REMOVE_ITEM MLX-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLX/GPU+Metal.swift
       ${CMAKE_CURRENT_LIST_DIR}/Source/MLX/MLXArray+Metal.swift)
endif()

if(NOT MLX_BUILD_CUDA)
  list(REMOVE_ITEM MLX-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLX/GPU+CUDA.swift)
endif()

add_library(MLX STATIC ${MLX-src})
target_include_directories(MLX
                           PUBLIC ${CMAKE_CURRENT_LIST_DIR}/Source/Cmlx/include)
# Note: mlxc is target name exported by mlx-c
target_link_libraries(MLX PUBLIC mlxc Numerics)
target_compile_options(MLX PRIVATE -package-name MLX)

# MLXRandom package
file(GLOB MLXRandom-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLXRandom/*.swift)
add_library(MLXRandom STATIC ${MLXRandom-src})
target_link_libraries(MLXRandom PRIVATE MLX)

# MLXFast package
file(GLOB MLXFast-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLXFast/*.swift)
add_library(MLXFast STATIC ${MLXFast-src})
target_link_libraries(MLXFast PRIVATE MLX)

# MLXNN package
file(GLOB MLXNN-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLXNN/*.swift)
add_library(MLXNN STATIC ${MLXNN-src})
target_link_libraries(MLXNN PRIVATE MLX MLXRandom MLXFast)

# MLXOptimizers package
file(GLOB MLXOptimizers-src
     ${CMAKE_CURRENT_LIST_DIR}/Source/MLXOptimizers/*.swift)
add_library(MLXOptimizers STATIC ${MLXOptimizers-src})
target_link_libraries(MLXOptimizers PRIVATE MLX MLXNN)

# MLXFFT package
file(GLOB MLXFFT-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLXFFT/*.swift)
add_library(MLXFFT STATIC ${MLXFFT-src})
target_link_libraries(MLXFFT PRIVATE MLX)

# MLXLinalg package
file(GLOB MLXLinalg-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLXLinalg/*.swift)
add_library(MLXLinalg STATIC ${MLXLinalg-src})
target_link_libraries(MLXLinalg PRIVATE MLX)

# Examples
if(MLX_SWIFT_BUILD_EXAMPLES)
  add_executable(example1
                 ${CMAKE_CURRENT_LIST_DIR}/Source/Examples/Example1.swift)
  target_link_libraries(example1 PRIVATE MLX)
  target_compile_options(example1 PRIVATE -parse-as-library)

  add_executable(tutorial
                 ${CMAKE_CURRENT_LIST_DIR}/Source/Examples/Tutorial.swift)
  target_link_libraries(tutorial PRIVATE MLX)
  target_compile_options(tutorial PRIVATE -parse-as-library)
endif()
