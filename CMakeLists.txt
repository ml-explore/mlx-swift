cmake_minimum_required(VERSION 3.16)
project(MLXSwift LANGUAGES C CXX Swift)

include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# mlx-c
FetchContent_Declare(
  mlx-c
  GIT_REPOSITORY "https://github.com/ml-explore/mlx-c.git"
  GIT_TAG "v0.0.5")
FetchContent_MakeAvailable(mlx-c)

# swift-numerics
set(swift_numerics_patch git apply
                         ${CMAKE_CURRENT_SOURCE_DIR}/cmake/swift-numerics.patch)
FetchContent_Declare(
  swift-numerics
  GIT_REPOSITORY "https://github.com/apple/swift-numerics.git"
  GIT_TAG "1.0.2"
  PATCH_COMMAND ${swift_numerics_patch} || true)
FetchContent_MakeAvailable(swift-numerics)

# MLX package
file(GLOB MLX-src ${CMAKE_CURRENT_LIST_DIR}/Source/MLX/*.swift)
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
add_executable(example1
               ${CMAKE_CURRENT_LIST_DIR}/Source/Examples/Example1.swift)
target_link_libraries(example1 PRIVATE MLX)
target_compile_options(example1 PRIVATE -parse-as-library)

add_executable(tutorial
               ${CMAKE_CURRENT_LIST_DIR}/Source/Examples/Tutorial.swift)
target_link_libraries(tutorial PRIVATE MLX)
target_compile_options(tutorial PRIVATE -parse-as-library)
