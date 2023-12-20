cmake_minimum_required(VERSION 3.16)
project(SwiftMLX LANGUAGES C CXX Swift)

set(MLX_C_BUILD_EXAMPLES OFF)
include(FetchContent)
# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

FetchContent_Declare(
  mlxc
  GIT_REPOSITORY
    "git@github.com:ml-explore/mlx-c.git" # https://github.com/ml-explore/mlx-c.git"
  GIT_TAG "main")
FetchContent_MakeAvailable(mlxc)

# Mlx lib
add_library(
  Mlx STATIC
  ${CMAKE_CURRENT_LIST_DIR}/Source/Mlx/Array.swift
  ${CMAKE_CURRENT_LIST_DIR}/Source/Mlx/Device.swift
  ${CMAKE_CURRENT_LIST_DIR}/Source/Mlx/Stream.swift)
target_include_directories(Mlx
                           PUBLIC ${CMAKE_CURRENT_LIST_DIR}/Source/Cmlx/include)
target_link_libraries(Mlx PRIVATE mlxc)

# Examples
add_executable(example1
               ${CMAKE_CURRENT_LIST_DIR}/Source/Examples/Example1.swift)
target_link_libraries(example1 PRIVATE Mlx)
