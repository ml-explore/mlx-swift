cmake_minimum_required(VERSION 3.16)
project(MLXCex LANGUAGES C CXX Swift)

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

add_library(
  Mlx STATIC
  ${CMAKE_CURRENT_LIST_DIR}/Source/Mlx/array.swift
  ${CMAKE_CURRENT_LIST_DIR}/Source/Mlx/device.swift
  ${CMAKE_CURRENT_LIST_DIR}/Source/Mlx/stream.swift)
target_include_directories(Mlx
                           PUBLIC ${CMAKE_CURRENT_LIST_DIR}/Source/Cmlx/include)
target_link_libraries(Mlx PRIVATE mlxc)

add_executable(example example.swift)
target_link_libraries(example PRIVATE Mlx)
