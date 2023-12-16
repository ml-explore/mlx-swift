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

add_executable(example example.swift)
target_include_directories(example PRIVATE ${CMAKE_CURRENT_LIST_DIR}/Cmlx)
target_link_libraries(example PRIVATE mlxc)
