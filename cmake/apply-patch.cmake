# Idempotent `git apply` for FetchContent PATCH_COMMAND.
#
# Usage (from a FetchContent_Declare PATCH_COMMAND):
#
# PATCH_COMMAND ${CMAKE_COMMAND} -DREPO=<SOURCE_DIR>
# -DPATCH=${CMAKE_CURRENT_SOURCE_DIR}/cmake/foo.patch -P
# ${CMAKE_CURRENT_SOURCE_DIR}/cmake/apply-patch.cmake
#
# Unlike `git apply ... || true` this does not swallow real failures: an
# already-applied patch is a no-op (the patch step re-runs on every update), but
# a patch that fails to apply is a hard error instead of a silently unpatched
# dependency.

cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED PATCH OR PATCH STREQUAL "")
  message(FATAL_ERROR "apply-patch.cmake: -DPATCH=<file> is required")
endif()

if(NOT DEFINED REPO
   OR REPO STREQUAL ""
   OR NOT IS_DIRECTORY "${REPO}")
  # the patch step runs with the dependency source dir as its working directory
  set(REPO "${CMAKE_CURRENT_BINARY_DIR}")
endif()

if(NOT EXISTS "${PATCH}")
  message(FATAL_ERROR "apply-patch.cmake: no such patch file: ${PATCH}")
endif()

# note: deliberately not `find_package(Git REQUIRED)` -- that pulls in
# FindPackageHandleStandardArgs in script mode for no benefit here.
if(NOT DEFINED GIT_EXECUTABLE OR GIT_EXECUTABLE STREQUAL "")
  find_program(GIT_EXECUTABLE NAMES git git.exe)
endif()

if(NOT GIT_EXECUTABLE)
  message(
    FATAL_ERROR "apply-patch.cmake: git not found (pass -DGIT_EXECUTABLE=)")
endif()

message(STATUS "apply-patch: repo=${REPO} patch=${PATCH} git=${GIT_EXECUTABLE}")

# already applied? (the patch step re-runs whenever the update step re-runs)
execute_process(
  COMMAND ${GIT_EXECUTABLE} apply --reverse --check "${PATCH}"
  WORKING_DIRECTORY "${REPO}"
  RESULT_VARIABLE reverse_check
  OUTPUT_QUIET ERROR_QUIET)

if(reverse_check EQUAL 0)
  message(STATUS "apply-patch: already applied: ${PATCH}")
  return()
endif()

# note: git's stdout/stderr are intentionally *not* captured so that the real
# reason a patch failed always reaches the build log, even if this script dies
# before it can format a message of its own.
execute_process(
  COMMAND ${GIT_EXECUTABLE} apply --verbose "${PATCH}"
  WORKING_DIRECTORY "${REPO}"
  RESULT_VARIABLE apply_result)

if(NOT apply_result EQUAL 0)
  message(FATAL_ERROR "apply-patch: failed to apply ${PATCH} in ${REPO} "
                      "(git apply exit ${apply_result}, see output above)")
endif()

message(STATUS "apply-patch: applied ${PATCH}")
