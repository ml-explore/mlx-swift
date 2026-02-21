#!/bin/bash
set -euo pipefail

# Builds MLX-C (and its MLX dependency) as a single merged static library
# with CUDA support for the current system architecture.
#
# Prerequisites:
#   - CUDA toolkit installed and TOOLKIT_VERSION env var set
#   - CMake >= 3.16 and Ninja installed
#   - Clang C/C++ compiler
#
# Usage:
#   TOOLKIT_VERSION=cuda-12.9 bash build-cuda-static.sh [--output DIR]
#
# Outputs to build/output/<arch>/:
#   lib/libCmlx.a         Merged static library (MLX-C + MLX)
#   include/mlx/c/*.h     Public C API headers

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
  *) echo "Error: Unsupported architecture: $ARCH" >&2; exit 1 ;;
esac

OUTPUT_DIR="$REPO_ROOT/build/output/$ARCH"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Error: Unknown option: $1" >&2; exit 1 ;;
  esac
done

# CUDA setup.
: "${TOOLKIT_VERSION:?Error: TOOLKIT_VERSION is not set.}"
export PATH="/usr/local/${TOOLKIT_VERSION}/bin:$PATH"

echo "==> Building MLX-C (CUDA) for $ARCH"

BUILD_DIR="$REPO_ROOT/build/cmake-cuda-$ARCH"
INSTALL_DIR="$BUILD_DIR/_install"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Configure MLX-C. It fetches MLX via FetchContent; the MLX_BUILD_* flags
# propagate through as CMake cache variables.
cmake -S "$REPO_ROOT/Source/Cmlx/mlx-c" -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DMLX_BUILD_METAL=OFF \
  -DMLX_BUILD_CUDA=ON \
  -DMLX_CUDA_ARCHITECTURES="70;75;80;86;87;89;90;100" \
  -DMLX_BUILD_TESTS=OFF \
  -DMLX_BUILD_EXAMPLES=OFF \
  -DMLX_BUILD_BENCHMARKS=OFF \
  -DMLX_BUILD_PYTHON_BINDINGS=OFF \
  -DMLX_C_BUILD_EXAMPLES=OFF

# ---------------------------------------------------------------------------
# Patch MLX visibility after FetchContent downloads sources.
#
# MLX sets CXX_VISIBILITY_PRESET=hidden for shared-library builds. When we
# merge objects with `ld -r` to create a single relocatable object, the
# hidden symbols become internal and unresolvable from the C wrapper objects.
# For a static-only artifact this visibility serves no purpose, so we remove
# it and re-configure before building.
# ---------------------------------------------------------------------------
MLX_CMAKE="$BUILD_DIR/_deps/mlx-src/mlx/CMakeLists.txt"
if [ -f "$MLX_CMAKE" ]; then
  echo "==> Patching MLX visibility to default (for ld -r compatibility)"
  sed -i 's/CXX_VISIBILITY_PRESET hidden/CXX_VISIBILITY_PRESET default/' "$MLX_CMAKE"
  sed -i 's/CUDA_VISIBILITY_PRESET hidden/CUDA_VISIBILITY_PRESET default/' "$MLX_CMAKE"
  sed -i 's/VISIBILITY_INLINES_HIDDEN ON/VISIBILITY_INLINES_HIDDEN OFF/' "$MLX_CMAKE"
  # Re-configure to pick up the visibility changes
  cmake -S "$REPO_ROOT/Source/Cmlx/mlx-c" -B "$BUILD_DIR" -G Ninja
fi

cmake --build "$BUILD_DIR" --config Release -j "${BUILD_JOBS:-$(nproc)}"
cmake --install "$BUILD_DIR" --config Release

# ---------------------------------------------------------------------------
# Merge all internal static libraries into a single libCmlx.a
# ---------------------------------------------------------------------------
echo "==> Merging static libraries"

# Collect all static libraries that should be merged. We use an associative
# array keyed by basename to avoid duplicates (prefer install-tree copies).
declare -A SEEN
LIBS=()

add_lib() {
  local f="$1"
  local base
  base="$(basename "$f")"
  if [[ -z "${SEEN[$base]+x}" ]]; then
    SEEN[$base]=1
    LIBS+=("$f")
  fi
}

# 1. All static libraries from the install tree.
for dir in lib lib64; do
  if [ -d "$INSTALL_DIR/$dir" ]; then
    while IFS= read -r -d '' f; do
      add_lib "$f"
    done < <(find "$INSTALL_DIR/$dir" -name "lib*.a" -print0 2>/dev/null)
  fi
done

# 2. All static libraries from the entire build tree (catches libmlx.a even
#    when CMake's FetchContent EXCLUDE_FROM_ALL prevents it from being
#    installed). Skip the install tree to avoid double-counting.
while IFS= read -r -d '' f; do
  add_lib "$f"
done < <(find "$BUILD_DIR" -path "$INSTALL_DIR" -prune -o -name "lib*.a" -print0 2>/dev/null)

if [ ${#LIBS[@]} -eq 0 ]; then
  echo "Error: no static libraries found to merge" >&2
  exit 1
fi
echo "==> Collected ${#LIBS[@]} libraries:"
for lib in "${LIBS[@]}"; do
  count=$(ar t "$lib" 2>/dev/null | wc -l)
  size=$(du -sh "$lib" | cut -f1)
  echo "    $lib ($count members, $size)"
done

# Verify libmlx.a was found (the core C++ library).
found_mlx=false
for lib in "${LIBS[@]}"; do
  if [[ "$(basename "$lib")" == "libmlx.a" ]]; then
    found_mlx=true
    break
  fi
done
if ! $found_mlx; then
  echo "Error: libmlx.a not found in collected libraries. The merged artifact will be incomplete." >&2
  echo "Searched: $INSTALL_DIR and $BUILD_DIR" >&2
  find "$BUILD_DIR" -name "libmlx*" 2>/dev/null | head -10 >&2
  exit 1
fi

MERGE=$(mktemp -d)
trap 'rm -rf "$MERGE"' EXIT

mkdir -p "$OUTPUT_DIR/lib"

echo "==> Merging ${#LIBS[@]} libraries with ld -r --whole-archive"

# Use ld -r (relocatable link) with --whole-archive to merge all static
# libraries into a single relocatable object. This avoids issues with ar x
# failing on archives whose members have directory-path names (e.g.
# CMakeFiles/mlx.dir/mlx/ops.cpp.o) where intermediate directories don't
# exist.
#
# The single merged .o is critical because ld.gold (used by Swift on Linux)
# does a single pass through archive members. With hundreds of cross-
# referencing objects it cannot resolve every symbol. A single merged .o
# guarantees that pulling in any symbol pulls in everything.
#
# Note: this requires MLX to be compiled with default visibility (patched
# above), otherwise ld -r internalizes hidden symbols making them unreachable.
ld -r --whole-archive -o "$MERGE/merged.o" "${LIBS[@]}"
ar rcs "$OUTPUT_DIR/lib/libCmlx.a" "$MERGE/merged.o"
ranlib "$OUTPUT_DIR/lib/libCmlx.a"

# Verify the merged library has the expected core symbols.
undef_mlx=$(nm "$OUTPUT_DIR/lib/libCmlx.a" 2>/dev/null | grep -c " U.*mlx" || true)
defined_mlx=$(nm "$OUTPUT_DIR/lib/libCmlx.a" 2>/dev/null | grep -c " [Tt].*mlx" || true)
echo "==> Symbol check: $defined_mlx defined mlx symbols, $undef_mlx undefined mlx symbols"
if [ "$undef_mlx" -gt 50 ]; then
  echo "Warning: $undef_mlx undefined mlx symbols in merged library" >&2
  nm "$OUTPUT_DIR/lib/libCmlx.a" 2>/dev/null | grep " U.*mlx" > "$MERGE/undef_syms.txt" || true
  echo "First 20 undefined mlx symbols:" >&2
  head -20 "$MERGE/undef_syms.txt" >&2
fi

# ---------------------------------------------------------------------------
# Copy public headers
# ---------------------------------------------------------------------------
echo "==> Copying headers"
mkdir -p "$OUTPUT_DIR/include"
cp -r "$INSTALL_DIR/include/mlx" "$OUTPUT_DIR/include/"

echo "==> Done: $OUTPUT_DIR/lib/libCmlx.a ($(du -sh "$OUTPUT_DIR/lib/libCmlx.a" | cut -f1))"
