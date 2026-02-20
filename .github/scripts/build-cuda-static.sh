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

cmake --build "$BUILD_DIR" --config Release -j "$(nproc)"
cmake --install "$BUILD_DIR" --config Release

# ---------------------------------------------------------------------------
# Merge all internal static libraries into a single libCmlx.a
# ---------------------------------------------------------------------------
echo "==> Merging static libraries"

LIBS=()

# All static libraries from the install tree (libmlxc.a and possibly libmlx.a
# if MLX's own install rules ran).
for dir in lib lib64; do
  if [ -d "$INSTALL_DIR/$dir" ]; then
    while IFS= read -r -d '' f; do
      LIBS+=("$f")
    done < <(find "$INSTALL_DIR/$dir" -maxdepth 1 -name "lib*.a" -print0 2>/dev/null)
  fi
done

# All static libraries built by FetchContent dependencies (libmlx.a, etc.).
# Skip any that were already collected from the install tree.
while IFS= read -r -d '' f; do
  basename_f="$(basename "$f")"
  already=false
  for l in "${LIBS[@]}"; do
    if [ "$(basename "$l")" = "$basename_f" ]; then
      already=true
      break
    fi
  done
  if ! $already; then
    LIBS+=("$f")
  fi
done < <(find "$BUILD_DIR/_deps" -name "lib*.a" -print0 2>/dev/null)

if [ ${#LIBS[@]} -eq 0 ]; then
  echo "Error: no static libraries found to merge" >&2
  exit 1
fi
printf '    %s\n' "${LIBS[@]}"

MERGE=$(mktemp -d)
trap 'rm -rf "$MERGE"' EXIT

# Extract object files from each archive into separate subdirectories
# to avoid filename collisions.
for i in "${!LIBS[@]}"; do
  d="$MERGE/$i"
  mkdir -p "$d"
  (cd "$d" && ar x "${LIBS[$i]}")
done

mkdir -p "$OUTPUT_DIR/lib"

# Use ld -r (relocatable link) to merge all object files into a single
# object. This is critical because ld.gold (used by Swift on Linux) does a
# single pass through archive members. With hundreds of cross-referencing
# objects the pass cannot resolve every symbol. A single merged .o guarantees
# that pulling in any symbol pulls in everything.
find "$MERGE" \( -name '*.o' -o -name '*.obj' \) -print0 \
  | xargs -0 ld -r -o "$MERGE/merged.o"
ar rcs "$OUTPUT_DIR/lib/libCmlx.a" "$MERGE/merged.o"
ranlib "$OUTPUT_DIR/lib/libCmlx.a"

# ---------------------------------------------------------------------------
# Copy public headers
# ---------------------------------------------------------------------------
echo "==> Copying headers"
mkdir -p "$OUTPUT_DIR/include"
cp -r "$INSTALL_DIR/include/mlx" "$OUTPUT_DIR/include/"

echo "==> Done: $OUTPUT_DIR/lib/libCmlx.a ($(du -sh "$OUTPUT_DIR/lib/libCmlx.a" | cut -f1))"
