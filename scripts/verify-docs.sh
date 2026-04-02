#!/bin/bash
# Verify documentation builds without warnings

set -e
cd "$(dirname "$0")/.."

export MLX_SWIFT_BUILD_DOC=1

TARGETS=("MLX" "MLXRandom" "MLXNN" "MLXOptimizers" "MLXFFT" "MLXLinalg" "MLXFast")
FAILED=0

for TARGET in "${TARGETS[@]}"; do
    echo "Building documentation for $TARGET..."
    if ! swift package generate-documentation --target "$TARGET" --warnings-as-errors; then
        FAILED=1
    fi
    echo ""
done

if [ "$FAILED" -ne 0 ]; then
    echo "Documentation build failed with warnings."
    exit 1
fi

echo "All documentation builds passed."
