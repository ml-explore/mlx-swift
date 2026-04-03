#!/bin/bash
# Verify documentation builds without warnings

cd "$(dirname "$0")/.."

export MLX_SWIFT_BUILD_DOC=1

# Discover library product targets from Package.swift, skipping test/macro/executable targets
TARGETS=$(swift package dump-package | python3 -c "
import json, sys
pkg = json.load(sys.stdin)
targets = set()
for p in pkg['products']:
    if p['type'].get('library') is not None:
        targets.update(p['targets'])
for t in sorted(targets):
    print(t)
")

if [ -z "$TARGETS" ]; then
    echo "No targets found."
    exit 1
fi

FAILED=0

while IFS= read -r TARGET; do
    echo "Building documentation for $TARGET..."
    if ! swift package generate-documentation --target "$TARGET" --warnings-as-errors; then
        FAILED=1
    fi
    echo ""
done <<< "$TARGETS"

if [ "$FAILED" -ne 0 ]; then
    echo "Documentation build failed with warnings."
    exit 1
fi

echo "All documentation builds passed."
