#!/bin/bash
# Verify documentation builds without warnings

cd "$(dirname "$0")/.."

export MLX_SWIFT_BUILD_DOC=1

# Targets to exclude from documentation verification
SKIP_TARGETS="Cmlx"

# Discover library product targets from Package.swift, skipping test/macro/executable targets
TARGETS=$(SKIP_TARGETS="$SKIP_TARGETS" swift package dump-package | python3 -c "
import json, os, sys
pkg = json.load(sys.stdin)
skip = set(os.environ.get('SKIP_TARGETS', '').split())
targets = set()
for p in pkg['products']:
    if p['type'].get('library') is not None:
        targets.update(p['targets'])
for t in sorted(targets - skip):
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
