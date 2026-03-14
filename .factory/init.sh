#!/bin/bash
set -e

# Idempotent environment setup for mlx-swift distributed mission
# No dependencies to install -- all C/C++ code is vendored via submodules

# Ensure git submodules are initialized
cd "$(dirname "$0")/.."
git submodule update --init --recursive

echo "mlx-swift environment ready"
