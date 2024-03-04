#!/bin/zsh

# See MAINTENANCE.md : Updating `mlx` and `mlx-c`

set -e

if [[ ! -d Source ]]
then
    echo "Please run from the root of the repository, e.g. ./tools/update-mlx.sh"
    exit 1
fi

# run the command to do the build-time code generation

/bin/bash Source/Cmlx/mlx/mlx/backend/metal/make_compiled_preamble.sh \
    Source/Cmlx/mlx-generated/metal_compiled_preamble.cpp \
    /usr/bin/cc \
    Source/Cmlx/mlx

/bin/bash Source/Cmlx/mlx/mlx/backend/common/make_compiled_preamble.sh \
    Source/Cmlx/mlx-generated/common_compiled_preamble.cpp \
    /usr/bin/cc \
    Source/Cmlx/mlx \
    TRUE

# remove any absolute paths and make them relative to the package root

sed -i .tmp -e "s:`pwd`::g" Source/Cmlx/mlx-generated/metal_compiled_preamble.cpp
sed -i .tmp -e "s:`pwd`::g" Source/Cmlx/mlx-generated/common_compiled_preamble.cpp
rm Source/Cmlx/mlx-generated/*.tmp
