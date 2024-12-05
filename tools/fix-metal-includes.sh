#!/bin/bash
# Fixing include path for mlx-swift metal headers

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(realpath "${SCRIPT_DIR}/..")

# Where the files end up
OUTPUT_DIR="${ROOT_DIR}/Source/Cmlx/mlx-generated/metal"

# The Cmlx source dir
CMLX_MLX_DIR="${ROOT_DIR}/Source/Cmlx/mlx"

# sub-directory of Cmlx source containing the kernels
KERNELS_INCLUDE_PATH="mlx/backend/metal/kernels"

KERNELS_DIR="${CMLX_MLX_DIR}/${KERNELS_INCLUDE_PATH}"

# list of kernels files to process
# see Source/Cmlx/mlx/mlx/backend/metal/kernels/CMakeLists.txt
KERNEL_LIST=" \
arg_reduce.metal \
conv.metal \
gemv.metal \
layer_norm.metal \
random.metal \
rms_norm.metal \
rope.metal \
scaled_dot_product_attention.metal \
steel/attn/kernels/steel_attention.metal"

# We fixup all the header files AND the listed kernel files
HEADERS=$(find "${KERNELS_DIR}" -name "*.h")
KERNELS=$(for file in ${KERNEL_LIST}; do  echo "${KERNELS_DIR}/${file}"; done)

# Regular expression to replace include directives
PATTERN="^#include \"${KERNELS_INCLUDE_PATH}/([^\"]*)\""

mkdir -p "${OUTPUT_DIR}"

# Mimic the original logic in  PrepareMetalShaders::transformIncludes
# Returns rootPath, a string containing a sequence of "../../" to prefix the
# include path
function replaceIncludePrefix {
    #Extract components up to the output dir and drop the last one
    #swift: let pathUnderKernels = url.pathComponents.drop { $0 != "output" }.dropLast()

    absolutePath=$(realpath "${1}")
    absoluteOut=$(realpath "${OUTPUT_DIR}")
    remainingPath=${absolutePath#"$absoluteOut"/}

    # Doing the `dropLast` with `dirname`, handling the case where it returns `.``
    remainingPath=$(dirname "${remainingPath}" | sed -E 's|^\.$||')

    # Build the root path
    # swift:  let rootPath =Array(repeating: "..", count: pathUnderKernels.count - 1).joined(separator: "/")
    #       + ((pathUnderKernels.count - 1 == 0) ? "" : "/")
    IFS='/' read -r -a path <<< "${remainingPath}"
    count=${#path[@]}

    if [ "$count" -le 0 ]; then
        root_path=""
    else
        root_path=$(printf "../%.0s" $(seq 1 "${count}"))
    fi
    echo "${root_path}"
}

# First pass : copy the files if needed
for src in ${HEADERS} ${KERNELS}; do

    relative_path=${src#"$KERNELS_DIR"/}
    dest=${OUTPUT_DIR}/${relative_path}

    # If destination file doesn't exist or if it's older than the source
    # copy from source and replace the #include directives
    if [ ! -e "$dest" ] || [ "$src" -nt "$dest" ]; then
        echo "${src} -> ${dest}"
        mkdir -p "$(dirname "${dest}")"
        cp -p "${src}" "${dest}"
    else
        echo "Skipping $src (more recent destination)"
    fi

done

# second pass: update the include lines
# iterating on src to only process the list of files we copied
# (in case the destination directory has other unrelated files)
for src in ${HEADERS} ${KERNELS}; do

    relative_path=${src#"$KERNELS_DIR"/}
    dest=${OUTPUT_DIR}/${relative_path}
    prefix=$(replaceIncludePrefix "${dest}")

    # for each matching input line, compute the relative path, then replace the line
    while read -r includeLine; do
        includePath=$(echo "${includeLine}" | sed -E -n "s|${PATTERN}|\1|p")

        # Note the absence of "/" between the prefix and the path
        replace="${prefix}${includePath}"

        # Replace the include line with the new one
        echo sed -i '' -e "s|${KERNELS_INCLUDE_PATH}/${includePath}|${replace}|" "${dest}"
        sed -i '' -e "s|${KERNELS_INCLUDE_PATH}/${includePath}|${replace}|" "${dest}"

    done < <(grep -E -o "${PATTERN}" "${dest}")
done
