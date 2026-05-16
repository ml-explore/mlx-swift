#!/bin/bash
# Generate C++ source that embeds the default Metal kernels for SwiftPM builds.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(realpath "${SCRIPT_DIR}/..")
METAL_DIR="${ROOT_DIR}/Source/Cmlx/mlx-generated/metal"
OUTPUT="${ROOT_DIR}/Source/Cmlx/mlx-generated/default_library.cpp"
TMP_SOURCE=$(mktemp)
TMP_OUTPUT=$(mktemp)
trap 'rm -f "${TMP_SOURCE}" "${TMP_OUTPUT}"' EXIT

KERNELS=(
  "arg_reduce.metal"
  "conv.metal"
  "gemv.metal"
  "layer_norm.metal"
  "random.metal"
  "rms_norm.metal"
  "rope.metal"
  "scaled_dot_product_attention.metal"
  "steel/attn/kernels/steel_attention.metal"
)

SEEN_FILES=""

emit_file() {
  local file
  file=$(realpath "$1")
  if printf '%s\n' "${SEEN_FILES}" | grep -Fqx "$file"; then
    return
  fi
  SEEN_FILES="${SEEN_FILES}
${file}"

  printf '\n// ---- embedded from %s ----\n' "${file#"$ROOT_DIR"/}" >> "${TMP_SOURCE}"
  local dir
  dir=$(dirname "$file")

  while IFS= read -r line || [[ -n "$line" ]]; do
    if [[ "$line" =~ ^[[:space:]]*#include[[:space:]]+\"([^\"]+)\" ]]; then
      local include="${BASH_REMATCH[1]}"
      local include_path="${dir}/${include}"
      if [[ -f "$include_path" ]]; then
        emit_file "$include_path"
      else
        printf '%s\n' "$line" >> "${TMP_SOURCE}"
      fi
    else
      printf '%s\n' "$line" >> "${TMP_SOURCE}"
    fi
  done < "$file"
}

for kernel in "${KERNELS[@]}"; do
  emit_file "${METAL_DIR}/${kernel}"
done

{
  printf '%s\n' 'namespace mlx::core::metal {'
  printf '%s\n' ''
  printf '%s\n' 'const char* embedded_default_library() {'
  printf '%s\n' '  return R"MLXEMB('
  cat "${TMP_SOURCE}"
  printf '%s\n' ')MLXEMB";'
  printf '%s\n' '}'
  printf '%s\n' ''
  printf '%s\n' '} // namespace mlx::core::metal'
} > "${TMP_OUTPUT}"

if [[ ! -f "${OUTPUT}" ]] || ! cmp -s "${TMP_OUTPUT}" "${OUTPUT}"; then
  cp "${TMP_OUTPUT}" "${OUTPUT}"
fi
