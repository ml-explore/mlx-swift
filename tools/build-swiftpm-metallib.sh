#!/bin/bash
# Build the default Metal library resource used by SwiftPM Cmlx builds.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_METALLIB" >&2
  exit 64
fi

OUTPUT="$1"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(realpath "${SCRIPT_DIR}/..")
KERNELS_DIR="${ROOT_DIR}/Source/Cmlx/mlx/mlx/backend/metal/kernels"

normalize_sdk_name() {
  local raw="$1"
  raw=$(basename "${raw}")
  raw=$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')
  case "${raw}" in
    macosx*) echo "macosx" ;;
    iphoneos*) echo "iphoneos" ;;
    iphonesimulator*) echo "iphonesimulator" ;;
    appletvos*) echo "appletvos" ;;
    appletvsimulator*) echo "appletvsimulator" ;;
    xros* | visionos*) echo "xros" ;;
    xrsimulator* | visionsimulator*) echo "xrsimulator" ;;
    *) echo "${raw}" ;;
  esac
}

requested_sdk="${SDK_NAME:-${PLATFORM_NAME:-}}"
if [[ -z "${requested_sdk}" && -n "${SDKROOT:-}" ]]; then
  requested_sdk=$(basename "${SDKROOT}")
fi
SDK=$(normalize_sdk_name "${requested_sdk:-macosx}")

case "${SDK}" in
  macosx)
    DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    deployment_flag=("-mmacosx-version-min=${DEPLOYMENT_TARGET}")
    ;;
  iphoneos | iphonesimulator)
    DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-${IOS_DEPLOYMENT_TARGET:-17.0}}"
    deployment_flag=("-mios-version-min=${DEPLOYMENT_TARGET}")
    ;;
  appletvos | appletvsimulator)
    DEPLOYMENT_TARGET="${TVOS_DEPLOYMENT_TARGET:-17.0}"
    deployment_flag=("-mtvos-version-min=${DEPLOYMENT_TARGET}")
    ;;
  xros)
    DEPLOYMENT_TARGET="${XROS_DEPLOYMENT_TARGET:-${VISIONOS_DEPLOYMENT_TARGET:-1.0}}"
    deployment_flag=("-mtargetos=xros${DEPLOYMENT_TARGET}")
    ;;
  xrsimulator)
    DEPLOYMENT_TARGET="${XROS_DEPLOYMENT_TARGET:-${VISIONOS_DEPLOYMENT_TARGET:-1.0}}"
    deployment_flag=("-mtargetos=xros${DEPLOYMENT_TARGET}-simulator")
    ;;
  *)
    echo "unsupported SDK '${SDK}'" >&2
    exit 65
    ;;
esac

METAL=$(xcrun -sdk "${SDK}" -find metal)
METALLIB=$(xcrun -sdk "${SDK}" -find metallib)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

metal_version=$(
  printf '%s\n' '__METAL_VERSION__' |
    "${METAL}" "${deployment_flag[@]}" -E -x metal -P - |
    tail -1 |
    tr -d '[:space:]'
)
metal_version=${metal_version:-0}

kernels=(
  "arg_reduce"
  "conv"
  "gemv"
  "layer_norm"
  "random"
  "rms_norm"
  "rope"
  "scaled_dot_product_attention"
)

if (( metal_version >= 320 )); then
  kernels+=("fence")
fi

metal_flags=(
  -x metal
  -Wall
  -Wextra
  -fno-fast-math
  -Wno-c++17-extensions
  -Wno-c++20-extensions
  "${deployment_flag[@]}"
)

if (( metal_version >= 400 )); then
  metal_flags+=(-std=metal4.0)
elif (( metal_version >= 320 )); then
  metal_flags+=(-std=metal3.2)
elif (( metal_version >= 310 )); then
  metal_flags+=(-std=metal3.1)
elif (( metal_version >= 300 )); then
  metal_flags+=(-std=metal3.0)
fi

air_files=()
for kernel in "${kernels[@]}"; do
  source="${KERNELS_DIR}/${kernel}.metal"
  air="${TMP_DIR}/${kernel}.air"
  "${METAL}" "${metal_flags[@]}" -c "${source}" -I"${ROOT_DIR}/Source/Cmlx/mlx" -o "${air}"
  air_files+=("${air}")
done

mkdir -p "$(dirname "${OUTPUT}")"
"${METALLIB}" "${air_files[@]}" -o "${TMP_DIR}/default.metallib"
mv "${TMP_DIR}/default.metallib" "${OUTPUT}"
