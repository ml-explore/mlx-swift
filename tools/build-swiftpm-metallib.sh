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

normalize_platform_name() {
  local raw="$1"
  raw=$(basename "${raw}")
  raw=$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')
  raw=${raw#-}
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

requested_platform="${PLATFORM_NAME:-${EFFECTIVE_PLATFORM_NAME:-${SDK_NAME:-}}}"
if [[ -z "${requested_platform}" && -n "${SDKROOT:-}" ]]; then
  requested_platform=$(basename "${SDKROOT}")
fi
platform_name=$(normalize_platform_name "${requested_platform:-macosx}")

case "${platform_name}" in
  macosx)
    sdk=macosx
    target_os="macos${MACOSX_DEPLOYMENT_TARGET:-14.0}"
    ;;
  iphoneos)
    sdk=iphoneos
    target_os="ios${IPHONEOS_DEPLOYMENT_TARGET:-${IOS_DEPLOYMENT_TARGET:-17.0}}"
    ;;
  iphonesimulator)
    sdk=iphonesimulator
    target_os="ios${IPHONEOS_DEPLOYMENT_TARGET:-${IOS_DEPLOYMENT_TARGET:-17.0}}-simulator"
    ;;
  appletvos)
    sdk=appletvos
    target_os="tvos${TVOS_DEPLOYMENT_TARGET:-17.0}"
    ;;
  appletvsimulator)
    sdk=appletvsimulator
    target_os="tvos${TVOS_DEPLOYMENT_TARGET:-17.0}-simulator"
    ;;
  xros)
    sdk=xros
    target_os="xros${XROS_DEPLOYMENT_TARGET:-${VISIONOS_DEPLOYMENT_TARGET:-1.0}}"
    ;;
  xrsimulator)
    sdk=xrsimulator
    target_os="xros${XROS_DEPLOYMENT_TARGET:-${VISIONOS_DEPLOYMENT_TARGET:-1.0}}-simulator"
    ;;
  *)
    echo "unsupported Apple platform '${platform_name}'" >&2
    exit 65
    ;;
esac

SDK_PATH=$(xcrun -sdk "${sdk}" -show-sdk-path)
METAL=$(xcrun -sdk "${sdk}" -find metal)
METALLIB=$(xcrun -sdk "${sdk}" -find metallib)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT
target_flag="-mtargetos=${target_os}"

echo "Building SwiftPM default.metallib for ${platform_name} (${target_os}; SDK ${sdk})"

metal_version=$(
  printf '%s\n' '__METAL_VERSION__' |
    SDKROOT="${SDK_PATH}" "${METAL}" "${target_flag}" -E -x metal -P - |
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
  "${target_flag}"
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
  SDKROOT="${SDK_PATH}" "${METAL}" "${metal_flags[@]}" -c "${source}" \
    -I"${ROOT_DIR}/Source/Cmlx/mlx" -o "${air}"
  air_files+=("${air}")
done

mkdir -p "$(dirname "${OUTPUT}")"
SDKROOT="${SDK_PATH}" "${METALLIB}" "${air_files[@]}" -o "${TMP_DIR}/default.metallib"
mv "${TMP_DIR}/default.metallib" "${OUTPUT}"
