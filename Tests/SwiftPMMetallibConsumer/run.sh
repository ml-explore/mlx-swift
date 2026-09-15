#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
TMP_DIR=$(mktemp -d)
trap 'rm -rf "${TMP_DIR}"' EXIT

scratch_path="${TMP_DIR}/build"
swift build \
  --package-path "${SCRIPT_DIR}" \
  --scratch-path "${scratch_path}" \
  -c release

bin_dir=$(
  swift build \
    --package-path "${SCRIPT_DIR}" \
    --scratch-path "${scratch_path}" \
    -c release \
    --show-bin-path
)

executable="${bin_dir}/SwiftPMMetallibConsumer"
resource_bundle="${bin_dir}/mlx-swift_Cmlx.bundle"
metallib="${resource_bundle}/default.metallib"

test -x "${executable}"
test -r "${metallib}"

mkdir -p "${TMP_DIR}/deploy" "${TMP_DIR}/working-directory"
cp "${executable}" "${TMP_DIR}/deploy/"
cp -R "${resource_bundle}" "${TMP_DIR}/deploy/"

output=$(
  cd "${TMP_DIR}/working-directory"
  "${TMP_DIR}/deploy/SwiftPMMetallibConsumer"
)

if [[ "${output}" != *"GPU computation succeeded"* ]]; then
  printf 'unexpected consumer output:\n%s\n' "${output}" >&2
  exit 1
fi

printf '%s\n' "${output}"
