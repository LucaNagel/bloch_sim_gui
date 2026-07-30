#!/usr/bin/env bash

set -euo pipefail

target="${MACOSX_DEPLOYMENT_TARGET:?MACOSX_DEPLOYMENT_TARGET is required}"
libomp_path="$(brew --prefix libomp)/lib/libomp.dylib"

if [[ ! -f "${libomp_path}" ]]; then
    echo "libomp was not found at ${libomp_path}" >&2
    exit 1
fi

actual="$(vtool -show-build "${libomp_path}" | awk '$1 == "minos" { print $2; exit }')"
if [[ -z "${actual}" ]]; then
    echo "Could not determine the minimum macOS version of ${libomp_path}" >&2
    exit 1
fi

echo "libomp minimum macOS version: ${actual}; requested target: ${target}"

if ! awk -v actual="${actual}" -v target="${target}" 'BEGIN {
    split(actual, actual_parts, ".")
    split(target, target_parts, ".")
    actual_major = actual_parts[1] + 0
    actual_minor = actual_parts[2] + 0
    target_major = target_parts[1] + 0
    target_minor = target_parts[2] + 0
    compatible = actual_major < target_major || \
        (actual_major == target_major && actual_minor <= target_minor)
    exit compatible ? 0 : 1
}'; then
    echo "libomp requires macOS ${actual}, which is newer than ${target}." >&2
    exit 1
fi
