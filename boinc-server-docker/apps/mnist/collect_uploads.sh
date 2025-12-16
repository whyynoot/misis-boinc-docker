#!/bin/bash
# Helper to copy all mnist_run* result files from upload/ into /results/mnist/.
set -euo pipefail

PROJECT_ROOT="/home/boincadm/project"
TARGET_PREFIX="${1:-mnist_run}"
UPLOAD_DIR="${PROJECT_ROOT}/upload"
SCRIPT="${PROJECT_ROOT}/apps/mnist/assimilate_mnist.sh"

cd "${PROJECT_ROOT}"
find "${UPLOAD_DIR}" -name "${TARGET_PREFIX}*_r*" -type f | while read -r path; do
    base="$(basename "${path}")"
    wu="${base%_r*}"
    "${SCRIPT}" "${base}" "${wu}"
done
