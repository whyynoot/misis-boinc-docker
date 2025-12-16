#!/bin/bash
# Lightweight assimilator for MNIST demo jobs.
# Copies the canonical result files into /results/mnist/<wu_id>/ and logs a short summary.
set -euo pipefail

FILES_ARG="${1:-}"
WU_ID="${2:-unknown_wu}"
RESULT_ID="${3:-unknown_result}"
RUNTIME="${4:-}"

PROJECT_ROOT="/home/boincadm/project"
UPLOAD_DIR="${PROJECT_ROOT}/upload"
OUT_BASE="/results/mnist"
LOG_FILE="${OUT_BASE}/assimilator.log"

IFS=' ' read -r -a FILES <<< "${FILES_ARG}"
WU_LABEL="${WU_ID}"
if [[ ${FILES[0]+isset} ]]; then
    base_name="$(basename "${FILES[0]}")"
    base_name="${base_name%_r*}"
    if [[ -n "${base_name}" ]]; then
        WU_LABEL="${base_name}"
    fi
fi

OUT_DIR="${OUT_BASE}/${WU_LABEL}"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_BASE}/${WU_ID}"

for fname in "${FILES[@]}"; do
    [[ -n "${fname}" ]] || continue
    base_file="$(basename "${fname}")"

    # locate the uploaded file anywhere under the fanout hierarchy or use the provided path
    if [[ -f "${fname}" ]]; then
        src_path="${fname}"
    else
        src_path="$(find "${UPLOAD_DIR}" -name "${base_file}" -print -quit || true)"
    fi
    if [[ -z "${src_path}" ]]; then
        echo "$(date -Is) [${WU_ID}] missing file ${fname}" >> "${LOG_FILE}"
        continue
    fi

    dest_path="${OUT_DIR}/${base_file}"
    cp "${src_path}" "${dest_path}"
    cp "${src_path}" "${OUT_DIR}/metrics.json"
    echo "$(date -Is) [${WU_ID}] saved ${base_file} from ${src_path} (result ${RESULT_ID}, runtime ${RUNTIME}s)" >> "${LOG_FILE}"
    if [[ "${OUT_DIR}" != "${OUT_BASE}/${WU_ID}" ]]; then
        cp "${src_path}" "${OUT_BASE}/${WU_ID}/${base_file}"
        cp "${src_path}" "${OUT_BASE}/${WU_ID}/metrics.json"
    fi
done
