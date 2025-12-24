#!/bin/bash
# Assimilator for CIFAR-100 demo jobs.
# Copies result files into /results/cifar100/<wu_id>/ and logs a short summary.
set -euo pipefail

# BOINC may prepend options like --error/--no_delete. Consume any --* flags first.
FILES_ARG=""
WU_ID="unknown_wu"
RESULT_ID="unknown_result"
RUNTIME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --*) shift ;;              # ignore flags
        *)   FILES_ARG="$1"; shift
             WU_ID="${1:-$WU_ID}"; shift || true
             RESULT_ID="${1:-$RESULT_ID}"; shift || true
             RUNTIME="${1:-$RUNTIME}"; shift || true
             break ;;
    esac
done

PROJECT_ROOT="/home/boincadm/project"
UPLOAD_DIR="${PROJECT_ROOT}/upload"
OUT_BASE="/results/cifar100"
LOG_FILE="${OUT_BASE}/assimilator.log"

IFS=' ' read -r -a FILES <<< "${FILES_ARG}"
WU_LABEL="${WU_ID}"

echo "$(date -Is) [${WU_ID}] start args=\"$*\" files=\"${FILES_ARG}\"" >> "${LOG_FILE}"

OUT_DIR="${OUT_BASE}/${WU_LABEL}"
mkdir -p "${OUT_DIR}"
mkdir -p "${OUT_BASE}/${WU_ID}"

for fname in "${FILES[@]}"; do
    [[ -n "${fname}" ]] || continue
    base_file="$(basename "${fname}")"

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
    cp "${src_path}" "${OUT_DIR}/metrics_${RESULT_ID}.json"
    echo "$(date -Is) [${WU_ID}] saved ${base_file} from ${src_path} (result ${RESULT_ID}, runtime ${RUNTIME}s)" >> "${LOG_FILE}"
    if [[ "${OUT_DIR}" != "${OUT_BASE}/${WU_ID}" ]]; then
        cp "${src_path}" "${OUT_BASE}/${WU_ID}/${base_file}"
        cp "${src_path}" "${OUT_BASE}/${WU_ID}/metrics_${RESULT_ID}.json"
    fi
done
