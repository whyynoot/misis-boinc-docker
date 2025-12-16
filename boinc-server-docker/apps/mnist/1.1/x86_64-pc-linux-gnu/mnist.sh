#!/bin/bash
set -e

resolve_path() {
    local target="$1"
    if [ -f "$target" ]; then
        local content
        content=$(cat "$target")
        if [[ "$content" == \<soft_link\>* ]]; then
            content=${content#<soft_link>}
            content=${content%</soft_link>}
            echo "$(readlink -f "$content")"
            return
        fi
    fi
    echo "$(readlink -f "$target")"
}

JOB_FILE="$(resolve_path "${1:-job.json}")"
OUT_FILE="${2:-metrics.json}"
SCRIPT_PATH="$(resolve_path mnist.py)"

python3 "$SCRIPT_PATH" "$JOB_FILE" "$OUT_FILE" 2>&1 | tee run.log
echo 0 > boinc_finish_called
