#!/usr/bin/env bash
set -euo pipefail

declare -A GPU_QUOTA=(
  ["yangli1-lab"]=15
  ["bweng-lab"]=5
)

ACCOUNTS=("yangli1-lab" "bweng-lab")
LONG_JOB_USER="dasante"

# -----------------------------
# Helpers
# -----------------------------
count_gpus() {
    local acct=$1
    local state=$2   # R or PD
    local user=${3:-}

    if [[ -n "$user" ]]; then
        squeue -A "$acct" -t "$state" -u "$user" -h -o "%b"
    else
        squeue -A "$acct" -t "$state" -h -o "%b"
    fi | awk '
        {
            match($0, /gpu:[^:]+:([0-9]+)/, m)
            if (m[1] != "") sum += m[1]
        }
        END {print sum+0}
    '
}

# -----------------------------
# Decision Logic
# -----------------------------
best_account=""
best_pressure=999999

for acct in "${ACCOUNTS[@]}"; do
    quota=${GPU_QUOTA[$acct]}

    running_total=$(count_gpus "$acct" "R")
    queued=$(count_gpus "$acct" "PD")
    dasante_running=$(count_gpus "$acct" "R" "$LONG_JOB_USER")

    # Remove dasante from consideration
    effective_quota=$(( quota - dasante_running ))
    effective_running=$(( running_total - dasante_running ))

    # Guard against zero / negative quota
    if (( effective_quota <= 0 )); then
        echo "[DEBUG] $acct skipped: effective_quota=$effective_quota" >&2
        continue
    fi

    # Immediate availability
    if (( effective_running + queued < effective_quota )); then
        echo "$acct"
        exit 0
    fi

    # Pressure = load / capacity
    pressure=$(awk -v r="$effective_running" -v q="$queued" -v c="$effective_quota" \
        'BEGIN { printf "%.4f", (r+q)/c }')

    echo "[DEBUG] $acct:
      quota=$quota
      running_total=$running_total
      dasante_running=$dasante_running
      effective_running=$effective_running
      queued=$queued
      effective_quota=$effective_quota
      pressure=$pressure" >&2

    # Choose lowest pressure
    awk -v p="$pressure" -v bp="$best_pressure" 'BEGIN { exit !(p < bp) }' \
        && { best_pressure="$pressure"; best_account="$acct"; }
done

echo "$best_account"
