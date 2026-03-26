#!/usr/bin/env bash
set -euo pipefail

declare -A GPU_QUOTA=(
  ["yangli1-lab"]=18
  ["bweng-lab"]=8
)

ACCOUNTS=("yangli1-lab" "bweng-lab")
LONG_JOB_USER="dasante"
NOVA_THRESHOLD="${NOVA_THRESHOLD:-1.0}"   # best_pressure <= this => use nova

# -----------------------------
# Helpers
# -----------------------------
count_gpus() {
    local acct=$1
    local state=$2   # R or PD
    local user=${3:-}

    if [[ "$state" == "PD" ]]; then
        # For pending jobs, exclude those held by the user.
        if [[ -n "$user" ]]; then
            squeue -A "$acct" -t "$state" -u "$user" -h -o "%b|%r"
        else
            squeue -A "$acct" -t "$state" -h -o "%b|%r"
        fi | awk -F'|' '
            $2 == "JobHeldUser" { next }
            {
                match($1, /gpu:[^:]+:([0-9]+)/, m)
                if (m[1] != "") sum += m[1]
            }
            END { print sum+0 }
        '
    else
        if [[ -n "$user" ]]; then
            squeue -A "$acct" -t "$state" -u "$user" -h -o "%b"
        else
            squeue -A "$acct" -t "$state" -h -o "%b"
        fi | awk '
            {
                match($0, /gpu:[^:]+:([0-9]+)/, m)
                if (m[1] != "") sum += m[1]
            }
            END { print sum+0 }
        '
    fi
}

# -----------------------------
# Decision Logic
# -----------------------------
best_account=""
best_pressure="999999"

for acct in "${ACCOUNTS[@]}"; do
    quota=${GPU_QUOTA[$acct]}

    running_total=$(count_gpus "$acct" "R")
    queued=$(count_gpus "$acct" "PD")
    mokarram_running=$(count_gpus "$acct" "R" "mokarram")
    dasante_running=$(count_gpus "$acct" "R" "$LONG_JOB_USER")

    queued=$(( queued ))
    effective_quota=$(( quota - dasante_running - mokarram_running ))
    effective_running=$(( running_total - dasante_running - mokarram_running ))

    if (( effective_quota <= 0 )); then
        echo "[DEBUG] $acct skipped: effective_quota=$effective_quota" >&2
        continue
    fi

    pressure=$(awk -v r="$effective_running" -v q="$queued" -v c="$effective_quota" \
        'BEGIN { printf "%.4f", (r+q)/c }')

    echo "[DEBUG] $acct:
      quota=$quota
      running_total=$running_total
      effective_running=$effective_running
      queued=$queued
      effective_quota=$effective_quota
      pressure=$pressure" >&2

    if awk -v p="$pressure" -v bp="$best_pressure" 'BEGIN { exit !(p < bp) }'; then
        best_pressure="$pressure"
        best_account="$acct"
    fi
done

if [[ -z "$best_account" ]]; then
    echo "ERROR: no valid account found" >&2
    exit 1
fi

if awk -v p="$best_pressure" -v t="$NOVA_THRESHOLD" 'BEGIN { exit !(p <= t) }'; then
    best_partition="nova"
else
    best_partition="scavenger"
fi

# output: account pressure partition
echo "$best_account $best_pressure $best_partition"