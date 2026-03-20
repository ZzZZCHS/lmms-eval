#!/usr/bin/env bash
set -u

retry_sleep=60
max_retries=-1              # -1 means infinite retries
retry_all_failures=false    # default: only retry signal-like termination

DEFAULT_TASKS=("videomme" "mlvu_dev" "longvideobench_val_v" "mvbench")

forward_args=()
user_specified_tasks=false

for arg in "$@"; do
    case "$arg" in
        --retry_sleep=*)
            retry_sleep="${arg#*=}"
            ;;
        --max_retries=*)
            max_retries="${arg#*=}"
            ;;
        --retry_all_failures)
            retry_all_failures=true
            ;;
        --tasks=*)
            user_specified_tasks=true
            forward_args+=("$arg")
            ;;
        *)
            forward_args+=("$arg")
            ;;
    esac
done

run_one_task_with_retries() {
    local task_name="$1"
    local attempt=0

    while true; do
        attempt=$((attempt + 1))
        export AUTO_RESTART_ATTEMPT="$attempt"

        echo
        echo "=================================================="
        echo "Task: $task_name"
        echo "Attempt $attempt started at $(date)"
        echo "=================================================="

        ./srun_llava_ov.sh "${forward_args[@]}" --tasks="$task_name"
        rc=$?

        echo "Task $task_name, attempt $attempt finished with exit code: $rc"

        if [[ $rc -eq 0 ]]; then
            echo "Task $task_name completed successfully on attempt $attempt."
            return 0
        fi

        should_retry=false

        if [[ "$retry_all_failures" == true ]]; then
            should_retry=true
            echo "retry_all_failures=true, will retry task $task_name."
        else
            case "$rc" in
                137)
                    should_retry=true
                    echo "Task $task_name likely killed by SIGKILL (137). Will retry."
                    ;;
                143)
                    should_retry=true
                    echo "Task $task_name likely terminated by SIGTERM (143). Will retry."
                    ;;
                *)
                    should_retry=false
                    echo "Task $task_name failed with non-signal-like exit code $rc. Will stop."
                    ;;
            esac
        fi

        if [[ "$should_retry" != true ]]; then
            return "$rc"
        fi

        if [[ "$max_retries" -ge 0 && "$attempt" -ge "$max_retries" ]]; then
            echo "Task $task_name reached max_retries=$max_retries. Stop retrying."
            return "$rc"
        fi

        jitter=$((RANDOM % 30))
        sleep_time=$((retry_sleep + jitter))
        echo "Sleeping for ${sleep_time}s before retrying task $task_name..."
        sleep "$sleep_time"
    done
}

echo "Auto-restart wrapper started at $(date)"
echo "retry_sleep=${retry_sleep}s"
echo "max_retries=${max_retries}"
echo "retry_all_failures=${retry_all_failures}"
echo "user_specified_tasks=${user_specified_tasks}"
echo "Forwarded base args: ${forward_args[*]}"

if [[ "$user_specified_tasks" == true ]]; then
    echo "User specified --tasks explicitly. Running exactly what user requested."
    run_one_task_with_retries "__USER_SPECIFIED_TASKS__"
    exit $?
fi

echo "No --tasks specified. Running default tasks one by one:"
printf '  - %s\n' "${DEFAULT_TASKS[@]}"

for task_name in "${DEFAULT_TASKS[@]}"; do
    run_one_task_with_retries "$task_name"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "Stopping because task $task_name failed with exit code $rc."
        exit "$rc"
    fi
done

echo "All default tasks completed successfully."
exit 0