#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LOG_ROOT="$ROOT/logs/ca1m_repro"
APPLE_SESSION="${BOXFUSION_CA1M_APPLE_SESSION:-ca1m_apple_recovery40}"
METADATA_SESSION="${BOXFUSION_CA1M_METADATA_SESSION:-ca1m_metadata_recovery40}"
APPLE_LOG="$LOG_ROOT/download_apple_recovery40.log"
METADATA_LOG="$LOG_ROOT/download_metadata_recovery40.log"
GPU_SPEC="${BOXFUSION_CA1M_GPUS:-0,1}"

wait_session() {
    local name="$1"
    local log="$2"
    while tmux has-session -t "$name" 2>/dev/null; do
        echo "[$(date '+%F %T')] Waiting for $name"
        sleep 60
    done
    [[ -f "$log" ]] && grep -qx 'EXIT=0' "$log" || {
        echo "Background prerequisite failed or lacks EXIT=0: $name ($log)" >&2
        exit 1
    }
}

wait_session "$METADATA_SESSION" "$METADATA_LOG"
wait_session "$APPLE_SESSION" "$APPLE_LOG"

echo "[$(date '+%F %T')] Downloads passed; converting and auditing recovery40"
bash "$ROOT/scripts/recover_ca1m_processed_recovery40.sh" \
    > "$LOG_ROOT/recover_processed_recovery40.log" 2>&1

while true; do
    compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sed '/^[[:space:]]*$/d')"
    if [[ -z "$compute_pids" ]]; then
        break
    fi
    echo "[$(date '+%F %T')] GPUs have active compute processes; waiting: ${compute_pids//$'\n'/,}"
    sleep 60
done

echo "[$(date '+%F %T')] Exact107 data passed and GPUs idle; starting corrected C0"
cd "$ROOT"
bash scripts/run_ca1m_c0_score04_real_score_full107.sh "$GPU_SPEC" \
    > "$LOG_ROOT/c0_score04_real_score_full107_driver.log" 2>&1
echo "[$(date '+%F %T')] Corrected CA-1M C0 full107 completed"
