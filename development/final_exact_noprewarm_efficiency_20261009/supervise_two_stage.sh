#!/usr/bin/env bash
set -uo pipefail

ROOT=/data/ZhaoX/BoxFusion
CACHE_SESSION=exact_final_paired_fps100_20261008
CACHE_RUNNER="$ROOT/development/final_exact_paired_efficiency_20261008/run_official100_prewarm.sh"
CACHE_REPORT="$ROOT/reports/original_boxfusion_vs_final_novis_calrv1_efficiency100_paired_20261008"
CACHE_SUMMARY="$CACHE_REPORT/summary.json"
CACHE_EXPECTED_FINGERPRINT=4469c1e769de9d23ee393e9ba401ce6c6303100bf77b6f451f015c06d0717d87
NOPREWARM_RUNNER="$ROOT/development/final_exact_noprewarm_efficiency_20261009/run_official100_noprewarm.sh"
NOPREWARM_REPORT="$ROOT/reports/original_boxfusion_vs_final_novis_calrv1_efficiency100_noprewarm_paired_20261009"
NOPREWARM_SUMMARY="$NOPREWARM_REPORT/summary.json"
SUPERVISOR_LOG="$ROOT/reports/exact_final_two_stage_fps_supervisor_20261008.log"
MAX_RETRIES=5

exec >>"$SUPERVISOR_LOG" 2>&1

valid_summary() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  /home/admin1/miniconda3/envs/boxfusion-online/bin/python - "$path" <<'PY' >/dev/null
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["scene_count"] == 100
assert d["arms"]["original"]["raw_frames"] == 169140
assert d["arms"]["recar"]["raw_frames"] == 169140
PY
}

cache_fingerprint() {
  find \
    "$ROOT/development/final_exact_paired_efficiency_20261008/tools" \
    "$ROOT/development/final_exact_paired_efficiency_20261008/recar_source" \
    "$ROOT/reports/paper_matched_runtime_full100_20260916/official_source_06ea629" \
    -type f ! -path '*/__pycache__/*' ! -path '*/.pytest_cache/*' \
    ! -name '*.pyc' -print0 \
    | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'
}

echo "[$(date '+%F %T')] supervisor started"
while tmux has-session -t "$CACHE_SESSION" 2>/dev/null; do
  sleep 60
done

cache_attempt=0
while ! valid_summary "$CACHE_SUMMARY"; do
  actual="$(cache_fingerprint)"
  if [[ "$actual" != "$CACHE_EXPECTED_FINGERPRINT" ]]; then
    echo "[$(date '+%F %T')] cache-symmetric source fingerprint changed; refusing mixed-source resume" >&2
    exit 1
  fi
  cache_attempt=$((cache_attempt + 1))
  if (( cache_attempt > MAX_RETRIES )); then
    echo "[$(date '+%F %T')] cache-symmetric stage exceeded retry limit" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] resuming cache-symmetric stage attempt=$cache_attempt"
  if bash "$CACHE_RUNNER"; then
    echo "[$(date '+%F %T')] cache-symmetric runner returned successfully"
  else
    status=$?
    echo "[$(date '+%F %T')] cache-symmetric runner failed status=$status; retrying after 60s"
    sleep 60
  fi
done

echo "[$(date '+%F %T')] cache-symmetric stage validated"
noprewarm_attempt=0
while ! valid_summary "$NOPREWARM_SUMMARY"; do
  noprewarm_attempt=$((noprewarm_attempt + 1))
  if (( noprewarm_attempt > MAX_RETRIES )); then
    echo "[$(date '+%F %T')] no-prewarm stage exceeded retry limit" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] starting/resuming no-prewarm stage attempt=$noprewarm_attempt"
  if bash "$NOPREWARM_RUNNER"; then
    echo "[$(date '+%F %T')] no-prewarm runner returned successfully"
  else
    status=$?
    echo "[$(date '+%F %T')] no-prewarm runner failed status=$status; retrying after 60s"
    sleep 60
  fi
done

echo "[$(date '+%F %T')] both FPS stages validated complete"
