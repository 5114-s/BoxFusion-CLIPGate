#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
FIRST_SESSION=exact_final_paired_fps100_20261008
FIRST_SUMMARY="$ROOT/reports/original_boxfusion_vs_final_novis_calrv1_efficiency100_paired_20261008/summary.json"
QUEUE_LOG="$ROOT/reports/original_boxfusion_vs_final_novis_calrv1_efficiency100_noprewarm_paired_20261009.queue.log"

mkdir -p "$(dirname "$QUEUE_LOG")"
exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[$(date '+%F %T')] waiting for cache-symmetric paired benchmark"
while tmux has-session -t "$FIRST_SESSION" 2>/dev/null; do
  sleep 60
done

if [[ ! -s "$FIRST_SUMMARY" ]]; then
  echo "[$(date '+%F %T')] cache-symmetric benchmark ended without summary; refusing to start" >&2
  exit 1
fi
/home/admin1/miniconda3/envs/boxfusion-online/bin/python - "$FIRST_SUMMARY" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["scene_count"] == 100
assert d["arms"]["original"]["raw_frames"] == 169140
assert d["arms"]["recar"]["raw_frames"] == 169140
PY

echo "[$(date '+%F %T')] cache-symmetric benchmark complete; starting no-prewarm paired benchmark"
exec bash "$ROOT/development/final_exact_noprewarm_efficiency_20261009/run_official100_noprewarm.sh"
