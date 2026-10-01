#!/usr/bin/env bash
set -euo pipefail

# official100: Cbest plus the single frozen route requested by the user:
# native-unmatched -> causal Diverse Top-K -> SAM3 mask-depth -> CLIP -> birth.

ROOT=/data/ZhaoX/BoxFusion
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
EXPECTED_SCENE_LIST_SHA=4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5
EXPERIMENT=scannet_cbest_sam3_diverse_clip_birth_score05
PRED_ROOT="$ROOT/results/$EXPERIMENT"
LOG_ROOT="$ROOT/logs/sam3_diverse_clip_birth/full100"
MATERIALIZER="$ROOT/tools/materialize_scannet_sam3_diverse_clip_birth_full100.py"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh"
BASELINE_ROOT="$ROOT/results/scannet_t05_boxer_replay_active_score05"
RAW_LOG_ROOT="$ROOT/logs/scannet_raw_boxer_full100_score05_v1"
SCHEDULE_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/scannet-score05-gap25-postfilter-v2
RGBD_ROOT="$ROOT/upstream_clean/scannet_readme_frames"
TEACHER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/cache/sam3_teacher/sam3_teacher_full100_c050_frozen_v1
V2_MANIFEST="$ROOT/results/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05/RAW_BOXER_PAST3_BIRTH_FULL100.json"
CLIP_SIDECAR="$ROOT/logs/scannet_cbest_raw_boxer_clip_vocab_shadow_score05/CLIP_VOCAB_SHADOW_FULL100.json"

for required in \
  "$PYTHON" "$SCENE_LIST" "$MATERIALIZER" "$EVAL_RUNNER" \
  "$BASELINE_ROOT" "$RAW_LOG_ROOT" "$SCHEDULE_ROOT" "$RGBD_ROOT" \
  "$TEACHER_ROOT" "$V2_MANIFEST" "$CLIP_SIDECAR"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == "$EXPECTED_SCENE_LIST_SHA" ]] || \
  { echo "Official scene-list hash mismatch" >&2; exit 1; }
[[ ! -e "$PRED_ROOT" ]] || \
  { echo "Refusing to overwrite prediction root: $PRED_ROOT" >&2; exit 1; }

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

echo "[$(date '+%F %T')] SAM3 Diverse-TopK CLIP birth official100 started"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" "$MATERIALIZER" \
  --scene-list "$SCENE_LIST" \
  --teacher-scene-list "$SCENE_LIST" \
  --expected-scene-count 100 \
  --baseline-root "$BASELINE_ROOT" \
  --raw-log-root "$RAW_LOG_ROOT" \
  --schedule-root "$SCHEDULE_ROOT" \
  --scene-rgbd-root "$RGBD_ROOT" \
  --teacher-root "$TEACHER_ROOT" \
  --v2-manifest "$V2_MANIFEST" \
  --clip-sidecar "$CLIP_SIDECAR" \
  --output-root "$PRED_ROOT"

bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
echo "[$(date '+%F %T')] SAM3 Diverse-TopK CLIP birth official100 complete"
