#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

TAG="${1:-c2_c1top10_fixed10_v1}"
[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$ ]] || {
  echo "usage: $0 [unique-run-tag]" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_C2_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
C1_RUN="${BOXFUSION_C2_C1_RUN:-$ROOT/artifacts/tr3d_c1_track/c1_r3active_fixed10_v2}"
C1_CACHE_ROOT="${BOXFUSION_C2_C1_CACHE_ROOT:-$C1_RUN/cache}"
R2_ROOT="${BOXFUSION_C2_R2_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev}"
PARENT_ROOT="${BOXFUSION_C2_PARENT_CACHE_ROOT:-$R2_ROOT/cache/tr3d_prefix_boxfusion_causal_p100_fixed10_v2}"
TEACHER_ROOT="${BOXFUSION_C2_TEACHER_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/cache/sam3_teacher/sam3_teacher_ablation10_c050_v3}"
SCENE_LIST="${BOXFUSION_C2_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
SCANS_ROOT="${BOXFUSION_C2_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_C2_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
ACTIVE_ROOT="${BOXFUSION_C2_ACTIVE_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r3_20260805/boxfusion_tr3d_terminal_paired_fixed10/results/b6_g0_tr3d_terminal/g0_tr3d_terminal_paired_fixed10_v1/scannetv2_val_ablation10_even-0b45515fe11a}"
RUN_ROOT="${BOXFUSION_C2_ARTIFACT_ROOT:-$ROOT/artifacts/tr3d_c2_maskrgbd}/$TAG"
CACHE_ROOT="$RUN_ROOT/cache"
REPORT_ROOT="$RUN_ROOT/reports"
SOURCE_BUDGET="${BOXFUSION_C2_SOURCE_BUDGET:-10}"

for path in "$PYTHON_BIN" "$SCENE_LIST"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
for path in "$C1_CACHE_ROOT" "$PARENT_ROOT" "$TEACHER_ROOT" "$ACTIVE_ROOT" "$SCANS_ROOT" "$GT_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 2; }
done
scene_count="$(awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "$SCENE_LIST")"
[[ "$scene_count" == "10" ]] || { echo "C2 fixed10 requires exactly 10 scenes" >&2; exit 2; }
[[ ! -e "$RUN_ROOT" ]] || { echo "Refusing immutable run overwrite: $RUN_ROOT" >&2; exit 2; }
mkdir -p "$CACHE_ROOT" "$REPORT_ROOT"

echo "C2 multi-view SAM3 Mask-RGBD confirmation observer"
echo "  source: C1 depth-feature ranking, Top-$SOURCE_BUDGET per scene"
echo "  teacher: $TEACHER_ROOT"
echo "  real depth/pose: $SCANS_ROOT"
echo "  output: $RUN_ROOT"
echo "  contract: observer_only=true; applied_count=0; CLIP/predictions unchanged"

env -u PYTHONPATH TMPDIR="${BOXFUSION_C2_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" tools/run_tr3d_c2_maskrgbd_observer.py \
    --scene-list "$SCENE_LIST" --active-prediction-root "$ACTIVE_ROOT" \
    --parent-cache-root "$PARENT_ROOT" --c1-cache-root "$C1_CACHE_ROOT" \
    --teacher-cache-root "$TEACHER_ROOT" --output-root "$CACHE_ROOT" \
    --source-budget "$SOURCE_BUDGET" --prefix-id p100 \
    --report "$REPORT_ROOT/export_report.json" \
    > "$REPORT_ROOT/export_stdout.json"

env -u PYTHONPATH TMPDIR="${BOXFUSION_C2_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$PYTHON_BIN" tools/audit_tr3d_c2_maskrgbd_observer.py \
    --scene-list "$SCENE_LIST" --export-report "$REPORT_ROOT/export_report.json" \
    --c2-cache-root "$CACHE_ROOT" --parent-cache-root "$PARENT_ROOT" \
    --active-prediction-root "$ACTIVE_ROOT" --scans-root "$SCANS_ROOT" \
    --gt-root "$GT_ROOT" --prefix-id p100 --report "$REPORT_ROOT/gt_audit.json" \
    > "$REPORT_ROOT/audit_stdout.json"

"$PYTHON_BIN" tools/summarize_tr3d_c2_maskrgbd.py "$REPORT_ROOT/gt_audit.json" \
  | tee "$REPORT_ROOT/summary.txt"
echo "C2 fixed10 complete: $REPORT_ROOT/gt_audit.json"
