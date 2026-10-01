#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
TAG="${2:-r5_spgroup_fixed10_v1}"
[[ "$GPU_ID" =~ ^[0-9]+$ ]] || { echo "GPU id must be a non-negative integer" >&2; exit 2; }

CPU_PYTHON="${BOXFUSION_R5_CPU_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
TR3D_PYTHON="${BOXFUSION_R5_TR3D_PYTHON:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/.conda/boxfusion-tr3d/bin/python}"
SCENE_LIST="${BOXFUSION_R5_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
SCANS_ROOT="${BOXFUSION_R5_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
ARTIFACT_BASE="${BOXFUSION_R5_ARTIFACT_BASE:-/extra/ZhaoX/codex_artifacts/boxfusion_r5_spgroup_fixed10}"
RUN_ROOT="$ARTIFACT_BASE/$TAG"
PARTITION_ROOT="$RUN_ROOT/partition"
FEATURE_ROOT="$RUN_ROOT/features"
R5_ROOT="$RUN_ROOT/r5_pairs"
REPORT_ROOT="$RUN_ROOT/reports"

OFFICIAL_ROOT="${BOXFUSION_R5_OFFICIAL_ROOT:-/data/ZhaoX/OVM3D-Dett/third_party/SPGroup3D_official}"
OFFICIAL_COMMIT="${BOXFUSION_R5_OFFICIAL_COMMIT:-181283547323d3bd54d0e9f58baf0cd413ccc107}"
CHECKPOINT="${BOXFUSION_R5_CHECKPOINT:-/extra/ZhaoX/codex_artifacts/spgroup3d_official/epoch_10.clean.pth}"
SEGMENTATOR="${BOXFUSION_R5_SEGMENTATOR:-/extra/ZhaoX/codex_artifacts/spgroup3d_official/segmentator}"
SEGMENTATOR_COMMIT="${BOXFUSION_R5_SEGMENTATOR_COMMIT:-4c6126551685166c6c300551e9ad63db988928c4}"
SEGMENTATOR_BINARY_SHA="${BOXFUSION_R5_SEGMENTATOR_BINARY_SHA256:-41e0ba70e8cbdd771aecad6157d5c671327c12a56e672af80496aa34b54f4cc8}"
R4D_ROOT="${BOXFUSION_R5_R4D_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r4_smov_fixed10/cache/r4d_fixed10_v1}"

R3_ROOT="${BOXFUSION_R5_R3_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r3_20260805/boxfusion_tr3d_terminal_paired_full100}"
R3_RUN="${BOXFUSION_R5_R3_RUN:-g0_tr3d_terminal_paired_full100_v1}"
R3_SCOPE="${BOXFUSION_R5_R3_SCOPE:-scannetv2_val-4b18fc586f7a}"
BASELINE_ROOT="$R3_ROOT/results/b6_g0_tr3d_terminal_same_run_baseline/$R3_RUN/$R3_SCOPE"
ACTIVE_ROOT="$R3_ROOT/results/b6_g0_tr3d_terminal/$R3_RUN/$R3_SCOPE"

for path in "$CPU_PYTHON" "$TR3D_PYTHON" "$SCENE_LIST" "$CHECKPOINT"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
for path in "$SCANS_ROOT" "$OFFICIAL_ROOT" "$SEGMENTATOR" "$R4D_ROOT" "$BASELINE_ROOT" "$ACTIVE_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing required directory: $path" >&2; exit 2; }
done
for report in partition_export.json feature_export.json r5_export.json r5_counterfactual.json; do
  [[ ! -e "$REPORT_ROOT/$report" ]] || { echo "Refusing immutable report overwrite: $REPORT_ROOT/$report" >&2; exit 2; }
done
mkdir -p "$PARTITION_ROOT" "$FEATURE_ROOT" "$R5_ROOT" "$REPORT_ROOT"

echo "R5 true SPGroup3D local-grouping observer"
echo "  scenes: $SCENE_LIST"
echo "  GPU: physical $GPU_ID"
echo "  official commit: $OFFICIAL_COMMIT"
echo "  output: $RUN_ROOT"
echo "  contract: observer-only; prediction geometry/score/order and CLIP are immutable"
echo "  scope: offline reconstructed mesh; not yet online-eligible"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  TMPDIR="${BOXFUSION_R5_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 \
  "$CPU_PYTHON" "$ROOT/tools/build_spgroup_partition_cache.py" \
    --scans-root "$SCANS_ROOT" --scene-list "$SCENE_LIST" \
    --output-root "$PARTITION_ROOT" --segmentator-root "$SEGMENTATOR" \
    --expected-segmentator-commit "$SEGMENTATOR_COMMIT" \
    --expected-segmentator-binary-sha256 "$SEGMENTATOR_BINARY_SHA" \
    --official-root "$OFFICIAL_ROOT" --expected-commit "$OFFICIAL_COMMIT" \
    --report "$REPORT_ROOT/partition_export.json"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  TMPDIR="${BOXFUSION_R5_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 \
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  "$TR3D_PYTHON" "$ROOT/tools/run_spgroup_feature_observer.py" \
    --partition-root "$PARTITION_ROOT" --feature-root "$FEATURE_ROOT" \
    --scene-list "$SCENE_LIST" --active-prediction-root "$ACTIVE_ROOT" \
    --official-root "$OFFICIAL_ROOT" --checkpoint "$CHECKPOINT" --device cuda \
    --report "$REPORT_ROOT/feature_export.json"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  TMPDIR="${BOXFUSION_R5_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$CPU_PYTHON" "$ROOT/tools/run_tr3d_r5_spgroup_observer.py" \
    --r4-depth-cache-root "$R4D_ROOT" --partition-root "$PARTITION_ROOT" \
    --feature-root "$FEATURE_ROOT" --output-root "$R5_ROOT" \
    --scene-list "$SCENE_LIST" --active-prediction-root "$ACTIVE_ROOT" \
    --report "$REPORT_ROOT/r5_export.json"

env -u PYTHONPATH -u LD_LIBRARY_PATH \
  TMPDIR="${BOXFUSION_R5_TMPDIR:-/dev/shm}" PYTHONDONTWRITEBYTECODE=1 \
  "$CPU_PYTHON" "$ROOT/tools/audit_tr3d_r5_spgroup.py" \
    --r5-cache-root "$R5_ROOT" --same-run-baseline-root "$BASELINE_ROOT" \
    --active-prediction-root "$ACTIVE_ROOT" --scene-list "$SCENE_LIST" \
    --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
    --scans-root "$SCANS_ROOT" --report "$REPORT_ROOT/r5_counterfactual.json"

echo "R5 fixed10 observer complete: $REPORT_ROOT/r5_counterfactual.json"
