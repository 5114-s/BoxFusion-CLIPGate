#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID="${1:-0}"
R4D_TAG="${2:-r4d_scene0277_smoke_v1}"
R4F_TAG="${3:-r4f_scene0277_smoke_v1}"
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
  echo "GPU id must be a non-negative integer" >&2
  exit 2
fi
PYTHON_BIN="${BOXFUSION_R4_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="${BOXFUSION_R4_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_smoke_scene0277_00.txt}"
ARTIFACT_BASE="${BOXFUSION_R4_ARTIFACT_BASE:-/extra/ZhaoX/codex_artifacts/boxfusion_r4_smov_smoke}"
R3_ARTIFACT_ROOT="${BOXFUSION_R4_R3_ARTIFACT_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r3_20260805/boxfusion_tr3d_terminal_paired_full100}"
R3_RUN_TAG="${BOXFUSION_R4_R3_RUN_TAG:-g0_tr3d_terminal_paired_full100_v1}"
LIST_SCOPE="${BOXFUSION_R4_R3_LIST_SCOPE:-scannetv2_val-4b18fc586f7a}"
PREFIX_MANIFEST="${BOXFUSION_R4_PREFIX_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/data/tr3d_prefix_val100_boxfusion_causal_p100_v3/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl}"
FRAMES_ROOT="${BOXFUSION_R4_FRAMES_ROOT:-/extra/ZhaoX/scannet_data/scans}"
OFFICIAL_BOXER_ROOT="${BOXFUSION_R4_OFFICIAL_BOXER_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer}"
BOXER_COMMIT="${BOXFUSION_R4_BOXER_COMMIT:-1f86542dc342a4b1d474c87c97c5d1d6566d9148}"
DINO_SHA="${BOXFUSION_R4_DINO_SHA256:-4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea}"
ACTIVE_ROOT="$R3_ARTIFACT_ROOT/results/b6_g0_tr3d_terminal/$R3_RUN_TAG/$LIST_SCOPE"
R4D_ROOT="$ARTIFACT_BASE/cache/$R4D_TAG"
R4F_ROOT="$ARTIFACT_BASE/cache/$R4F_TAG"
REPORT="$ARTIFACT_BASE/reports/$R4F_TAG/export_report.json"

for path in "$PYTHON_BIN" "$SCENE_LIST" "$PREFIX_MANIFEST"; do
  [[ -e "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
for directory in "$FRAMES_ROOT" "$OFFICIAL_BOXER_ROOT" "$ACTIVE_ROOT" "$R4D_ROOT"; do
  [[ -d "$directory" ]] || { echo "Missing required directory: $directory" >&2; exit 2; }
done
[[ ! -e "$REPORT" ]] || { echo "Refusing immutable report overwrite: $REPORT" >&2; exit 2; }
mkdir -p "$R4F_ROOT" "$(dirname "$REPORT")"

echo "R4-F paired Boxer-DINO feature observer"
echo "  scenes: $SCENE_LIST"
echo "  GPU: physical $GPU_ID"
echo "  R4-D parent: $R4D_ROOT"
echo "  output: $R4F_ROOT"
echo "  observer-only; standalone timing is not online latency"

TMPDIR="${BOXFUSION_R4_TMPDIR:-/dev/shm}" \
PYTHONDONTWRITEBYTECODE=1 \
CUDA_VISIBLE_DEVICES="$GPU_ID" \
"$PYTHON_BIN" "$ROOT/tools/run_tr3d_r4_feature_observer.py" \
  --r4-depth-cache-root "$R4D_ROOT" \
  --r4-feature-cache-root "$R4F_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$SCENE_LIST" \
  --active-prediction-root "$ACTIVE_ROOT" \
  --official-boxer-root "$OFFICIAL_BOXER_ROOT" \
  --expected-boxer-commit "$BOXER_COMMIT" \
  --expected-dino-sha256 "$DINO_SHA" \
  --prefix-id p100 \
  --precision bfloat16 \
  --device cuda \
  --min-support-points 2 \
  --min-feature-cells 1 \
  --report "$REPORT"

echo "R4-F observer complete: $REPORT"
