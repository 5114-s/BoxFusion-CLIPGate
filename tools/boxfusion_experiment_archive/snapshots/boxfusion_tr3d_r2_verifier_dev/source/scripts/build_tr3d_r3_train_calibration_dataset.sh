#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"
RUN_TAG="${1:-${BOXFUSION_R3_TRAIN_RUN_TAG:-}}"
[[ -n "$RUN_TAG" && "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$ ]] || {
  echo "Usage: $0 <train-r3-run-tag>" >&2
  exit 2
}
PYTHON_BIN="${BOXFUSION_R3_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
FROZEN_MANIFEST="$ROOT/manifests/frozen_g0_selective_boxer_train100.json"
R3_EXPORT="$ROOT/reports/tr3d_r3_train/$RUN_TAG/export_report.json"
R3_CACHE_ROOT="$ROOT/cache/tr3d_r3_train/$RUN_TAG"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt"
VAL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
OFFICIAL_TRAIN_SCENE_LIST="/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/third_party/mmdetection3d/data/scannet/meta_data/scannetv2_train.txt"
SCANS_ROOT="/extra/ZhaoX/scannet_data/scans"
GT_ROOT="/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data"
OUTPUT="$ROOT/datasets/tr3d_r3_calibration/$RUN_TAG.npz"
REPORT="$ROOT/reports/tr3d_r3_calibration/$RUN_TAG/dataset_report.json"

for path in "$PYTHON_BIN" "$FROZEN_MANIFEST" "$R3_EXPORT" "$R3_CACHE_ROOT" \
  "$SCENE_LIST" "$VAL_SCENE_LIST" "$OFFICIAL_TRAIN_SCENE_LIST" "$SCANS_ROOT" "$GT_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing R3 calibration-dataset input: $path" >&2; exit 2; }
done

"$PYTHON_BIN" tools/build_tr3d_r3_calibration_dataset.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --r3-export-report "$R3_EXPORT" \
  --r3-cache-root "$R3_CACHE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --forbidden-scene-list "$VAL_SCENE_LIST" \
  --official-train-scene-list "$OFFICIAL_TRAIN_SCENE_LIST" \
  --scans-root "$SCANS_ROOT" \
  --gt-root "$GT_ROOT" \
  --prefix-id p100 \
  --output "$OUTPUT" \
  --report "$REPORT"

echo "Immutable train-only calibration dataset: $OUTPUT"
echo "This artifact discloses that epoch12 TR3D was trained on ScanNet train."
