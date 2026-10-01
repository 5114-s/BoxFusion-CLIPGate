#!/usr/bin/env bash
set -euo pipefail

# Link the deterministic ScanNet train-only subset used by B6.
# This machine already has complete extracted RGB-D data under scans.sens;
# the repository only stores lightweight <scene>/frames symlinks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
EXTRACTED_ROOT="${BOXFUSION_SCANNET_EXTRACTED_ROOT:-/extra/ZhaoX/scannet_data/scans.sens}"
SCENE_LIST="${BOXFUSION_B6_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$ROOT/data/scannet_train}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing B6 Python environment: $PYTHON" >&2
    exit 1
fi
if [[ ! -f "$SCENE_LIST" ]]; then
    echo "Missing B6 train-only scene list: $SCENE_LIST" >&2
    exit 1
fi
if [[ "$(basename "$SCENE_LIST")" == *val* ]]; then
    echo "Refusing validation-labelled frame preparation: $SCENE_LIST" >&2
    exit 1
fi
if [[ ! -d "$EXTRACTED_ROOT" ]]; then
    echo "Missing extracted ScanNet RGB-D root: $EXTRACTED_ROOT" >&2
    exit 1
fi

echo "Preparing B6 ScanNet train frame links"
echo "Scene list: $SCENE_LIST"
echo "Extracted data root: $EXTRACTED_ROOT"
echo "Linked frames root: $FRAMES_ROOT"

"$PYTHON" "$ROOT/scripts/link_scannet_scene_frames.py" \
    --scene-list "$SCENE_LIST" \
    --source-root "$EXTRACTED_ROOT" \
    --frames-root "$FRAMES_ROOT"

echo "B6 ScanNet train frame links completed"
