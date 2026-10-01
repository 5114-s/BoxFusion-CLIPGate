#!/usr/bin/env bash
set -euo pipefail

# CPU-only YiDu A6 training protocol:
# A5 train diagnostics -> strict geometry adapter -> train-only IoU labels
# -> generic 91-D AP50 gate -> provenance validation.
#
# The default is check-only. No file is created until
# BOXFUSION_YIDU_GATE_TRAIN_EXECUTE=1 is supplied.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_YIDU_ENV_ROOT:-${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion2}}"
PYTHON="$ENV_ROOT/bin/python"
EXECUTE="${BOXFUSION_YIDU_GATE_TRAIN_EXECUTE:-0}"

DIAGNOSTICS_ROOT="${BOXFUSION_YIDU_TRAIN_DIAGNOSTICS_ROOT:-}"
PREDICTION_ROOT="${BOXFUSION_YIDU_TRAIN_PRED_ROOT:-}"
SCENE_LIST="${BOXFUSION_YIDU_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FORBIDDEN_LIST="${BOXFUSION_YIDU_FORBIDDEN_VAL_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
GT_ROOT="${BOXFUSION_YIDU_GT_ROOT:-$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data}"
SCAN_ROOT="${BOXFUSION_YIDU_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"

GEOMETRY_ROOT="${BOXFUSION_YIDU_GATE_GEOMETRY_ROOT:-$ROOT/datasets/yidu_a5_train_geometry_v1}"
TRAINING_ARCHIVE="${BOXFUSION_YIDU_GATE_TRAINING_ARCHIVE:-$ROOT/datasets/scannet_yidu_ap50_gate_trainonly_v1.npz}"
CHECKPOINT="${BOXFUSION_YIDU_GATE_CHECKPOINT:-$ROOT/models/scannet_yidu_ap50_gate_trainonly_v1.npz}"
EPOCHS="${BOXFUSION_YIDU_GATE_EPOCHS:-400}"
SEED="${BOXFUSION_YIDU_GATE_SEED:-1337}"

die() {
    echo "YiDu A6 trainer: $*" >&2
    exit 1
}

canonical() {
    realpath -m -- "$1"
}

require_output_inside_checkout() {
    local role="$1"
    local path="$2"
    local root_real
    local path_real
    root_real="$(canonical "$ROOT")"
    path_real="$(canonical "$path")"
    case "$path_real" in
        "$root_real"/*) ;;
        *) die "$role must remain inside the isolated checkout: $path" ;;
    esac
}

case "$EXECUTE" in
    0|1) ;;
    *) die "BOXFUSION_YIDU_GATE_TRAIN_EXECUTE must be 0 or 1" ;;
esac
[[ "$EPOCHS" =~ ^[1-9][0-9]*$ ]] \
    || die "BOXFUSION_YIDU_GATE_EPOCHS must be positive"
[[ "$SEED" =~ ^[0-9]+$ ]] \
    || die "BOXFUSION_YIDU_GATE_SEED must be non-negative"
[[ -x "$PYTHON" ]] || die "missing Python runtime: $PYTHON"
[[ -n "$DIAGNOSTICS_ROOT" ]] \
    || die "BOXFUSION_YIDU_TRAIN_DIAGNOSTICS_ROOT is required"
[[ -n "$PREDICTION_ROOT" ]] \
    || die "BOXFUSION_YIDU_TRAIN_PRED_ROOT is required"
for directory in \
    "$DIAGNOSTICS_ROOT" "$PREDICTION_ROOT" "$GT_ROOT" "$SCAN_ROOT"; do
    [[ -d "$directory" ]] || die "missing input directory: $directory"
done
for path in "$SCENE_LIST" "$FORBIDDEN_LIST"; do
    [[ -s "$path" ]] || die "missing or empty scene list: $path"
done
[[ "${TRAINING_ARCHIVE,,}" == *.npz ]] \
    || die "training archive must end in .npz"
[[ "${CHECKPOINT,,}" == *.npz ]] \
    || die "checkpoint must end in .npz"

require_output_inside_checkout "geometry root" "$GEOMETRY_ROOT"
require_output_inside_checkout "training archive" "$TRAINING_ARCHIVE"
require_output_inside_checkout "checkpoint" "$CHECKPOINT"
[[ "$(canonical "$GEOMETRY_ROOT")" != "$(canonical "$DIAGNOSTICS_ROOT")" ]] \
    || die "geometry and diagnostics roots must differ"
[[ "$(canonical "$GEOMETRY_ROOT")" != "$(canonical "$PREDICTION_ROOT")" ]] \
    || die "geometry and prediction roots must differ"
[[ ! -e "$GEOMETRY_ROOT" ]] \
    || die "geometry root already exists; choose a fresh output path"
[[ ! -e "$TRAINING_ARCHIVE" ]] \
    || die "training archive already exists; choose a fresh output path"
[[ ! -e "$CHECKPOINT" ]] \
    || die "checkpoint already exists; choose a fresh output path"

"$PYTHON" - "$SCENE_LIST" "$FORBIDDEN_LIST" <<'PY'
import re
import sys
from pathlib import Path

pattern = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")

def read(path):
    rows = [
        row.strip()
        for row in Path(path).read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    if not rows or any(pattern.fullmatch(row) is None for row in rows):
        raise SystemExit(f"invalid scene list: {path}")
    if len(rows) != len(set(rows)):
        raise SystemExit(f"duplicate scene ID in {path}")
    return set(rows)

train = read(sys.argv[1])
forbidden = read(sys.argv[2])
overlap = sorted(train & forbidden)
if overlap:
    raise SystemExit(
        "train/validation leakage: " + ", ".join(overlap[:8])
    )
PY

echo "YiDu A6 train-only protocol"
echo "  mode: $([[ "$EXECUTE" == "1" ]] && echo execute || echo check-only)"
echo "  A5 diagnostics: $DIAGNOSTICS_ROOT"
echo "  A5 predictions: $PREDICTION_ROOT"
echo "  train scenes: $SCENE_LIST"
echo "  forbidden scenes: $FORBIDDEN_LIST"
echo "  geometry output: $GEOMETRY_ROOT"
echo "  training archive: $TRAINING_ARCHIVE"
echo "  checkpoint: $CHECKPOINT"
echo "  epochs/seed: $EPOCHS/$SEED"
echo "  device: CPU only"

if [[ "$EXECUTE" != "1" ]]; then
    echo "CHECK PASSED. No files were created and no GPU process was started."
    exit 0
fi

export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

"$PYTHON" "$ROOT/tools/export_yidu_geometry_candidates.py" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --prediction-root "$PREDICTION_ROOT" \
    --scene-list "$SCENE_LIST" \
    --output-root "$GEOMETRY_ROOT" \
    --stage A5

"$PYTHON" "$ROOT/tools/build_ap50_gate_training_from_trifusion.py" \
    --geometry-root "$GEOMETRY_ROOT" \
    --prediction-root "$PREDICTION_ROOT" \
    --scene-list "$SCENE_LIST" \
    --forbidden-scene-list "$FORBIDDEN_LIST" \
    --gt-root "$GT_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --output "$TRAINING_ARCHIVE" \
    --verified-only

"$PYTHON" "$ROOT/tools/train_ap50_safety_gate.py" \
    "$TRAINING_ARCHIVE" \
    --output "$CHECKPOINT" \
    --forbidden-scene-list "$FORBIDDEN_LIST" \
    --epochs "$EPOCHS" \
    --seed "$SEED"

"$PYTHON" "$ROOT/tools/validate_yidu_gate_provenance.py" \
    --checkpoint "$CHECKPOINT" \
    --training-archive "$TRAINING_ARCHIVE" \
    --train-scene-list "$SCENE_LIST" \
    --forbidden-scene-list "$FORBIDDEN_LIST"

echo "YiDu A6 checkpoint trained and provenance-validated."
echo "Use all four provenance paths when running the A6 observer."
