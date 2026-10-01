#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-6}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$STAGE" =~ ^[1-6]$ ]] || { echo "stage must be 1..6" >&2; exit 2; }
TAG="${BOXFUSION_LIGHTWEIGHT_TRAIN_TAG:-tr3d_lightweight_l${STAGE}_train100_v1}"
TRAIN_LIST="${BOXFUSION_LIGHTWEIGHT_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_LIGHTWEIGHT_FORBIDDEN_VAL_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
GT_ROOT="${BOXFUSION_LIGHTWEIGHT_TRAIN_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_LIGHTWEIGHT_TRAIN_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
PYTHON_BIN="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
list_sha="$(sha256sum "$TRAIN_LIST" | awk '{print $1}')"
scope="$(basename "$TRAIN_LIST" .txt)-${list_sha:0:12}"
DIAGNOSTICS="${BOXFUSION_LIGHTWEIGHT_TRAIN_DIAGNOSTICS:-$ROOT/diagnostics/b6_g0_tr3d_terminal/$TAG/$scope/online/tr3d_incremental}"
DATASET="${BOXFUSION_LIGHTWEIGHT_DATASET:-$ROOT/datasets/tr3d_lightweight_l${STAGE}_train100_v1.npz}"
REPORT="${BOXFUSION_LIGHTWEIGHT_DATASET_REPORT:-$ROOT/reports/tr3d_lightweight_l${STAGE}_train100_v1/dataset.json}"
POLICY="${BOXFUSION_LIGHTWEIGHT_POLICY:-$ROOT/models/tr3d_lightweight_l${STAGE}_gate_train100_v1.json}"

for path in "$TRAIN_LIST" "$VAL_LIST"; do [[ -f "$path" ]] || { echo "Missing list: $path" >&2; exit 2; }; done
for path in "$DIAGNOSTICS" "$GT_ROOT" "$SCANS_ROOT"; do [[ -d "$path" ]] || { echo "Missing root: $path" >&2; exit 2; }; done

"$PYTHON_BIN" "$ROOT/tools/build_tr3d_incremental_novelty_dataset.py" \
    --train-scene-list "$TRAIN_LIST" --forbidden-validation-scene-list "$VAL_LIST" \
    --diagnostics-root "$DIAGNOSTICS" --ground-truth-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" --output "$DATASET" --report "$REPORT" \
    --lightweight-stage "$STAGE"
"$PYTHON_BIN" "$ROOT/tools/train_tr3d_incremental_novelty_gate.py" \
    --dataset "$DATASET" --train-scene-list "$TRAIN_LIST" \
    --forbidden-validation-scene-list "$VAL_LIST" --output "$POLICY"
"$PYTHON_BIN" - "$POLICY" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("activation_authorized:",d["activation_authorized"])
print("OOF selection:",d["oof"]["selection"])
if not d["activation_authorized"]:
    raise SystemExit("Train-only gate failed; validation active mode is forbidden")
PY
echo "Authorized L$STAGE policy: $POLICY"
