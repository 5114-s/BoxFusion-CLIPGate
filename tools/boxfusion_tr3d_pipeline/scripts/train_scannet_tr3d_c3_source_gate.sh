#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_LIST="${BOXFUSION_C3_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_LIST="${BOXFUSION_C3_FORBIDDEN_VAL_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
DIAGNOSTICS="${BOXFUSION_C3_TRAIN_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/tr3d_c3_online_train100_v1}"
GT_ROOT="${BOXFUSION_SCANNET_TRAIN_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
DATASET="${BOXFUSION_C3_TRAIN_DATASET:-$ROOT/datasets/tr3d_c3_source_gate_train100_v1.npz}"
DATASET_REPORT="${BOXFUSION_C3_TRAIN_DATASET_REPORT:-$ROOT/reports/tr3d_c3_source_gate_train100_v1/dataset.json}"
POLICY="${BOXFUSION_C3_ACTIVE_POLICY:-$ROOT/models/tr3d_c3_source_gate_train100_v1.json}"
PYTHON_BIN="${BOXFUSION_PYTHON:-python}"

for path in "$TRAIN_LIST" "$VAL_LIST"; do
    [[ -f "$path" ]] || { echo "Missing scene list: $path" >&2; exit 1; }
done
[[ -d "$DIAGNOSTICS" ]] || {
    echo "Missing train-only online C3 diagnostics: $DIAGNOSTICS" >&2
    echo "Do not substitute validation diagnostics." >&2
    exit 1
}
[[ -d "$GT_ROOT" ]] || { echo "Missing ScanNet train GT root: $GT_ROOT" >&2; exit 1; }

echo "Building leakage-checked train-only C3 dataset"
"$PYTHON_BIN" "$ROOT/tools/build_tr3d_c3_source_gate_dataset.py" \
    --train-scene-list "$TRAIN_LIST" \
    --forbidden-validation-scene-list "$VAL_LIST" \
    --diagnostics-root "$DIAGNOSTICS" \
    --ground-truth-root "$GT_ROOT" \
    --output "$DATASET" \
    --report "$DATASET_REPORT"

echo "Training scene-grouped OOF C3 source gate"
"$PYTHON_BIN" "$ROOT/tools/train_tr3d_c3_source_gate.py" \
    --dataset "$DATASET" \
    --train-scene-list "$TRAIN_LIST" \
    --forbidden-validation-scene-list "$VAL_LIST" \
    --output "$POLICY"

"$PYTHON_BIN" - "$POLICY" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
print(f"policy: {path}")
print(f"activation_authorized: {payload['activation_authorized']}")
print(f"OOF selection: {payload['oof']['selection']}")
if not payload["activation_authorized"]:
    raise SystemExit(
        "Train-only gate failed; policy remains observer-only and validation "
        "active materialization is forbidden."
    )
PY
