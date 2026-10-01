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
DATASET="$ROOT/datasets/tr3d_r3_calibration/$RUN_TAG.npz"
MODEL="$ROOT/models/tr3d_r3_veto/$RUN_TAG.json"
REPORT="$ROOT/reports/tr3d_r3_calibration/$RUN_TAG/training_report.json"
[[ -e "$PYTHON_BIN" && -f "$DATASET" ]] || {
  echo "Missing Python or calibration dataset: $DATASET" >&2
  exit 2
}

"$PYTHON_BIN" tools/train_tr3d_r3_veto_calibrator.py \
  --dataset "$DATASET" \
  --model "$MODEL" \
  --report "$REPORT"

"$PYTHON_BIN" - "$MODEL" "$REPORT" <<'PY'
import json
from pathlib import Path
import sys
model = json.loads(Path(sys.argv[1]).read_text())
report = json.loads(Path(sys.argv[2]).read_text())
print("train gate:", "PASS" if report["gate_pass"] else "FAIL")
print("model activation_authorized:", model["activation_authorized"])
print("OOF delta AP15/AP25/AP50:", report["oof_deltas"])
print("OOF veto minus raw primary:", report["oof_veto_minus_raw"])
if not report["gate_pass"]:
    print("Calibration is observer-only; do not run validation active materialization.")
PY
