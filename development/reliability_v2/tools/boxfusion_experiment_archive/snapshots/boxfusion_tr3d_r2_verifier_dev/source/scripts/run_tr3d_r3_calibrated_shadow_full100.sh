#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

RUN_TAG="${1:-${BOXFUSION_R3_CALIBRATED_RUN_TAG:-}}"
CALIBRATOR_MODEL="${2:-${BOXFUSION_R3_CALIBRATOR_MODEL:-}}"
[[ -n "$RUN_TAG" && "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$ ]] || {
  echo "Usage: $0 <unique-val-run-tag> <authorized-calibrator.json>" >&2
  exit 2
}
[[ -n "$CALIBRATOR_MODEL" ]] || {
  echo "Missing authorized train-only calibrator model." >&2
  echo "Usage: $0 <unique-val-run-tag> <authorized-calibrator.json>" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_R3_ACTIVE_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
FROZEN_MANIFEST="$ROOT/manifests/frozen_g0_selective_boxer_full100.json"
R3_CACHE_ROOT="$ROOT/cache/tr3d_r3/r3_near_full100_v1"
R3_EXPORT_REPORT="$ROOT/reports/tr3d_r3/r3_near_full100_v1/export_report.json"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SCANS_ROOT="/extra/ZhaoX/scannet_data/scans"
GT_ROOT="/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data"
ACTIVE_ROOT="$ROOT/results/tr3d_r3_calibrated_shadow/$RUN_TAG"
REPORT_ROOT="$ROOT/reports/tr3d_r3_calibrated_shadow/$RUN_TAG"
LOG_ROOT="$ROOT/logs/tr3d_r3_calibrated_shadow/$RUN_TAG"
EVAL_ROOT="$ROOT/evaluation/tr3d_r3_calibrated_shadow/$RUN_TAG"
MATERIALIZE_REPORT="$REPORT_ROOT/materialize_manifest.json"
FROZEN_SHA="327b0cfb07265db04db3af2f631e27e1165a65c9367a9db1d09a31299911342e"
R3_EXPORT_SHA="bb76491e4e7139be57038134a7f6b55e39e9aeb988f01ad3355209a430fcad4c"

CALIBRATOR_MODEL="$(readlink -f "$CALIBRATOR_MODEL")"
for path in \
  "$PYTHON_BIN" "$FROZEN_MANIFEST" "$R3_EXPORT_REPORT" "$R3_CACHE_ROOT" \
  "$CALIBRATOR_MODEL" "$SCENE_LIST" "$SCANS_ROOT" "$GT_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing calibrated-shadow input: $path" >&2; exit 2; }
done

check_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "Pinned SHA mismatch: $path" >&2
    echo "  expected: $expected" >&2
    echo "  observed: $observed" >&2
    exit 2
  }
}
check_sha "$FROZEN_MANIFEST" "$FROZEN_SHA"
check_sha "$R3_EXPORT_REPORT" "$R3_EXPORT_SHA"

# Fail closed on the train gate before claiming any validation namespace.
"$PYTHON_BIN" - "$CALIBRATOR_MODEL" <<'PY'
from pathlib import Path
import sys
from boxfusion.tr3d_r3_calibrator import load_calibrator
path = Path(sys.argv[1])
if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
    raise SystemExit(f"calibrator must be a regular immutable file: {path}")
model = load_calibrator(path)
if not model.activation_authorized or model.metadata.get("train_gate_pass") is not True:
    raise SystemExit("calibrator is not authorized by the train-only gate")
print("Authorized train-only calibrator:", path)
PY

for path in "$ACTIVE_ROOT" "$REPORT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"; do
  [[ ! -e "$path" ]] || {
    echo "Refusing existing calibrated-shadow namespace: $path" >&2
    echo "Choose a new run tag; immutable outputs are never overwritten." >&2
    exit 2
  }
done
mkdir -p "$REPORT_ROOT" "$LOG_ROOT" "$EVAL_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "Another process holds $LOG_ROOT/run.lock" >&2; exit 2; }

MODEL_SHA="$(sha256sum "$CALIBRATOR_MODEL" | awk '{print $1}')"
echo "R3 veto-calibrated shadow-active full100 validation"
echo "  baseline: frozen G0 + Selective Boxer"
echo "  primary: frozen R3 anchor-near rule"
echo "  new module: train-authorized harm veto only"
echo "  calibrator file SHA256: $MODEL_SHA"
echo "  policy: geometry-only; label/score/order/count remain byte-exact"
echo "  validation is one-shot shadow evaluation; no validation tuning is permitted"

"$PYTHON_BIN" tools/materialize_tr3d_r3_calibrated_shadow.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --r3-export-report "$R3_EXPORT_REPORT" \
  --r3-cache-root "$R3_CACHE_ROOT" \
  --calibrator-model "$CALIBRATOR_MODEL" \
  --scene-list "$SCENE_LIST" \
  --scans-root "$SCANS_ROOT" \
  --output-root "$ACTIVE_ROOT" \
  --manifest "$MATERIALIZE_REPORT" \
  --prefix-id p100 \
  > "$LOG_ROOT/materialize_stdout.json"

echo "Immutable materialization passed; starting the unmodified ScanNet evaluator"
(
  cd "$ROOT/evaluation"
  PYTHONDONTWRITEBYTECODE=1 \
  MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
  "$PYTHON_BIN" eval_scannet.py \
    --dataset scannet \
    --data_path "$SCANS_ROOT" \
    --gt_root "$GT_ROOT" \
    --dump_dir "$EVAL_ROOT" \
    --num_point 40000 \
    --cluster_sampling seed_fps \
    --use_3d_nms \
    --use_cls_nms \
    --per_class_proposal \
    --num_workers 0 \
    --gpu 0 \
    --seed 0 \
    --scene_list "$SCENE_LIST" \
    --pred_root "$ACTIVE_ROOT"
) > "$LOG_ROOT/eval_stdout.log" 2>&1

grep -E '^eval (mAP|APrec|ARecall):' "$LOG_ROOT/eval_stdout.log"
echo "R3 calibrated shadow full100 evaluation completed"
echo "  predictions: $ACTIVE_ROOT"
echo "  immutable lineage manifest: $MATERIALIZE_REPORT"
echo "  standard evaluator log: $LOG_ROOT/eval_stdout.log"
