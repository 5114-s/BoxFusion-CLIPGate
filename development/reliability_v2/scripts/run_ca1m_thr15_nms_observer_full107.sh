#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
GPU="${1:-1}"
SCENES="$ROOT/tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt"
CONFIG="$ROOT/config/ca1m_thr15_nms_observer.yaml"
PRED="$ROOT/results/ca1m_thr15_nms_observer_full107"
DIAG="$ROOT/diagnostics/ca1m_thr15_nms_observer_full107"
BOXER_DIAG="$ROOT/diagnostics/ca1m_thr15_nms_observer_boxer_full107"
LOG_ROOT="$ROOT/logs/ca1m_thr15_nms_observer_full107"
REPORT_ROOT="$ROOT/reports/ca1m_thr15_nms_child_headroom"
DATA_ROOT=/tmp/ca1m_clean_root
EVAL_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/evaluation

# Preserve earlier runs. A partial run requires an explicit new experiment path.
for directory in "$PRED" "$DIAG" "$BOXER_DIAG" "$LOG_ROOT" "$REPORT_ROOT"; do
  if [[ -e "$directory" ]]; then
    echo "Refusing to reuse existing artifact directory: $directory" >&2
    exit 1
  fi
done
cd "$ROOT"

# Validate observer wiring, source parity, dataset coverage, and the evaluator
# before allocating a GPU. This performs no inference and writes no predictions.
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" tools/audit_ca1m_nms_child_headroom.py \
  --preflight-only --config "$CONFIG" --reference-config config/ca1m_thr15.yaml \
  --scenes "$SCENES" --expected-scenes 107 --data-root "$DATA_ROOT" \
  --evaluator-root "$EVAL_ROOT" \
  --baseline-root "$PRED" --diagnostics-root "$DIAG"

mkdir -p "$PRED" "$DIAG" "$BOXER_DIAG" "$LOG_ROOT" "$REPORT_ROOT"
sha256sum "$CONFIG" "$SCENES" demo.py boxfusion/instances.py boxfusion/pvq_ar.py \
  scripts/run_boxfusion_sequences.py tools/audit_ca1m_nms_child_headroom.py \
  "$EVAL_ROOT/eval_ca1m.py" "$EVAL_ROOT/utils/box_util.py" \
  "$EVAL_ROOT/utils/eval_det.py" > "$LOG_ROOT/input_sha256.txt"
printf '%s\n' \
  'CA1M full107, native threshold=0.15, TopK3+Boxer-active+real-score' \
  'PVQ shadow observer enabled; NMS active adjudication disabled' \
  'Headroom uses final predictions from this same observer run' \
  'Coverage is an offline GT-assisted candidate diagnostic, not AP or a birth module' \
  'Anchor box3d_iou_v2 computes AABB IoU; GT world coordinates remain unchanged' \
  > "$LOG_ROOT/protocol.txt"

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
HF_HUB_OFFLINE=1 \
"$PYTHON" scripts/run_boxfusion_sequences.py \
  --dataset CA1M --seq-list "$SCENES" --config "$CONFIG" \
  --model-path "$ROOT/models/cutr_rgbd.pth" --device cuda --gpu "$GPU" \
  --log-dir "$LOG_ROOT/scenes" 2>&1 | tee "$LOG_ROOT/generate.log"

# Fail if source/config changed during generation; never mix provenance silently.
sha256sum --check "$LOG_ROOT/input_sha256.txt" > "$LOG_ROOT/source_integrity.log"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON" tools/audit_ca1m_nms_child_headroom.py \
  --config "$CONFIG" --reference-config config/ca1m_thr15.yaml \
  --scenes "$SCENES" --expected-scenes 107 --data-root "$DATA_ROOT" \
  --evaluator-root "$EVAL_ROOT" \
  --baseline-root "$PRED" --diagnostics-root "$DIAG" \
  --output "$REPORT_ROOT/headroom.json" --markdown "$REPORT_ROOT/REPORT.md" \
  2>&1 | tee "$LOG_ROOT/audit.log"

# The anchor evaluator writes relative evaluation artifacts; keep them here.
mkdir -p "$LOG_ROOT/eval_work"
cd "$LOG_ROOT/eval_work"
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
MPLCONFIGDIR=/tmp/mpl_ca1m_thr15_nms_observer \
"$PYTHON" "$EVAL_ROOT/eval_ca1m.py" \
  --dataset ca1m --data_path "$DATA_ROOT" --pred_root "$PRED" \
  --gpu "$GPU" --use_3d_nms --per_class_proposal \
  2>&1 | tee "$LOG_ROOT/eval_observer.log"

echo "CA1M_THR15_NMS_OBSERVER_FULL107_DONE"
