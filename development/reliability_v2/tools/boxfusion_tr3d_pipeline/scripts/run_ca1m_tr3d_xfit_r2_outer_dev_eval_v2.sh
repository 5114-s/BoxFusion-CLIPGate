#!/usr/bin/env bash
set -euo pipefail

# Post-training entry point.  The v2 evaluation config additionally requires
# the fixed outer training wrapper log to terminate with unique TRAIN_EXIT=0.
GPU_ID="${1:-0}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [non-negative-gpu-id]" >&2
  exit 2
fi

PIPELINE_ROOT="/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline"
PYTHON_BIN="/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/.conda/boxfusion-tr3d/bin/python"
RUNNER="${PIPELINE_ROOT}/tools/run_ca1m_tr3d_xfit_r2_outer_dev_eval.py"
CONFIG="${PIPELINE_ROOT}/config/ca1m_tr3d_xfit_r2_outer_dev_eval_v2.json"

export PYTHONPATH="${PIPELINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG}" preflight
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG}" all --device cuda:0
