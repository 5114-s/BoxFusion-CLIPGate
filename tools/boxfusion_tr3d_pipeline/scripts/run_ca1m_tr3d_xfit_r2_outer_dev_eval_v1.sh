#!/usr/bin/env bash
set -euo pipefail

# Single post-training entry point for the sealed xfit-R2 outer-dev protocol.
# The host GPU id is mapped to cuda:0 inside the worker process.  No fold-1 or
# official-validation path is accepted by the bound JSON configuration.
GPU_ID="${1:-0}"
if [[ ! "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "usage: $0 [non-negative-gpu-id]" >&2
  exit 2
fi

PIPELINE_ROOT="/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline"
PYTHON_BIN="/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/.conda/boxfusion-tr3d/bin/python"
RUNNER="${PIPELINE_ROOT}/tools/run_ca1m_tr3d_xfit_r2_outer_dev_eval.py"
CONFIG="${PIPELINE_ROOT}/config/ca1m_tr3d_xfit_r2_outer_dev_eval_v1.json"

export PYTHONPATH="${PIPELINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG}" preflight
CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  "${PYTHON_BIN}" "${RUNNER}" --config "${CONFIG}" all --device cuda:0
