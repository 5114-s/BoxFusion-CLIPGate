#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DEVICE="${1:-cpu}"
ENV_REF="${BOXFUSION_TR3D_ENV:-${ROOT_DIR}/.conda/boxfusion-tr3d}"

if [[ "${ENV_REF}" == */* ]]; then
  if [[ "${ENV_REF}" != /* ]]; then
    ENV_REF="${ROOT_DIR}/${ENV_REF}"
  fi
  CONDA_SELECTOR=(-p "${ENV_REF}")
else
  CONDA_SELECTOR=(-n "${ENV_REF}")
fi

cd "${ROOT_DIR}"
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  conda run --no-capture-output "${CONDA_SELECTOR[@]}" \
  env PYTHONNOUSERSITE=1 \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/boxfusion-tr3d-matplotlib}" \
      PYTHONPATH="${ROOT_DIR}/third_party/mmdetection3d:${ROOT_DIR}" \
  python tools/smoke_load_tr3d_foreground_checkpoint.py --device "${DEVICE}"
