#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
ENV_REF="${1:-${BOXFUSION_TR3D_ENV:-${ROOT_DIR}/.conda/boxfusion-tr3d}}"
if (( $# > 0 )); then
  shift
fi
if [[ "${ENV_REF}" == */* ]]; then
  if [[ "${ENV_REF}" != /* ]]; then
    ENV_REF="${ROOT_DIR}/${ENV_REF}"
  fi
  CONDA_SELECTOR=(-p "${ENV_REF}")
else
  CONDA_SELECTOR=(-n "${ENV_REF}")
fi

"${SCRIPT_DIR}/check_tr3d_vendor.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required to select the isolated TR3D environment." >&2
  exit 2
fi

# Clear paths inherited from an activated research environment. They can make
# Python load libtorch or packages from a different conda prefix.
env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  conda run --no-capture-output "${CONDA_SELECTOR[@]}" \
  env PYTHONNOUSERSITE=1 \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/boxfusion-tr3d-matplotlib}" \
      PYTHONPATH="${ROOT_DIR}/third_party/mmdetection3d:${ROOT_DIR}" \
  python "${ROOT_DIR}/tools/check_tr3d_environment.py" "$@"
