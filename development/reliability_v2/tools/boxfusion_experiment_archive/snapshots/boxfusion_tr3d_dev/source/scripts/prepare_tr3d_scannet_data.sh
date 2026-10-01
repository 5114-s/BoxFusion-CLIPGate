#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${BOXFUSION_TR3D_DATA_PYTHON:-python}"
SOURCE_ROOT="${BOXFUSION_TR3D_SCANNET_ROOT:-/extra/ZhaoX/scannet_data}"
OUTPUT_ROOT="${BOXFUSION_TR3D_DATA_ROOT:-${ROOT_DIR}/data/tr3d_scannet}"
FRAMES_ROOT="${BOXFUSION_TR3D_FRAMES_ROOT:-${SOURCE_ROOT}/scans.sens}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/prepare_tr3d_scannet.py \
  --source-root "${SOURCE_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --frames-root "${FRAMES_ROOT}" \
  "$@"
