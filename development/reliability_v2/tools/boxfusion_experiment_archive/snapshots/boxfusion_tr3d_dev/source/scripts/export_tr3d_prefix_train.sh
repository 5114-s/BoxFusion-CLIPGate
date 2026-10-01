#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PYTHON_BIN="${BOXFUSION_TR3D_DATA_PYTHON:-python}"
SOURCE_ROOT="${BOXFUSION_TR3D_SCANNET_ROOT:-/extra/ZhaoX/scannet_data}"
PREPARED_ROOT="${BOXFUSION_TR3D_DATA_ROOT:-${ROOT_DIR}/data/tr3d_scannet}"
FRAMES_ROOT="${BOXFUSION_TR3D_FRAMES_ROOT:-${SOURCE_ROOT}/scans.sens}"
SCENE_LIST="${BOXFUSION_TR3D_PREFIX_SCENE_LIST:-${PREPARED_ROOT}/splits/trajectory_available_train.txt}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" tools/export_tr3d_trajectory_prefixes.py \
  --prepared-root "${PREPARED_ROOT}" \
  --frames-root "${FRAMES_ROOT}" \
  --scene-list "${SCENE_LIST}" \
  --source-info "${SOURCE_ROOT}/scannet_infos_train.pkl" \
  --source-points "${SOURCE_ROOT}/points" \
  "$@"
