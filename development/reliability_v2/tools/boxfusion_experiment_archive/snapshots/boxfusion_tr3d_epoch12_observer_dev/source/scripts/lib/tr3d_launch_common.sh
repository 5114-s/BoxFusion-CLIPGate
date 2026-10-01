#!/usr/bin/env bash
# Shared fail-closed helpers for the isolated genuine-TR3D launchers.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This file is a library; source it from a TR3D launcher." >&2
  exit 2
fi

tr3d_die() {
  echo "TR3D launch refused: $*" >&2
  exit 2
}

tr3d_require_file() {
  [[ -f "$1" ]] || tr3d_die "missing required file: $1"
}

tr3d_require_tag() {
  local value="$1"
  [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$ ]] \
    || tr3d_die "run tag must be 3-80 safe characters: $value"
}

tr3d_select_env() {
  TR3D_ENV_REF="$1"
  if [[ "$TR3D_ENV_REF" == */* ]]; then
    [[ "$TR3D_ENV_REF" == /* ]] || TR3D_ENV_REF="$TR3D_ROOT/$TR3D_ENV_REF"
    TR3D_CONDA_SELECTOR=(-p "$TR3D_ENV_REF")
  else
    TR3D_CONDA_SELECTOR=(-n "$TR3D_ENV_REF")
  fi
  command -v conda >/dev/null 2>&1 || tr3d_die "conda is unavailable"
}

tr3d_conda_run() {
  env -u PYTHONPATH -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
    conda run --no-capture-output "${TR3D_CONDA_SELECTOR[@]}" \
    env PYTHONNOUSERSITE=1 \
      CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
      OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" \
      MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/boxfusion-tr3d-matplotlib}" \
      PYTHONPATH="$TR3D_ROOT/third_party/mmdetection3d:$TR3D_ROOT" \
      "$@"
}

tr3d_parse_gpus() {
  local spec="$1"
  IFS=',' read -r -a TR3D_GPUS <<< "$spec"
  (( ${#TR3D_GPUS[@]} > 0 )) || tr3d_die "at least one GPU is required"
  local gpu
  for gpu in "${TR3D_GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || tr3d_die "invalid GPU index: $gpu"
  done
  TR3D_GPU_SPEC="$(IFS=,; echo "${TR3D_GPUS[*]}")"
}

tr3d_prepare_unique_root() {
  local path="$1"
  local resume="$2"
  if [[ "$resume" == "1" ]]; then
    [[ -d "$path" ]] || tr3d_die "resume root does not exist: $path"
  else
    [[ ! -e "$path" ]] || tr3d_die "output root already exists: $path"
    mkdir -p "$path"
  fi
}

tr3d_require_new_root() {
  local path="$1"
  [[ ! -e "$path" ]] || tr3d_die "output root already exists: $path"
  mkdir -p "$path"
}

tr3d_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

tr3d_check_environment() {
  local config="$1"
  local visible_gpu="$2"
  local check_dataset="${3:-0}"
  local dataset_args=()
  [[ "$check_dataset" == "1" ]] && dataset_args=(--check-dataset-sample)
  CUDA_VISIBLE_DEVICES="$visible_gpu" \
    "$TR3D_ROOT/scripts/check_tr3d_environment.sh" \
      "$TR3D_ENV_REF" \
      --config "$config" \
      --build-model \
      "${dataset_args[@]}" \
      --require-cuda
}
