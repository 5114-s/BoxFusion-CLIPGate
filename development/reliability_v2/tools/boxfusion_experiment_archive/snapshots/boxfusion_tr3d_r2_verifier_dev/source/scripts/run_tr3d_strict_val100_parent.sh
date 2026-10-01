#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TR3D_ROOT="$ROOT"
# shellcheck source=scripts/lib/tr3d_launch_common.sh
source "$ROOT/scripts/lib/tr3d_launch_common.sh"
cd "$ROOT"

GPU_SPEC="${1:-0,1}"
RESUME="${BOXFUSION_TR3D_RESUME:-}"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || tr3d_die \
  "set BOXFUSION_TR3D_RESUME explicitly to 0 (new) or 1 (resume)"
tr3d_parse_gpus "$GPU_SPEC"
[[ "${#TR3D_GPUS[@]}" == "2" ]] || tr3d_die \
  "strict val100 parent inference requires exactly two GPUs"
[[ "${TR3D_GPUS[0]}" != "${TR3D_GPUS[1]}" ]] || tr3d_die \
  "the two GPU indices must be distinct"

SOURCE_ROOT="${BOXFUSION_TR3D_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev}"
tr3d_select_env "${BOXFUSION_TR3D_ENV:-$SOURCE_ROOT/.conda/boxfusion-tr3d}"
CHECKPOINT="${BOXFUSION_TR3D_CHECKPOINT:-$SOURCE_ROOT/work_dirs/tr3d/tr3d_fg_full_seed0_fp32_v1/epoch_12.pth}"
CONFIG="$ROOT/config/tr3d/tr3d_scannet_foreground_official_val.py"
EXPECTED_CHECKPOINT_SHA="a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
EXPECTED_CONFIG_SHA="709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785"
FULL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
FIXED_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
PREFIX_ROOT="$ROOT/data/tr3d_prefix_val100_boxfusion_causal_p100_v3"
FULL_MANIFEST="$PREFIX_ROOT/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl"
FULL_INFO="$PREFIX_ROOT/annotations/scannet_infos_prefix_val100_boxfusion_causal_p100_v3.pkl"
FIXED_PREFIX_ROOT="$ROOT/data/tr3d_prefix_val10_boxfusion_causal_p100_v2"
FIXED_MANIFEST="$FIXED_PREFIX_ROOT/manifests/trajectory_prefix_val10_boxfusion_causal_p100_v2.jsonl"
FIXED_INFO="$FIXED_PREFIX_ROOT/annotations/scannet_infos_prefix_val10_boxfusion_causal_p100_v2.pkl"
CACHE_ROOT="$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3"
FIXED_CACHE_ROOT="$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_fixed10_v2"
ATTEMPT_TAG="${BOXFUSION_TR3D_ATTEMPT_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
LOG_ROOT="${BOXFUSION_TR3D_LOG_ROOT:-$ROOT/logs/tr3d_parent/full100_v3/$ATTEMPT_TAG}"
PYTHON_BIN="${BOXFUSION_TR3D_CONTROL_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"

for path in \
  "$CHECKPOINT" "$CONFIG" "$FULL_SCENE_LIST" "$FIXED_SCENE_LIST" \
  "$FULL_MANIFEST" "$FULL_INFO" "$FIXED_MANIFEST" "$FIXED_INFO" \
  "$FIXED_CACHE_ROOT" "$PYTHON_BIN"; do
  [[ -e "$path" ]] || tr3d_die "missing strict-parent input: $path"
done
[[ "$(tr3d_sha256 "$CHECKPOINT")" == "$EXPECTED_CHECKPOINT_SHA" ]] \
  || tr3d_die "epoch12 checkpoint SHA256 mismatch"
[[ "$(tr3d_sha256 "$CONFIG")" == "$EXPECTED_CONFIG_SHA" ]] \
  || tr3d_die "TR3D config SHA256 mismatch"

audit_prefix_and_optional_cache() {
  local cache_args=()
  if [[ "$1" == "with-cache" ]]; then
    cache_args=(
      --full-cache-root "$CACHE_ROOT"
      --fixed-cache-root "$FIXED_CACHE_ROOT"
      --prefix-id p100
      --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA"
      --config-sha256 "$EXPECTED_CONFIG_SHA"
    )
  fi
  "$PYTHON_BIN" "$ROOT/tools/audit_tr3d_prefix_superset.py" \
    --full-manifest "$FULL_MANIFEST" \
    --fixed-manifest "$FIXED_MANIFEST" \
    --full-info "$FULL_INFO" \
    --fixed-info "$FIXED_INFO" \
    --full-scene-list "$FULL_SCENE_LIST" \
    --fixed-scene-list "$FIXED_SCENE_LIST" \
    --expected-full-scene-count 100 \
    "${cache_args[@]}"
}

# Always prove that the GPU inputs are a strict fixed10-content superset before
# loading CUDA or touching the immutable parent-cache namespace.
audit_prefix_and_optional_cache without-cache

if "$PYTHON_BIN" "$ROOT/tools/validate_tr3d_residual_cache.py" \
    --cache-root "$CACHE_ROOT" \
    --scene-list "$FULL_SCENE_LIST" \
    --prefix-id p100 \
    --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA" \
    --config-sha256 "$EXPECTED_CONFIG_SHA" >/dev/null 2>&1; then
  audit_prefix_and_optional_cache with-cache
  echo "Strict val100 parent cache already complete and fixed10-array-identical."
  echo "No GPU/environment initialization was performed."
  exit 0
fi

cache_root_preexisting=0
if [[ -e "$CACHE_ROOT" ]]; then
  cache_root_preexisting=1
  [[ "$RESUME" == "1" ]] || tr3d_die \
    "partial cache exists; rerun only with BOXFUSION_TR3D_RESUME=1"
else
  [[ "$RESUME" == "0" ]] || tr3d_die \
    "resume requested but cache root does not exist: $CACHE_ROOT"
fi

gpu_compute_pids() {
  local gpu="$1"
  local output
  local attempt
  # The large prefix-content audit opens and hashes every point artifact.
  # On this host NVML can transiently fail immediately after that I/O-heavy
  # process exits, even though a fresh query one second later succeeds.  Retry
  # the read-only occupancy query, but still fail closed if it stays unhealthy.
  for attempt in 1 2 3 4 5; do
    if output="$(nvidia-smi -i "$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>&1)"; then
      printf '%s\n' "$output" \
        | awk '/^[[:space:]]*[0-9]+[[:space:]]*$/ {gsub(/[[:space:]]/, ""); print}'
      return 0
    fi
    [[ "$attempt" == "5" ]] || sleep 1
  done
  echo "nvidia-smi failed for GPU $gpu after 5 attempts: $output" >&2
  return 2
}

for gpu in "${TR3D_GPUS[@]}"; do
  busy="$(gpu_compute_pids "$gpu")" || tr3d_die \
    "GPU occupancy could not be verified for GPU $gpu"
  [[ -z "$busy" ]] || tr3d_die \
    "GPU $gpu is busy with compute PIDs: $(echo "$busy" | tr '\n' ' ')"
done

tr3d_require_new_root "$LOG_ROOT"
tr3d_check_environment "$CONFIG" "${TR3D_GPUS[0]}"
if [[ "$cache_root_preexisting" == "0" ]]; then
  # Claim the immutable namespace only after the GPU/environment preflight has
  # passed.  A busy GPU therefore cannot strand an empty "partial" cache.
  mkdir "$CACHE_ROOT"
fi

echo "Strict val100 p100 epoch12 parent-cache inference"
echo "  GPUs: $TR3D_GPU_SPEC"
echo "  resume: $RESUME (explicit)"
echo "  input: $FULL_MANIFEST"
echo "  cache: $CACHE_ROOT"
echo "  checkpoint/config SHA locked"
echo "  score/max proposals/voxel: 0.01/1000/0.01"

BOXFUSION_TR3D_ENV="$TR3D_ENV_REF" \
BOXFUSION_TR3D_CONFIG="$CONFIG" \
BOXFUSION_TR3D_CHECKPOINT="$CHECKPOINT" \
BOXFUSION_TR3D_INPUT_MANIFEST="$FULL_MANIFEST" \
BOXFUSION_TR3D_SCENE_LIST="$FULL_SCENE_LIST" \
BOXFUSION_TR3D_PREFIX_ID=p100 \
BOXFUSION_TR3D_SCORE_THRESHOLD=0.01 \
BOXFUSION_TR3D_MAX_PROPOSALS=1000 \
BOXFUSION_TR3D_VOXEL_SIZE=0.01 \
BOXFUSION_TR3D_CACHE_ROOT="$CACHE_ROOT" \
BOXFUSION_TR3D_LOG_ROOT="$LOG_ROOT" \
BOXFUSION_TR3D_RUN_TAG=tr3d_prefix_boxfusion_causal_p100_full100_v3 \
  bash "$ROOT/scripts/run_tr3d_cache_inference.sh" "$TR3D_GPU_SPEC"

"$PYTHON_BIN" "$ROOT/tools/validate_tr3d_residual_cache.py" \
  --cache-root "$CACHE_ROOT" \
  --scene-list "$FULL_SCENE_LIST" \
  --prefix-id p100 \
  --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA" \
  --config-sha256 "$EXPECTED_CONFIG_SHA"
audit_prefix_and_optional_cache with-cache
echo "Strict val100 parent cache complete and fixed10-array-identical."
