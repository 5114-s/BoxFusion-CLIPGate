#!/usr/bin/env bash
set -euo pipefail

# Run the frozen epoch-12 class-agnostic TR3D parent on the strict train100
# p100 export.  All GPU/cache mutations happen only after dataset, hash,
# environment, and GPU-idle preflight checks have passed.

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
  "strict train100 parent inference requires exactly two GPUs"
[[ "${TR3D_GPUS[0]}" != "${TR3D_GPUS[1]}" ]] || tr3d_die \
  "the two GPU indices must be distinct"

SOURCE_ROOT="${BOXFUSION_TR3D_SOURCE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev}"
tr3d_select_env "${BOXFUSION_TR3D_ENV:-$SOURCE_ROOT/.conda/boxfusion-tr3d}"
CHECKPOINT="${BOXFUSION_TR3D_CHECKPOINT:-$SOURCE_ROOT/work_dirs/tr3d/tr3d_fg_full_seed0_fp32_v1/epoch_12.pth}"
CONFIG="$ROOT/config/tr3d/tr3d_scannet_foreground_official_val.py"
EXPECTED_CHECKPOINT_SHA="a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
EXPECTED_CONFIG_SHA="709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785"
TRAIN_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt"
VAL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
PREFIX_ROOT="$ROOT/data/tr3d_prefix_train100_boxfusion_causal_p100_v1"
STEM="trajectory_prefix_train100_boxfusion_causal_p100_v1"
MANIFEST="$PREFIX_ROOT/manifests/$STEM.jsonl"
SUMMARY="$PREFIX_ROOT/manifests/$STEM.summary.json"
INFO="$PREFIX_ROOT/annotations/scannet_infos_prefix_train100_boxfusion_causal_p100_v1.pkl"
CACHE_ROOT="$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_train100_v1"
ATTEMPT_TAG="${BOXFUSION_TR3D_ATTEMPT_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
LOG_ROOT="${BOXFUSION_TR3D_LOG_ROOT:-$ROOT/logs/tr3d_parent/train100_v1/$ATTEMPT_TAG}"
PYTHON_BIN="${BOXFUSION_TR3D_CONTROL_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"

for path in \
  "$CHECKPOINT" "$CONFIG" "$TRAIN_SCENE_LIST" "$VAL_SCENE_LIST" \
  "$MANIFEST" "$SUMMARY" "$INFO" "$PYTHON_BIN"; do
  [[ -e "$path" ]] || tr3d_die "missing strict train-parent input: $path"
done
[[ "$(tr3d_sha256 "$CHECKPOINT")" == "$EXPECTED_CHECKPOINT_SHA" ]] \
  || tr3d_die "epoch12 checkpoint SHA256 mismatch"
[[ "$(tr3d_sha256 "$CONFIG")" == "$EXPECTED_CONFIG_SHA" ]] \
  || tr3d_die "TR3D config SHA256 mismatch"

audit_prefix() {
  "$PYTHON_BIN" - \
    "$ROOT" "$PREFIX_ROOT" "$TRAIN_SCENE_LIST" "$VAL_SCENE_LIST" \
    "$MANIFEST" "$SUMMARY" "$INFO" <<'PY'
from pathlib import Path
import json
import sys

root, prefix_root, train_path, val_path, manifest_path, summary_path, info_path = (
    map(Path, sys.argv[1:])
)
sys.path.insert(0, str(root.resolve()))
from tools.tr3d_data import load_info, read_scene_list, scene_id_from_info

prefix_root = prefix_root.resolve()
train = read_scene_list(train_path.resolve())
validation = set(read_scene_list(val_path.resolve()))
if len(train) != 100 or len(set(train)) != 100:
    raise SystemExit("strict parent requires exactly 100 unique train scenes")
overlap = sorted(set(train) & validation)
if overlap:
    raise SystemExit(f"strict parent train/validation overlap: {overlap[:8]}")
rows = [
    json.loads(line)
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 100 or [row.get("scene_id") for row in rows] != train:
    raise SystemExit("strict parent prefix manifest scene order/set mismatch")
expected_points = set()
for scene, row in zip(train, rows):
    expected = prefix_root / "points" / "prefixes" / scene / f"{scene}__p100.bin"
    if (
        row.get("schema") != "boxfusion.tr3d.trajectory_prefix.v1"
        or row.get("status") != "exported"
        or row.get("tag") != "p100"
        or float(row.get("fraction", -1)) != 1.0
        or row.get("clock_policy") != "g0_post_frame_tail_guard_v1"
        or row.get("pose_policy") != "previous_valid_inf_only_v1"
    ):
        raise SystemExit(f"{scene}: strict p100 provenance mismatch")
    if Path(row.get("point_path", "")).resolve() != expected or not expected.is_file():
        raise SystemExit(f"{scene}: strict p100 point path is missing or redirected")
    expected_points.add(expected)
if {path.resolve() for path in (prefix_root / "points").rglob("*.bin")} != expected_points:
    raise SystemExit("strict parent point artifact set is missing or has extras")
_, info_rows = load_info(info_path.resolve())
if len(info_rows) != 100 or [scene_id_from_info(row) for row in info_rows] != train:
    raise SystemExit("strict parent annotation rows disagree with train100")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
for key, value in {
    "scene_count_requested": 100,
    "scene_count_exported": 100,
    "prefix_count": 100,
    "annotation_row_count": 100,
    "manifest_only": False,
}.items():
    if summary.get(key) != value:
        raise SystemExit(f"strict parent summary {key} mismatch")
if summary.get("errors") != []:
    raise SystemExit("strict parent refuses an export with recorded errors")
PY
}

# Prove the exact train-only input set before touching CUDA or cache/log roots.
audit_prefix

if "$PYTHON_BIN" "$ROOT/tools/validate_tr3d_residual_cache.py" \
    --cache-root "$CACHE_ROOT" \
    --scene-list "$TRAIN_SCENE_LIST" \
    --prefix-id p100 \
    --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA" \
    --config-sha256 "$EXPECTED_CONFIG_SHA" >/dev/null 2>&1; then
  echo "Strict train100 epoch12 parent cache already complete and validated."
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
  local output=""
  local attempt
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
  mkdir "$CACHE_ROOT"
fi

echo "Strict train100 p100 epoch12 parent-cache inference"
echo "  GPUs: $TR3D_GPU_SPEC"
echo "  resume: $RESUME (explicit)"
echo "  train scenes: 100; validation overlap=0"
echo "  input: $MANIFEST"
echo "  cache: $CACHE_ROOT"
echo "  checkpoint/config SHA locked"
echo "  score/max proposals/voxel: 0.01/1000/0.01"

BOXFUSION_TR3D_ENV="$TR3D_ENV_REF" \
BOXFUSION_TR3D_CONFIG="$CONFIG" \
BOXFUSION_TR3D_CHECKPOINT="$CHECKPOINT" \
BOXFUSION_TR3D_INPUT_MANIFEST="$MANIFEST" \
BOXFUSION_TR3D_SCENE_LIST="$TRAIN_SCENE_LIST" \
BOXFUSION_TR3D_PREFIX_ID=p100 \
BOXFUSION_TR3D_SCORE_THRESHOLD=0.01 \
BOXFUSION_TR3D_MAX_PROPOSALS=1000 \
BOXFUSION_TR3D_VOXEL_SIZE=0.01 \
BOXFUSION_TR3D_CACHE_ROOT="$CACHE_ROOT" \
BOXFUSION_TR3D_LOG_ROOT="$LOG_ROOT" \
BOXFUSION_TR3D_RUN_TAG=tr3d_prefix_boxfusion_causal_p100_train100_v1 \
  bash "$ROOT/scripts/run_tr3d_cache_inference.sh" "$TR3D_GPU_SPEC"

"$PYTHON_BIN" "$ROOT/tools/validate_tr3d_residual_cache.py" \
  --cache-root "$CACHE_ROOT" \
  --scene-list "$TRAIN_SCENE_LIST" \
  --prefix-id p100 \
  --checkpoint-sha256 "$EXPECTED_CHECKPOINT_SHA" \
  --config-sha256 "$EXPECTED_CONFIG_SHA"
audit_prefix
echo "Strict train100 epoch12 parent cache complete and validated."
