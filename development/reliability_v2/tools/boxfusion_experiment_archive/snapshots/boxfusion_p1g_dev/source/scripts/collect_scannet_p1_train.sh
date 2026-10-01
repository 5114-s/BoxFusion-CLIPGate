#!/usr/bin/env bash
set -euo pipefail

# Collect raw P1 sparse voxel inputs on ScanNet train scenes only.
# This command intentionally does not run ScanNet evaluation.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SCENE_LIST="${BOXFUSION_P1_TRAIN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FORBIDDEN="${BOXFUSION_P1_FORBIDDEN_SCENES:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
FRAMES_ROOT="${BOXFUSION_P1_TRAIN_FRAMES_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev/data/scannet_train}"
QUALITY_CHECKPOINT="${BOXFUSION_P_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
RUN_TAG="${BOXFUSION_P1_TRAIN_RUN_TAG:-p1_residual_inputs_train100_v1}"

for path in "$SCENE_LIST" "$FORBIDDEN" "$QUALITY_CHECKPOINT"; do
    [[ -f "$path" ]] || { echo "Missing P1 training input: $path" >&2; exit 1; }
done
[[ -d "$FRAMES_ROOT" ]] || {
    echo "Missing ScanNet train frames: $FRAMES_ROOT" >&2; exit 1;
}
python_bin="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}/bin/python"
"$python_bin" - "$SCENE_LIST" "$FORBIDDEN" <<'PY'
import sys
from pathlib import Path
train = {x.strip() for x in Path(sys.argv[1]).read_text().splitlines() if x.strip()}
forbidden = {x.strip() for x in Path(sys.argv[2]).read_text().splitlines() if x.strip()}
overlap = sorted(train & forbidden)
if overlap:
    raise SystemExit("train/validation leakage: " + ", ".join(overlap[:8]))
PY

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_REFINER_CHECKPOINT BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT BOXFUSION_YIDU_GATE_CHECKPOINT
unset BOXFUSION_P1_RESIDUAL_CHECKPOINT
unset BOXFUSION_P_STAGE BOXFUSION_P_MANIFEST
unset BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY
unset BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE
unset BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY
export BOXFUSION_ENV_ROOT="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}"
export BOXFUSION_ONLINE_CONFIG="$ROOT/config/scannet_online_refinement.yaml"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_ONLINE_ABLATION_PROFILE=p1_residual_proposal_observer
export BOXFUSION_PROPOSAL_PROVIDER=yoloe
export BOXFUSION_YOLOE_CHECKPOINT="${BOXFUSION_P_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
export BOXFUSION_P1_RESIDUAL_MODE=collect
export BOXFUSION_P1_COLLECT_VOXELS=1
export BOXFUSION_SKIP_EVALUATION=1
export BOXFUSION_QUALITY_MODE=iou_mlp
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_P_B6_DETECTOR_BLEND:-0.40}"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_P_B6_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_P_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK=provider_call
export BOXFUSION_CANDIDATE_TRACK_TTL=3
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS=0
export BOXFUSION_INFERENCE_SEED=0
export BOXFUSION_ONLINE_PRED_ROOT="$ROOT/results/p1_training/$RUN_TAG"
export BOXFUSION_ONLINE_LOG_ROOT="$ROOT/logs/p1_training/$RUN_TAG"
export BOXFUSION_DIAGNOSTICS_ROOT="$ROOT/diagnostics/p1_training/$RUN_TAG"
export BOXFUSION_EVAL_ROOT="$ROOT/evaluation/p1_training/$RUN_TAG"

echo "P1 train-only collection: scenes=$SCENE_LIST, tag=$RUN_TAG, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
