#!/usr/bin/env bash
set -euo pipefail

# Frozen-B6 P0/P1/P2 protocol.
#
# Fixed ten scenes:
#   bash scripts/run_scannet_p_ablation.sh P0 0,1
#   BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
#     bash scripts/run_scannet_p_ablation.sh P1 0,1
#   BOXFUSION_P1_RESIDUAL_CHECKPOINT="$PWD/models/scannet_p1_residual.pt" \
#   BOXFUSION_P2_OCCUPANCY_CHECKPOINT="$PWD/models/scannet_p2_occupancy_topk.pt" \
#     bash scripts/run_scannet_p_ablation.sh P2 0,1
#
# Full validation is deliberately opt-in:
#   BOXFUSION_P_FULL100=1 ... bash scripts/run_scannet_p_ablation.sh P2 0,1

STAGE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

case "${STAGE^^}" in
    P0) STAGE=P0; PROFILE=p0_frozen_b6 ;;
    P1) STAGE=P1; PROFILE=p1_residual_proposal_observer ;;
    P2) STAGE=P2; PROFILE=p2_occupancy_topk_observer ;;
    *) echo "Stage must be P0, P1, or P2" >&2; exit 2 ;;
esac

FULL100="${BOXFUSION_P_FULL100:-0}"
case "$FULL100" in 0|1) ;; *)
    echo "BOXFUSION_P_FULL100 must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -n "${BOXFUSION_P_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_P_SCENE_LIST"
    SCOPE=custom
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
    SCOPE=full100
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
    SCOPE=ablation10
fi

QUALITY_CHECKPOINT="${BOXFUSION_P_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_P_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
P1_CHECKPOINT="${BOXFUSION_P1_RESIDUAL_CHECKPOINT:-}"
P2_CHECKPOINT="${BOXFUSION_P2_OCCUPANCY_CHECKPOINT:-}"
CONFIG="${BOXFUSION_P_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-${CONDA_PREFIX:-/home/admin1/miniconda3/envs/boxfusion2}}"
PYTHON="$ENV_ROOT/bin/python"
B6_DETECTOR_BLEND="${BOXFUSION_P_B6_DETECTOR_BLEND:-0.40}"
B6_MIN_EXTENT="${BOXFUSION_P_B6_MIN_EXTENT:-0.40}"
PROPOSAL_INTERVAL="${BOXFUSION_P_PROPOSAL_INTERVAL:-5}"
for path in "$PYTHON" "$SCENE_LIST" "$CONFIG" "$QUALITY_CHECKPOINT" "$YOLOE_CHECKPOINT"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P ablation input: $path" >&2
        exit 1
    fi
done
if [[ "$STAGE" != "P0" && ! -f "$P1_CHECKPOINT" ]]; then
    echo "P1/P2 require BOXFUSION_P1_RESIDUAL_CHECKPOINT" >&2
    exit 1
fi
if [[ "$STAGE" == "P2" && ! -f "$P2_CHECKPOINT" ]]; then
    echo "P2 requires BOXFUSION_P2_OCCUPANCY_CHECKPOINT" >&2
    exit 1
fi
if [[ "$STAGE" == "P0" ]]; then
    P1_CHECKPOINT=""
    P2_CHECKPOINT=""
elif [[ "$STAGE" == "P1" ]]; then
    P2_CHECKPOINT=""
fi

RUN_TAG="${BOXFUSION_P_RUN_TAG:-${STAGE,,}_${SCOPE}_b6frozen_v1}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "BOXFUSION_P_RUN_TAG contains unsafe path characters" >&2
    exit 2
fi
PRED_ROOT="${BOXFUSION_P_PRED_ROOT:-$ROOT/results/p_ablation/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_P_LOG_ROOT:-$ROOT/logs/p_ablation/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_P_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/p_ablation/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_P_EVAL_ROOT:-$ROOT/evaluation/p_ablation/$RUN_TAG}"
"$PYTHON" - \
    "$ROOT" "$SCENE_LIST" \
    "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_ROOT" <<'PY'
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
scene_list = Path(sys.argv[2])
scenes = [
    row.strip()
    for row in scene_list.read_text(encoding="utf-8").splitlines()
    if row.strip()
]
if (
    not scenes
    or len(scenes) != len(set(scenes))
    or any(re.fullmatch(r"scene[0-9]{4}_[0-9]{2}", row) is None for row in scenes)
):
    raise SystemExit("P scene list must contain unique canonical ScanNet IDs")
outputs = [Path(value).resolve() for value in sys.argv[3:]]
if len(outputs) != len(set(outputs)):
    raise SystemExit("P output roots must be pairwise distinct")
for output in outputs:
    try:
        output.relative_to(root)
    except ValueError:
        raise SystemExit("P output roots must remain inside boxfusion_p2_dev")
PY

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_REFINER_CHECKPOINT BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT BOXFUSION_YIDU_GATE_CHECKPOINT
unset BOXFUSION_SCANNET_POST_MIN_EXTENT
unset BOXFUSION_PROPOSAL_CACHE_DIRECTORY
unset BOXFUSION_PROPOSAL_CACHE_NAMESPACE
unset BOXFUSION_PROPOSAL_CACHE_MISSING_POLICY
unset BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY
unset BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE
unset BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY
export BOXFUSION_ENV_ROOT="$ENV_ROOT"
export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="$PROFILE"
export BOXFUSION_PROPOSAL_PROVIDER=yoloe
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_QUALITY_MODE=iou_mlp
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
# Freeze the user's strongest established B6 anchor.
export BOXFUSION_QUALITY_DETECTOR_BLEND="$B6_DETECTOR_BLEND"
export BOXFUSION_SCANNET_MIN_EXTENT="$B6_MIN_EXTENT"
export BOXFUSION_PROPOSAL_INTERVAL="$PROPOSAL_INTERVAL"
export BOXFUSION_CANDIDATE_TTL_CLOCK=provider_call
export BOXFUSION_CANDIDATE_TRACK_TTL=3
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS=0
export BOXFUSION_INFERENCE_SEED=0
export BOXFUSION_EVAL_SEED=0
export BOXFUSION_SKIP_EVALUATION=0
export BOXFUSION_P1_COLLECT_VOXELS=0
if [[ "$STAGE" == "P1" || "$STAGE" == "P2" ]]; then
    export BOXFUSION_P1_RESIDUAL_MODE=infer
else
    export BOXFUSION_P1_RESIDUAL_MODE=
fi
export BOXFUSION_P1_RESIDUAL_CHECKPOINT="$P1_CHECKPOINT"
export BOXFUSION_P2_OCCUPANCY_CHECKPOINT="$P2_CHECKPOINT"
export BOXFUSION_P_STAGE="$STAGE"

export BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"
export BOXFUSION_P_MANIFEST="$LOG_ROOT/run_manifest.json"

echo "P ablation: stage=$STAGE, scope=$SCOPE, tag=$RUN_TAG, GPUs=$GPU_SPEC"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
