#!/usr/bin/env bash
set -euo pipefail

# B6 + Selective Boxer G0 with a post-B6, fixed-Top-K geometry ablation.
#
# Fixed 10 scenes:
#   bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f0_control 0,1
#   bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f1_observer 0,1
#   bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f2_active 0,1
#
# Full 100 scenes is deliberately explicit:
#   BOXFUSION_B6_BOXER_FINAL_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
#     bash scripts/run_scannet_b6_boxer_uncertainty_final.sh f2_active 0,1

PROFILE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
SCENE_LIST="${BOXFUSION_B6_BOXER_FINAL_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
CONFIG="$ROOT/config/scannet_b6_selective_boxer.yaml"
QUALITY_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"

case "$PROFILE" in
    f0_control)
        FINAL_MODE="disabled"
        ;;
    f1_observer)
        FINAL_MODE="observer"
        ;;
    f2_active)
        FINAL_MODE="active"
        ;;
    *)
        echo "Profile must be f0_control, f1_observer, or f2_active" >&2
        exit 2
        ;;
esac

for required in "$SCENE_LIST" "$CONFIG" "$QUALITY_CHECKPOINT" "$YOLOE_CHECKPOINT"; do
    if [[ ! -f "$required" ]]; then
        echo "Missing required file: $required" >&2
        exit 1
    fi
done

quality_sha="$(sha256sum "$QUALITY_CHECKPOINT" | awk '{print $1}')"
if [[ "$quality_sha" != "d60abf798edbfa3d7902b42651be7d6053727948f740e05795de6feed60a7071" ]]; then
    echo "Frozen B6 quality checkpoint SHA256 mismatch: $quality_sha" >&2
    exit 1
fi
yoloe_sha="$(sha256sum "$YOLOE_CHECKPOINT" | awk '{print $1}')"
if [[ "$yoloe_sha" != "292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d" ]]; then
    echo "Frozen YOLOE checkpoint SHA256 mismatch: $yoloe_sha" >&2
    exit 1
fi

duplicate_scene="$(
    awk 'NF && $1 !~ /^#/ {count[$1] += 1} END {
        for (scene in count) if (count[scene] > 1) {print scene; exit}
    }' "$SCENE_LIST"
)"
if [[ -n "$duplicate_scene" ]]; then
    echo "Duplicate scene in scene list: $duplicate_scene" >&2
    exit 1
fi

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
artifact_root="$ROOT/results/b6_boxer_uncertainty_final/$PROFILE/$list_scope"
log_root="$ROOT/logs/b6_boxer_uncertainty_final/$PROFILE/$list_scope"
diagnostics_root="$ROOT/diagnostics/b6_boxer_uncertainty_final/$PROFILE/$list_scope/online"
boxer_diagnostics_root="$ROOT/diagnostics/b6_boxer_uncertainty_final/$PROFILE/$list_scope/boxer"
final_diagnostics_root="$ROOT/diagnostics/b6_boxer_uncertainty_final/$PROFILE/$list_scope/final"
eval_root="$ROOT/evaluation/b6_boxer_uncertainty_final/$PROFILE/$list_scope"

export BOXFUSION_ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
export BOXFUSION_LIVE_ROOT="$LIVE_ROOT"
export BOXFUSION_SCANNET_FRAMES_ROOT="$LIVE_ROOT/upstream_clean/scannet_readme_frames"
export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="quality_only"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"
export BOXFUSION_SCANNET_MIN_EXTENT="0.40"
export BOXFUSION_PROPOSAL_INTERVAL="5"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_DISABLE_ONLINE_REFINEMENT="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
unset BOXFUSION_REFINER_CHECKPOINT

# Freeze the validated G0 online trajectory.  Online uncertainty must remain
# disabled; only the post-B6 module varies across F0/F1/F2.
export BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="0.10"
export BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="0.50"
export BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="2.00"
export BOXFUSION_BOXER_UNCERTAINTY_MODE="disabled"
unset BOXFUSION_BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT
export BOXFUSION_BOXER_FINAL_UNCERTAINTY_MODE="$FINAL_MODE"

export BOXFUSION_ONLINE_PRED_ROOT="$artifact_root"
export BOXFUSION_ONLINE_LOG_ROOT="$log_root"
export BOXFUSION_DIAGNOSTICS_ROOT="$diagnostics_root"
export BOXFUSION_BOXER_DIAGNOSTICS_ROOT="$boxer_diagnostics_root"
if [[ "$FINAL_MODE" == "disabled" ]]; then
    unset BOXFUSION_BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT
else
    export BOXFUSION_BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT="$final_diagnostics_root"
fi
export BOXFUSION_EVAL_ROOT="$eval_root"

echo "B6 + Selective Boxer G0 + post-B6 fixed-Top-K uncertainty"
echo "  profile/mode: $PROFILE/$FINAL_MODE"
echo "  scenes: $SCENE_LIST"
echo "  scene-list SHA256: $list_sha"
echo "  GPUs: $GPU_SPEC"
echo "  score/min-extent: 0.40/0.40"
echo "  online path: frozen G0; online uncertainty disabled"
echo "  final path: fixed G0 Top-K=3; reweight only; scores/order/count locked"
echo "  predictions: $artifact_root"
echo "  final diagnostics: ${BOXFUSION_BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT:-disabled}"

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
