#!/usr/bin/env bash
set -euo pipefail

# Paired B6 + Selective Boxer ablation.
#
# Usage (fixed 10 scenes by default):
#   bash scripts/run_scannet_b6_selective_boxer.sh s0_control 0,1
#   bash scripts/run_scannet_b6_selective_boxer.sh s0_observer 0,1
#   bash scripts/run_scannet_b6_selective_boxer.sh s1_selective 0,1
#   bash scripts/run_scannet_b6_selective_boxer.sh g1 0,1
#   bash scripts/run_scannet_b6_selective_boxer.sh g2 0,1
#   bash scripts/run_scannet_b6_selective_boxer.sh g3 0,1
#
# Full 100 scenes must be requested explicitly:
#   BOXFUSION_B6_BOXER_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
#     bash scripts/run_scannet_b6_selective_boxer.sh s1_selective 0,1

PROFILE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
SCENE_LIST="${BOXFUSION_B6_BOXER_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
QUALITY_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"

case "$PROFILE" in
    s0_control)
        CONFIG="$ROOT/config/scannet_b6_cutr_replay.yaml"
        GATE_CENTER=""
        GATE_MIN_VOLUME=""
        GATE_MAX_VOLUME=""
        ;;
    s0_observer)
        CONFIG="$ROOT/config/scannet_b6_boxer_observer.yaml"
        GATE_CENTER=""
        GATE_MIN_VOLUME=""
        GATE_MAX_VOLUME=""
        ;;
    s1_selective|g0)
        CONFIG="$ROOT/config/scannet_b6_selective_boxer.yaml"
        GATE_CENTER="0.10"
        GATE_MIN_VOLUME="0.50"
        GATE_MAX_VOLUME="2.00"
        ;;
    g1)
        CONFIG="$ROOT/config/scannet_b6_selective_boxer.yaml"
        GATE_CENTER="0.075"
        GATE_MIN_VOLUME="0.50"
        GATE_MAX_VOLUME="2.00"
        ;;
    g2)
        CONFIG="$ROOT/config/scannet_b6_selective_boxer.yaml"
        GATE_CENTER="0.075"
        GATE_MIN_VOLUME="0.67"
        GATE_MAX_VOLUME="1.50"
        ;;
    g3)
        CONFIG="$ROOT/config/scannet_b6_selective_boxer.yaml"
        GATE_CENTER="0.05"
        GATE_MIN_VOLUME="0.67"
        GATE_MAX_VOLUME="1.50"
        ;;
    *)
        echo "Profile must be s0_control, s0_observer, s1_selective, g0, g1, g2, or g3" >&2
        exit 2
        ;;
esac

for required in "$SCENE_LIST" "$QUALITY_CHECKPOINT" "$YOLOE_CHECKPOINT" "$CONFIG"; do
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
artifact_root="$ROOT/results/b6_selective_boxer/$PROFILE/$list_scope"
log_root="$ROOT/logs/b6_selective_boxer/$PROFILE/$list_scope"
diagnostics_root="$ROOT/diagnostics/b6_selective_boxer/$PROFILE/$list_scope/online"
boxer_diagnostics_root="$ROOT/diagnostics/b6_selective_boxer/$PROFILE/$list_scope/boxer"
eval_root="$ROOT/evaluation/b6_selective_boxer/$PROFILE/$list_scope"

export BOXFUSION_ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
export BOXFUSION_LIVE_ROOT="$LIVE_ROOT"
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
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_ONLINE_PRED_ROOT="$artifact_root"
export BOXFUSION_ONLINE_LOG_ROOT="$log_root"
export BOXFUSION_DIAGNOSTICS_ROOT="$diagnostics_root"
if [[ "$PROFILE" == "s0_control" ]]; then
    unset BOXFUSION_BOXER_DIAGNOSTICS_ROOT
else
    export BOXFUSION_BOXER_DIAGNOSTICS_ROOT="$boxer_diagnostics_root"
fi
if [[ -n "$GATE_CENTER" ]]; then
    export BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M="$GATE_CENTER"
    export BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO="$GATE_MIN_VOLUME"
    export BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO="$GATE_MAX_VOLUME"
else
    unset BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M
    unset BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO
    unset BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO
fi
export BOXFUSION_EVAL_ROOT="$eval_root"

echo "B6 + Selective Boxer paired run"
echo "  profile: $PROFILE"
echo "  scenes: $SCENE_LIST"
echo "  scene-list SHA256: $list_sha"
echo "  GPUs: $GPU_SPEC"
echo "  score/min-extent: 0.40/0.40"
echo "  B6 quality: iou_mlp, detector blend=0.40"
if [[ -n "$GATE_CENTER" ]]; then
    echo "  gate: center<=$GATE_CENTER m, volume=[$GATE_MIN_VOLUME,$GATE_MAX_VOLUME]"
else
    echo "  gate: config-default (inactive for CuTR control)"
fi
echo "  predictions: $artifact_root"
if [[ "$PROFILE" == "s0_control" ]]; then
    echo "  Boxer diagnostics: disabled (CuTR control)"
else
    echo "  Boxer diagnostics: $boxer_diagnostics_root"
fi

exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
