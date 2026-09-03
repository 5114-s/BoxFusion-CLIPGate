#!/usr/bin/env bash
set -euo pipefail

# B6 + Selective Boxer G0 + terminal-p100 TR3D R3 cache-replay active hook.
#
# Usage (fixed 10 scenes by default):
#   bash scripts/run_scannet_b6_g0_tr3d_terminal_active.sh 0,1
#
# Full 100 scenes must be requested explicitly:
#   BOXFUSION_B6_BOXER_SCENE_LIST="$PWD/evaluation/data_util/meta_data/scannetv2_val.txt" \
#     bash scripts/run_scannet_b6_g0_tr3d_terminal_active.sh 0,1

PROFILE="${BOXFUSION_TR3D_TERMINAL_RUN_TAG:-g0_tr3d_terminal_paired_v2}"
GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ARTIFACT_BASE="${BOXFUSION_TR3D_TERMINAL_ARTIFACT_BASE:-$ROOT}"
SCENE_LIST="${BOXFUSION_B6_BOXER_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
QUALITY_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"

if [[ ! "$PROFILE" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$ ]]; then
    echo "Invalid BOXFUSION_TR3D_TERMINAL_RUN_TAG: $PROFILE" >&2
    exit 1
fi

CONFIG="${BOXFUSION_ONLINE_CONFIG:-$ROOT/config/scannet_b6_selective_boxer.yaml}"
GATE_CENTER="0.10"
GATE_MIN_VOLUME="0.50"
GATE_MAX_VOLUME="2.00"

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
artifact_root="$ARTIFACT_BASE/results/b6_g0_tr3d_terminal/$PROFILE/$list_scope"
same_run_baseline_root="$ARTIFACT_BASE/results/b6_g0_tr3d_terminal_same_run_baseline/$PROFILE/$list_scope"
log_root="$ARTIFACT_BASE/logs/b6_g0_tr3d_terminal/$PROFILE/$list_scope"
diagnostics_root="$ARTIFACT_BASE/diagnostics/b6_g0_tr3d_terminal/$PROFILE/$list_scope/online"
boxer_diagnostics_root="$ARTIFACT_BASE/diagnostics/b6_g0_tr3d_terminal/$PROFILE/$list_scope/boxer"
r3_diagnostics_root="$ARTIFACT_BASE/diagnostics/b6_g0_tr3d_terminal/$PROFILE/$list_scope/tr3d_terminal"
eval_root="$ARTIFACT_BASE/evaluation/b6_g0_tr3d_terminal/$PROFILE/$list_scope"
same_run_eval_root="$ARTIFACT_BASE/evaluation/b6_g0_tr3d_terminal_same_run_baseline/$PROFILE/$list_scope"

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
export BOXFUSION_R3_SAME_RUN_BASELINE_ROOT="$same_run_baseline_root"
export BOXFUSION_ONLINE_LOG_ROOT="$log_root"
export BOXFUSION_DIAGNOSTICS_ROOT="$diagnostics_root"
export BOXFUSION_BOXER_DIAGNOSTICS_ROOT="$boxer_diagnostics_root"
export BOXFUSION_R3_PREFIX_MANIFEST="${BOXFUSION_R3_PREFIX_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/data/tr3d_prefix_val100_boxfusion_causal_p100_v3/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl}"
export BOXFUSION_R3_PARENT_CACHE_ROOT="${BOXFUSION_R3_PARENT_CACHE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3}"
export BOXFUSION_R3_FROZEN_G0_ROOT="${BOXFUSION_R3_FROZEN_G0_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/results/b6_selective_boxer/s1_selective/scannetv2_val-4b18fc586f7a}"
export BOXFUSION_R3_SHADOW_GOLD_ROOT="${BOXFUSION_R3_SHADOW_GOLD_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/results/tr3d_r3_shadow_active/r3_shadow_active_full100_v1}"
export BOXFUSION_R3_FROZEN_MANIFEST="${BOXFUSION_R3_FROZEN_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/manifests/frozen_g0_selective_boxer_full100.json}"
export BOXFUSION_R3_SHADOW_MANIFEST="${BOXFUSION_R3_SHADOW_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/reports/tr3d_r3_shadow_active/r3_shadow_active_full100_v1/materialize_manifest.json}"
export BOXFUSION_R3_DIAGNOSTICS_ROOT="$r3_diagnostics_root"
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
export BOXFUSION_R3_SAME_RUN_EVAL_ROOT="$same_run_eval_root"

echo "B6 + Selective Boxer G0 + terminal-p100 TR3D active-hook run"
echo "  profile: $PROFILE"
echo "  scenes: $SCENE_LIST"
echo "  scene-list SHA256: $list_sha"
echo "  GPUs: $GPU_SPEC"
echo "  artifact base: $ARTIFACT_BASE"
echo "  score/min-extent: 0.40/0.40"
echo "  B6 quality: iou_mlp, detector blend=0.40"
if [[ -n "$GATE_CENTER" ]]; then
    echo "  gate: center<=$GATE_CENTER m, volume=[$GATE_MIN_VOLUME,$GATE_MAX_VOLUME]"
else
    echo "  gate: config-default (inactive for CuTR control)"
fi
echo "  predictions: $artifact_root"
echo "  same-run post-B6/pre-R3 baseline: $same_run_baseline_root"
echo "  Boxer diagnostics: $boxer_diagnostics_root"
echo "  R3 provider: immutable causal p100 parent-cache replay"
echo "  R3 prefix manifest: $BOXFUSION_R3_PREFIX_MANIFEST"
echo "  R3 parent cache: $BOXFUSION_R3_PARENT_CACHE_ROOT"
echo "  R3 diagnostics: $BOXFUSION_R3_DIAGNOSTICS_ROOT"
echo "  same-run baseline evaluation: $same_run_eval_root"
echo "  active evaluation: $eval_root"
echo "  warning: cached TR3D runtime is not authoritative live latency"

exec bash "$ROOT/scripts/run_scannet_tr3d_terminal_active.sh" "$GPU_SPEC"
