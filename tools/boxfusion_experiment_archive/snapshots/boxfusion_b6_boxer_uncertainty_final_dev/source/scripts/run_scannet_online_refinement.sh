#!/usr/bin/env bash
set -euo pipefail

# Isolated Stage-3+ runner. It never writes to the live BoxFusion repository.
#
# Usage:
#   BOXFUSION_YOLOE_CHECKPOINT=/path/yoloe-11s-seg-pf.pt \
#     bash scripts/run_scannet_online_refinement.sh 0
#
#   # Two independent scene shards:
#   BOXFUSION_YOLOE_CHECKPOINT=/path/yoloe-11s-seg-pf.pt \
#     bash scripts/run_scannet_online_refinement.sh 0,1
#
# Optional learned heads:
#   BOXFUSION_REFINER_CHECKPOINT=/path/box_refiner.pt
#   BOXFUSION_QUALITY_CHECKPOINT=/path/quality_linear.npz
#   BOXFUSION_QUALITY_MODE=linear  # or iou_mlp

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
CONFIG="${BOXFUSION_ONLINE_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
META="${BOXFUSION_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
PRED_ROOT="${BOXFUSION_ONLINE_PRED_ROOT:-$ROOT/results/scannet_online_refinement}"
LOG_ROOT="${BOXFUSION_ONLINE_LOG_ROOT:-$ROOT/logs/scannet_online_refinement}"
DIAGNOSTICS_ROOT="${BOXFUSION_DIAGNOSTICS_ROOT:-$ROOT/results/scannet_online_refinement_diagnostics}"
BOXER_DIAGNOSTICS_OVERRIDE="${BOXFUSION_BOXER_DIAGNOSTICS_ROOT:-}"
BOXER_GATE_CENTER="${BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M:-}"
BOXER_GATE_MIN_VOLUME="${BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO:-}"
BOXER_GATE_MAX_VOLUME="${BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO:-}"
BOXER_UNCERTAINTY_MODE="${BOXFUSION_BOXER_UNCERTAINTY_MODE:-}"
BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT="${BOXFUSION_BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT:-}"
BOXER_FINAL_UNCERTAINTY_MODE="${BOXFUSION_BOXER_FINAL_UNCERTAINTY_MODE:-}"
BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT="${BOXFUSION_BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT:-}"
GT_ROOT="$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$LIVE_ROOT/upstream_clean/scannet_readme_frames}"
EVAL_ROOT="${BOXFUSION_EVAL_ROOT:-$ROOT/evaluation/scannet_online_refinement}"
EVAL_SEED="${BOXFUSION_EVAL_SEED:-0}"
INFERENCE_SEED="${BOXFUSION_INFERENCE_SEED:-0}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
REFINER_CHECKPOINT="${BOXFUSION_REFINER_CHECKPOINT:-}"
QUALITY_CHECKPOINT="${BOXFUSION_QUALITY_CHECKPOINT:-}"
QUALITY_MODE="${BOXFUSION_QUALITY_MODE:-linear}"
QUALITY_DETECTOR_BLEND="${BOXFUSION_QUALITY_DETECTOR_BLEND:-}"
SCANNET_MIN_EXTENT="${BOXFUSION_SCANNET_MIN_EXTENT:-}"
PROPOSAL_INTERVAL="${BOXFUSION_PROPOSAL_INTERVAL:-5}"
CANDIDATE_TTL_CLOCK="${BOXFUSION_CANDIDATE_TTL_CLOCK:-}"
CANDIDATE_TRACK_TTL="${BOXFUSION_CANDIDATE_TRACK_TTL:-}"
ARCHIVE_CONFIRMED="${BOXFUSION_ARCHIVE_CONFIRMED_TRACKS:-}"
DISABLE_ONLINE="${BOXFUSION_DISABLE_ONLINE_REFINEMENT:-0}"
ABLATION_PROFILE="${BOXFUSION_ONLINE_ABLATION_PROFILE:-}"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

mkdir -p "$PRED_ROOT" "$LOG_ROOT/scenes" "$LOG_ROOT/mplconfig" "$DIAGNOSTICS_ROOT"
if [[ -n "$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT" ]]; then
    mkdir -p "$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT"
fi
if [[ -n "$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT" ]]; then
    mkdir -p "$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT"
fi
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
    echo "Another online-refinement driver holds $LOG_ROOT/run.lock" >&2
    exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
if [[ "${#GPUS[@]}" -lt 1 ]]; then
    echo "No GPU was specified" >&2
    exit 1
fi
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "Invalid GPU index: $gpu" >&2
        exit 1
    fi
done
if [[ ! "$PROPOSAL_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOXFUSION_PROPOSAL_INTERVAL must be a positive integer" >&2
    exit 1
fi
if [[ -n "$CANDIDATE_TTL_CLOCK" \
      && "$CANDIDATE_TTL_CLOCK" != "keyframe" \
      && "$CANDIDATE_TTL_CLOCK" != "provider_call" ]]; then
    echo "BOXFUSION_CANDIDATE_TTL_CLOCK must be keyframe or provider_call" >&2
    exit 1
fi
if [[ -n "$CANDIDATE_TRACK_TTL" \
      && ! "$CANDIDATE_TRACK_TTL" =~ ^[0-9]+$ ]]; then
    echo "BOXFUSION_CANDIDATE_TRACK_TTL must be a non-negative integer" >&2
    exit 1
fi
if [[ -n "$ARCHIVE_CONFIRMED" \
      && "$ARCHIVE_CONFIRMED" != "0" \
      && "$ARCHIVE_CONFIRMED" != "1" ]]; then
    echo "BOXFUSION_ARCHIVE_CONFIRMED_TRACKS must be 0 or 1" >&2
    exit 1
fi
if [[ "$QUALITY_MODE" != "linear" \
      && "$QUALITY_MODE" != "mlp" \
      && "$QUALITY_MODE" != "iou_mlp" ]]; then
    echo "BOXFUSION_QUALITY_MODE must be linear, mlp, or iou_mlp" >&2
    exit 1
fi
if [[ -n "$SCANNET_MIN_EXTENT" \
      && ! "$SCANNET_MIN_EXTENT" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
    echo "BOXFUSION_SCANNET_MIN_EXTENT must be non-negative" >&2
    exit 1
fi
if [[ -n "$QUALITY_DETECTOR_BLEND" ]]; then
    if [[ ! "$QUALITY_DETECTOR_BLEND" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        echo "BOXFUSION_QUALITY_DETECTOR_BLEND must be numeric" >&2
        exit 1
    fi
    if ! "$PYTHON" -c \
        "value=float('$QUALITY_DETECTOR_BLEND'); assert 0.0 <= value <= 1.0" \
        >/dev/null 2>&1; then
        echo "BOXFUSION_QUALITY_DETECTOR_BLEND must lie in [0, 1]" >&2
        exit 1
    fi
fi
boxer_gate_value_count=0
for boxer_gate_value in \
    "$BOXER_GATE_CENTER" "$BOXER_GATE_MIN_VOLUME" "$BOXER_GATE_MAX_VOLUME"; do
    if [[ -n "$boxer_gate_value" ]]; then
        boxer_gate_value_count=$((boxer_gate_value_count + 1))
    fi
done
if [[ "$boxer_gate_value_count" != "0" \
      && "$boxer_gate_value_count" != "3" ]]; then
    echo "All three BOXFUSION_BOXER_GATE_* overrides must be set together" >&2
    exit 1
fi
if [[ "$boxer_gate_value_count" == "3" ]]; then
    if ! "$PYTHON" -c \
        "import math; c=float('$BOXER_GATE_CENTER'); lo=float('$BOXER_GATE_MIN_VOLUME'); hi=float('$BOXER_GATE_MAX_VOLUME'); assert math.isfinite(c) and c >= 0.0; assert math.isfinite(lo) and lo > 0.0; assert math.isfinite(hi) and hi >= lo" \
        >/dev/null 2>&1; then
        echo "Invalid selective-Boxer gate: require finite center>=0 and 0<min<=max" >&2
        exit 1
    fi
fi
if [[ -n "$BOXER_UNCERTAINTY_MODE" ]]; then
    case "$BOXER_UNCERTAINTY_MODE" in
        disabled|observer|active) ;;
        *)
            echo "BOXFUSION_BOXER_UNCERTAINTY_MODE must be disabled, observer, or active" >&2
            exit 1
            ;;
    esac
    if [[ "$BOXER_UNCERTAINTY_MODE" != "disabled" \
          && -z "$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT" ]]; then
        echo "Observer/active Boxer uncertainty mode requires diagnostics root" >&2
        exit 1
    fi
fi
if [[ -n "$BOXER_FINAL_UNCERTAINTY_MODE" ]]; then
    case "$BOXER_FINAL_UNCERTAINTY_MODE" in
        disabled|observer|active) ;;
        *)
            echo "BOXFUSION_BOXER_FINAL_UNCERTAINTY_MODE must be disabled, observer, or active" >&2
            exit 1
            ;;
    esac
    if [[ "$BOXER_FINAL_UNCERTAINTY_MODE" != "disabled" \
          && -z "$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT" ]]; then
        echo "Observer/active final Boxer uncertainty mode requires diagnostics root" >&2
        exit 1
    fi
fi
if [[ -n "$BOXER_UNCERTAINTY_MODE" \
      && "$BOXER_UNCERTAINTY_MODE" != "disabled" \
      && -n "$BOXER_FINAL_UNCERTAINTY_MODE" \
      && "$BOXER_FINAL_UNCERTAINTY_MODE" != "disabled" ]]; then
    echo "Online and final-only Boxer uncertainty modes are mutually exclusive" >&2
    exit 1
fi
if [[ "$DISABLE_ONLINE" != "0" && "$DISABLE_ONLINE" != "1" ]]; then
    echo "BOXFUSION_DISABLE_ONLINE_REFINEMENT must be 0 or 1" >&2
    exit 1
fi
if [[ -n "$ABLATION_PROFILE" ]]; then
    case "$ABLATION_PROFILE" in
        observer|quality_observer|refit_only|supplemental_only|supplemental_conservative|quality_only|full) ;;
        *)
            echo "Invalid BOXFUSION_ONLINE_ABLATION_PROFILE: $ABLATION_PROFILE" >&2
            exit 1
            ;;
    esac
    if [[ "$DISABLE_ONLINE" == "1" ]]; then
        echo "Online ablation profile conflicts with disabled refinement" >&2
        exit 1
    fi
fi
if [[ ! "$EVAL_SEED" =~ ^[0-9]+$ ]]; then
    echo "BOXFUSION_EVAL_SEED must be a non-negative integer" >&2
    exit 1
fi
if [[ ! "$INFERENCE_SEED" =~ ^[0-9]+$ ]]; then
    echo "BOXFUSION_INFERENCE_SEED must be a non-negative integer" >&2
    exit 1
fi

required_files=(
    "$PYTHON"
    "$CONFIG"
    "$META"
    "$LIVE_ROOT/models/cutr_rgbd.pth"
    "$LIVE_ROOT/models/open_clip_pytorch_model.bin"
    "$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
    "$LIVE_ROOT/data/class_features.pt"
    "$LIVE_ROOT/data/pst_1024_0.tiff"
)
if [[ "$DISABLE_ONLINE" == "0" ]]; then
    required_files+=("$YOLOE_CHECKPOINT")
fi
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        if [[ "$path" == "$YOLOE_CHECKPOINT" ]]; then
            echo "Set BOXFUSION_YOLOE_CHECKPOINT to the local YOLOE segmentation checkpoint." >&2
        fi
        exit 1
    fi
done
if [[ -n "$REFINER_CHECKPOINT" && ! -f "$REFINER_CHECKPOINT" ]]; then
    echo "Missing BoxRefiner checkpoint: $REFINER_CHECKPOINT" >&2
    exit 1
fi
if [[ -n "$QUALITY_CHECKPOINT" && ! -f "$QUALITY_CHECKPOINT" ]]; then
    echo "Missing quality checkpoint: $QUALITY_CHECKPOINT" >&2
    exit 1
fi
if [[ ! -d "$FRAMES_ROOT" ]]; then
    echo "Missing ScanNet frames root: $FRAMES_ROOT" >&2
    exit 1
fi
if [[ ! -d "$GT_ROOT" ]]; then
    echo "Missing ScanNet evaluation ground truth: $GT_ROOT" >&2
    exit 1
fi

CACHE_MODE="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("mode", "disabled"))
PY
)"
CACHE_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("root", ""))
PY
)"
CACHE_NAMESPACE="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("namespace", ""))
PY
)"
CACHE_BASELINE_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("proposal_cache", {}).get("baseline_prediction_root", ""))
PY
)"
BOXER_DIAGNOSTICS_ROOT="$("$PYTHON" - "$CONFIG" <<'PY'
import sys
import yaml
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
print(cfg.get("lifting", {}).get("boxer", {}).get("diagnostics_dir", ""))
PY
)"
if [[ -n "$BOXER_DIAGNOSTICS_OVERRIDE" ]]; then
    BOXER_DIAGNOSTICS_ROOT="$BOXER_DIAGNOSTICS_OVERRIDE"
fi
if [[ "$CACHE_MODE" != "disabled" && "$CACHE_MODE" != "replay" ]]; then
    echo "This runner supports disabled or replay proposal cache, got: $CACHE_MODE" >&2
    exit 1
fi
if [[ "$CACHE_MODE" == "replay" ]]; then
    for cache_path in "$CACHE_ROOT" "$CACHE_BASELINE_ROOT"; do
        if [[ -z "$cache_path" || ! -d "$cache_path" ]]; then
            echo "Missing replay cache path: $cache_path" >&2
            exit 1
        fi
    done
    if [[ -z "$CACHE_NAMESPACE" ]]; then
        echo "Replay cache namespace is required" >&2
        exit 1
    fi
fi

LIST_SHA256="$(sha256sum "$META" | awk '{print $1}')"
CODE_FINGERPRINT="$(
    {
        sha256sum \
            "$CONFIG" \
            "$ROOT/demo.py" \
            "$ROOT/boxfusion/boxer_lifter.py" \
            "$ROOT/boxfusion/boxer_uncertainty.py" \
            "$ROOT/boxfusion/reliable_views.py" \
            "$ROOT/boxfusion/instances.py" \
            "$ROOT/boxfusion/boxes.py" \
            "$ROOT/boxfusion/proposal_cache.py" \
            "$ROOT/boxfusion/online_refinement.py" \
            "$ROOT/boxfusion/online_ablation.py" \
            "$ROOT/boxfusion/quality_score.py" \
            "$ROOT/boxfusion/box_fusion.py" \
            "$ROOT/boxfusion/box_manager.py" \
            "$ROOT/scripts/run_scannet_online_refinement.sh" \
            "$LIVE_ROOT/models/cutr_rgbd.pth" \
            "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
            "$YOLOE_CHECKPOINT"
        if [[ -n "$QUALITY_CHECKPOINT" ]]; then
            sha256sum "$QUALITY_CHECKPOINT"
        fi
        printf '%s\n' \
            "scene_list_sha256=$LIST_SHA256" \
            "python=$(readlink -f "$PYTHON")" \
            "python_version=$("$PYTHON" --version 2>&1)" \
            "quality_mode=$QUALITY_MODE" \
            "quality_blend=$QUALITY_DETECTOR_BLEND" \
            "min_extent=$SCANNET_MIN_EXTENT" \
            "boxer_gate_center=$BOXER_GATE_CENTER" \
            "boxer_gate_min_volume=$BOXER_GATE_MIN_VOLUME" \
            "boxer_gate_max_volume=$BOXER_GATE_MAX_VOLUME" \
            "boxer_uncertainty_mode=$BOXER_UNCERTAINTY_MODE" \
            "boxer_uncertainty_diagnostics=$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT" \
            "boxer_final_uncertainty_mode=$BOXER_FINAL_UNCERTAINTY_MODE" \
            "boxer_final_uncertainty_diagnostics=$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT" \
            "ablation_profile=$ABLATION_PROFILE"
    } | sha256sum | awk '{print $1}'
)"

if [[ "$DISABLE_ONLINE" == "0" ]]; then
    PREFLIGHT_IMPORTS="import torch, torchvision, open_clip, ultralytics"
else
    PREFLIGHT_IMPORTS="import torch, torchvision, open_clip"
fi
CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
    "$PREFLIGHT_IMPORTS; assert torch.cuda.is_available(); print('Online environment OK:', torch.__version__, torch.version.cuda)"

total="$(awk 'END {print NR}' "$META")"
worker_count="${#GPUS[@]}"
echo "[$(date '+%F %T')] Starting isolated online-refinement inference"
echo "[$(date '+%F %T')] Code root: $ROOT"
echo "[$(date '+%F %T')] Live assets (read-only): $LIVE_ROOT"
echo "[$(date '+%F %T')] GPUs: $GPU_SPEC; workers: $worker_count; scenes: $total"
echo "[$(date '+%F %T')] Proposal interval: every $PROPOSAL_INTERVAL keyframes"
echo "[$(date '+%F %T')] Candidate TTL clock: ${CANDIDATE_TTL_CLOCK:-config-default}"
echo "[$(date '+%F %T')] Candidate track TTL: ${CANDIDATE_TRACK_TTL:-config-default}"
echo "[$(date '+%F %T')] Archive confirmed tracks: ${ARCHIVE_CONFIRMED:-config-default}"
echo "[$(date '+%F %T')] Inference/evaluation seeds: $INFERENCE_SEED/$EVAL_SEED"
echo "[$(date '+%F %T')] Online refinement disabled: $DISABLE_ONLINE"
echo "[$(date '+%F %T')] Online ablation profile: ${ABLATION_PROFILE:-config-default}"
echo "[$(date '+%F %T')] ScanNet minimum extent: ${SCANNET_MIN_EXTENT:-config-default}"
echo "[$(date '+%F %T')] Detector/quality blend: ${QUALITY_DETECTOR_BLEND:-config-default}"
if [[ "$boxer_gate_value_count" == "3" ]]; then
    echo "[$(date '+%F %T')] Selective Boxer gate: center<=$BOXER_GATE_CENTER m; volume=[$BOXER_GATE_MIN_VOLUME,$BOXER_GATE_MAX_VOLUME]"
else
    echo "[$(date '+%F %T')] Selective Boxer gate: config-default"
fi
echo "[$(date '+%F %T')] Boxer uncertainty fusion: ${BOXER_UNCERTAINTY_MODE:-config-default}"
echo "[$(date '+%F %T')] Boxer uncertainty diagnostics: ${BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT:-disabled}"
echo "[$(date '+%F %T')] Prediction root: $PRED_ROOT"
echo "[$(date '+%F %T')] Diagnostics root: $DIAGNOSTICS_ROOT"
echo "[$(date '+%F %T')] ScanNet frames root: $FRAMES_ROOT"
echo "[$(date '+%F %T')] Proposal cache: $CACHE_MODE:${CACHE_NAMESPACE:-disabled}"
echo "[$(date '+%F %T')] Code fingerprint: $CODE_FINGERPRINT"
echo "[$(date '+%F %T')] Scene-list SHA256: $LIST_SHA256"

run_worker() {
    local gpu="$1"
    local shard="$2"
    local shards="$3"
    local index=0
    local completed=0

    while IFS= read -r scene || [[ -n "$scene" ]]; do
        if (( index % shards != shard )); then
            index=$((index + 1))
            continue
        fi
        local prediction="$PRED_ROOT/${scene}_boxes.pkl"
        local marker="$PRED_ROOT/${scene}.run_fingerprint"
        local scene_log="$LOG_ROOT/scenes/${scene}.log"
        local boxer_diagnostic=""
        local boxer_uncertainty_diagnostic=""
        local boxer_final_uncertainty_diagnostic=""
        local cache_manifest=""
        local cache_expected_fingerprint=""
        local scene_frames_root="$FRAMES_ROOT/$scene/frames"
        local scene_input_fingerprint
        local scene_fingerprint
        if [[ ! -d "$scene_frames_root" ]]; then
            echo "Missing ScanNet frame directory: $scene_frames_root" >&2
            return 1
        fi
        scene_input_fingerprint="$(
            find "$scene_frames_root" -type f \
                -printf '%P\t%s\t%T@\n' \
                | LC_ALL=C sort \
                | sha256sum \
                | awk '{print $1}'
        )"
        scene_fingerprint="$(
            printf '%s\n%s\n' "$CODE_FINGERPRINT" "$scene_input_fingerprint" \
                | sha256sum \
                | awk '{print $1}'
        )"
        if [[ -n "$BOXER_DIAGNOSTICS_ROOT" ]]; then
            boxer_diagnostic="$BOXER_DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl"
        fi
        if [[ -n "$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT" \
              && "$BOXER_UNCERTAINTY_MODE" != "disabled" ]]; then
            boxer_uncertainty_diagnostic="$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT/${scene}_boxer_uncertainty.json"
        fi
        if [[ -n "$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT" \
              && "$BOXER_FINAL_UNCERTAINTY_MODE" != "disabled" ]]; then
            boxer_final_uncertainty_diagnostic="$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT/${scene}_final_boxer_uncertainty.json"
        fi
        if [[ "$CACHE_MODE" == "replay" ]]; then
            cache_manifest="$CACHE_ROOT/$CACHE_NAMESPACE/$scene/manifest.json"
            local baseline_marker="$CACHE_BASELINE_ROOT/${scene}.run_fingerprint"
            if [[ ! -s "$baseline_marker" || ! -s "$cache_manifest" ]]; then
                echo "Missing frozen CuTR marker/cache for replay: $scene" >&2
                return 1
            fi
            cache_expected_fingerprint="$(tr -d '\n' < "$baseline_marker")"
            local manifest_producer
            manifest_producer="$(
                "$PYTHON" - "$cache_manifest" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as handle:
    print(json.load(handle)["producer_fingerprint"])
PY
            )"
            if [[ "$manifest_producer" != "$cache_expected_fingerprint" ]]; then
                echo "Frozen cache does not match CuTR baseline marker: $scene" >&2
                return 1
            fi
            scene_fingerprint="$(
                printf '%s\n%s\n' \
                    "$scene_fingerprint" \
                    "$(sha256sum "$cache_manifest" | awk '{print $1}')" \
                    | sha256sum \
                    | awk '{print $1}'
            )"
        fi
        if [[ -s "$prediction" ]]; then
            if [[ ! -s "$marker" \
                  || "$(tr -d '\n' < "$marker")" != "$scene_fingerprint" ]]; then
                echo "Refusing stale/untracked prediction: $prediction" >&2
                return 1
            fi
            if [[ -n "$boxer_diagnostic" && ! -s "$boxer_diagnostic" ]]; then
                echo "Prediction exists but Boxer diagnostic is missing: $boxer_diagnostic" >&2
                return 1
            fi
            if [[ -n "$boxer_uncertainty_diagnostic" \
                  && ! -s "$boxer_uncertainty_diagnostic" ]]; then
                echo "Prediction exists but uncertainty diagnostic is missing: $boxer_uncertainty_diagnostic" >&2
                return 1
            fi
            if [[ -n "$boxer_final_uncertainty_diagnostic" \
                  && ! -s "$boxer_final_uncertainty_diagnostic" ]]; then
                echo "Prediction exists but final uncertainty diagnostic is missing: $boxer_final_uncertainty_diagnostic" >&2
                return 1
            fi
            completed=$((completed + 1))
            echo "[$(date '+%F %T')] [GPU $gpu] $scene already complete"
            index=$((index + 1))
            continue
        fi
        if [[ -n "$boxer_diagnostic" && -e "$boxer_diagnostic" ]]; then
            echo "Refusing orphan Boxer diagnostic: $boxer_diagnostic" >&2
            return 1
        fi
        if [[ -n "$boxer_uncertainty_diagnostic" \
              && -e "$boxer_uncertainty_diagnostic" ]]; then
            echo "Refusing orphan uncertainty diagnostic: $boxer_uncertainty_diagnostic" >&2
            return 1
        fi
        if [[ -n "$boxer_final_uncertainty_diagnostic" \
              && -e "$boxer_final_uncertainty_diagnostic" ]]; then
            echo "Refusing orphan final uncertainty diagnostic: $boxer_final_uncertainty_diagnostic" >&2
            return 1
        fi

        local optional_args=()
        if [[ -n "$BOXER_DIAGNOSTICS_ROOT" ]]; then
            optional_args+=(
                --boxer-diagnostics-root "$BOXER_DIAGNOSTICS_ROOT"
            )
        fi
        if [[ "$boxer_gate_value_count" == "3" ]]; then
            optional_args+=(
                --boxer-selective-max-center-shift-m "$BOXER_GATE_CENTER"
                --boxer-selective-min-volume-ratio "$BOXER_GATE_MIN_VOLUME"
                --boxer-selective-max-volume-ratio "$BOXER_GATE_MAX_VOLUME"
            )
        fi
        if [[ -n "$BOXER_UNCERTAINTY_MODE" ]]; then
            optional_args+=(--boxer-uncertainty-mode "$BOXER_UNCERTAINTY_MODE")
        fi
        if [[ -n "$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT" ]]; then
            optional_args+=(
                --boxer-uncertainty-diagnostics-root
                "$BOXER_UNCERTAINTY_DIAGNOSTICS_ROOT"
            )
        fi
        if [[ -n "$BOXER_FINAL_UNCERTAINTY_MODE" ]]; then
            optional_args+=(
                --boxer-final-uncertainty-mode
                "$BOXER_FINAL_UNCERTAINTY_MODE"
            )
        fi
        if [[ -n "$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT" ]]; then
            optional_args+=(
                --boxer-final-uncertainty-diagnostics-root
                "$BOXER_FINAL_UNCERTAINTY_DIAGNOSTICS_ROOT"
            )
        fi
        if [[ -n "$REFINER_CHECKPOINT" ]]; then
            optional_args+=(--online-refiner-checkpoint "$REFINER_CHECKPOINT")
        fi
        if [[ -n "$QUALITY_CHECKPOINT" ]]; then
            optional_args+=(
                --online-quality-checkpoint "$QUALITY_CHECKPOINT"
                --online-quality-mode "$QUALITY_MODE"
            )
        fi
        if [[ -n "$QUALITY_DETECTOR_BLEND" ]]; then
            optional_args+=(
                --online-quality-detector-blend "$QUALITY_DETECTOR_BLEND"
            )
        fi
        if [[ "$DISABLE_ONLINE" == "1" ]]; then
            optional_args+=(--disable-online-refinement)
        fi
        if [[ -n "$ABLATION_PROFILE" ]]; then
            optional_args+=(--online-ablation-profile "$ABLATION_PROFILE")
        fi
        if [[ -n "$CANDIDATE_TTL_CLOCK" ]]; then
            optional_args+=(
                --online-candidate-ttl-clock "$CANDIDATE_TTL_CLOCK"
            )
        fi
        if [[ -n "$CANDIDATE_TRACK_TTL" ]]; then
            optional_args+=(
                --online-candidate-track-ttl "$CANDIDATE_TRACK_TTL"
            )
        fi
        if [[ -n "$SCANNET_MIN_EXTENT" ]]; then
            optional_args+=(
                --scannet-min-extent "$SCANNET_MIN_EXTENT"
            )
        fi
        if [[ "$ARCHIVE_CONFIRMED" == "1" ]]; then
            optional_args+=(--online-archive-confirmed-tracks)
        elif [[ "$ARCHIVE_CONFIRMED" == "0" ]]; then
            optional_args+=(--no-online-archive-confirmed-tracks)
        fi

        echo "[$(date '+%F %T')] [GPU $gpu] Running $scene (list index $((index + 1))/$total)"
        if ! (
            cd "$ROOT"
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONHASHSEED="$INFERENCE_SEED" \
            CUBLAS_WORKSPACE_CONFIG=:4096:8 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONUNBUFFERED=1 \
            BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$cache_expected_fingerprint" \
            BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT="$cache_expected_fingerprint" \
            MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
            XDG_CACHE_HOME="$LOG_ROOT/model_cache_gpu${gpu}" \
            LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
            "$PYTHON" demo.py scannet \
                --model-path "$LIVE_ROOT/models/cutr_rgbd.pth" \
                --clip_path "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
                --class_txt "$LIVE_ROOT/data/panoptic_categories_nomerge.txt" \
                --class-features "$LIVE_ROOT/data/class_features.pt" \
                --config "$CONFIG" \
                --output-dir "$PRED_ROOT" \
                --diagnostics-root "$DIAGNOSTICS_ROOT" \
                --scannet-frames-root "$FRAMES_ROOT" \
                --online-proposal-checkpoint "$YOLOE_CHECKPOINT" \
                --online-proposal-every-keyframes "$PROPOSAL_INTERVAL" \
                --device cuda \
                --seed "$INFERENCE_SEED" \
                --seq "$scene" \
                "${optional_args[@]}"
        ) >"$scene_log" 2>&1; then
            echo "[$(date '+%F %T')] ERROR: GPU $gpu failed on $scene" >&2
            tail -n 40 "$scene_log" >&2 || true
            return 1
        fi

        if [[ ! -s "$prediction" ]]; then
            echo "[$(date '+%F %T')] ERROR: GPU $gpu did not produce $prediction" >&2
            tail -n 40 "$scene_log" >&2 || true
            return 1
        fi
        if [[ -n "$boxer_diagnostic" && ! -s "$boxer_diagnostic" ]]; then
            echo "[$(date '+%F %T')] ERROR: Boxer diagnostic missing: $boxer_diagnostic" >&2
            return 1
        fi
        if [[ -n "$boxer_uncertainty_diagnostic" \
              && ! -s "$boxer_uncertainty_diagnostic" ]]; then
            echo "[$(date '+%F %T')] ERROR: uncertainty diagnostic missing: $boxer_uncertainty_diagnostic" >&2
            return 1
        fi
        if [[ -n "$boxer_final_uncertainty_diagnostic" \
              && ! -s "$boxer_final_uncertainty_diagnostic" ]]; then
            echo "[$(date '+%F %T')] ERROR: final uncertainty diagnostic missing: $boxer_final_uncertainty_diagnostic" >&2
            return 1
        fi
        local marker_tmp="${marker}.tmp.$$"
        printf '%s\n' "$scene_fingerprint" > "$marker_tmp"
        mv "$marker_tmp" "$marker"
        completed=$((completed + 1))
        local summary
        summary="$(
            grep -E 'Online refinement summary|Boxer lifting summary' "$scene_log" \
                | tail -n 2 \
                | tr '\n' ' ' \
                || true
        )"
        echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $summary"
        index=$((index + 1))
    done < "$META"
    echo "[$(date '+%F %T')] [GPU $gpu] Worker completed $completed scenes"
}

child_pids=()
cleanup() {
    local pid
    for pid in "${child_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM

for shard in "${!GPUS[@]}"; do
    run_worker "${GPUS[$shard]}" "$shard" "$worker_count" &
    child_pids+=("$!")
done

worker_status=0
for pid in "${child_pids[@]}"; do
    if ! wait "$pid"; then
        worker_status=1
    fi
done
trap - INT TERM
if [[ "$worker_status" -ne 0 ]]; then
    echo "At least one worker failed; evaluation was not started" >&2
    exit 1
fi

prediction_count="$(
    find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l
)"
if [[ "$prediction_count" -ne "$total" ]]; then
    echo "Expected $total predictions, found $prediction_count" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Completed inference; starting ScanNet evaluation"
(
    cd "$ROOT/evaluation"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    "$PYTHON" eval_scannet.py \
        --dataset scannet \
        --data_path /extra/ZhaoX/scannet_data/scans \
        --gt_root "$GT_ROOT" \
        --dump_dir "$EVAL_ROOT" \
        --num_point 40000 \
        --cluster_sampling seed_fps \
        --use_3d_nms \
        --use_cls_nms \
        --per_class_proposal \
        --num_workers 0 \
        --gpu 0 \
        --seed "$EVAL_SEED" \
        --scene_list "$META" \
        --pred_root "$PRED_ROOT"
) >"$LOG_ROOT/eval_stdout.log" 2>&1

grep -E 'eval mAP|eval APrec|eval ARecall' "$LOG_ROOT/eval_stdout.log" || true
echo "[$(date '+%F %T')] Online-refinement inference and evaluation completed"
