#!/usr/bin/env bash
set -euo pipefail

# One-scene end-to-end smoke test for the Stage-2-based online refiner.
#
# Usage:
#   bash scripts/run_single_scene_online_smoke.sh 0 scene0702_01

GPU="${1:-0}"
SCENE="${2:-scene0702_01}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
CONFIG="${BOXFUSION_ONLINE_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
PROPOSAL_INTERVAL="${BOXFUSION_PROPOSAL_INTERVAL:-5}"
CANDIDATE_TTL_CLOCK="${BOXFUSION_CANDIDATE_TTL_CLOCK:-}"
CANDIDATE_TRACK_TTL="${BOXFUSION_CANDIDATE_TRACK_TTL:-}"
ARCHIVE_CONFIRMED="${BOXFUSION_ARCHIVE_CONFIRMED_TRACKS:-}"
ABLATION_PROFILE="${BOXFUSION_ONLINE_ABLATION_PROFILE:-}"
PRED_ROOT="${BOXFUSION_SMOKE_PRED_ROOT:-$ROOT/results/scannet_online_smoke_$SCENE}"
DIAGNOSTICS_ROOT="${BOXFUSION_SMOKE_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/scannet_online_smoke_$SCENE}"
LOG_ROOT="${BOXFUSION_SMOKE_LOG_ROOT:-$ROOT/logs/scannet_online_smoke_$SCENE}"
PREDICTION="$PRED_ROOT/${SCENE}_boxes.pkl"
DIAGNOSTICS="$DIAGNOSTICS_ROOT/${SCENE}_tracks.npz"
LOG_FILE="$LOG_ROOT/smoke.log"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "GPU must be a non-negative integer, received: $GPU" >&2
    exit 2
fi
if [[ ! "$SCENE" =~ ^scene[0-9]{4}_[0-9]{2}$ ]]; then
    echo "Invalid ScanNet scene id: $SCENE" >&2
    exit 2
fi
if [[ ! "$PROPOSAL_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOXFUSION_PROPOSAL_INTERVAL must be a positive integer" >&2
    exit 2
fi
if [[ -n "$CANDIDATE_TTL_CLOCK" \
      && "$CANDIDATE_TTL_CLOCK" != "keyframe" \
      && "$CANDIDATE_TTL_CLOCK" != "provider_call" ]]; then
    echo "BOXFUSION_CANDIDATE_TTL_CLOCK must be keyframe or provider_call" >&2
    exit 2
fi
if [[ -n "$CANDIDATE_TRACK_TTL" \
      && ! "$CANDIDATE_TRACK_TTL" =~ ^[0-9]+$ ]]; then
    echo "BOXFUSION_CANDIDATE_TRACK_TTL must be a non-negative integer" >&2
    exit 2
fi
if [[ -n "$ARCHIVE_CONFIRMED" \
      && "$ARCHIVE_CONFIRMED" != "0" \
      && "$ARCHIVE_CONFIRMED" != "1" ]]; then
    echo "BOXFUSION_ARCHIVE_CONFIRMED_TRACKS must be 0 or 1" >&2
    exit 2
fi
if [[ -n "$ABLATION_PROFILE" ]]; then
    case "$ABLATION_PROFILE" in
        observer|refit_only|supplemental_only|supplemental_conservative|full) ;;
        *)
            echo "Invalid BOXFUSION_ONLINE_ABLATION_PROFILE: $ABLATION_PROFILE" >&2
            exit 2
            ;;
    esac
fi

required_files=(
    "$PYTHON"
    "$CONFIG"
    "$YOLOE_CHECKPOINT"
    "$LIVE_ROOT/models/cutr_rgbd.pth"
    "$LIVE_ROOT/models/open_clip_pytorch_model.bin"
    "$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
    "$LIVE_ROOT/data/class_features.pt"
    "$LIVE_ROOT/data/pst_1024_0.tiff"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$LIVE_ROOT/upstream_clean/scannet_readme_frames/$SCENE/frames" ]]; then
    echo "Missing ScanNet RGB-D frames for $SCENE" >&2
    exit 1
fi
if [[ -e "$PREDICTION" ]]; then
    echo "Refusing to reuse an existing smoke prediction: $PREDICTION" >&2
    exit 1
fi

mkdir -p \
    "$PRED_ROOT" \
    "$DIAGNOSTICS_ROOT" \
    "$LOG_ROOT/mplconfig" \
    "$LOG_ROOT/model_cache" \
    "$LOG_ROOT/ultralytics_config"

exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
    echo "Another smoke test holds $LOG_ROOT/run.lock" >&2
    exit 1
fi
exec > >(tee "$LOG_FILE") 2>&1

echo "[$(date '+%F %T')] Starting one-scene online-refinement smoke test"
echo "[$(date '+%F %T')] Scene: $SCENE; host GPU: $GPU"
echo "[$(date '+%F %T')] Config: $CONFIG"
echo "[$(date '+%F %T')] YOLOE checkpoint: $YOLOE_CHECKPOINT"
echo "[$(date '+%F %T')] Proposal interval: $PROPOSAL_INTERVAL"
echo "[$(date '+%F %T')] Candidate TTL clock: ${CANDIDATE_TTL_CLOCK:-config-default}"
echo "[$(date '+%F %T')] Candidate track TTL: ${CANDIDATE_TRACK_TTL:-config-default}"
echo "[$(date '+%F %T')] Archive confirmed tracks: ${ARCHIVE_CONFIRMED:-config-default}"
echo "[$(date '+%F %T')] Online ablation profile: ${ABLATION_PROFILE:-config-default}"
echo "[$(date '+%F %T')] Prediction root: $PRED_ROOT"

CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
YOLO_CONFIG_DIR="$LOG_ROOT/ultralytics_config" \
XDG_CACHE_HOME="$LOG_ROOT/model_cache" \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
    "import torch, torchvision, open_clip, ultralytics; assert torch.cuda.is_available(); print('Smoke environment OK:', torch.__version__, torchvision.__version__, ultralytics.__version__)"

started="$(date +%s)"
optional_args=()
if [[ -n "$CANDIDATE_TTL_CLOCK" ]]; then
    optional_args+=(--online-candidate-ttl-clock "$CANDIDATE_TTL_CLOCK")
fi
if [[ -n "$CANDIDATE_TRACK_TTL" ]]; then
    optional_args+=(--online-candidate-track-ttl "$CANDIDATE_TRACK_TTL")
fi
if [[ "$ARCHIVE_CONFIRMED" == "1" ]]; then
    optional_args+=(--online-archive-confirmed-tracks)
elif [[ "$ARCHIVE_CONFIRMED" == "0" ]]; then
    optional_args+=(--no-online-archive-confirmed-tracks)
fi
if [[ -n "$ABLATION_PROFILE" ]]; then
    optional_args+=(--online-ablation-profile "$ABLATION_PROFILE")
fi
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1 \
MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
YOLO_CONFIG_DIR="$LOG_ROOT/ultralytics_config" \
XDG_CACHE_HOME="$LOG_ROOT/model_cache" \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" "$ROOT/demo.py" scannet \
    --model-path "$LIVE_ROOT/models/cutr_rgbd.pth" \
    --clip_path "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
    --class_txt "$LIVE_ROOT/data/panoptic_categories_nomerge.txt" \
    --class-features "$LIVE_ROOT/data/class_features.pt" \
    --config "$CONFIG" \
    --output-dir "$PRED_ROOT" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --online-proposal-checkpoint "$YOLOE_CHECKPOINT" \
    --online-proposal-every-keyframes "$PROPOSAL_INTERVAL" \
    --device cuda \
    --seq "$SCENE" \
    "${optional_args[@]}"
elapsed="$(( $(date +%s) - started ))"

if [[ ! -s "$PREDICTION" ]]; then
    echo "Smoke test did not produce $PREDICTION" >&2
    exit 1
fi
if [[ ! -s "$DIAGNOSTICS" ]]; then
    echo "Smoke test did not produce $DIAGNOSTICS" >&2
    exit 1
fi
if ! grep -q "Online refinement summary" "$LOG_FILE"; then
    echo "Smoke log is missing the online-refinement summary" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Smoke test passed in ${elapsed}s"
echo "[$(date '+%F %T')] Prediction: $PREDICTION"
echo "[$(date '+%F %T')] Diagnostics: $DIAGNOSTICS"
