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
#   BOXFUSION_JOINT_CHECKPOINT=/path/joint_b3_b5_b6v2.pt
#   BOXFUSION_SGCDET_SPARSE_CHECKPOINT=/path/sgcdet_sparse_refiner.pt

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
JOINT_CHECKPOINT="${BOXFUSION_JOINT_CHECKPOINT:-}"
JOINT_DETECTOR_BLEND="${BOXFUSION_JOINT_DETECTOR_BLEND:-}"
SGCDET_SPARSE_CHECKPOINT="${BOXFUSION_SGCDET_SPARSE_CHECKPOINT:-}"
SCANNET_MIN_EXTENT="${BOXFUSION_SCANNET_MIN_EXTENT:-}"
PROPOSAL_INTERVAL="${BOXFUSION_PROPOSAL_INTERVAL:-5}"
CANDIDATE_TTL_CLOCK="${BOXFUSION_CANDIDATE_TTL_CLOCK:-}"
CANDIDATE_TRACK_TTL="${BOXFUSION_CANDIDATE_TRACK_TTL:-}"
ARCHIVE_CONFIRMED="${BOXFUSION_ARCHIVE_CONFIRMED_TRACKS:-}"
DISABLE_ONLINE="${BOXFUSION_DISABLE_ONLINE_REFINEMENT:-0}"
ABLATION_PROFILE="${BOXFUSION_ONLINE_ABLATION_PROFILE:-}"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

mkdir -p "$PRED_ROOT" "$LOG_ROOT/scenes" "$LOG_ROOT/mplconfig" "$DIAGNOSTICS_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
    echo "Another online-refinement driver holds $LOG_ROOT/run.lock" >&2
    exit 1
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

# A here-string needs a temporary file on some Bash builds.  The previous
# protocol therefore emitted a misleading parse error when /tmp was full.
# Process substitution uses a pipe and keeps GPU parsing functional long
# enough for the explicit free-space preflight below to report the real issue.
IFS=',' read -r -a GPUS < <(printf '%s\n' "$GPU_SPEC")
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

MIN_TMP_FREE_MB="${BOXFUSION_MIN_TMP_FREE_MB:-1024}"
if [[ ! "$MIN_TMP_FREE_MB" =~ ^[1-9][0-9]*$ ]]; then
    echo "BOXFUSION_MIN_TMP_FREE_MB must be a positive integer" >&2
    exit 1
fi
tmp_root="${TMPDIR:-/tmp}"
if [[ ! -d "$tmp_root" ]]; then
    echo "Temporary directory does not exist: $tmp_root" >&2
    exit 1
fi
tmp_free_kb="$(df -Pk "$tmp_root" | awk 'NR == 2 {print $4}')"
if [[ ! "$tmp_free_kb" =~ ^[0-9]+$ ]]; then
    echo "Could not determine free space for temporary directory: $tmp_root" >&2
    exit 1
fi
if (( tmp_free_kb < MIN_TMP_FREE_MB * 1024 )); then
    echo "Insufficient free space on $tmp_root: $((tmp_free_kb / 1024)) MB available; ${MIN_TMP_FREE_MB} MB required" >&2
    echo "Free only disposable caches/temp files; do not remove predictions, checkpoints, or diagnostics." >&2
    exit 1
fi
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
if [[ -n "$JOINT_DETECTOR_BLEND" ]]; then
    if [[ ! "$JOINT_DETECTOR_BLEND" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]]; then
        echo "BOXFUSION_JOINT_DETECTOR_BLEND must be numeric" >&2
        exit 1
    fi
    if ! "$PYTHON" -c \
        "value=float('$JOINT_DETECTOR_BLEND'); assert 0.0 <= value <= 1.0" \
        >/dev/null 2>&1; then
        echo "BOXFUSION_JOINT_DETECTOR_BLEND must lie in [0, 1]" >&2
        exit 1
    fi
fi
if [[ -n "$JOINT_CHECKPOINT" \
      && ( -n "$REFINER_CHECKPOINT" || -n "$QUALITY_CHECKPOINT" ) ]]; then
    echo "The joint checkpoint is mutually exclusive with legacy B5/B6 checkpoints" >&2
    exit 1
fi
if [[ -n "$SGCDET_SPARSE_CHECKPOINT" \
      && ( -n "$REFINER_CHECKPOINT" || -n "$JOINT_CHECKPOINT" ) ]]; then
    echo "The SGCDet sparse checkpoint is mutually exclusive with legacy/joint geometry checkpoints" >&2
    exit 1
fi
if [[ "$DISABLE_ONLINE" != "0" && "$DISABLE_ONLINE" != "1" ]]; then
    echo "BOXFUSION_DISABLE_ONLINE_REFINEMENT must be 0 or 1" >&2
    exit 1
fi
if [[ -n "$ABLATION_PROFILE" ]]; then
    case "$ABLATION_PROFILE" in
        observer|quality_observer|refit_only|supplemental_only|supplemental_conservative|quality_only|b3_memory_observer|b3_topk_refit_only|b3_b6|b3v2_memory_observer|b3v2_visibility_refit_only|b3v2_b6|b5v2_memory_observer|b5v2_refiner_only|b5v2_b6|joint_b3_b5_b6v2_observer|joint_b3_b5_b6v2|sgcdet_sparse_observer|sgcdet_sparse_identity|sgcdet_sparse_active|full) ;;
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
if [[ -n "$JOINT_CHECKPOINT" && ! -f "$JOINT_CHECKPOINT" ]]; then
    echo "Missing joint B3/B5/B6-v2 checkpoint: $JOINT_CHECKPOINT" >&2
    exit 1
fi
if [[ -n "$SGCDET_SPARSE_CHECKPOINT" \
      && ! -f "$SGCDET_SPARSE_CHECKPOINT" ]]; then
    echo "Missing SGCDet local sparse-refiner checkpoint: $SGCDET_SPARSE_CHECKPOINT" >&2
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
echo "[$(date '+%F %T')] Joint checkpoint: ${JOINT_CHECKPOINT:-disabled}"
echo "[$(date '+%F %T')] Joint detector blend: ${JOINT_DETECTOR_BLEND:-config-default}"
echo "[$(date '+%F %T')] SGCDet sparse checkpoint: ${SGCDET_SPARSE_CHECKPOINT:-disabled}"
echo "[$(date '+%F %T')] Prediction root: $PRED_ROOT"
echo "[$(date '+%F %T')] Diagnostics root: $DIAGNOSTICS_ROOT"
echo "[$(date '+%F %T')] ScanNet frames root: $FRAMES_ROOT"

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
        local scene_log="$LOG_ROOT/scenes/${scene}.log"
        if [[ -s "$prediction" ]]; then
            completed=$((completed + 1))
            echo "[$(date '+%F %T')] [GPU $gpu] $scene already complete"
            index=$((index + 1))
            continue
        fi

        local optional_args=()
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
        if [[ -n "$JOINT_CHECKPOINT" ]]; then
            optional_args+=(
                --online-joint-checkpoint "$JOINT_CHECKPOINT"
            )
        fi
        if [[ -n "$JOINT_DETECTOR_BLEND" ]]; then
            optional_args+=(
                --online-joint-detector-blend "$JOINT_DETECTOR_BLEND"
            )
        fi
        if [[ -n "$SGCDET_SPARSE_CHECKPOINT" ]]; then
            optional_args+=(
                --online-sgcdet-sparse-checkpoint "$SGCDET_SPARSE_CHECKPOINT"
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
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONUNBUFFERED=1 \
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
        completed=$((completed + 1))
        local summary
        summary="$(grep 'Online refinement summary' "$scene_log" | tail -n 1 || true)"
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
if ! (
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
) >"$LOG_ROOT/eval_stdout.log" 2>&1; then
    echo "[$(date '+%F %T')] ERROR: ScanNet evaluation failed" >&2
    echo "Evaluation log: $LOG_ROOT/eval_stdout.log" >&2
    tail -n 80 "$LOG_ROOT/eval_stdout.log" >&2 || true
    exit 1
fi

grep -E 'eval mAP|eval APrec|eval ARecall' "$LOG_ROOT/eval_stdout.log" || true
echo "[$(date '+%F %T')] Online-refinement inference and evaluation completed"
