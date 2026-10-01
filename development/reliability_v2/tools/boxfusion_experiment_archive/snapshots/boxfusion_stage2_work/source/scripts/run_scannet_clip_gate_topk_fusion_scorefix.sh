#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   Single GPU:
#     bash scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh 0
#   Dual GPU:
#     bash scripts/run_scannet_clip_gate_topk_fusion_scorefix.sh 0,1
GPU_SPEC="${1:-0}"

ROOT=/data/ZhaoX/BoxFusion
CODE="$ROOT"
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion2
PYTHON="$ENV_ROOT/bin/python"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
META="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
EXPERIMENT_NAME="${BOXFUSION_EXPERIMENT_NAME:-scannet_clip_gate_topk_fusion_scorefix}"
CONFIG="${BOXFUSION_CONFIG:-$ROOT/config/${EXPERIMENT_NAME}.yaml}"
PRED_ROOT="$ROOT/results/$EXPERIMENT_NAME"
LOG_ROOT="$ROOT/logs/$EXPERIMENT_NAME"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
GT_ROOT="$ROOT/evaluation/data_util/scannet_train_detection_data"
STAGE1_EXPERIMENT="${BOXFUSION_STAGE1_EXPERIMENT:-scannet_clip_gate_scorefix}"
STAGE1_LOG="$ROOT/logs/$STAGE1_EXPERIMENT/driver.log"
EVAL_ROOT="$ROOT/evaluation/$EXPERIMENT_NAME"
REFERENCE_TEXT="${BOXFUSION_REFERENCE_TEXT:-Baseline reference: AP15/AP25/AP50 = 32.53/27.11/9.46}"

mkdir -p "$PRED_ROOT" "$SCENE_LOG_ROOT" "$LOG_ROOT/mplconfig"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
    echo "Another Top-K fusion driver already holds $LOG_ROOT/run.lock" >&2
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

required_files=(
    "$PYTHON"
    "$META"
    "$CONFIG"
    "$ROOT/models/cutr_rgbd.pth"
    "$ROOT/models/open_clip_pytorch_model.bin"
)
for path in "${required_files[@]}"; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required path: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$GT_ROOT" ]]; then
    echo "Missing ScanNet GT: $GT_ROOT" >&2
    exit 1
fi
if [[ ! -d "$ROOT/upstream_clean/scannet_readme_frames" ]]; then
    echo "Missing 32.53 baseline frames: $ROOT/upstream_clean/scannet_readme_frames" >&2
    exit 1
fi
if ! grep -q "def appearance_gate_decisions" "$ROOT/boxfusion/instances.py"; then
    echo "Root BoxFusion code does not contain the appearance gate" >&2
    exit 1
fi
if ! grep -q "Saving score-preserving predictions" "$ROOT/demo.py"; then
    echo "Root BoxFusion code does not contain score-preserving export" >&2
    exit 1
fi
if [[ ! -f "$ROOT/boxfusion/reliable_views.py" ]]; then
    echo "Root BoxFusion code does not contain reliable-view selection" >&2
    exit 1
fi
if ! grep -q "use_view_weights" "$ROOT/boxfusion/box_fusion.py"; then
    echo "Root BoxFusion code does not contain weighted fusion" >&2
    exit 1
fi
if ! grep -A2 -q "reliable_views:.*" "$CONFIG"; then
    echo "Top-K reliable-view configuration is missing" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
    "import torch, torchvision, open_clip; assert torch.cuda.is_available(); print('Python environment OK:', torch.__version__, torch.version.cuda)"

total=$(awk 'END {print NR}' "$META")
worker_count="${#GPUS[@]}"
echo "[$(date '+%F %T')] Starting CLIP-gate + Top-K reliable-view fusion inference: $EXPERIMENT_NAME"
echo "[$(date '+%F %T')] GPUs: $GPU_SPEC; workers: $worker_count; scenes: $total"
echo "[$(date '+%F %T')] $REFERENCE_TEXT"
echo "[$(date '+%F %T')] Config: $CONFIG"
echo "[$(date '+%F %T')] Stage-1 log: $STAGE1_LOG"
echo "[$(date '+%F %T')] Input frames: $ROOT/upstream_clean/scannet_readme_frames"
echo "[$(date '+%F %T')] Prediction root: $PRED_ROOT"

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

        local pred_path="$PRED_ROOT/${scene}_boxes.pkl"
        local scene_log="$SCENE_LOG_ROOT/${scene}.log"
        if [[ -s "$pred_path" ]]; then
            completed=$((completed + 1))
            echo "[$(date '+%F %T')] [GPU $gpu] $scene already complete"
            index=$((index + 1))
            continue
        fi

        echo "[$(date '+%F %T')] [GPU $gpu] Running $scene (list index $((index + 1))/$total)"
        (
            cd "$CODE"
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONUNBUFFERED=1 \
            MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
            LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
            "$PYTHON" demo.py scannet \
                --model-path "$ROOT/models/cutr_rgbd.pth" \
                --clip_path "$ROOT/models/open_clip_pytorch_model.bin" \
                --config "$CONFIG" \
                --device cuda \
                --seq "$scene"
        ) >"$scene_log" 2>&1

        if [[ ! -s "$pred_path" ]]; then
            echo "[$(date '+%F %T')] ERROR: GPU $gpu did not produce $pred_path" >&2
            return 1
        fi

        completed=$((completed + 1))
        local gate_summary
        local fusion_summary
        gate_summary=$(grep 'Appearance gate summary' "$scene_log" | tail -n 1 || true)
        fusion_summary=$(grep 'Reliable-view fusion summary' "$scene_log" | tail -n 1 || true)
        echo "[$(date '+%F %T')] [GPU $gpu] Completed $scene $gate_summary $fusion_summary"
        index=$((index + 1))
    done < "$META"
    echo "[$(date '+%F %T')] [GPU $gpu] Worker completed $completed scenes in shard $shard/$shards"
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
    echo "At least one inference worker failed; evaluation was not started" >&2
    exit 1
fi

prediction_count=$(find "$PRED_ROOT" -maxdepth 1 -type f -name 'scene*_boxes.pkl' | wc -l)
if [[ "$prediction_count" -ne "$total" ]]; then
    echo "Expected $total predictions, found $prediction_count; evaluation was not started" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Completed all $total scenes; starting score-preserving ScanNet evaluation"
(
    cd "$CODE/evaluation"
    CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    "$PYTHON" eval_scannet.py \
        --dataset scannet \
        --data_path /extra/ZhaoX/scannet_data/scans \
        --dump_dir "$EVAL_ROOT" \
        --num_point 40000 \
        --cluster_sampling seed_fps \
        --use_3d_nms \
        --use_cls_nms \
        --per_class_proposal \
        --num_workers 0 \
        --gpu 0 \
        --pred_root "$PRED_ROOT"
) >"$LOG_ROOT/eval_stdout.log" 2>&1

grep -E 'eval mAP|eval APrec|eval ARecall' "$LOG_ROOT/eval_stdout.log" || true
echo "[$(date '+%F %T')] CLIP-gate + Top-K fusion inference and evaluation completed"
