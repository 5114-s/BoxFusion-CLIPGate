#!/usr/bin/env bash
set -euo pipefail

HOST_GPU="${1:-0}"
ENV_NAME=boxfusion_official_cu118
EXPECTED_COMMIT=b2e0219a7284249bad4a4a8925066839fe2fa33b
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion_official_cu118
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
WORK_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_repro
BOXFUSION_ROOT=/data/ZhaoX/BoxFusion
CODE="$BOXFUSION_ROOT/upstream_clean/BoxFusion_scorefix"
META="$CODE/evaluation/data_util/meta_data/scannetv2_val.txt"
CONFIG="$WORK_ROOT/scannet_sens_rgb_scorefix.yaml"
PRED_ROOT="$WORK_ROOT/results/scannet_sens_rgb_fixedk_score"
LOG_ROOT="$WORK_ROOT/logs/scannet_sens_rgb_fixedk_score"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
GT_ROOT="$BOXFUSION_ROOT/evaluation/data_util/scannet_train_detection_data"

mkdir -p "$PRED_ROOT" "$SCENE_LOG_ROOT"

if [[ ! -f "$BOXFUSION_ROOT/models/cutr_rgbd.pth" ]]; then
    echo "Missing model: $BOXFUSION_ROOT/models/cutr_rgbd.pth" >&2
    exit 1
fi
if [[ ! -f "$BOXFUSION_ROOT/models/open_clip_pytorch_model.bin" ]]; then
    echo "Missing model: $BOXFUSION_ROOT/models/open_clip_pytorch_model.bin" >&2
    exit 1
fi
if [[ ! -d "$GT_ROOT" ]]; then
    echo "Missing ScanNet GT: $GT_ROOT" >&2
    exit 1
fi
if [[ "$(git -C "$CODE" rev-parse HEAD)" != "$EXPECTED_COMMIT" ]]; then
    echo "Unexpected BoxFusion commit; expected $EXPECTED_COMMIT" >&2
    exit 1
fi
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    conda run --no-capture-output -n "$ENV_NAME" python -c \
    "import cv2, torch, torchvision, torchaudio, pycuda, open_clip, rerun; assert torch.__version__.startswith('2.6.0+cu118'), torch.__version__; assert torchvision.__version__.startswith('0.21.0+cu118'), torchvision.__version__; assert torchaudio.__version__.startswith('2.6.0+cu118'), torchaudio.__version__; assert torch.version.cuda == '11.8', torch.version.cuda; assert torch.cuda.is_available()"

total=$(awk 'END {print NR}' "$META")
echo "[$(date '+%F %T')] Starting official-configuration inference for $total ScanNet scenes"
echo "[$(date '+%F %T')] Host GPU: $HOST_GPU"
echo "[$(date '+%F %T')] Code commit: $(git -C "$CODE" rev-parse HEAD)"
echo "[$(date '+%F %T')] Conda environment: $ENV_NAME"
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    conda run --no-capture-output -n "$ENV_NAME" python -c \
    "import sys, torch, torchvision, torchaudio; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Torchaudio:', torchaudio.__version__); print('PyTorch CUDA:', torch.version.cuda)"
echo "[$(date '+%F %T')] RGB root: $WORK_ROOT/scannet_sens_rgb"
echo "[$(date '+%F %T')] Prediction root: $PRED_ROOT"

completed=0
while IFS= read -r scene || [[ -n "$scene" ]]; do
    pred_path="$PRED_ROOT/${scene}_boxes.pkl"
    scene_log="$SCENE_LOG_ROOT/${scene}.log"
    rgb_marker="$WORK_ROOT/scannet_sens_rgb/$scene/.sens_rgb_complete.json"

    if [[ ! -s "$rgb_marker" ]]; then
        echo "[$(date '+%F %T')] ERROR: missing verified RGB marker $rgb_marker" >&2
        exit 1
    fi

    if [[ -s "$pred_path" ]]; then
        completed=$((completed + 1))
        echo "[$(date '+%F %T')] [$completed/$total] $scene already complete"
        continue
    fi

    echo "[$(date '+%F %T')] [$((completed + 1))/$total] Running $scene"
    cd "$CODE"
    CUDA_VISIBLE_DEVICES="$HOST_GPU" PYTHONDONTWRITEBYTECODE=1 \
        LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
        conda run --no-capture-output -n "$ENV_NAME" python demo.py scannet \
        --model-path "$BOXFUSION_ROOT/models/cutr_rgbd.pth" \
        --clip_path "$BOXFUSION_ROOT/models/open_clip_pytorch_model.bin" \
        --config "$CONFIG" \
        --device cuda \
        --seq "$scene" \
        >"$scene_log" 2>&1

    if [[ ! -s "$pred_path" ]]; then
        echo "[$(date '+%F %T')] ERROR: $scene did not produce $pred_path" >&2
        exit 1
    fi

    completed=$((completed + 1))
    score_summary=$(grep 'Saving score-preserving predictions:' "$scene_log" | tail -n 1 || true)
    echo "[$(date '+%F %T')] [$completed/$total] Completed $scene $score_summary"
done < "$META"

echo "[$(date '+%F %T')] Completed all $completed scenes; starting official ScanNet evaluation"
cd "$CODE/evaluation"
CUDA_VISIBLE_DEVICES="$HOST_GPU" PYTHONDONTWRITEBYTECODE=1 \
    LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    conda run --no-capture-output -n "$ENV_NAME" python eval_scannet.py \
    --dataset scannet \
    --data_path /extra/ZhaoX/scannet_data/scans \
    --dump_dir "$WORK_ROOT/evaluation/scannet_sens_rgb_fixedk_score" \
    --num_point 40000 \
    --cluster_sampling seed_fps \
    --use_3d_nms \
    --use_cls_nms \
    --per_class_proposal \
    --gpu 0 \
    --pred_root "$PRED_ROOT" \
    >"$LOG_ROOT/eval_stdout.log" 2>&1

grep -E 'eval mAP|eval APrec|eval ARecall' "$LOG_ROOT/eval_stdout.log" || true
echo "[$(date '+%F %T')] Inference and evaluation completed"
