#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=boxfusion_official_cu118
WORK_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_repro
WHEEL_ROOT="$WORK_ROOT/wheels/cu118"
INPUT_FILE="$WORK_ROOT/cu118_wheels.txt"

mkdir -p "$WHEEL_ROOT"

echo "[$(date '+%F %T')] Downloading official cu118 wheels with persistent resume"
aria2c \
    --input-file="$INPUT_FILE" \
    --dir="$WHEEL_ROOT" \
    --continue=true \
    --max-connection-per-server=4 \
    --split=4 \
    --min-split-size=20M \
    --max-tries=0 \
    --retry-wait=5 \
    --connect-timeout=30 \
    --timeout=60 \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=false \
    --check-integrity=true \
    --summary-interval=30

wheel_count=$(find "$WHEEL_ROOT" -maxdepth 1 -type f -name '*.whl' | wc -l)
if [[ "$wheel_count" -ne 14 ]]; then
    echo "Expected 14 verified wheels, found $wheel_count" >&2
    exit 1
fi

echo "[$(date '+%F %T')] Removing cloned cu124 Torch packages"
mapfile -t old_packages < <(
    conda run --no-capture-output -n "$ENV_NAME" pip list --format=freeze |
        sed -n -e 's/^\(torch[^=]*\)==.*/\1/p' -e 's/^\(nvidia-[^=]*-cu12\)==.*/\1/p'
)
if (( ${#old_packages[@]} > 0 )); then
    conda run --no-capture-output -n "$ENV_NAME" pip uninstall -y "${old_packages[@]}"
fi

echo "[$(date '+%F %T')] Installing verified wheels without network access"
conda run --no-capture-output -n "$ENV_NAME" pip install \
    --no-index \
    --find-links="$WHEEL_ROOT" \
    torch==2.6.0+cu118 \
    torchvision==0.21.0+cu118 \
    torchaudio==2.6.0+cu118

echo "[$(date '+%F %T')] Verifying imports and exact CUDA build"
conda run --no-capture-output -n "$ENV_NAME" python -c \
    "import sys, torch, torchvision, torchaudio, pycuda, open_clip, rerun; assert torch.__version__.startswith('2.6.0+cu118'), torch.__version__; assert torchvision.__version__.startswith('0.21.0+cu118'), torchvision.__version__; assert torchaudio.__version__.startswith('2.6.0+cu118'), torchaudio.__version__; assert torch.version.cuda == '11.8', torch.version.cuda; assert torch.cuda.is_available(); print('python', sys.version.split()[0]); print('torch', torch.__version__); print('torchvision', torchvision.__version__); print('torchaudio', torchaudio.__version__); print('cuda_build', torch.version.cuda); print('cuda_device', torch.cuda.get_device_name(0)); print('all_imports_ok')"

echo "[$(date '+%F %T')] Official cu118 environment is ready"
