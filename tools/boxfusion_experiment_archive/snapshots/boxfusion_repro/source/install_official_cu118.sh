#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=boxfusion_official_cu118
MAX_ATTEMPTS=5

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    echo "[$(date '+%F %T')] cu118 install attempt $attempt/$MAX_ATTEMPTS"
    if conda run --no-capture-output -n "$ENV_NAME" pip install \
        --force-reinstall \
        --retries 20 \
        --resume-retries 20 \
        --timeout 600 \
        torch==2.6.0 \
        torchvision==0.21.0 \
        torchaudio==2.6.0 \
        --index-url https://download.pytorch.org/whl/cu118; then
        echo "[$(date '+%F %T')] cu118 packages installed"
        conda run --no-capture-output -n "$ENV_NAME" python -c \
            "import torch, torchvision, torchaudio; print(torch.__version__, torchvision.__version__, torchaudio.__version__, torch.version.cuda)"
        exit 0
    fi
    if (( attempt < MAX_ATTEMPTS )); then
        echo "[$(date '+%F %T')] Installation failed; retrying in 30 seconds"
        sleep 30
    fi
done

echo "[$(date '+%F %T')] ERROR: cu118 installation failed after $MAX_ATTEMPTS attempts" >&2
exit 1
