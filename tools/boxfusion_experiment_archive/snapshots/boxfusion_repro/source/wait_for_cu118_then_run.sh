#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=boxfusion_official_cu118
WORK_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_repro
INSTALL_LOG="$WORK_ROOT/install_cu118_offline.log"

echo "[$(date '+%F %T')] Waiting for verified cu118 environment"
while true; do
    if conda run --no-capture-output -n "$ENV_NAME" python -c \
        "import torch, torchvision, torchaudio, pycuda, open_clip, rerun; assert torch.__version__.startswith('2.6.0+cu118'); assert torchvision.__version__.startswith('0.21.0+cu118'); assert torchaudio.__version__.startswith('2.6.0+cu118'); assert torch.version.cuda == '11.8'; assert torch.cuda.is_available()" \
        >/dev/null 2>&1; then
        echo "[$(date '+%F %T')] Verified cu118 environment is ready"
        exec bash "$WORK_ROOT/wait_for_gpu_and_run.sh" 318655 0
    fi

    if grep -q 'ERROR:.*cu118\\|Expected 14 verified wheels' "$INSTALL_LOG" 2>/dev/null &&
        ! tmux has-session -t boxfusion_cu118_install 2>/dev/null; then
        echo "[$(date '+%F %T')] ERROR: cu118 installer exited unsuccessfully" >&2
        exit 1
    fi

    echo "[$(date '+%F %T')] cu118 environment not ready yet"
    sleep 60
done
