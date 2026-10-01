#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIVE_BOXFUSION="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
SHARED_BOXER="${BOXFUSION_SHARED_BOXER_ROOT:-/data/ZhaoX/OVM3D-Dett/third_party/boxer}"
BOXER_ROOT="$CODE_ROOT/third_party/boxer"
CKPT_DIR="$BOXER_ROOT/ckpts"
BOXER_CKPT="$CKPT_DIR/boxernet_hw960in2x6d768-c88128f8.ckpt"
BOXER_SHA256="d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
DINO_NAME="dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
DINO_SHA256="4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
DINO_SHARED="$SHARED_BOXER/ckpts/$DINO_NAME"
DINO_LOCAL="$CKPT_DIR/$DINO_NAME"

link_read_only_asset() {
  local target="$1"
  local link_path="$2"
  if [[ -L "$link_path" ]]; then
    if [[ "$(readlink -f "$link_path")" != "$(readlink -f "$target")" ]]; then
      echo "Refusing to replace unexpected symlink: $link_path" >&2
      exit 1
    fi
    return
  fi
  if [[ -e "$link_path" ]]; then
    echo "Refusing to replace existing path: $link_path" >&2
    exit 1
  fi
  if [[ ! -e "$target" ]]; then
    echo "Missing read-only asset target: $target" >&2
    exit 1
  fi
  ln -s "$target" "$link_path"
}

cd "$CODE_ROOT"
git submodule update --init --recursive

commit="$(git -C "$BOXER_ROOT" rev-parse HEAD)"
if [[ "$commit" != "1f86542dc342a4b1d474c87c97c5d1d6566d9148" ]]; then
  echo "Unexpected Boxer commit: $commit" >&2
  exit 1
fi

mkdir -p "$CKPT_DIR"
link_read_only_asset "$DINO_SHARED" "$DINO_LOCAL"
link_read_only_asset "$LIVE_BOXFUSION/data" "$CODE_ROOT/data"
link_read_only_asset "$LIVE_BOXFUSION/models" "$CODE_ROOT/models"
link_read_only_asset \
  "$LIVE_BOXFUSION/upstream_clean/BoxFusion_scorefix/evaluation/data_util/scannet_train_detection_data" \
  "$CODE_ROOT/evaluation/data_util/scannet_train_detection_data"

if [[ ! -s "$BOXER_CKPT" ]]; then
  wget -c -O "$BOXER_CKPT" \
    "https://huggingface.co/facebook/boxer/resolve/main/boxernet_hw960in2x6d768-c88128f8.ckpt"
fi

printf '%s  %s\n' "$BOXER_SHA256" "$BOXER_CKPT" | sha256sum --check --strict
printf '%s  %s\n' "$DINO_SHA256" "$DINO_LOCAL" | sha256sum --check --strict

PYTHON_BIN="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
PYTHONPATH="$BOXER_ROOT:$CODE_ROOT" "$PYTHON_BIN" -c \
  "import torch; from boxernet.boxernet import BoxerNet; from boxfusion.boxer_lifter import OFFICIAL_BOXER_COMMIT; print('Boxer import OK:', torch.__version__, OFFICIAL_BOXER_COMMIT)"

echo "Boxer lifting assets are ready."
