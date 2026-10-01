#!/usr/bin/env bash
set -euo pipefail

# Official MultiScan five-scene validation pilot.  The caller must first accept
# the CC BY-NC 4.0 dataset terms at
# https://huggingface.co/datasets/3dlg-hcvc/MultiScan and run `hf auth login`.

OUTPUT_ROOT="${1:-/extra/ZhaoX/MultiScan}"
HF_BIN="${HF_BIN:-hf}"

"$HF_BIN" auth whoami >/dev/null
mkdir -p "$OUTPUT_ROOT"
"$HF_BIN" download 3dlg-hcvc/MultiScan \
  scans/scans.txt \
  scans/scene_00002_00.zip \
  scans/scene_00004_00.zip \
  scans/scene_00007_00.zip \
  scans/scene_00008_00.zip \
  scans/scene_00009_00.zip \
  --repo-type dataset \
  --local-dir "$OUTPUT_ROOT"

echo "Downloaded the official MultiScan validation pilot to $OUTPUT_ROOT"
