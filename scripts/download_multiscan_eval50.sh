#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${1:-/extra/ZhaoX/MultiScan}"
SCENE_LIST="${SCENE_LIST:-/data/ZhaoX/BoxFusion/data/multiscan_eval50.txt}"
HF_BIN="${HF_BIN:-/home/admin1/miniconda3/envs/temp/bin/hf}"

"$HF_BIN" auth whoami >/dev/null
[[ -s "$SCENE_LIST" ]] || { echo "Missing scene list: $SCENE_LIST" >&2; exit 1; }
[[ "$(grep -c . "$SCENE_LIST")" -eq 50 ]] || { echo "Expected 50 scene IDs" >&2; exit 1; }

files=(scans/scans.txt)
while IFS= read -r scene; do
  [[ -n "$scene" ]] || continue
  files+=("scans/${scene}.zip")
done < "$SCENE_LIST"

mkdir -p "$OUTPUT_ROOT"
"$HF_BIN" download 3dlg-hcvc/MultiScan "${files[@]}" \
  --repo-type dataset \
  --local-dir "$OUTPUT_ROOT"

echo "Downloaded 50-scene MultiScan subset to $OUTPUT_ROOT"
