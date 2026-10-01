#!/usr/bin/env bash
set -euo pipefail

MAIN=/data/ZhaoX/BoxFusion
SCAN_ROOT=/extra/ZhaoX/MultiScan/scans
SCENE_LIST="$MAIN/data/multiscan_eval50.txt"

mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 50 ]] || { echo "Expected 50 scenes" >&2; exit 1; }

for index in "${!SCENES[@]}"; do
  scene="${SCENES[$index]}"
  archive="$SCAN_ROOT/${scene}.zip"
  target="$SCAN_ROOT/$scene"
  [[ -s "$archive" ]] || { echo "Missing archive: $archive" >&2; exit 1; }
  unzip -tq "$archive" >/dev/null
  complete=true
  for suffix in .json .jsonl .mp4 .depth.zlib .align.json .annotations.json .ply; do
    [[ -s "$target/${scene}${suffix}" ]] || complete=false
  done
  if [[ "$complete" != true ]]; then
    unzip -q -o "$archive" -d "$SCAN_ROOT"
  fi
  echo "[$((index+1))/50] ready $scene"
done

touch "$SCAN_ROOT/MULTISCAN_EVAL50_EXTRACT_COMPLETE"
echo "All 50 MultiScan scenes passed ZIP validation and extraction checks."
