#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

RUN_TAG="${1:-${BOXFUSION_R3_TRAIN_RUN_TAG:-}}"
[[ -n "$RUN_TAG" && "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$ ]] || {
  echo "Usage: $0 <unique-train-run-tag>" >&2
  exit 2
}
RESUME="${BOXFUSION_R3_TRAIN_RESUME:-0}"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] || {
  echo "BOXFUSION_R3_TRAIN_RESUME must be 0 or 1" >&2
  exit 2
}

PYTHON_BIN="${BOXFUSION_R3_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
FROZEN_MANIFEST="$ROOT/manifests/frozen_g0_selective_boxer_train100.json"
PARENT_CACHE_ROOT="$ROOT/cache/tr3d_prefix_boxfusion_causal_p100_train100_v1"
PREFIX_ROOT="$ROOT/data/tr3d_prefix_train100_boxfusion_causal_p100_v1"
PREFIX_MANIFEST="$PREFIX_ROOT/manifests/trajectory_prefix_train100_boxfusion_causal_p100_v1.jsonl"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt"
VAL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SCANS_ROOT="/extra/ZhaoX/scannet_data/scans"
R3_CACHE_ROOT="$ROOT/cache/tr3d_r3_train/$RUN_TAG"
REPORT_ROOT="$ROOT/reports/tr3d_r3_train/$RUN_TAG"
CHECKPOINT_SHA="a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448"
CONFIG_SHA="709b66d9e244ef4385dfa9bbc89895ad06c78534f9d14bb7149b687fd58da785"

for path in \
  "$PYTHON_BIN" "$FROZEN_MANIFEST" "$PREFIX_MANIFEST" "$SCENE_LIST" \
  "$VAL_SCENE_LIST" "$SCANS_ROOT"; do
  [[ -e "$path" ]] || { echo "Missing train R3 input: $path" >&2; exit 2; }
done
"$PYTHON_BIN" - "$SCENE_LIST" "$VAL_SCENE_LIST" <<'PY'
from pathlib import Path
import sys

def read(path: Path):
    return [line.split()[0] for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
train, val = read(Path(sys.argv[1])), set(read(Path(sys.argv[2])))
if len(train) != 100 or len(set(train)) != 100 or set(train) & val:
    raise SystemExit("R3 train observer requires 100 unique train-only scenes")
PY

if [[ ! -d "$PARENT_CACHE_ROOT" ]]; then
  echo "Missing train100 TR3D parent cache: $PARENT_CACHE_ROOT" >&2
  echo "Run after NVIDIA devices are available:" >&2
  echo "  BOXFUSION_TR3D_RESUME=0 bash scripts/run_tr3d_strict_train100_parent.sh 0,1" >&2
  exit 2
fi
"$PYTHON_BIN" tools/validate_tr3d_residual_cache.py \
  --cache-root "$PARENT_CACHE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id p100 \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --config-sha256 "$CONFIG_SHA"

resume_args=()
if [[ "$RESUME" == "1" ]]; then
  [[ -d "$R3_CACHE_ROOT" ]] || { echo "Resume cache absent: $R3_CACHE_ROOT" >&2; exit 2; }
  [[ ! -e "$REPORT_ROOT/export_report.json" ]] || {
    echo "Train R3 export is already complete: $REPORT_ROOT/export_report.json" >&2
    exit 2
  }
  resume_args+=(--resume)
else
  [[ ! -e "$R3_CACHE_ROOT" ]] || { echo "Train R3 cache exists: $R3_CACHE_ROOT" >&2; exit 2; }
fi
[[ ! -e "$REPORT_ROOT" ]] || {
  [[ "$RESUME" == "1" && ! -e "$REPORT_ROOT/export_report.json" ]] || {
    echo "Train R3 report namespace exists: $REPORT_ROOT" >&2
    exit 2
  }
}

echo "R3 train100 anchor-near observer"
echo "  train/validation overlap: 0"
echo "  R2a/R2b: disabled"
echo "  GT/CLIP access: disabled"
echo "  output: immutable sidecars only"

"$PYTHON_BIN" tools/run_tr3d_r3_near_observer.py \
  --frozen-manifest "$FROZEN_MANIFEST" \
  --parent-cache-root "$PARENT_CACHE_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --scene-list "$SCENE_LIST" \
  --scans-root "$SCANS_ROOT" \
  --r3-cache-root "$R3_CACHE_ROOT" \
  --prefix-id p100 \
  --expected-parent-checkpoint-sha256 "$CHECKPOINT_SHA" \
  --expected-parent-config-sha256 "$CONFIG_SHA" \
  --report "$REPORT_ROOT/export_report.json" \
  "${resume_args[@]}"

echo "R3 train100 observer complete: $REPORT_ROOT/export_report.json"
