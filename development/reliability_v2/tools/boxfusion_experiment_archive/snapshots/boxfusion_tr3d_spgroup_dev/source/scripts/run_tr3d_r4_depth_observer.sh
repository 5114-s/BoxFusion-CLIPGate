#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BOXFUSION_R4_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
RUN_TAG="${1:-r4d_smoke_scene0277_v1}"
SCENE_LIST="${BOXFUSION_R4_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_smoke_scene0277_00.txt}"
R3_ARTIFACT_ROOT="${BOXFUSION_R4_R3_ARTIFACT_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r3_20260805/boxfusion_tr3d_terminal_paired_full100}"
R3_RUN_TAG="${BOXFUSION_R4_R3_RUN_TAG:-g0_tr3d_terminal_paired_full100_v1}"
LIST_SCOPE="${BOXFUSION_R4_R3_LIST_SCOPE:-scannetv2_val-4b18fc586f7a}"
ARTIFACT_BASE="${BOXFUSION_R4_ARTIFACT_BASE:-/extra/ZhaoX/codex_artifacts/boxfusion_r4_smov}"

PARENT_ROOT="${BOXFUSION_R4_PARENT_CACHE_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/cache/tr3d_prefix_boxfusion_causal_p100_full100_v3}"
PREFIX_MANIFEST="${BOXFUSION_R4_PREFIX_MANIFEST:-/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev/data/tr3d_prefix_val100_boxfusion_causal_p100_v3/manifests/trajectory_prefix_val100_boxfusion_causal_p100_v3.jsonl}"
# The terminal-prefix manifest is cryptographically bound to this decoded
# ScanNet tree.  The BoxFusion convenience symlink contains equivalent images
# but is intentionally rejected because its provenance root is different.
FRAMES_ROOT="${BOXFUSION_R4_FRAMES_ROOT:-/extra/ZhaoX/scannet_data/scans}"
BASELINE_ROOT="$R3_ARTIFACT_ROOT/results/b6_g0_tr3d_terminal_same_run_baseline/$R3_RUN_TAG/$LIST_SCOPE"
ACTIVE_ROOT="$R3_ARTIFACT_ROOT/results/b6_g0_tr3d_terminal/$R3_RUN_TAG/$LIST_SCOPE"
R3_DIAGNOSTICS="$R3_ARTIFACT_ROOT/diagnostics/b6_g0_tr3d_terminal/$R3_RUN_TAG/$LIST_SCOPE/tr3d_terminal"
R4_CACHE_ROOT="$ARTIFACT_BASE/cache/$RUN_TAG"
REPORT="$ARTIFACT_BASE/reports/$RUN_TAG/export_report.json"

for path in "$PYTHON_BIN" "$SCENE_LIST" "$PREFIX_MANIFEST"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 1
  fi
done
for directory in "$PARENT_ROOT" "$FRAMES_ROOT" "$BASELINE_ROOT" "$ACTIVE_ROOT" "$R3_DIAGNOSTICS"; do
  if [[ ! -d "$directory" ]]; then
    echo "Missing required directory: $directory" >&2
    exit 1
  fi
done
if [[ -e "$REPORT" ]]; then
  echo "Refusing immutable report overwrite: $REPORT" >&2
  exit 1
fi

mkdir -p "$(dirname "$REPORT")" "$R4_CACHE_ROOT"
echo "R4-D/FS paired SMOV3D-inspired observer"
echo "  scenes: $SCENE_LIST"
echo "  frozen raw R3 active: $ACTIVE_ROOT"
echo "  same-run G0 anchors: $BASELINE_ROOT"
echo "  output sidecars: $R4_CACHE_ROOT"
echo "  observer-only: predictions/scores/order/CLIP are never written"

TMPDIR="${BOXFUSION_R4_TMPDIR:-/dev/shm}" \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON_BIN" "$ROOT/tools/run_tr3d_r4_depth_observer.py" \
  --parent-cache-root "$PARENT_ROOT" \
  --prefix-manifest "$PREFIX_MANIFEST" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$SCENE_LIST" \
  --same-run-baseline-root "$BASELINE_ROOT" \
  --active-prediction-root "$ACTIVE_ROOT" \
  --r3-diagnostics-root "$R3_DIAGNOSTICS" \
  --r4-cache-root "$R4_CACHE_ROOT" \
  --prefix-id p100 \
  --top-k 5 \
  --pixel-stride 4 \
  --depth-scale 1000 \
  --margin 0.05 \
  --min-depth 0.10 \
  --max-depth 8.0 \
  --image-height 480 \
  --image-width 640 \
  --report "$REPORT"

echo "R4-D/FS observer complete: $REPORT"
