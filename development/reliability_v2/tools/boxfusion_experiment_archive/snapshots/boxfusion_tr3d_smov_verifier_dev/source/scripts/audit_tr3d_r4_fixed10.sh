#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${BOXFUSION_R4_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
ARTIFACT_BASE="${BOXFUSION_R4_ARTIFACT_BASE:-/extra/ZhaoX/codex_artifacts/boxfusion_r4_smov_fixed10}"
R4D_TAG="${1:-r4d_fixed10_v1}"
R4F_TAG="${2:-r4f_fixed10_v1}"
REPORT_TAG="${3:-r4_counterfactual_fixed10_v1}"
R3_ROOT="${BOXFUSION_R4_R3_ARTIFACT_ROOT:-/extra/ZhaoX/codex_artifacts/boxfusion_r3_20260805/boxfusion_tr3d_terminal_paired_full100}"
R3_RUN="${BOXFUSION_R4_R3_RUN_TAG:-g0_tr3d_terminal_paired_full100_v1}"
SCOPE="${BOXFUSION_R4_R3_LIST_SCOPE:-scannetv2_val-4b18fc586f7a}"
SCENE_LIST="${BOXFUSION_R4_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
BASELINE="$R3_ROOT/results/b6_g0_tr3d_terminal_same_run_baseline/$R3_RUN/$SCOPE"
ACTIVE="$R3_ROOT/results/b6_g0_tr3d_terminal/$R3_RUN/$SCOPE"
REPORT="$ARTIFACT_BASE/reports/$REPORT_TAG.json"

[[ ! -e "$REPORT" ]] || { echo "Refusing immutable audit overwrite: $REPORT" >&2; exit 2; }
mkdir -p "$(dirname "$REPORT")"
TMPDIR="${BOXFUSION_R4_TMPDIR:-/dev/shm}" \
PYTHONDONTWRITEBYTECODE=1 \
"$PYTHON_BIN" "$ROOT/tools/audit_tr3d_r4_verifier.py" \
  --r4-depth-cache-root "$ARTIFACT_BASE/cache/$R4D_TAG" \
  --r4-feature-cache-root "$ARTIFACT_BASE/cache/$R4F_TAG" \
  --same-run-baseline-root "$BASELINE" \
  --active-prediction-root "$ACTIVE" \
  --scene-list "$SCENE_LIST" \
  --gt-root /data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data \
  --scans-root /extra/ZhaoX/scannet_data/scans \
  --report "$REPORT"
