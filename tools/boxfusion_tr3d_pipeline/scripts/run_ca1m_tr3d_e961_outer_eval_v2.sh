#!/usr/bin/env bash
set -euo pipefail

PIPELINE_ROOT="/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline"
PYTHON_BIN="/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/.conda/boxfusion-tr3d/bin/python"
RUNNER="$PIPELINE_ROOT/tools/run_ca1m_tr3d_e961_outer_eval_v2.py"
CONFIG="$PIPELINE_ROOT/config/ca1m_tr3d_e961_outer_dev_eval_v2.json"
PROTOCOL_V2="$PIPELINE_ROOT/manifests/ca1m_tr3d_e961_outer_dev_eval_v1/PREREGISTRATION_PROTOCOL_V2.json"
INVALID_V1="$PIPELINE_ROOT/manifests/ca1m_tr3d_e961_outer_dev_eval_v1/PREREGISTRATION_PROTOCOL_V1_INVALID.json"
LOCK_ROOT="/extra/ZhaoX/ca1m_tr3d_e961_outer_dev_eval_v1/RUN_V2.lock"

usage() {
  cat >&2 <<'EOF'
Usage:
  bash scripts/run_ca1m_tr3d_e961_outer_eval_v2.sh --preflight
  bash scripts/run_ca1m_tr3d_e961_outer_eval_v2.sh --run OUTER_R2_RUN_TAG GPU_ID

The --run form is the sole future formal command. It first seals the
run-tag-specific create-only preregistration without touching RUN_RECEIPT,
checkpoint, anchor, or GT. It then invokes the hash-pinned R2 receipt verifier,
builds/seals the point-only exact20 proposals, and evaluates the reused-dev
continuation gate. It never launches inner training and never opens fold1 or
official validation.
EOF
  exit 2
}

mode="${1:-}"
case "$mode" in
  --preflight)
    [[ "$#" == 1 ]] || usage
    export PYTHONPATH="$PIPELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    exec "$PYTHON_BIN" "$RUNNER" --config "$CONFIG" preflight
    ;;
  --run)
    [[ "$#" == 3 ]] || usage
    run_tag="$2"
    gpu_id="$3"
    [[ "$run_tag" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$ ]] || usage
    [[ "$run_tag" != *..* ]] || usage
    [[ "$gpu_id" =~ ^[0-9]+$ ]] || usage
    ;;
  *) usage ;;
esac

[[ -x "$PYTHON_BIN" ]] || { echo "missing executable evaluator Python: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$RUNNER" && ! -L "$RUNNER" ]] || { echo "missing regular E961 V2 evaluator" >&2; exit 3; }
[[ -f "$CONFIG" && ! -L "$CONFIG" ]] || { echo "missing regular E961 V2 config" >&2; exit 3; }
[[ -f "$INVALID_V1" && ! -L "$INVALID_V1" ]] || { echo "missing protocol-v1 invalidation receipt" >&2; exit 3; }
[[ -f "$PROTOCOL_V2" && ! -L "$PROTOCOL_V2" ]] || {
  echo "formal E961 evaluation requires the reviewed R2-bound protocol V2" >&2
  exit 5
}

lock_parent="${LOCK_ROOT%/*}"
[[ ! -L "$lock_parent" ]] || { echo "evaluation lock parent must not be a symlink" >&2; exit 3; }
mkdir -p "$lock_parent"
if ! mkdir "$LOCK_ROOT" 2>/dev/null; then
  echo "another E961 outer evaluation is active: $LOCK_ROOT" >&2
  exit 4
fi
cleanup() { rmdir "$LOCK_ROOT" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

export PYTHONPATH="$PIPELINE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES="$gpu_id" \
  "$PYTHON_BIN" "$RUNNER" --config "$CONFIG" all \
  --outer-run-tag "$run_tag" --device cuda:0
