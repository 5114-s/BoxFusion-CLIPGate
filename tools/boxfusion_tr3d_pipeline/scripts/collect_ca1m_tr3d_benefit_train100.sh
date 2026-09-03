#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:---preflight}"
case "$MODE" in
  --preflight|--run) ;;
  *) echo "usage: $0 [--preflight|--run]" >&2; exit 2 ;;
esac

TAG=ca1m_tr3d_benefit_train100_v1
PY=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
SCENE_LIST="$ROOT/manifests/ca1m_native_b6_train100_v1/scene_ids.txt"
TRAIN_DATASET="$ROOT/datasets/ca1m_native_b6_train100_v1.npz"
DATA_ROOT=/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1
ANCHOR_ROOT="$ROOT/results/ca1m_native_b6_train100_v1/g0_observer_same_run_anchor"
NATIVE_ROOT="$ROOT/diagnostics/ca1m_native_b6_train100_v1/native_b6"
B6_CHECKPOINT="$ROOT/models/ca1m_native_b6_iou_mlp_v1.npz"
B6_MANIFEST="$ROOT/models/ca1m_native_b6_iou_mlp_v1.manifest.json"
TERMINAL_ROOT="$ROOT/diagnostics/$TAG/terminal"
CANDIDATE_EVIDENCE_ROOT="$ROOT/diagnostics/$TAG/candidate_native_evidence"
REPORT_ROOT="$ROOT/reports/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
TERMINAL_AUDIT="$REPORT_ROOT/terminal_observer_audit.json"
CANDIDATE_AUDIT="$REPORT_ROOT/candidate_evidence_audit.json"
TR3D_PY=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/.conda/boxfusion-tr3d/bin/python
RUNTIME_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev
TR3D_CONFIG=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d/tr3d_scannet_foreground.py
TR3D_CHECKPOINT=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/work_dirs/tr3d/tr3d_fg_full_seed0_fp32_v1/epoch_12.pth
TR3D_PROJECT=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev
TR3D_VENDOR="$TR3D_PROJECT/third_party/mmdetection3d"
GPU="${BOXFUSION_TR3D_GPU:-1}"
LOCK=/tmp/ca1m_tr3d_benefit_train100_v1.lock

require_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 1
  }
}

require_sha "$SCENE_LIST" 35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd
require_sha "$TRAIN_DATASET" 6dbcb8f996dee76d77261b7bf9a42ee9bbb2562c60384c920b7e1fff12a4ff04
require_sha "$B6_CHECKPOINT" d19b3471c84144634c4f50cc339d772a25ada33f873875235087636e8188ca77
require_sha "$B6_MANIFEST" b941c1008dd6a8703010e731c3b0d3675b981c146ceb4cf8a065698c96c560ea
require_sha "$TR3D_CONFIG" e74b29335f32baa6595bcc84a9b3e4fdd14b92a7044abd408a44de95fc360dc4
require_sha "$TR3D_CHECKPOINT" a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448
test "$(sed '/^[[:space:]]*$/d' "$SCENE_LIST" | wc -l)" -eq 100
while IFS= read -r scene; do
  [[ "$scene" =~ ^[0-9]{8}$ ]] || { echo "bad scene: $scene" >&2; exit 1; }
  test -d "$DATA_ROOT/$scene"
  test -s "$ANCHOR_ROOT/${scene}_boxes.pkl"
  test -s "$NATIVE_ROOT/${scene}_ca1m_native_b6.npz"
done < "$SCENE_LIST"

if [[ "$MODE" == --preflight ]]; then
  printf 'preflight_ok=true tag=%s scenes=100 terminal_root=%s candidate_evidence_root=%s\n' \
    "$TAG" "$TERMINAL_ROOT" "$CANDIDATE_EVIDENCE_ROOT"
  exit 0
fi

exec 9>"$LOCK"
flock -n 9 || { echo "collection lock is held: $LOCK" >&2; exit 1; }
mkdir -p "$TERMINAL_ROOT" "$CANDIDATE_EVIDENCE_ROOT" "$REPORT_ROOT" "$LOG_ROOT"

common_terminal_audit=(
  "$PY" tools/audit_ca1m_tr3d_terminal_observer.py
  --scene-list "$SCENE_LIST"
  --data-root "$DATA_ROOT"
  --anchor-root "$ANCHOR_ROOT"
  --native-b6-diagnostics-root "$NATIVE_ROOT"
  --native-b6-checkpoint "$B6_CHECKPOINT"
  --native-b6-manifest "$B6_MANIFEST"
  --observer-root "$TERMINAL_ROOT"
  --worker-script "$ROOT/tools/ca1m_tr3d_terminal_worker.py"
  --runtime-root "$RUNTIME_ROOT"
  --tr3d-config "$TR3D_CONFIG"
  --tr3d-checkpoint "$TR3D_CHECKPOINT"
  --require-genuine
)

if [[ ! -e "$TERMINAL_AUDIT" ]]; then
  terminal_todo=()
  while IFS= read -r scene; do
    artifact="$TERMINAL_ROOT/${scene}_ca1m_tr3d_terminal.npz"
    if [[ -e "$artifact" ]]; then
      "${common_terminal_audit[@]}" --scene "$scene" >/dev/null
    else
      terminal_todo+=(--scene "$scene")
    fi
  done < "$SCENE_LIST"
  if (( ${#terminal_todo[@]} )); then
    env -u PYTHONPATH \
      LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 \
      CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
      MPLCONFIGDIR="/tmp/${TAG}_mpl" OMP_NUM_THREADS=12 TMPDIR=/tmp \
      "$PY" tools/run_ca1m_tr3d_terminal_observer.py \
        --scene-list "$SCENE_LIST" "${terminal_todo[@]}" \
        --data-root "$DATA_ROOT" \
        --anchor-root "$ANCHOR_ROOT" \
        --native-b6-diagnostics-root "$NATIVE_ROOT" \
        --native-b6-checkpoint "$B6_CHECKPOINT" \
        --native-b6-manifest "$B6_MANIFEST" \
        --output-root "$TERMINAL_ROOT" \
        --worker-python "$TR3D_PY" \
        --worker-script "$ROOT/tools/ca1m_tr3d_terminal_worker.py" \
        --runtime-root "$RUNTIME_ROOT" \
        --tr3d-config "$TR3D_CONFIG" \
        --tr3d-checkpoint "$TR3D_CHECKPOINT" \
        --tr3d-project-root "$TR3D_PROJECT" \
        --tr3d-vendor-root "$TR3D_VENDOR" \
        --device cuda:0 2>&1 | tee -a "$LOG_ROOT/terminal_collection.log"
  fi
  "${common_terminal_audit[@]}" --output "$TERMINAL_AUDIT" \
    2>&1 | tee "$LOG_ROOT/terminal_audit.log"
fi

evidence_todo=()
while IFS= read -r scene; do
  artifact="$CANDIDATE_EVIDENCE_ROOT/${scene}_ca1m_native_b6.npz"
  [[ -e "$artifact" ]] || evidence_todo+=(--scene "$scene")
done < "$SCENE_LIST"
if (( ${#evidence_todo[@]} )); then
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TMPDIR=/tmp \
    "$PY" tools/run_ca1m_tr3d_candidate_evidence.py \
      --scene-list "$SCENE_LIST" "${evidence_todo[@]}" \
      --data-root "$DATA_ROOT" \
      --terminal-cache-root "$TERMINAL_ROOT" \
      --output-root "$CANDIDATE_EVIDENCE_ROOT" \
      2>&1 | tee -a "$LOG_ROOT/candidate_evidence_collection.log"
fi

if [[ ! -e "$CANDIDATE_AUDIT" ]]; then
  "$PY" tools/audit_ca1m_tr3d_candidate_evidence.py \
    --scene-list "$SCENE_LIST" \
    --terminal-cache-root "$TERMINAL_ROOT" \
    --candidate-evidence-root "$CANDIDATE_EVIDENCE_ROOT" \
    --terminal-audit "$TERMINAL_AUDIT" \
    --output "$CANDIDATE_AUDIT" \
    2>&1 | tee "$LOG_ROOT/candidate_evidence_audit.log"
fi

printf 'COLLECTION_EXIT=0\n' | tee -a "$LOG_ROOT/collection.log"
