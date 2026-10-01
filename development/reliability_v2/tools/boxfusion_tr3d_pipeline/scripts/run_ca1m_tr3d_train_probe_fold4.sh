#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:---preflight}"
case "$MODE" in
  --preflight|--run|--evaluate) ;;
  *) echo "usage: $0 [--preflight|--run|--evaluate]" >&2; exit 2 ;;
esac

TAG=ca1m_tr3d_train_probe_fold4_v1
PY=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
SCENE_LIST="$ROOT/manifests/ca1m_tr3d_train_probe_fold4_v1/scene_ids.txt"
SUBSET_MANIFEST="$ROOT/manifests/ca1m_tr3d_train_probe_fold4_v1/subset_manifest.json"
TRAIN_DATASET="$ROOT/datasets/ca1m_native_b6_train100_v1.npz"
DATA_ROOT=/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1
ANCHOR_ROOT="$ROOT/results/ca1m_native_b6_train100_v1/g0_observer_same_run_anchor"
NATIVE_ROOT="$ROOT/diagnostics/ca1m_native_b6_train100_v1/native_b6"
B6_CHECKPOINT="$ROOT/models/ca1m_native_b6_iou_mlp_v1.npz"
B6_MANIFEST="$ROOT/models/ca1m_native_b6_iou_mlp_v1.manifest.json"
OBSERVER_ROOT="$ROOT/diagnostics/$TAG/observer"
REPORT_ROOT="$ROOT/reports/$TAG"
LOG_ROOT="$ROOT/logs/$TAG"
AUDIT_REPORT="$REPORT_ROOT/observer_audit.json"
PROBE_REPORT="$REPORT_ROOT/transfer_probe_report.json"
TR3D_PY=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/.conda/boxfusion-tr3d/bin/python
RUNTIME_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev
TR3D_CONFIG=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/config/tr3d/tr3d_scannet_foreground.py
TR3D_CHECKPOINT=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/work_dirs/tr3d/tr3d_fg_full_seed0_fp32_v1/epoch_12.pth
TR3D_PROJECT=/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev
TR3D_VENDOR="$TR3D_PROJECT/third_party/mmdetection3d"
GPU="${BOXFUSION_TR3D_GPU:-0}"
LOCK=/tmp/ca1m_tr3d_train_probe_fold4_v1.lock

require_sha() {
  local path="$1" expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "SHA256 mismatch: $path expected=$expected actual=$actual" >&2
    exit 1
  }
}

require_sha "$SCENE_LIST" 44b9d8574b72c9c6811ea540e142c7482a5a90c2c05c07f2e1ce56bd454c7839
require_sha "$TRAIN_DATASET" 6dbcb8f996dee76d77261b7bf9a42ee9bbb2562c60384c920b7e1fff12a4ff04
require_sha "$TR3D_CONFIG" e74b29335f32baa6595bcc84a9b3e4fdd14b92a7044abd408a44de95fc360dc4
require_sha "$TR3D_CHECKPOINT" a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448
test "$(sed '/^[[:space:]]*$/d' "$SCENE_LIST" | wc -l)" -eq 20
while IFS= read -r scene; do
  [[ "$scene" =~ ^[0-9]{8}$ ]] || { echo "bad scene: $scene" >&2; exit 1; }
  test -d "$DATA_ROOT/$scene"
  test -s "$ANCHOR_ROOT/${scene}_boxes.pkl"
  test -s "$NATIVE_ROOT/${scene}_ca1m_native_b6.npz"
done < "$SCENE_LIST"

if [[ "$MODE" == --preflight ]]; then
  printf 'preflight_ok=true tag=%s scenes=20 observer_root=%s\n' "$TAG" "$OBSERVER_ROOT"
  exit 0
fi

exec 9>"$LOCK"
flock -n 9 || { echo "probe lock is held: $LOCK" >&2; exit 1; }

common_audit=(
  "$PY" tools/audit_ca1m_tr3d_terminal_observer.py
  --scene-list "$SCENE_LIST"
  --data-root "$DATA_ROOT"
  --anchor-root "$ANCHOR_ROOT"
  --native-b6-diagnostics-root "$NATIVE_ROOT"
  --native-b6-checkpoint "$B6_CHECKPOINT"
  --native-b6-manifest "$B6_MANIFEST"
  --observer-root "$OBSERVER_ROOT"
  --worker-script "$ROOT/tools/ca1m_tr3d_terminal_worker.py"
  --runtime-root "$RUNTIME_ROOT"
  --tr3d-config "$TR3D_CONFIG"
  --tr3d-checkpoint "$TR3D_CHECKPOINT"
  --require-genuine
)

if [[ "$MODE" == --evaluate ]]; then
  test -s "$AUDIT_REPORT" || { echo "sealed observer audit is absent" >&2; exit 1; }
  test ! -e "$PROBE_REPORT" || { echo "probe report already exists" >&2; exit 1; }
  "$PY" tools/evaluate_ca1m_tr3d_train_probe.py \
    --scene-list "$SCENE_LIST" \
    --subset-manifest "$SUBSET_MANIFEST" \
    --train-dataset "$TRAIN_DATASET" \
    --official-val-list /data/ZhaoX/BoxFusion/data/val.txt \
    --data-root "$DATA_ROOT" \
    --observer-root "$OBSERVER_ROOT" \
    --observer-audit "$AUDIT_REPORT" \
    --output "$PROBE_REPORT"
  exit 0
fi

test ! -e "$AUDIT_REPORT" || {
  echo "sealed observer audit already exists; refusing another collection" >&2
  exit 1
}
mkdir -p "$OBSERVER_ROOT" "$REPORT_ROOT" "$LOG_ROOT"

comm -13 \
  <(sed 's/$/_ca1m_tr3d_terminal.npz/' "$SCENE_LIST" | sort) \
  <(find "$OBSERVER_ROOT" -maxdepth 1 -type f -printf '%f\n' | sort) \
  | grep -q . && {
  echo "observer root contains unexpected files" >&2
  exit 1
}

todo_args=()
while IFS= read -r scene; do
  artifact="$OBSERVER_ROOT/${scene}_ca1m_tr3d_terminal.npz"
  if [[ -e "$artifact" ]]; then
    "${common_audit[@]}" --scene "$scene" >/dev/null
  else
    todo_args+=(--scene "$scene")
  fi
done < "$SCENE_LIST"

if (( ${#todo_args[@]} )); then
  env -u PYTHONPATH \
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR="/tmp/${TAG}_mpl" OMP_NUM_THREADS=12 TMPDIR=/tmp \
    "$PY" tools/run_ca1m_tr3d_terminal_observer.py \
      --scene-list "$SCENE_LIST" "${todo_args[@]}" \
      --data-root "$DATA_ROOT" \
      --anchor-root "$ANCHOR_ROOT" \
      --native-b6-diagnostics-root "$NATIVE_ROOT" \
      --native-b6-checkpoint "$B6_CHECKPOINT" \
      --native-b6-manifest "$B6_MANIFEST" \
      --output-root "$OBSERVER_ROOT" \
      --worker-python "$TR3D_PY" \
      --worker-script "$ROOT/tools/ca1m_tr3d_terminal_worker.py" \
      --runtime-root "$RUNTIME_ROOT" \
      --tr3d-config "$TR3D_CONFIG" \
      --tr3d-checkpoint "$TR3D_CHECKPOINT" \
      --tr3d-project-root "$TR3D_PROJECT" \
      --tr3d-vendor-root "$TR3D_VENDOR" \
      --device cuda:0 \
      2>&1 | tee -a "$LOG_ROOT/collection.log"
fi

"${common_audit[@]}" --output "$AUDIT_REPORT" | tee "$LOG_ROOT/audit.log"
printf 'COLLECTION_EXIT=0\n' | tee -a "$LOG_ROOT/collection.log"
