#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="preflight"
GPU_SPEC="0,1"
case "${1:-}" in
  --preflight) MODE="preflight"; GPU_SPEC="${2:-0,1}" ;;
  --run) MODE="run"; GPU_SPEC="${2:-0,1}" ;;
  "") ;;
  *) echo "Usage: $0 [--preflight|--run] [gpu0,gpu1]" >&2; exit 2 ;;
esac
[[ "$#" -le 2 ]] || { echo "Usage: $0 [--preflight|--run] [gpu0,gpu1]" >&2; exit 2; }

LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="${BOXFUSION_PYTHON:-$ENV_ROOT/bin/python}"
TAG="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_TAG:-ca1m_native_final_base_train100_v1}"
MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_MANIFEST:-$ROOT/manifests/ca1m_native_b6_train100_v1/subset_manifest.json}"
SCENE_LIST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENE_LIST:-$ROOT/manifests/ca1m_native_b6_train100_v1/scene_ids.txt}"
DATA_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1}"
CONFIG="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_CONFIG:-$ROOT/config/ca1m_native_final_base_train100_v1.yaml}"
CONTROL_CONFIG="${BOXFUSION_CA1M_FINAL_BASE_CONTROL_CONFIG:-$ROOT/config/ca1m_c4_final_base_g0_control_fixed10_v1.yaml}"
FIXED10_CONFIG="${BOXFUSION_CA1M_FINAL_BASE_CONFIG:-$ROOT/config/ca1m_c4_final_base_g0_clip_topk3_fixed10_v1.yaml}"
CACHE_NAMESPACE="ca1m-native-b6-train100-score04-gap20-cutr-v1"
CACHE_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_SOURCE_CACHE_ROOT:-$ROOT/cache/ca1m_native_b6_train100_v1}"
BASELINE_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_SOURCE_BASELINE_ROOT:-$ROOT/results/ca1m_native_b6_train100_v1/cutr_record}"
ACTIVE_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_ROOT:-$ROOT/results/$TAG/final_base}"
IDENTITY_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_IDENTITY_ROOT:-$ROOT/results/$TAG/same_run_identity}"
BOXER_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_BOXER_ROOT:-$ROOT/diagnostics/$TAG/boxer}"
LOG_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_LOG_ROOT:-$ROOT/logs/$TAG}"
REPORT_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_REPORT_ROOT:-$ROOT/reports/$TAG}"
STAGING_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_STAGING_ROOT:-$ROOT/staging/$TAG}"
LOCK_ROOT="${BOXFUSION_RUN_LOCK_ROOT:-/tmp/boxfusion_ca1m_runlocks}"
LOCK_DIR="$LOCK_ROOT/$TAG.lock"
RUNTIME_TMP="${BOXFUSION_RUNTIME_TMP_ROOT:-/tmp/bfc-$TAG}"

MODEL="$LIVE_ROOT/models/cutr_rgbd.pth"
CLIP="$LIVE_ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$LIVE_ROOT/data/class_features.pt"
PST="$LIVE_ROOT/data/pst_1024_0.tiff"
AUDITOR="$ROOT/tools/audit_ca1m_final_base_anchor.py"
VALIDATOR="$ROOT/tools/validate_ca1m_prediction_file.py"

EXPECTED_MANIFEST_SHA="29a32e92cfece667e9fef4389227eacba2b96c55737569fa6219ca7ab527fd23"
EXPECTED_SCENE_LIST_SHA="35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"
EXPECTED_MODEL_SHA="856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217"
EXPECTED_CLIP_SHA="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
EXPECTED_CLASS_FEATURES_SHA="49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197"
EXPECTED_CLASS_TXT_SHA="0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9"
EXPECTED_PST_SHA="867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660"
EXPECTED_CACHE_FINGERPRINT="a6f51310bdda49cf1bd5a6443e4f82eeb43027a2f9327eb69488498bc3fc260e"

die() { echo "$*" >&2; exit 2; }
file_sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path="$1" expected="$2" actual
  [[ -f "$path" && ! -L "$path" ]] || die "Missing regular frozen input: $path"
  actual="$(file_sha "$path")"
  [[ "$actual" == "$expected" ]] || die "SHA256 mismatch: $path ($actual != $expected)"
}

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
(( ${#GPUS[@]} >= 1 )) || die "No GPUs specified"
for gpu in "${GPUS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU identifier: $gpu"; done
for path in "$MANIFEST" "$SCENE_LIST" "$CONFIG" "$CONTROL_CONFIG" \
  "$FIXED10_CONFIG" "$MODEL" "$CLIP" "$CLASS_TXT" "$CLASS_FEATURES" \
  "$PST" "$AUDITOR" "$VALIDATOR" "$ROOT/demo.py"; do
  [[ -f "$path" && ! -L "$path" ]] || die "Missing regular input: $path"
done
[[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON"
require_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA"
require_sha "$SCENE_LIST" "$EXPECTED_SCENE_LIST_SHA"
require_sha "$MODEL" "$EXPECTED_MODEL_SHA"
require_sha "$CLIP" "$EXPECTED_CLIP_SHA"
require_sha "$CLASS_FEATURES" "$EXPECTED_CLASS_FEATURES_SHA"
require_sha "$CLASS_TXT" "$EXPECTED_CLASS_TXT_SHA"
require_sha "$PST" "$EXPECTED_PST_SHA"

"$PYTHON" "$AUDITOR" contract --control-config "$CONTROL_CONFIG" \
  --active-config "$FIXED10_CONFIG" --train-config "$CONFIG"
mapfile -t SCENES < <(sed -e 's/[[:space:]]*$//' -e '/^$/d' "$SCENE_LIST")
[[ "${#SCENES[@]}" == "100" ]] || die "train100 requires exactly 100 scenes"
[[ "$(printf '%s\n' "${SCENES[@]}" | sort -u | wc -l)" == "100" ]] \
  || die "train100 scene list contains duplicates"

# This phase reads only train RGB-D, immutable train CuTR cache manifests, and
# cache-bound CuTR predictions.  It has no validation GT/evaluator argument.
for scene in "${SCENES[@]}"; do
  scene_root="$DATA_ROOT/$scene"
  manifest="$CACHE_ROOT/$CACHE_NAMESPACE/$scene/manifest.json"
  baseline="$BASELINE_ROOT/${scene}_boxes.pkl"
  [[ -d "$scene_root" && ! -L "$scene_root" ]] || die "Missing CA train scene: $scene_root"
  [[ -s "$manifest" && ! -L "$manifest" ]] || die "Missing train cache manifest: $manifest"
  [[ -s "$baseline" && ! -L "$baseline" ]] || die "Missing train cache baseline: $baseline"
  "$PYTHON" -c 'import json,sys; from pathlib import Path; p,s,n,f=Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4]; v=json.loads(p.read_text()); assert v.get("namespace")==n and v.get("producer_fingerprint")==f and v.get("prediction_file")==s+"_boxes.pkl", f"cache contract drift: {p}"' \
    "$manifest" "$scene" "$CACHE_NAMESPACE" "$EXPECTED_CACHE_FINGERPRINT"
done

for path in "$(dirname "$ACTIVE_ROOT")" "$(dirname "$BOXER_ROOT")" \
  "$LOG_ROOT" "$REPORT_ROOT" "$STAGING_ROOT" "$RUNTIME_TMP"; do
  [[ ! -e "$path" ]] || die "Refusing existing train100 final-base namespace: $path"
done

if [[ "$MODE" == "preflight" ]]; then
  echo "CA-1M native final-base train100 preflight passed"
  echo "  target namespace: $TAG"
  echo "  source cache: existing immutable CA train100 CuTR cache (replay only)"
  echo "  outputs: G0 + frozen CLIP gate + reliable Top-K=3"
  echo "  validation GT/evaluator/training invoked: false"
  echo "  downstream native B6: recollection and CA-only retraining required"
  exit 0
fi

mkdir -p "$LOCK_ROOT"
mkdir "$LOCK_DIR" || die "Another run owns the train100 final-base lock: $LOCK_DIR"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
mkdir -p "$ACTIVE_ROOT" "$IDENTITY_ROOT" "$BOXER_ROOT" "$LOG_ROOT" \
  "$REPORT_ROOT" "$STAGING_ROOT" "$RUNTIME_TMP"
for shard in "${!GPUS[@]}"; do
  mkdir -p "$RUNTIME_TMP/mpl_$shard" "$RUNTIME_TMP/model_$shard"
done

CODE_SOURCES=(
  demo.py boxfusion/instances.py boxfusion/box_fusion.py boxfusion/reliable_views.py
  boxfusion/boxer_lifter.py boxfusion/proposal_cache.py boxfusion/tr3d_terminal_active.py
  config/ca1m_c4_final_base_g0_control_fixed10_v1.yaml
  config/ca1m_c4_final_base_g0_clip_topk3_fixed10_v1.yaml
  config/ca1m_native_final_base_train100_v1.yaml
  tools/audit_ca1m_final_base_anchor.py
  scripts/collect_ca1m_native_final_base_train100.sh
)
CODE_MANIFEST="$REPORT_ROOT/code_manifest.tsv"
for source in "${CODE_SOURCES[@]}"; do
  printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_MANIFEST"
done
CODE_FINGERPRINT="$(file_sha "$CODE_MANIFEST")"
"$PYTHON" "$AUDITOR" contract --control-config "$CONTROL_CONFIG" \
  --active-config "$FIXED10_CONFIG" --train-config "$CONFIG" \
  --output "$REPORT_ROOT/contract.json"

run_scene() {
  local scene="$1" gpu="$2" stage
  stage="$STAGING_ROOT/${scene}.$BASHPID"
  mkdir "$stage" "$stage/pred" "$stage/identity" "$stage/boxer"
  env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$EXPECTED_CACHE_FINGERPRINT" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    MPLCONFIGDIR="$RUNTIME_TMP/mpl_$gpu" XDG_CACHE_HOME="$RUNTIME_TMP/model_$gpu" \
    "$PYTHON" "$ROOT/demo.py" CA1M --model-path "$MODEL" --clip_path "$CLIP" \
      --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES" \
      --config "$CONFIG" --output-dir "$stage/pred" \
      --prediction-same-run-anchor-root "$stage/identity" \
      --boxer-diagnostics-root "$stage/boxer" --device cuda --seq "$scene" --seed 0 \
      > "$stage/run.log" 2>&1
  "$PYTHON" "$VALIDATOR" --prediction "$stage/pred/${scene}_boxes.pkl" >/dev/null
  "$PYTHON" "$VALIDATOR" --prediction "$stage/identity/${scene}_boxes.pkl" >/dev/null
  [[ "$stage/pred/${scene}_boxes.pkl" -ef "$stage/identity/${scene}_boxes.pkl" ]] \
    || die "$scene: staged prediction is not a hard-link identity pair"
  [[ -s "$stage/boxer/${scene}_boxer_lifting.jsonl" ]] \
    || die "$scene: staged Selective Boxer diagnostic is missing"
  mv "$stage/pred/${scene}_boxes.pkl" "$ACTIVE_ROOT/${scene}_boxes.pkl"
  mv "$stage/identity/${scene}_boxes.pkl" "$IDENTITY_ROOT/${scene}_boxes.pkl"
  mv "$stage/boxer/${scene}_boxer_lifting.jsonl" "$BOXER_ROOT/${scene}_boxer_lifting.jsonl"
  mv "$stage/run.log" "$LOG_ROOT/${scene}.log"
  rmdir "$stage/pred" "$stage/identity" "$stage/boxer" "$stage"
  chmod 0444 "$ACTIVE_ROOT/${scene}_boxes.pkl" "$IDENTITY_ROOT/${scene}_boxes.pkl" \
    "$BOXER_ROOT/${scene}_boxer_lifting.jsonl" "$LOG_ROOT/${scene}.log"
  echo "[$(date '+%F %T')] [GPU $gpu] final-base train collection complete: $scene"
}

workers="${#GPUS[@]}"
pids=()
failures=0
for shard in "${!GPUS[@]}"; do
  (
    for index in "${!SCENES[@]}"; do
      (( index % workers == shard )) || continue
      run_scene "${SCENES[$index]}" "${GPUS[$shard]}"
    done
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || failures=1; done
(( failures == 0 )) || die "At least one final-base train100 worker failed"

"$PYTHON" "$AUDITOR" identity --scene-list "$SCENE_LIST" --expected-scenes 100 \
  --split train100 --active-root "$ACTIVE_ROOT" --identity-root "$IDENTITY_ROOT" \
  --boxer-root "$BOXER_ROOT" --log-root "$LOG_ROOT" \
  --output "$REPORT_ROOT/collection_manifest.json"

CODE_AFTER="$RUNTIME_TMP/code_after.tsv"
for source in "${CODE_SOURCES[@]}"; do
  printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_AFTER"
done
[[ "$(file_sha "$CODE_AFTER")" == "$CODE_FINGERPRINT" ]] \
  || die "Final-base collection code changed during train100 inference"
echo "CA-1M native final-base train100 collection complete (no GT/evaluation/training):"
echo "  $REPORT_ROOT/collection_manifest.json"
echo "Next required step: recollect native-B6 evidence from this anchor and retrain B6 on CA train only."
