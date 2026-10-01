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
TAG="${BOXFUSION_CA1M_FINAL_BASE_TAG:-ca1m_c4_final_base_g0_clip_topk3_fixed10_v1}"
CONTROL_TAG="${BOXFUSION_CA1M_FINAL_BASE_CONTROL_TAG:-ca1m_c4_final_base_g0_control_fixed10_v1}"
SCENE_LIST="${BOXFUSION_CA1M_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_ablation10_even.txt}"
DATA_ROOT="${BOXFUSION_CA1M_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
CONTROL_CONFIG="${BOXFUSION_CA1M_FINAL_BASE_CONTROL_CONFIG:-$ROOT/config/ca1m_c4_final_base_g0_control_fixed10_v1.yaml}"
ACTIVE_CONFIG="${BOXFUSION_CA1M_FINAL_BASE_CONFIG:-$ROOT/config/ca1m_c4_final_base_g0_clip_topk3_fixed10_v1.yaml}"
TRAIN_CONFIG="${BOXFUSION_CA1M_FINAL_BASE_TRAIN_CONFIG:-$ROOT/config/ca1m_native_final_base_train100_v1.yaml}"
CACHE_ROOT="${BOXFUSION_CA1M_CUTR_CACHE_ROOT:-$ROOT/cache/ca1m_cutr_proposals/ca1m-score04-gap20-c0-v2}"
CACHE_NAMESPACE="ca1m-score04-gap20-c0-v2"
BASELINE_ROOT="${BOXFUSION_CA1M_C0_PRED_ROOT:-$ROOT/results/ca1m_port/c0_original_fixed10_v2}"
CONTROL_ROOT="${BOXFUSION_CA1M_FINAL_BASE_CONTROL_ROOT:-$ROOT/results/ca1m_port/$CONTROL_TAG}"
ACTIVE_ROOT="${BOXFUSION_CA1M_FINAL_BASE_ROOT:-$ROOT/results/ca1m_port/$TAG}"
IDENTITY_ROOT="${BOXFUSION_CA1M_FINAL_BASE_IDENTITY_ROOT:-$ROOT/results/ca1m_port/${TAG}_same_run_identity}"
CONTROL_BOXER_ROOT="${BOXFUSION_CA1M_FINAL_BASE_CONTROL_BOXER_ROOT:-$ROOT/diagnostics/ca1m_port/$CONTROL_TAG/boxer}"
ACTIVE_BOXER_ROOT="${BOXFUSION_CA1M_FINAL_BASE_BOXER_ROOT:-$ROOT/diagnostics/ca1m_port/$TAG/boxer}"
LOG_ROOT="${BOXFUSION_CA1M_FINAL_BASE_LOG_ROOT:-$ROOT/logs/ca1m_port/$TAG}"
REPORT_ROOT="${BOXFUSION_CA1M_FINAL_BASE_REPORT_ROOT:-$ROOT/reports/ca1m_port/$TAG}"
EVAL_VIEW="${BOXFUSION_CA1M_FINAL_BASE_EVAL_VIEW:-$ROOT/data/ca1m_eval_$TAG}"
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

EXPECTED_SCENE_LIST_SHA="b81bd6a2f147f964c6a94f3ed838edc1d0f3e801ae642d8ba30b85c643aebeab"
EXPECTED_MODEL_SHA="856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217"
EXPECTED_CLIP_SHA="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
EXPECTED_CLASS_FEATURES_SHA="49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197"
EXPECTED_CLASS_TXT_SHA="0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9"
EXPECTED_PST_SHA="867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660"
EXPECTED_CACHE_FINGERPRINT="991d51281617a2731784d09be4d8b2839290b6304e5ca5aa735ad1d4e6f8ccb6"

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
for path in "$SCENE_LIST" "$CONTROL_CONFIG" "$ACTIVE_CONFIG" "$TRAIN_CONFIG" \
  "$MODEL" "$CLIP" "$CLASS_TXT" "$CLASS_FEATURES" "$PST" "$AUDITOR" \
  "$VALIDATOR" "$ROOT/demo.py" "$ROOT/evaluation/eval_ca1m.py"; do
  [[ -f "$path" && ! -L "$path" ]] || die "Missing regular input: $path"
done
[[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON"
require_sha "$SCENE_LIST" "$EXPECTED_SCENE_LIST_SHA"
require_sha "$MODEL" "$EXPECTED_MODEL_SHA"
require_sha "$CLIP" "$EXPECTED_CLIP_SHA"
require_sha "$CLASS_FEATURES" "$EXPECTED_CLASS_FEATURES_SHA"
require_sha "$CLASS_TXT" "$EXPECTED_CLASS_TXT_SHA"
require_sha "$PST" "$EXPECTED_PST_SHA"

"$PYTHON" "$AUDITOR" contract --control-config "$CONTROL_CONFIG" \
  --active-config "$ACTIVE_CONFIG" --train-config "$TRAIN_CONFIG"

mapfile -t SCENES < <(sed -e 's/[[:space:]]*$//' -e '/^$/d' "$SCENE_LIST")
[[ "${#SCENES[@]}" == "10" ]] || die "fixed10 requires exactly 10 scenes"
[[ "$(printf '%s\n' "${SCENES[@]}" | sort -u | wc -l)" == "10" ]] \
  || die "fixed10 scene list contains duplicates"
for scene in "${SCENES[@]}"; do
  scene_root="$DATA_ROOT/$scene"
  manifest="$CACHE_ROOT/$scene/manifest.json"
  baseline="$BASELINE_ROOT/${scene}_boxes.pkl"
  [[ -d "$scene_root" && ! -L "$scene_root" ]] || die "Missing CA-1M scene: $scene_root"
  [[ -s "$manifest" && ! -L "$manifest" ]] || die "Missing immutable cache manifest: $manifest"
  [[ -s "$baseline" && ! -L "$baseline" ]] || die "Missing cache baseline prediction: $baseline"
  "$PYTHON" -c 'import json,sys; from pathlib import Path; p,s,n,f=Path(sys.argv[1]),sys.argv[2],sys.argv[3],sys.argv[4]; v=json.loads(p.read_text()); assert v.get("namespace")==n and v.get("producer_fingerprint")==f and v.get("prediction_file")==s+"_boxes.pkl", f"cache contract drift: {p}"' \
    "$manifest" "$scene" "$CACHE_NAMESPACE" "$EXPECTED_CACHE_FINGERPRINT"
done

for path in "$CONTROL_ROOT" "$ACTIVE_ROOT" "$IDENTITY_ROOT" \
  "$(dirname "$CONTROL_BOXER_ROOT")" "$(dirname "$ACTIVE_BOXER_ROOT")" \
  "$LOG_ROOT" "$REPORT_ROOT" "$EVAL_VIEW" "$RUNTIME_TMP"; do
  [[ ! -e "$path" ]] || die "Refusing existing final-base namespace: $path"
done

if [[ "$MODE" == "preflight" ]]; then
  echo "CA-1M final-base fixed10 preflight passed"
  echo "  active: $TAG"
  echo "  control: $CONTROL_TAG"
  echo "  cache: existing immutable CA fixed10 CuTR cache (read-only replay)"
  echo "  GT/evaluation/training invoked: false"
  exit 0
fi

mkdir -p "$LOCK_ROOT"
mkdir "$LOCK_DIR" || die "Another run owns the final-base lock: $LOCK_DIR"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
mkdir -p "$CONTROL_ROOT" "$ACTIVE_ROOT" "$IDENTITY_ROOT" \
  "$CONTROL_BOXER_ROOT" "$ACTIVE_BOXER_ROOT" "$LOG_ROOT/control" \
  "$LOG_ROOT/active" "$REPORT_ROOT" "$RUNTIME_TMP"
for shard in "${!GPUS[@]}"; do
  mkdir -p "$RUNTIME_TMP/mpl_control_$shard" "$RUNTIME_TMP/mpl_active_$shard" \
    "$RUNTIME_TMP/model_control_$shard" "$RUNTIME_TMP/model_active_$shard"
done

CODE_SOURCES=(
  demo.py boxfusion/instances.py boxfusion/box_fusion.py boxfusion/reliable_views.py
  boxfusion/boxer_lifter.py boxfusion/proposal_cache.py boxfusion/tr3d_terminal_active.py
  config/ca1m_c4_final_base_g0_control_fixed10_v1.yaml
  config/ca1m_c4_final_base_g0_clip_topk3_fixed10_v1.yaml
  config/ca1m_native_final_base_train100_v1.yaml
  tools/audit_ca1m_final_base_anchor.py scripts/run_ca1m_c4_final_base_fixed10.sh
)
CODE_MANIFEST="$REPORT_ROOT/code_manifest.tsv"
for source in "${CODE_SOURCES[@]}"; do
  printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_MANIFEST"
done
CODE_FINGERPRINT="$(file_sha "$CODE_MANIFEST")"
"$PYTHON" "$AUDITOR" contract --control-config "$CONTROL_CONFIG" \
  --active-config "$ACTIVE_CONFIG" --train-config "$TRAIN_CONFIG" \
  --output "$REPORT_ROOT/contract.json"

run_phase() {
  local phase="$1" config="$2" output="$3" boxer_root="$4" identity_root="${5:-}"
  local workers="${#GPUS[@]}" pids=() failures=0
  for shard in "${!GPUS[@]}"; do
    (
      for index in "${!SCENES[@]}"; do
        (( index % workers == shard )) || continue
        scene="${SCENES[$index]}"
        args=(
          "$PYTHON" "$ROOT/demo.py" CA1M --model-path "$MODEL" --clip_path "$CLIP"
          --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES"
          --config "$config" --output-dir "$output" --boxer-diagnostics-root "$boxer_root"
          --device cuda --seq "$scene" --seed 0
        )
        if [[ -n "$identity_root" ]]; then
          args+=(--prediction-same-run-anchor-root "$identity_root")
        fi
        env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" PYTHONHASHSEED=0 \
          PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
          BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$EXPECTED_CACHE_FINGERPRINT" \
          LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
          MPLCONFIGDIR="$RUNTIME_TMP/mpl_${phase}_$shard" \
          XDG_CACHE_HOME="$RUNTIME_TMP/model_${phase}_$shard" \
          "${args[@]}" > "$LOG_ROOT/$phase/${scene}.log" 2>&1
      done
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failures=1; done
  (( failures == 0 )) || die "$phase worker failed"
}

run_phase control "$CONTROL_CONFIG" "$CONTROL_ROOT" "$CONTROL_BOXER_ROOT"
run_phase active "$ACTIVE_CONFIG" "$ACTIVE_ROOT" "$ACTIVE_BOXER_ROOT" "$IDENTITY_ROOT"

for scene in "${SCENES[@]}"; do
  "$PYTHON" "$VALIDATOR" --prediction "$CONTROL_ROOT/${scene}_boxes.pkl"
  "$PYTHON" "$VALIDATOR" --prediction "$ACTIVE_ROOT/${scene}_boxes.pkl"
  "$PYTHON" "$VALIDATOR" --prediction "$IDENTITY_ROOT/${scene}_boxes.pkl"
  chmod 0444 "$CONTROL_ROOT/${scene}_boxes.pkl" "$ACTIVE_ROOT/${scene}_boxes.pkl" \
    "$IDENTITY_ROOT/${scene}_boxes.pkl" \
    "$CONTROL_BOXER_ROOT/${scene}_boxer_lifting.jsonl" \
    "$ACTIVE_BOXER_ROOT/${scene}_boxer_lifting.jsonl" \
    "$LOG_ROOT/control/${scene}.log" "$LOG_ROOT/active/${scene}.log"
done

# Mandatory no-GT audit precedes creation of the evaluation view/evaluator.
"$PYTHON" "$AUDITOR" identity --scene-list "$SCENE_LIST" --expected-scenes 10 \
  --split validation_fixed10 --control-root "$CONTROL_ROOT" --active-root "$ACTIVE_ROOT" \
  --control-boxer-root "$CONTROL_BOXER_ROOT" --control-log-root "$LOG_ROOT/control" \
  --identity-root "$IDENTITY_ROOT" --boxer-root "$ACTIVE_BOXER_ROOT" \
  --log-root "$LOG_ROOT/active" --output "$REPORT_ROOT/identity_and_paired_audit.json"

CODE_AFTER="$RUNTIME_TMP/code_after.tsv"
for source in "${CODE_SOURCES[@]}"; do
  printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_AFTER"
done
[[ "$(file_sha "$CODE_AFTER")" == "$CODE_FINGERPRINT" ]] \
  || die "Final-base code changed during paired inference"

mkdir "$EVAL_VIEW"
for scene in "${SCENES[@]}"; do ln -s "$DATA_ROOT/$scene" "$EVAL_VIEW/$scene"; done
evaluate() {
  local phase="$1" prediction_root="$2"
  (
    cd "$ROOT/evaluation"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
      MPLCONFIGDIR="$RUNTIME_TMP/mpl_eval_$phase" \
      "$PYTHON" eval_ca1m.py --dataset ca1m --data_path "$EVAL_VIEW" \
        --pred_root "$prediction_root" --ap_iou_thresholds 0.15,0.25,0.5 \
        --num_workers 0 --cluster_sampling seed_fps --use_3d_nms --use_cls_nms \
        --per_class_proposal --gpu 0
  ) > "$LOG_ROOT/eval_${phase}.log" 2>&1
  chmod 0444 "$LOG_ROOT/eval_${phase}.log"
}
evaluate control "$CONTROL_ROOT"
evaluate active "$ACTIVE_ROOT"

echo "=== CA-1M Selective Boxer G0 control ==="
grep -E 'eval (mAP|APrec|ARecall):|mAP:' "$LOG_ROOT/eval_control.log" | tail -12
echo "=== CA-1M final base: G0 + CLIP gate + reliable Top-K=3 ==="
grep -E 'eval (mAP|APrec|ARecall):|mAP:' "$LOG_ROOT/eval_active.log" | tail -12
echo "Fixed10 same-run identity, paired difference audit, and paired evaluation complete: $REPORT_ROOT"
