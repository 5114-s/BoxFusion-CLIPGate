#!/usr/bin/env bash
set -euo pipefail

# Query the sealed CA-1M final-base train100 predictions with native depth
# evidence.  This stage is CPU/offline: it never reruns CuTR, Boxer, CLIP, or
# BoxFusion and never reads train GT or any official-validation artifact.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="preflight"
WORKERS="${BOXFUSION_CA1M_NATIVE_B6_OFFLINE_WORKERS:-2}"
case "${1:-}" in
  --preflight) MODE="preflight"; WORKERS="${2:-$WORKERS}" ;;
  --run) MODE="run"; WORKERS="${2:-$WORKERS}" ;;
  "") ;;
  *) echo "Usage: $0 [--preflight|--run] [workers]" >&2; exit 2 ;;
esac
[[ "$#" -le 2 ]] || { echo "Usage: $0 [--preflight|--run] [workers]" >&2; exit 2; }
[[ "$WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "workers must be positive" >&2; exit 2; }
(( WORKERS <= 16 )) || { echo "workers must not exceed 16" >&2; exit 2; }

PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
TAG="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_TAG:-ca1m_native_b6_final_base_train100_v2}"
SUBSET_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_MANIFEST:-$ROOT/manifests/ca1m_native_b6_train100_v1/subset_manifest.json}"
SCENE_LIST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENE_LIST:-$ROOT/manifests/ca1m_native_b6_train100_v1/scene_ids.txt}"
DATA_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1}"
FINAL_BASE_CONFIG="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_CONFIG:-$ROOT/config/ca1m_native_final_base_train100_v1.yaml}"
OFFLINE_CONFIG="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_OFFLINE_CONFIG:-$ROOT/config/ca1m_native_b6_final_base_train100_v2_offline.yaml}"
FINAL_BASE_TAG="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_TAG:-ca1m_native_final_base_train100_v1}"
FINAL_BASE_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_ROOT:-$ROOT/results/$FINAL_BASE_TAG/final_base}"
FINAL_BASE_MANIFEST="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_MANIFEST:-$ROOT/reports/$FINAL_BASE_TAG/collection_manifest.json}"
FIXED10_TAG="${BOXFUSION_CA1M_FINAL_BASE_TAG:-ca1m_c4_final_base_g0_clip_topk3_fixed10_v1}"
FIXED10_AUDIT="${BOXFUSION_CA1M_FINAL_BASE_FIXED10_AUDIT:-$ROOT/reports/ca1m_port/$FIXED10_TAG/identity_and_paired_audit.json}"
FIXED10_EVAL="${BOXFUSION_CA1M_FINAL_BASE_FIXED10_EVAL:-$ROOT/logs/ca1m_port/$FIXED10_TAG/eval_active.log}"
FIXED10_PAIRED_REPORT="${BOXFUSION_CA1M_FINAL_BASE_PAIRED_REPORT:-$ROOT/reports/ca1m_port/$FIXED10_TAG/paired_eval_report.json}"

DIAGNOSTICS_ROOT="$ROOT/diagnostics/$TAG"
NATIVE_ROOT="$DIAGNOSTICS_ROOT/native_b6"
REPORT_ROOT="$ROOT/reports/$TAG"
RECEIPT_ROOT="$REPORT_ROOT/offline_receipts"
COMPLETION_ROOT="$REPORT_ROOT/completion/offline_native_b6"
RUNTIME_TMP="${BOXFUSION_RUNTIME_TMP_ROOT:-/tmp/bfc-$TAG}"
LOCK_ROOT="${BOXFUSION_RUN_LOCK_ROOT:-/tmp/boxfusion_ca1m_runlocks}"
LOCK_DIR="$LOCK_ROOT/$TAG.lock"

COLLECTOR="$ROOT/tools/collect_ca1m_native_b6_final_base_offline.py"
FINALIZER="$ROOT/tools/finalize_ca1m_native_b6_final_base_v2.py"
EXPECTED_MANIFEST_SHA="29a32e92cfece667e9fef4389227eacba2b96c55737569fa6219ca7ab527fd23"
EXPECTED_SCENE_LIST_SHA="35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"

die() { echo "$*" >&2; exit 2; }
file_sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
  local path="$1" expected="$2" actual
  [[ -f "$path" && ! -L "$path" ]] || die "Missing regular frozen input: $path"
  actual="$(file_sha "$path")"
  [[ "$actual" == "$expected" ]] || die "SHA256 mismatch: $path ($actual != $expected)"
}

[[ -x "$PYTHON" ]] || die "Python is not executable: $PYTHON"
for path in "$SUBSET_MANIFEST" "$SCENE_LIST" "$FINAL_BASE_CONFIG" \
  "$OFFLINE_CONFIG" "$FINAL_BASE_MANIFEST" "$FIXED10_AUDIT" \
  "$FIXED10_EVAL" "$FIXED10_PAIRED_REPORT" "$COLLECTOR" "$FINALIZER"; do
  [[ -f "$path" && ! -L "$path" ]] || die "Missing regular prerequisite: $path"
done
require_sha "$SUBSET_MANIFEST" "$EXPECTED_MANIFEST_SHA"
require_sha "$SCENE_LIST" "$EXPECTED_SCENE_LIST_SHA"
FIXED10_PAIRED_REPORT_SHA="$(file_sha "$FIXED10_PAIRED_REPORT")"

"$PYTHON" "$FINALIZER" contract --final-base-config "$FINAL_BASE_CONFIG" \
  --offline-config "$OFFLINE_CONFIG" --paired-report "$FIXED10_PAIRED_REPORT" \
  >/dev/null
"$PYTHON" -c 'import json,sys; from pathlib import Path; p=json.loads(Path(sys.argv[1]).read_text()); assert p.get("schema")=="boxfusion.ca1m_final_base_identity_audit.v1" and p.get("split")=="validation_fixed10" and p.get("scene_count")==10 and p.get("clip_appearance_gate_active") is True and p.get("reliable_view_top_k")==3 and p.get("ground_truth_access") is False and p.get("training_invoked") is False' "$FIXED10_AUDIT"
grep -q 'eval mAP:' "$FIXED10_EVAL" || die "Fixed10 active evaluation is incomplete: $FIXED10_EVAL"

mapfile -t SCENES < <(sed -e 's/[[:space:]]*$//' -e '/^$/d' "$SCENE_LIST")
[[ "${#SCENES[@]}" == 100 ]] || die "offline native-B6 v2 requires 100 train scenes"
[[ "$(printf '%s\n' "${SCENES[@]}" | sort -u | wc -l)" == 100 ]] \
  || die "frozen train scene list contains duplicates"
[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "Unsafe CA train root: $DATA_ROOT"
[[ -d "$FINAL_BASE_ROOT" && ! -L "$FINAL_BASE_ROOT" ]] \
  || die "Sealed final-base train100 root is not ready: $FINAL_BASE_ROOT"
"$PYTHON" "$FINALIZER" source --scene-list "$SCENE_LIST" --expected-scenes 100 \
  --final-base-root "$FINAL_BASE_ROOT" --final-base-manifest "$FINAL_BASE_MANIFEST" \
  --paired-report "$FIXED10_PAIRED_REPORT" \
  >/dev/null

if [[ "$MODE" == "preflight" ]]; then
  for path in "$DIAGNOSTICS_ROOT" "$REPORT_ROOT" "$RUNTIME_TMP"; do
    [[ ! -e "$path" ]] || die "Refusing existing offline native-B6 v2 namespace: $path"
  done
fi

# Decode the exact selected depth/K/pose lineage for every scene before any
# output namespace is created.  This is read-only and does not invoke B6.
for scene in "${SCENES[@]}"; do
  "$PYTHON" "$COLLECTOR" --mode preflight --scene "$scene" \
    --config "$OFFLINE_CONFIG" --data-root "$DATA_ROOT" \
    --final-base-root "$FINAL_BASE_ROOT" \
    --final-base-manifest "$FINAL_BASE_MANIFEST" \
    --diagnostics-root "$NATIVE_ROOT" \
    --receipt "$RECEIPT_ROOT/${scene}.json" >/dev/null
done

if [[ "$MODE" == "preflight" ]]; then
  echo "CA-1M final-base native-B6 v2 offline preflight passed"
  echo "  geometry authority: sealed train100 G0 + CLIP + reliable Top-K=3"
  echo "  evidence: direct native 14-D depth observer Top-K=5"
  echo "  BoxFusion/CuTR/Boxer/CLIP replay: false"
  echo "  RGB pixels/train GT/official validation/evaluator access: false"
  echo "  run requires BOXFUSION_CA1M_FINAL_BASE_FIXED10_ACCEPTED=1"
  exit 0
fi

[[ "${BOXFUSION_CA1M_FINAL_BASE_FIXED10_ACCEPTED:-0}" == 1 ]] \
  || die "Set BOXFUSION_CA1M_FINAL_BASE_FIXED10_ACCEPTED=1 after fixed10 acceptance"
mkdir -p "$LOCK_ROOT"
mkdir "$LOCK_DIR" || die "Another process owns offline native-B6 v2 lock: $LOCK_DIR"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
mkdir -p "$NATIVE_ROOT" "$RECEIPT_ROOT" "$COMPLETION_ROOT" "$RUNTIME_TMP"

CODE_SOURCES=(
  boxfusion/ca1m_native_b6_observer.py
  boxfusion/tr3d_r2_geometry.py
  boxfusion/tr3d_r4_smov_observer.py
  boxfusion/orientation.py
  config/ca1m_native_b6_final_base_train100_v2_offline.yaml
  tools/collect_ca1m_native_b6_final_base_offline.py
  tools/finalize_ca1m_native_b6_final_base_v2.py
  scripts/collect_ca1m_native_b6_final_base_train100_v2.sh
)
CODE_MANIFEST="$REPORT_ROOT/code_manifest.tsv"
CODE_CURRENT="$RUNTIME_TMP/code_current.$BASHPID.tsv"
[[ ! -e "$CODE_CURRENT" ]] || die "Refusing existing runtime code manifest: $CODE_CURRENT"
for source in "${CODE_SOURCES[@]}"; do
  printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_CURRENT"
done
if [[ -e "$CODE_MANIFEST" ]]; then
  [[ -f "$CODE_MANIFEST" && ! -L "$CODE_MANIFEST" ]] \
    || die "Unsafe existing v2 code manifest: $CODE_MANIFEST"
  cmp -s "$CODE_CURRENT" "$CODE_MANIFEST" \
    || die "Offline native-B6 v2 code differs from resumed collection"
else
  cp "$CODE_CURRENT" "$CODE_MANIFEST"
  chmod 0444 "$CODE_MANIFEST"
fi
CODE_FINGERPRINT="$(file_sha "$CODE_MANIFEST")"
"$PYTHON" "$FINALIZER" contract --final-base-config "$FINAL_BASE_CONFIG" \
  --offline-config "$OFFLINE_CONFIG" --paired-report "$FIXED10_PAIRED_REPORT" \
  --output "$REPORT_ROOT/contract.json" >/dev/null
"$PYTHON" "$FINALIZER" source --scene-list "$SCENE_LIST" --expected-scenes 100 \
  --final-base-root "$FINAL_BASE_ROOT" --final-base-manifest "$FINAL_BASE_MANIFEST" \
  --paired-report "$FIXED10_PAIRED_REPORT" \
  --output "$REPORT_ROOT/source_final_base_audit.json" >/dev/null

run_scene() {
  local scene="$1" diagnostic receipt completion
  diagnostic="$NATIVE_ROOT/${scene}_ca1m_native_b6.npz"
  receipt="$RECEIPT_ROOT/${scene}.json"
  completion="$COMPLETION_ROOT/${scene}.json"
  if [[ -e "$receipt" ]]; then
    [[ -f "$receipt" && ! -L "$receipt" && -f "$diagnostic" && ! -L "$diagnostic" ]] \
      || die "$scene: receipt exists without a safe diagnostic; refusing recovery"
  else
    # The collector can recover the only recognized partial state: a regular
    # diagnostic without a receipt.  It recomputes every semantic NPZ field
    # before creating the missing receipt.  No artifact is deleted.
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0 \
      "$PYTHON" "$COLLECTOR" --mode run --scene "$scene" \
        --config "$OFFLINE_CONFIG" --data-root "$DATA_ROOT" \
        --final-base-root "$FINAL_BASE_ROOT" \
        --final-base-manifest "$FINAL_BASE_MANIFEST" \
        --diagnostics-root "$NATIVE_ROOT" --receipt "$receipt" >/dev/null
  fi
  "$PYTHON" "$FINALIZER" scene --scene "$scene" \
    --final-base-root "$FINAL_BASE_ROOT" --final-base-manifest "$FINAL_BASE_MANIFEST" \
    --diagnostic "$diagnostic" --offline-receipt "$receipt" \
    --output "$completion" >/dev/null
  echo "[$(date '+%F %T')] offline final-base native-B6 evidence complete: $scene"
}

pids=()
failures=0
for shard in $(seq 0 $((WORKERS - 1))); do
  (
    for index in "${!SCENES[@]}"; do
      (( index % WORKERS == shard )) || continue
      run_scene "${SCENES[$index]}"
    done
  ) &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || failures=1; done
(( failures == 0 )) || die "At least one offline native-B6 worker failed"

"$PYTHON" "$FINALIZER" collection --subset-manifest "$SUBSET_MANIFEST" \
  --expected-scenes 100 --completion-root "$COMPLETION_ROOT" \
  --final-base-root "$FINAL_BASE_ROOT" --final-base-manifest "$FINAL_BASE_MANIFEST" \
  --paired-report "$FIXED10_PAIRED_REPORT" \
  --output "$REPORT_ROOT/collection_manifest.json" >/dev/null
CODE_AFTER="$RUNTIME_TMP/code_after.$BASHPID.tsv"
[[ ! -e "$CODE_AFTER" ]] || die "Refusing existing runtime code audit: $CODE_AFTER"
for source in "${CODE_SOURCES[@]}"; do
  printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_AFTER"
done
[[ "$(file_sha "$CODE_AFTER")" == "$CODE_FINGERPRINT" ]] \
  || die "Offline native-B6 v2 code changed during collection"
[[ "$(file_sha "$FIXED10_PAIRED_REPORT")" == "$FIXED10_PAIRED_REPORT_SHA" ]] \
  || die "Authoritative fixed10 paired report changed during collection"
echo "CA-1M final-base native-B6 v2 train100 offline collection complete:"
echo "  $REPORT_ROOT/collection_manifest.json"
echo "Next: build the isolated v2 dataset and train the CA-only B6 head."
