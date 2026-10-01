#!/usr/bin/env bash
set -euo pipefail

# Isolated score-only validation of the train-authorized CA-1M-native B6 model.
# This runner cannot start until the canonical103 GT-free collection is sealed.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${1:---preflight}"
GPU="${2:-0}"
case "$MODE" in
    --preflight) PHASE="preflight" ;;
    --run) PHASE="run" ;;
    *) echo "Usage: $0 [--preflight|--run] [evaluation_gpu]" >&2; exit 2 ;;
esac
[[ "$#" -le 2 && "$GPU" =~ ^[0-9]+$ ]] \
    || { echo "Usage: $0 [--preflight|--run] [evaluation_gpu]" >&2; exit 2; }

PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
SOURCE_TAG="${BOXFUSION_CA1M_NATIVE_B6_PAIRED_SOURCE_TAG:-ca1m_c3_native_b6_observer_canonical103_v1}"
TAG="${BOXFUSION_CA1M_NATIVE_B6_PAIRED_TAG:-ca1m_native_b6_paired_canonical103_v1}"
TAG_SHA="$(printf '%s' "$TAG" | sha256sum | cut -c1-12)"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/ca1m_val_canonical103.txt"
FULL_VAL_LIST="$ROOT/evaluation/data_util/meta_data/ca1m_val_full107.txt"
DATA_ROOT="${BOXFUSION_CA1M_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
EVAL_VIEW="${BOXFUSION_CA1M_NATIVE_B6_EVAL_VIEW:-$ROOT/data/ca1m_eval_canonical103_v1}"
OFFICIAL_ROOT="${BOXFUSION_OFFICIAL_ROOT:-/data/ZhaoX/BoxFusion/upstream_clean/BoxFusion_shallow}"

COLLECTION_REPORT_ROOT="$ROOT/reports/ca1m_port/$SOURCE_TAG"
COLLECTION_MANIFEST="$COLLECTION_REPORT_ROOT/collection_manifest.json"
IDENTITY_AUDIT="$COLLECTION_REPORT_ROOT/identity_audit.json"
RECORD_COMPLETIONS="$COLLECTION_REPORT_ROOT/completion/cutr_record"
OBSERVER_COMPLETIONS="$COLLECTION_REPORT_ROOT/completion/g0_observer"
ANCHOR_ROOT="$ROOT/results/ca1m_port/${SOURCE_TAG}_same_run_anchor"
OBSERVER_ROOT="$ROOT/results/ca1m_port/$SOURCE_TAG"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/ca1m_port/$SOURCE_TAG/native_b6"

CHECKPOINT="$ROOT/models/ca1m_native_b6_iou_mlp_v1.npz"
CHECKPOINT_MANIFEST="$ROOT/models/ca1m_native_b6_iou_mlp_v1.manifest.json"
TRAIN_DATASET="$ROOT/datasets/ca1m_native_b6_train100_v1.npz"
TRAIN_DATASET_MANIFEST="$ROOT/datasets/ca1m_native_b6_train100_v1.manifest.json"
TRAIN_SUBSET_MANIFEST="$ROOT/manifests/ca1m_native_b6_train100_v1/subset_manifest.json"
TRAIN_COLLECTION_MANIFEST="$ROOT/reports/ca1m_native_b6_train100_v1/collection_manifest.json"

ACTIVE_ROOT="$ROOT/results/ca1m_native_b6_score/$TAG"
REPORT_ROOT="$ROOT/reports/ca1m_native_b6_score/$TAG"
ACTIVE_REPORT="$REPORT_ROOT/active.json"
PAIRED_REPORT="$REPORT_ROOT/paired_report.json"
LOG_ROOT="$ROOT/logs/ca1m_native_b6_score/$TAG/evaluation"
# Keep this deliberately short: the official evaluator uses a multiprocessing
# DataLoader whose AF_UNIX listener path is nested below TMPDIR.
TMP_ROOT="${BOXFUSION_RUNTIME_TMP_ROOT:-/tmp/bfc103b6-$TAG_SHA}"
TOOL="$ROOT/tools/evaluate_ca1m_native_b6_paired.py"
SCORE_RUNNER="$ROOT/scripts/run_ca1m_native_b6_score_canonical103.sh"

die() { echo "$*" >&2; exit 2; }
for path in "$PYTHON" "$SCENE_LIST" "$FULL_VAL_LIST" "$COLLECTION_MANIFEST" \
    "$IDENTITY_AUDIT" "$CHECKPOINT" "$CHECKPOINT_MANIFEST" "$TRAIN_DATASET" \
    "$TRAIN_DATASET_MANIFEST" "$TRAIN_SUBSET_MANIFEST" \
    "$TRAIN_COLLECTION_MANIFEST" "$TOOL" "$SCORE_RUNNER"; do
    [[ -f "$path" ]] || die "Missing required sealed input: $path"
done
for path in "$DATA_ROOT" "$EVAL_VIEW" "$OFFICIAL_ROOT" "$RECORD_COMPLETIONS" \
    "$OBSERVER_COMPLETIONS" "$ANCHOR_ROOT" "$OBSERVER_ROOT" "$DIAGNOSTICS_ROOT"; do
    [[ -d "$path" ]] || die "Missing required sealed root: $path"
done

common=(
    --scene-list "$SCENE_LIST"
    --collection-manifest "$COLLECTION_MANIFEST"
    --identity-audit "$IDENTITY_AUDIT"
    --record-completion-root "$RECORD_COMPLETIONS"
    --observer-completion-root "$OBSERVER_COMPLETIONS"
    --anchor-root "$ANCHOR_ROOT"
    --observer-root "$OBSERVER_ROOT"
    --diagnostics-root "$DIAGNOSTICS_ROOT"
    --checkpoint "$CHECKPOINT"
    --checkpoint-manifest "$CHECKPOINT_MANIFEST"
    --training-dataset "$TRAIN_DATASET"
    --training-dataset-manifest "$TRAIN_DATASET_MANIFEST"
    --training-subset-manifest "$TRAIN_SUBSET_MANIFEST"
    --training-collection-manifest "$TRAIN_COLLECTION_MANIFEST"
    --full-val-scene-list "$FULL_VAL_LIST"
    --official-root "$OFFICIAL_ROOT"
    --eval-view "$EVAL_VIEW"
    --data-root "$DATA_ROOT"
    --python "$PYTHON"
)

echo "CA-1M native-B6 canonical103 paired ${PHASE}"
echo "  sealed source: $SOURCE_TAG"
echo "  active namespace: $TAG"
echo "  official upstream: b2e0219a7284249bad4a4a8925066839fe2fa33b"
echo "  metric: official box3d_iou_v2 world enclosing-AABB"
"$PYTHON" "$TOOL" preflight "${common[@]}"
if [[ "$PHASE" == "preflight" ]]; then
    echo "Paired preflight passed; no active prediction, GT, or evaluator was opened."
    exit 0
fi

for path in "$ACTIVE_ROOT" "$ACTIVE_REPORT" "$PAIRED_REPORT" "$LOG_ROOT" "$TMP_ROOT"; do
    [[ ! -e "$path" && ! -L "$path" ]] || die "Refusing existing formal output: $path"
done
echo "  interruption policy: create-only; use a new paired TAG to recover"

BOXFUSION_CA1M_NATIVE_B6_SCORE_SCENE_LIST="$SCENE_LIST" \
BOXFUSION_CA1M_NATIVE_B6_SCORE_SOURCE_TAG="$SOURCE_TAG" \
BOXFUSION_CA1M_NATIVE_B6_SCORE_CHECKPOINT="$CHECKPOINT" \
BOXFUSION_CA1M_NATIVE_B6_SCORE_CHECKPOINT_MANIFEST="$CHECKPOINT_MANIFEST" \
BOXFUSION_CA1M_NATIVE_B6_SCORE_TAG="$TAG" \
BOXFUSION_CA1M_NATIVE_B6_SCORE_ACTIVE_ROOT="$ACTIVE_ROOT" \
BOXFUSION_CA1M_NATIVE_B6_SCORE_REPORT="$ACTIVE_REPORT" \
BOXFUSION_PYTHON="$PYTHON" \
    bash "$SCORE_RUNNER" --active

"$PYTHON" "$TOOL" evaluate "${common[@]}" \
    --active-root "$ACTIVE_ROOT" --active-report "$ACTIVE_REPORT" \
    --log-root "$LOG_ROOT" --tmp-root "$TMP_ROOT" \
    --output "$PAIRED_REPORT" --gpu "$GPU"

echo "CA-1M native-B6 canonical103 paired evaluation completed: $PAIRED_REPORT"
