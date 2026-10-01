#!/usr/bin/env bash
set -euo pipefail

# Build and train the CA-only native-B6 head for the final-base anchor.
# Default --preflight is read-only.  Dataset creation and fitting are separate,
# explicit, create-only stages.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
TAG="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_TAG:-ca1m_native_b6_final_base_train100_v2}"
SUBSET_ROOT="${BOXFUSION_CA1M_NATIVE_B6_SUBSET_ROOT:-$ROOT/manifests/ca1m_native_b6_train100_v1}"
SCENE_LIST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENES:-$SUBSET_ROOT/scene_ids.txt}"
SUBSET_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_SUBSET_MANIFEST:-$SUBSET_ROOT/subset_manifest.json}"
VAL_URL_LIST="${BOXFUSION_CA1M_VAL_URL_LIST:-/data/ZhaoX/BoxFusion/data/val.txt}"
OBSERVER_ROOT="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_OBSERVER_ROOT:-$ROOT/diagnostics/$TAG/native_b6}"
FINAL_BASE_TAG="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_TAG:-ca1m_native_final_base_train100_v1}"
PREDICTION_ROOT="${BOXFUSION_CA1M_NATIVE_FINAL_BASE_ROOT:-$ROOT/results/$FINAL_BASE_TAG/final_base}"
GT_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_GT_ROOT:-/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1}"
COLLECTION_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_COLLECTION_MANIFEST:-$ROOT/reports/$TAG/collection_manifest.json}"
OBSERVER_COMPLETION_ROOT="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_COMPLETION_ROOT:-$ROOT/reports/$TAG/completion/offline_native_b6}"
DATASET="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_DATASET:-$ROOT/datasets/ca1m_native_b6_final_base_train100_v2.npz}"
DATASET_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_DATASET_MANIFEST:-$ROOT/datasets/ca1m_native_b6_final_base_train100_v2.manifest.json}"
CHECKPOINT="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_CHECKPOINT:-$ROOT/models/ca1m_native_b6_final_base_iou_mlp_v2.npz}"
CHECKPOINT_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_CHECKPOINT_MANIFEST:-$ROOT/models/ca1m_native_b6_final_base_iou_mlp_v2.manifest.json}"
OOF_ROW_SCORES="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_OOF_ROW_SCORES:-$ROOT/models/ca1m_native_b6_final_base_oof_row_scores_v2.npz}"
OOF_ROW_SCORES_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_FINAL_BASE_OOF_ROW_SCORES_MANIFEST:-$ROOT/models/ca1m_native_b6_final_base_oof_row_scores_v2.manifest.json}"
SPLIT_NAMESPACE="boxfusion.ca1m-native-b6.scene-folds.v1"
V2_COLLECTION_SCHEMA="boxfusion.ca1m_native_b6_final_base_train_collection.v2"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --build-dataset
  bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --preflight
  bash scripts/train_ca1m_native_b6_final_base_quality_v2.sh --train

--build-dataset joins only sealed final-base boxes plus offline v2 evidence with CA
train-derived GT.  --train preserves the established deterministic five-fold
CA protocol: deployable model on folds 1--4, untouched fold0 development, and
all-fold OOF gating.  Official CA validation GT/predictions are never read.
The v2 training action also seals a row-identical all-fold OOF score sidecar;
later stacked gates must use it instead of the deployable fold-0 model scores.
Exit 3 means the train-only activation gate failed; the checkpoint remains
observer-only (activation_authorized=false).
EOF
}

mode="${1:---preflight}"
[[ "$#" -le 1 ]] || { usage >&2; exit 2; }
case "$mode" in
  --build-dataset|--preflight|--train) ;;
  --help|-h) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }

verify_v2_collection() {
  [[ -f "$COLLECTION_MANIFEST" && ! -L "$COLLECTION_MANIFEST" ]] \
    || { echo "Missing sealed v2 collection: $COLLECTION_MANIFEST" >&2; exit 2; }
  "$PYTHON" -c 'import json,sys; from pathlib import Path; p=json.loads(Path(sys.argv[1]).read_text()); assert p.get("schema")==sys.argv[2] and p.get("complete") is True and p.get("train_only") is True and p.get("evaluation_invoked") is False and p.get("validation_ground_truth_access") is False and p.get("validation_prediction_access") is False and p.get("geometry_authority")=="sealed_final_base_prediction" and p.get("offline_direct_observer") is True and p.get("cross_run_boxfusion_replay_invoked") is False and p.get("cross_run_exact_identity_required") is False and p.get("old_native_b6_diagnostics_reused") is False and p.get("old_native_b6_checkpoint_reused") is False' \
    "$COLLECTION_MANIFEST" "$V2_COLLECTION_SCHEMA"
}

if [[ "$mode" == "--build-dataset" ]]; then
  verify_v2_collection
  "$PYTHON" "$ROOT/tools/build_ca1m_native_b6_dataset.py" \
    --observer-root "$OBSERVER_ROOT" --prediction-root "$PREDICTION_ROOT" \
    --gt-root "$GT_ROOT" --scene-list "$SCENE_LIST" \
    --subset-manifest "$SUBSET_MANIFEST" \
    --collection-manifest "$COLLECTION_MANIFEST" \
    --observer-completion-root "$OBSERVER_COMPLETION_ROOT" \
    --val-url-list "$VAL_URL_LIST" --split-namespace "$SPLIT_NAMESPACE" \
    --output "$DATASET" --manifest-output "$DATASET_MANIFEST"
  exit 0
fi

[[ -f "$DATASET_MANIFEST" && ! -L "$DATASET_MANIFEST" ]] \
  || { echo "Missing isolated v2 dataset manifest: $DATASET_MANIFEST" >&2; exit 2; }
"$PYTHON" -c 'import json,sys; from pathlib import Path; p=json.loads(Path(sys.argv[1]).read_text()); c=p.get("train_collection") or {}; assert c.get("schema")==sys.argv[2] and c.get("geometry_authority")=="sealed_final_base_prediction" and c.get("offline_direct_observer") is True and c.get("cross_run_boxfusion_replay_invoked") is False and c.get("cross_run_exact_identity_required") is False and c.get("old_native_b6_diagnostics_reused") is False and c.get("old_native_b6_checkpoint_reused") is False and (p.get("split") or {}).get("namespace")==sys.argv[3]' \
  "$DATASET_MANIFEST" "$V2_COLLECTION_SCHEMA" "$SPLIT_NAMESPACE"

arguments=(
  "$PYTHON" "$ROOT/tools/train_ca1m_native_b6_quality.py"
  --dataset "$DATASET" --dataset-manifest "$DATASET_MANIFEST"
)
if [[ "$mode" == "--preflight" ]]; then
  "${arguments[@]}" --preflight
  exit 0
fi

[[ "$CHECKPOINT" != "$ROOT/models/ca1m_native_b6_iou_mlp_v1.npz" ]] \
  || { echo "Refusing legacy native-B6 checkpoint path" >&2; exit 2; }
"${arguments[@]}" --train --output "$CHECKPOINT" \
  --manifest-output "$CHECKPOINT_MANIFEST" \
  --oof-output "$OOF_ROW_SCORES" \
  --oof-manifest-output "$OOF_ROW_SCORES_MANIFEST" \
  --epochs "${BOXFUSION_CA1M_NATIVE_B6_EPOCHS:-400}" \
  --learning-rate "${BOXFUSION_CA1M_NATIVE_B6_LR:-0.001}" \
  --l2-weight "${BOXFUSION_CA1M_NATIVE_B6_WEIGHT_DECAY:-0.0001}" \
  --hidden-dims "${BOXFUSION_CA1M_NATIVE_B6_HIDDEN_DIMS:-64,32}" \
  --ranking-weights "${BOXFUSION_CA1M_NATIVE_B6_RANKING_WEIGHTS:-0.10,0.20,0.30,0.40}" \
  --detector-blend "${BOXFUSION_CA1M_NATIVE_B6_DETECTOR_BLEND:-0.40}" \
  --seed "${BOXFUSION_CA1M_NATIVE_B6_SEED:-1337}"
