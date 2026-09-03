#!/usr/bin/env bash
set -euo pipefail

# Train-only CA-1M native-B6 dataset join and quality-head gate.
# Default is a non-writing training preflight.  Dataset materialization is an
# explicit stage; model fitting additionally requires --train.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
SUBSET_ROOT="${BOXFUSION_CA1M_NATIVE_B6_SUBSET_ROOT:-$ROOT/manifests/ca1m_native_b6_train100_v1}"
SCENE_LIST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENES:-$SUBSET_ROOT/scene_ids.txt}"
SUBSET_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_SUBSET_MANIFEST:-$SUBSET_ROOT/subset_manifest.json}"
VAL_URL_LIST="${BOXFUSION_CA1M_VAL_URL_LIST:-/data/ZhaoX/BoxFusion/data/val.txt}"
OBSERVER_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_OBSERVER_ROOT:-$ROOT/diagnostics/ca1m_native_b6_train100_v1/native_b6}"
PREDICTION_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_PRED_ROOT:-$ROOT/results/ca1m_native_b6_train100_v1/g0_observer}"
GT_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_GT_ROOT:-/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1}"
COLLECTION_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_COLLECTION_MANIFEST:-$ROOT/reports/ca1m_native_b6_train100_v1/collection_manifest.json}"
OBSERVER_COMPLETION_ROOT="${BOXFUSION_CA1M_NATIVE_B6_OBSERVER_COMPLETION_ROOT:-$ROOT/reports/ca1m_native_b6_train100_v1/completion/g0_observer}"
DATASET="${BOXFUSION_CA1M_NATIVE_B6_DATASET:-$ROOT/datasets/ca1m_native_b6_train100_v1.npz}"
DATASET_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_DATASET_MANIFEST:-$ROOT/datasets/ca1m_native_b6_train100_v1.manifest.json}"
CHECKPOINT="${BOXFUSION_CA1M_NATIVE_B6_CHECKPOINT:-$ROOT/models/ca1m_native_b6_iou_mlp_v1.npz}"
CHECKPOINT_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_CHECKPOINT_MANIFEST:-$ROOT/models/ca1m_native_b6_iou_mlp_v1.manifest.json}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/train_ca1m_native_b6_quality.sh --build-dataset
  bash scripts/train_ca1m_native_b6_quality.sh --preflight
  bash scripts/train_ca1m_native_b6_quality.sh --train

--build-dataset joins frozen train-only observer/G0/derived-GT artifacts.
--preflight (default) audits an existing joined dataset without fitting.
--train fits five OOF models, gates only on train OOF/dev, and writes the
fold-0 checkpoint trained on folds 1--4.  Exit status 3 means the gate failed;
the checkpoint remains observer-only and activation_authorized=false.
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

if [[ "$mode" == "--build-dataset" ]]; then
    "$PYTHON" "$ROOT/tools/build_ca1m_native_b6_dataset.py" \
        --observer-root "$OBSERVER_ROOT" \
        --prediction-root "$PREDICTION_ROOT" \
        --gt-root "$GT_ROOT" \
        --scene-list "$SCENE_LIST" \
        --subset-manifest "$SUBSET_MANIFEST" \
        --collection-manifest "$COLLECTION_MANIFEST" \
        --observer-completion-root "$OBSERVER_COMPLETION_ROOT" \
        --val-url-list "$VAL_URL_LIST" \
        --output "$DATASET" \
        --manifest-output "$DATASET_MANIFEST"
    exit 0
fi

arguments=(
    "$PYTHON" "$ROOT/tools/train_ca1m_native_b6_quality.py"
    --dataset "$DATASET"
    --dataset-manifest "$DATASET_MANIFEST"
)
if [[ "$mode" == "--preflight" ]]; then
    "${arguments[@]}" --preflight
    exit 0
fi

"${arguments[@]}" --train \
    --output "$CHECKPOINT" \
    --manifest-output "$CHECKPOINT_MANIFEST" \
    --epochs "${BOXFUSION_CA1M_NATIVE_B6_EPOCHS:-400}" \
    --learning-rate "${BOXFUSION_CA1M_NATIVE_B6_LR:-0.001}" \
    --l2-weight "${BOXFUSION_CA1M_NATIVE_B6_WEIGHT_DECAY:-0.0001}" \
    --hidden-dims "${BOXFUSION_CA1M_NATIVE_B6_HIDDEN_DIMS:-64,32}" \
    --detector-blend "${BOXFUSION_CA1M_NATIVE_B6_DETECTOR_BLEND:-0.40}" \
    --seed "${BOXFUSION_CA1M_NATIVE_B6_SEED:-1337}"
