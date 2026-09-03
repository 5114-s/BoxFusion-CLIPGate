#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion2
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_graw_e2_preflight3.txt"
REPORT_ROOT="$ROOT/logs/scannet_puf_gclean_formal3_score05"
MPL_ROOT="$REPORT_ROOT/mplconfig"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a non-negative integer" >&2
  exit 2
fi
for required in \
  "$ROOT/results/scannet_graw_e2_replay1_score05" \
  "$ROOT/results/scannet_puf_gclean_shadow_replay_score05" \
  "$ROOT/results/scannet_puf_gclean_counterfactual_score05_preflight3"; do
  if [[ ! -d "$required" ]]; then
    echo "Missing prediction root: $required" >&2
    exit 1
  fi
done
if [[ -e "$REPORT_ROOT/t05_constant.log" ]]; then
  echo "Formal PUF preflight report already exists: $REPORT_ROOT" >&2
  exit 1
fi

EVAL_ROOT=$(mktemp -d /tmp/boxfusion_puf3_eval.XXXXXX)
trap 'echo "Temporary evaluator root retained at: $EVAL_ROOT"' EXIT
mkdir -p "$REPORT_ROOT" "$MPL_ROOT"
cp -a "$ROOT/upstream_clean/BoxFusion_shallow/evaluation" "$EVAL_ROOT/constant"
cp -a "$ROOT/upstream_clean/BoxFusion_shallow/evaluation" "$EVAL_ROOT/native"
cp "$ROOT/evaluation/eval_scannet.py" "$EVAL_ROOT/native/eval_scannet.py"
cp "$SCENE_LIST" "$EVAL_ROOT/constant/data_util/meta_data/scannetv2_val.txt"
cp "$SCENE_LIST" "$EVAL_ROOT/native/data_util/meta_data/scannetv2_val.txt"
sha256sum \
  "$EVAL_ROOT/constant/eval_scannet.py" \
  "$EVAL_ROOT/native/eval_scannet.py" \
  "$SCENE_LIST" | tee "$REPORT_ROOT/input_sha256.txt"

evaluate_one() {
  local mode arm prediction_root workdir log_path
  local -a extra=()
  mode="$1"
  arm="$2"
  prediction_root="$3"
  workdir="$EVAL_ROOT/$mode"
  log_path="$REPORT_ROOT/${arm}_${mode}.log"
  if [[ "$mode" == native ]]; then
    extra=(--num_workers 0)
  fi
  mkdir -p "$MPL_ROOT/${arm}_${mode}"
  (
    cd "$workdir"
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONNOUSERSITE=1 \
    MPLCONFIGDIR="$MPL_ROOT/${arm}_${mode}" \
    LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    "$PYTHON" eval_scannet.py \
      --dataset scannet \
      --data_path /extra/ZhaoX/scannet_data/scans \
      --dump_dir eval_scannet \
      --num_point 40000 \
      --cluster_sampling seed_fps \
      --ap_iou_thresholds 0.15,0.25,0.5 \
      --use_3d_nms \
      --use_cls_nms \
      --per_class_proposal \
      --gpu "$GPU" \
      "${extra[@]}" \
      --pred_root "$prediction_root"
  ) 2>&1 | tee "$log_path"
  if ! grep -q 'kept 3 scans out of 3' "$log_path"; then
    echo "Evaluator did not use the sealed three-scene list: $log_path" >&2
    exit 1
  fi
  if [[ $(grep -c '^eval mAP:' "$log_path") -ne 3 ]]; then
    echo "Evaluator did not emit exactly three AP values: $log_path" >&2
    exit 1
  fi
}

evaluate_one constant t05 \
  "$ROOT/results/scannet_graw_e2_replay1_score05"
evaluate_one constant puf_native \
  "$ROOT/results/scannet_puf_gclean_shadow_replay_score05"
evaluate_one constant puf_counterfactual \
  "$ROOT/results/scannet_puf_gclean_counterfactual_score05_preflight3"
evaluate_one native t05 \
  "$ROOT/results/scannet_graw_e2_replay1_score05"
evaluate_one native puf_native \
  "$ROOT/results/scannet_puf_gclean_shadow_replay_score05"
evaluate_one native puf_counterfactual \
  "$ROOT/results/scannet_puf_gclean_counterfactual_score05_preflight3"

for log_path in \
  "$REPORT_ROOT"/{t05,puf_native,puf_counterfactual}_{constant,native}.log; do
  awk '
    /iou_thresh:/ {iou=int($3*100+0.5)}
    /^eval mAP:/ {ap[iou]=$3*100}
    /^eval APrec:/ {precision[iou]=$3*100}
    /^eval ARecall:/ {recall[iou]=$3*100}
    END {
      printf "%s\tAP15=%.4f\tAP25=%.4f\tAP50=%.4f\tR15=%.4f\tR25=%.4f\tR50=%.4f\n", \
        FILENAME, ap[15], ap[25], ap[50], recall[15], recall[25], recall[50]
    }
  ' "$log_path"
done | tee "$REPORT_ROOT/metrics.tsv"

