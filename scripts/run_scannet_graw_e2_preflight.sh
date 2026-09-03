#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/run_scannet_graw_e2_preflight.sh record|replay1|replay2 [GPU]
ARM="${1:-record}"
GPU="${2:-0}"

ROOT=/data/ZhaoX/BoxFusion
PYTHON=/home/admin1/miniconda3/envs/boxfusion2/bin/python
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion2
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_graw_e2_preflight3.txt"
CACHE_ROOT="$ROOT/cache/cutr_postfilter_v3"
NAMESPACE=scannet-graw-e2-score05-preflight3-v3-r1
INDEX_PATH="$CACHE_ROOT/$NAMESPACE/index.json"

case "$ARM" in
  record)
    CONFIG="$ROOT/config/scannet_graw_e2_record_score05.yaml"
    PRED_ROOT="$ROOT/results/scannet_graw_e2_record_score05"
    FINGERPRINT_ENV=BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT
    ;;
  replay1)
    CONFIG="$ROOT/config/scannet_graw_e2_replay1_score05.yaml"
    PRED_ROOT="$ROOT/results/scannet_graw_e2_replay1_score05"
    FINGERPRINT_ENV=BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT
    ;;
  replay2)
    CONFIG="$ROOT/config/scannet_graw_e2_replay2_score05.yaml"
    PRED_ROOT="$ROOT/results/scannet_graw_e2_replay2_score05"
    FINGERPRINT_ENV=BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT
    ;;
  *)
    echo "ARM must be record, replay1, or replay2" >&2
    exit 2
    ;;
esac

LOG_ROOT="$ROOT/logs/scannet_graw_e2_${ARM}_score05"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
mkdir -p "$PRED_ROOT" "$SCENE_LOG_ROOT" "$LOG_ROOT/mplconfig"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
  echo "Another $ARM preflight process holds $LOG_ROOT/run.lock" >&2
  exit 1
fi

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
  echo "GPU must be a non-negative integer" >&2
  exit 2
fi

if [[ "$ARM" == record ]]; then
  if [[ -e "$CACHE_ROOT/$NAMESPACE" ]]; then
    echo "Create-only cache namespace already exists: $CACHE_ROOT/$NAMESPACE" >&2
    exit 1
  fi
  FINGERPRINT=$("$PYTHON" "$ROOT/tools/proposal_cache_fingerprint.py" compute \
    --entry "cutr_checkpoint=$ROOT/models/cutr_rgbd.pth" \
    --entry "clip_checkpoint=$ROOT/models/open_clip_pytorch_model.bin" \
    --entry "class_features=$ROOT/data/class_features.pt" \
    --entry "pst=$ROOT/data/pst_1024_0.tiff" \
    --entry "record_config=$CONFIG" \
    --entry "scene_list=$SCENE_LIST" \
    --entry "demo=$ROOT/demo.py" \
    --entry "proposal_cache=$ROOT/boxfusion/proposal_cache.py" \
    --entry "instances=$ROOT/boxfusion/instances.py" \
    --entry "boxes=$ROOT/boxfusion/boxes.py" \
    --entry "preprocessor=$ROOT/boxfusion/preprocessor.py" \
    --entry "capture_stream=$ROOT/boxfusion/capture_stream.py" \
    --entry "cubify_transformer=$ROOT/boxfusion/cubify_transformer.py" \
    --entry "box_manager=$ROOT/boxfusion/box_manager.py" \
    --entry "box_fusion=$ROOT/boxfusion/box_fusion.py" \
    --entry "reliable_views=$ROOT/boxfusion/reliable_views.py" \
    --entry "utils=$ROOT/tools/utils.py" \
    --entry "runner=$ROOT/scripts/run_scannet_graw_e2_preflight.sh")
else
  if [[ ! -f "$INDEX_PATH" ]]; then
    echo "Sealed cache index is missing: $INDEX_PATH" >&2
    exit 1
  fi
  FINGERPRINT=$("$PYTHON" "$ROOT/tools/proposal_cache_fingerprint.py" \
    from-index --index "$INDEX_PATH")
fi

if [[ ! "$FINGERPRINT" =~ ^[0-9a-f]{64}$ ]]; then
  echo "Invalid producer fingerprint" >&2
  exit 1
fi
export "$FINGERPRINT_ENV=$FINGERPRINT"

echo "[$(date '+%F %T')] Starting $ARM on GPU $GPU"
echo "[$(date '+%F %T')] Fingerprint: $FINGERPRINT"
echo "[$(date '+%F %T')] Scene list: $SCENE_LIST"

while IFS= read -r scene || [[ -n "$scene" ]]; do
  if [[ -z "$scene" ]]; then
    echo "Scene list contains an empty row" >&2
    exit 1
  fi
  prediction="$PRED_ROOT/${scene}_boxes.pkl"
  scene_log="$SCENE_LOG_ROOT/${scene}.log"
  if [[ -e "$prediction" ]]; then
    echo "Fresh-arm output already exists: $prediction" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] Running $scene"
  (
    cd "$ROOT"
    CUDA_VISIBLE_DEVICES="$GPU" \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONHASHSEED=0 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
    LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
    "$PYTHON" demo.py scannet \
      --model-path "$ROOT/models/cutr_rgbd.pth" \
      --clip_path "$ROOT/models/open_clip_pytorch_model.bin" \
      --config "$CONFIG" \
      --device cuda \
      --seq "$scene"
  ) >"$scene_log" 2>&1
  if [[ ! -s "$prediction" ]]; then
    echo "No prediction produced for $scene; see $scene_log" >&2
    exit 1
  fi
  echo "[$(date '+%F %T')] Completed $scene"
done < "$SCENE_LIST"

if [[ "$ARM" == record ]]; then
  "$PYTHON" "$ROOT/tools/seal_proposal_cache_index.py" \
    --root "$CACHE_ROOT" \
    --namespace "$NAMESPACE" \
    --scene-list "$SCENE_LIST"
fi

echo "[$(date '+%F %T')] Completed $ARM"
