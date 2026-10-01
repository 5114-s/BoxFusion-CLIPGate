#!/usr/bin/env bash
set -euo pipefail

# official100: Cbest (Top-K + Boxer active) plus the single requested route:
# official EdgeTAM box-prompt masks, Diverse Top-K, rollback 7D MaskDepth-PFO.

GPU_SPEC="${1:-0,1}"
ROOT=/data/ZhaoX/BoxFusion
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
CONFIG="$ROOT/config/scannet_t05_boxer_edgetam_diverse_maskdepth_pfo_score05.yaml"
SCENE_LIST="${BOXFUSION_EDGETAM_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
PRED_ROOT="$ROOT/results/scannet_t05_boxer_edgetam_diverse_maskdepth_pfo_score05"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/t05_boxer/edgetam_diverse_maskdepth_pfo_score05"
LOG_ROOT="$ROOT/logs/edgetam_diverse_maskdepth_pfo/full100"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$ROOT/data/class_features.pt"
EDGE_ROOT="$ROOT/third_party/EdgeTAM"
EDGE_CHECKPOINT="$EDGE_ROOT/checkpoints/edgetam.pt"
BOXER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
BOXER_CHECKPOINT="$BOXER_ROOT/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CHECKPOINT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_cgf_paper100_constant_score.sh"
ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

for required in \
  "$PYTHON" "$CONFIG" "$SCENE_LIST" "$ROOT/demo.py" "$MODEL" \
  "$CLIP_MODEL" "$CLASS_TXT" "$CLASS_FEATURES" "$EDGE_CHECKPOINT" \
  "$EDGE_ROOT/sam2/configs/edgetam.yaml" "$BOXER_CHECKPOINT" \
  "$DINO_CHECKPOINT" "$EVAL_RUNNER"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

[[ "$(sha256sum "$EDGE_CHECKPOINT" | awk '{print $1}')" == \
  ed2d4850b8792c239689b043c47046ec239b6e808a3d9b6ae676c803fd8780df ]] || \
  { echo "EdgeTAM checkpoint mismatch" >&2; exit 1; }
[[ "$(sha256sum "$EDGE_ROOT/sam2/configs/edgetam.yaml" | awk '{print $1}')" == \
  25c2fda8490e7684f924abba130775487ca6eccee87b1d9cf92ddccf2436afe1 ]] || \
  { echo "EdgeTAM config mismatch" >&2; exit 1; }
[[ "$(git -C "$BOXER_ROOT" rev-parse HEAD)" == \
  1f86542dc342a4b1d474c87c97c5d1d6566d9148 ]] || \
  { echo "Boxer commit mismatch" >&2; exit 1; }
[[ "$(sha256sum "$BOXER_CHECKPOINT" | awk '{print $1}')" == \
  d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f ]] || \
  { echo "Boxer checkpoint mismatch" >&2; exit 1; }
[[ "$(sha256sum "$DINO_CHECKPOINT" | awk '{print $1}')" == \
  4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea ]] || \
  { echo "DINO checkpoint mismatch" >&2; exit 1; }

"$PYTHON" - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    cfg = yaml.safe_load(handle)
assert cfg["dataset"] == "scannet"
assert cfg["data"]["gap"] == 25
assert cfg["detection"]["score_thresh"] == 0.5
assert cfg["lifting"]["backend"] == "boxer"
assert cfg["lifting"]["proposal_cache"]["mode"] == "replay"
assert cfg["lifting"]["boxer"]["mode"] == "active"
assert cfg["box_fusion"]["reliable_views"]["diversity_enabled"] is True
assert cfg["box_fusion"]["edgetam_maskdepth"]["enabled"] is True
assert cfg["box_fusion"]["maskdepth_pfo"]["enabled"] is True
print("Protocol semantics OK")
PY

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "No GPU supplied" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
done
# Repeated GPU ids intentionally create independent scene workers on one card.
# The route uses about 6.2 GB per worker on a 24 GB RTX 3090.

OFFICIAL_SHA="$(sha256sum "$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt" | awk '{print $1}')"
LIST_SHA="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
mkdir -p "$PRED_ROOT" "$DIAGNOSTICS_ROOT" "$SCENE_LOG_ROOT" "$LOG_ROOT/mpl"

FINGERPRINT="$({
  sha256sum \
    "$CONFIG" "$ROOT/demo.py" "$ROOT/boxfusion/boxer_lifter.py" \
    "$ROOT/boxfusion/sealed_boxer_proposal_cache.py" \
    "$ROOT/boxfusion/edgetam_maskdepth.py" \
    "$ROOT/boxfusion/maskdepth_pfo.py" \
    "$ROOT/boxfusion/reliable_views.py" "$ROOT/boxfusion/box_fusion.py" \
    "$ROOT/boxfusion/box_manager.py" "$ROOT/boxfusion/instances.py" \
    "$ROOT/boxfusion/boxes.py" "$ROOT/data/pst_1024_0.tiff" \
    "$MODEL" "$CLIP_MODEL" "$CLASS_FEATURES" "$EDGE_CHECKPOINT" \
    "$EDGE_ROOT/sam2/configs/edgetam.yaml" "$EDGE_ROOT/sam2/build_sam.py" \
    "$EDGE_ROOT/sam2/sam2_image_predictor.py" "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT"
  printf '%s\n' "python=$(readlink -f "$PYTHON")" "python_version=$($PYTHON --version 2>&1)"
} | sha256sum | awk '{print $1}')"

exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1
total="$(awk 'NF && $1 !~ /^#/ {n += 1} END {print n + 0}' "$SCENE_LIST")"
echo "[$(date '+%F %T')] EdgeTAM+Diverse+MaskDepth-PFO official run"
echo "[$(date '+%F %T')] scenes=$total GPUs=$GPU_SPEC fingerprint=$FINGERPRINT"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" "$PYTHON" -c \
  "import torch,open_clip,hydra,iopath; assert torch.cuda.is_available(); print('Environment OK',torch.__version__)"

run_worker() {
  local gpu="$1" shard="$2" shards="$3" index=0 completed=0
  while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" || "$scene" == \#* ]] && continue
    if (( index % shards != shard )); then index=$((index + 1)); continue; fi
    local pred="$PRED_ROOT/${scene}_boxes.pkl"
    local marker="$PRED_ROOT/${scene}.run_fingerprint"
    local diagnostic="$DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl"
    local log="$SCENE_LOG_ROOT/${scene}.log"
    local producer_file=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/results/boxer_lifting/x0_cutr/${scene}.run_fingerprint
    local producer
    [[ -s "$producer_file" ]] || { echo "Missing producer marker: $scene" >&2; return 1; }
    producer="$(tr -d '\n' < "$producer_file")"
    if [[ -s "$pred" ]]; then
      [[ -s "$marker" && "$(tr -d '\n' < "$marker")" == "$FINGERPRINT" ]] || \
        { echo "Stale output refused: $scene" >&2; return 1; }
      [[ -s "$diagnostic" ]] || { echo "Missing Boxer diagnostic: $scene" >&2; return 1; }
      echo "[$(date '+%F %T')] [GPU $gpu] already complete $scene"
      completed=$((completed + 1)); index=$((index + 1)); continue
    fi
    [[ ! -e "$marker" && ! -e "$diagnostic" ]] || \
      { echo "Orphan output metadata refused: $scene" >&2; return 1; }
    echo "[$(date '+%F %T')] [GPU $gpu] running $scene ($((index + 1))/$total)"
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES="$gpu" CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 \
      BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$producer" \
      OMP_NUM_THREADS=8 MPLCONFIGDIR="$LOG_ROOT/mpl/gpu${gpu}_slot${shard}" \
      LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
      "$PYTHON" demo.py scannet --model-path "$MODEL" \
        --clip_path "$CLIP_MODEL" --class_txt "$CLASS_TXT" \
        --config "$CONFIG" --device cuda --seq "$scene"
    ) >"$log" 2>&1
    [[ -s "$pred" && -s "$diagnostic" ]] || \
      { echo "Scene failed to produce outputs: $scene" >&2; return 1; }
    printf '%s\n' "$FINGERPRINT" >"${marker}.tmp.$$"
    mv "${marker}.tmp.$$" "$marker"
    completed=$((completed + 1))
    local summary
    summary="$(rg 'EdgeTAM mask-depth summary|MaskDepth-PFO summary|Cost:' "$log" | tail -n 3 | tr '\n' ' ' || true)"
    echo "[$(date '+%F %T')] [GPU $gpu] completed $scene $summary"
    index=$((index + 1))
  done < "$SCENE_LIST"
  echo "[$(date '+%F %T')] [GPU $gpu] worker completed=$completed"
}

pids=()
for shard in "${!GPUS[@]}"; do
  mkdir -p "$LOG_ROOT/mpl/gpu${GPUS[$shard]}_slot${shard}"
  run_worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" -eq 0 ]] || { echo "Worker failure; evaluation not started" >&2; exit 1; }

while IFS= read -r scene || [[ -n "$scene" ]]; do
  [[ -z "$scene" || "$scene" == \#* ]] && continue
  [[ -s "$PRED_ROOT/${scene}_boxes.pkl" ]] || { echo "Missing result: $scene" >&2; exit 1; }
done < "$SCENE_LIST"
[[ "$LIST_SHA" == "$OFFICIAL_SHA" ]] || { echo "Non-official list; AP evaluation refused" >&2; exit 1; }

bash "$EVAL_RUNNER" scannet_t05_boxer_edgetam_diverse_maskdepth_pfo_score05 "$PRED_ROOT"
echo "[$(date '+%F %T')] official100 complete"
