#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,0,1,1}"
ROOT=/data/ZhaoX/BoxFusion
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
CONFIG="$ROOT/config/scannet_boxer_vapf_lite_real_score05.yaml"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
EXPERIMENT=scannet_boxer_vapf_lite_real_score05
PRED_ROOT="$ROOT/results/$EXPERIMENT"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/boxer_vapf_lite/official100"
LOG_ROOT="$ROOT/logs/$EXPERIMENT"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
BOXER_ROOT=/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer
BOXER_CHECKPOINT="$BOXER_ROOT/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CHECKPOINT="$BOXER_ROOT/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
EVAL_RUNNER="$ROOT/scripts/eval_scannet_official100_real_score.sh"
ENV_ROOT="$(dirname "$(dirname "$PYTHON")")"
ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"

for required in \
  "$PYTHON" "$CONFIG" "$SCENE_LIST" "$ROOT/demo.py" "$MODEL" \
  "$CLIP_MODEL" "$CLASS_TXT" "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT" \
  "$EVAL_RUNNER" "$ROOT/boxfusion/vapf_lite.py"; do
  [[ -e "$required" ]] || { echo "Missing required path: $required" >&2; exit 1; }
done

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
assert cfg["association"]["appearance_gate"]["enabled"] is False
assert "causal_hungarian" not in cfg["association"]
assert cfg["box_fusion"]["reliable_views"]["enabled"] is False
assert cfg["box_fusion"]["vapf_lite"]["enabled"] is True
assert "maskdepth_pfo" not in cfg["box_fusion"]
print("VAPF-lite official100 protocol semantics OK")
PY

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -gt 0 ]] || { echo "No GPU supplied" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU: $gpu" >&2; exit 2; }
done

OFFICIAL_SHA="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
[[ "$OFFICIAL_SHA" == 4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5 ]] || \
  { echo "Official scene-list hash mismatch" >&2; exit 1; }
mkdir -p "$PRED_ROOT" "$DIAGNOSTICS_ROOT" "$SCENE_LOG_ROOT" "$LOG_ROOT/mpl"

FINGERPRINT="$({
  sha256sum \
    "$CONFIG" "$ROOT/demo.py" "$ROOT/boxfusion/vapf_lite.py" \
    "$ROOT/boxfusion/box_fusion.py" "$ROOT/boxfusion/box_manager.py" \
    "$ROOT/boxfusion/instances.py" "$ROOT/boxfusion/boxes.py" \
    "$ROOT/boxfusion/boxer_lifter.py" \
    "$ROOT/boxfusion/sealed_boxer_proposal_cache.py" \
    "$ROOT/data/pst_1024_0.tiff" "$MODEL" "$CLIP_MODEL" \
    "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT"
  printf '%s\n' "python=$(readlink -f "$PYTHON")" "python_version=$($PYTHON --version 2>&1)"
} | sha256sum | awk '{print $1}')"

exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1
total="$(awk 'NF && $1 !~ /^#/ {n += 1} END {print n + 0}' "$SCENE_LIST")"
echo "[$(date '+%F %T')] VAPF-lite official100 started scenes=$total GPUs=$GPU_SPEC fingerprint=$FINGERPRINT"

CUDA_VISIBLE_DEVICES="${GPUS[0]}" PYTHONNOUSERSITE=1 \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" "$PYTHON" -c \
  "import torch,open_clip,scipy,shapely; assert torch.cuda.is_available(); print('Environment OK',torch.__version__)"

run_worker() {
  local gpu="$1" shard="$2" shards="$3" index=0 completed=0
  while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" || "$scene" == \#* ]] && continue
    if (( index % shards != shard )); then index=$((index + 1)); continue; fi
    local pred="$PRED_ROOT/${scene}_boxes.pkl"
    local marker="$PRED_ROOT/${scene}.run_fingerprint"
    local diagnostic="$DIAGNOSTICS_ROOT/${scene}_boxer_lifting.jsonl"
    local log="$SCENE_LOG_ROOT/${shard}_${scene}.log"
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
    summary="$(rg 'VAPF-lite summary|Cost:' "$log" | tail -n 2 | tr '\n' ' ' || true)"
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

bash "$EVAL_RUNNER" "$EXPERIMENT" "$PRED_ROOT"
echo "[$(date '+%F %T')] VAPF-lite official100 complete"
