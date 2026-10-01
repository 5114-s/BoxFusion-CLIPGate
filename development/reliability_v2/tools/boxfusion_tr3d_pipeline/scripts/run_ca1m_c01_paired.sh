#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON_BIN="$ENV_ROOT/bin/python"
SCENE_LIST="${BOXFUSION_CA1M_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_ablation10_even.txt}"
DATA_ROOT="${BOXFUSION_CA1M_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
C0_CONFIG="${BOXFUSION_CA1M_C0_CONFIG:-$ROOT/config/ca1m_c0_original_paired.yaml}"
C1_CONFIG="${BOXFUSION_CA1M_C1_CONFIG:-$ROOT/config/ca1m_c1_selective_boxer_paired.yaml}"
C0_ROOT="$ROOT/results/ca1m_port/c0_original_fixed10_v2"
C1_ROOT="$ROOT/results/ca1m_port/c1_selective_boxer_fixed10_v2"
CACHE_ROOT="$ROOT/cache/ca1m_cutr_proposals/ca1m-score04-gap20-c0-v2"
LOG_ROOT="$ROOT/logs/ca1m_port/c01_paired_fixed10_v2"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/ca1m_port/c1_selective_boxer_fixed10_v2"
EVAL_VIEW="$ROOT/data/ca1m_eval_fixed10_v2"
MODEL="$LIVE_ROOT/models/cutr_rgbd.pth"
CLIP="$LIVE_ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$LIVE_ROOT/data/class_features.pt"

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -ge 1 ]] || { echo "No GPUs specified" >&2; exit 2; }
for path in "$PYTHON_BIN" "$SCENE_LIST" "$C0_CONFIG" "$C1_CONFIG" "$MODEL" "$CLIP" "$CLASS_TXT" "$CLASS_FEATURES"; do
    [[ -f "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
for root in "$C0_ROOT" "$C1_ROOT" "$CACHE_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_VIEW"; do
    [[ ! -e "$root" ]] || { echo "Refusing existing formal namespace: $root" >&2; exit 2; }
done
mkdir -p "$C0_ROOT" "$C1_ROOT" "$LOG_ROOT/c0" "$LOG_ROOT/c1" "$EVAL_VIEW"

while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -n "$scene" ]] || continue
    source="$DATA_ROOT/$scene"
    [[ -d "$source" && ! -L "$source" ]] || { echo "Missing CA-1M scene: $source" >&2; exit 2; }
    ln -s "$source" "$EVAL_VIEW/$scene"
done < "$SCENE_LIST"

LIST_SHA="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
MODEL_SHA="$(sha256sum "$MODEL" | awk '{print $1}')"
PROPOSAL_FINGERPRINT="$(printf '%s\n%s\n%s\n' 'ca1m-cutr-score04-gap20-v2' "$LIST_SHA" "$MODEL_SHA" | sha256sum | awk '{print $1}')"

run_phase() {
    local phase="$1" config="$2" output="$3" fingerprint_env="$4"
    local workers="${#GPUS[@]}" pids=() failures=0
    echo "[$(date '+%F %T')] Starting CA-1M $phase with GPUs=$GPU_SPEC"
    for shard in "${!GPUS[@]}"; do
        (
            index=0
            while IFS= read -r scene || [[ -n "$scene" ]]; do
                [[ -n "$scene" ]] || continue
                if (( index % workers != shard )); then index=$((index + 1)); continue; fi
                log="$LOG_ROOT/$phase/${scene}.log"
                echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] $phase $scene"
                env CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
                    PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
                    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
                    MPLCONFIGDIR="$LOG_ROOT/mplconfig_${phase}_${shard}" \
                    XDG_CACHE_HOME="$LOG_ROOT/model_cache_${phase}_${shard}" \
                    "$fingerprint_env=$PROPOSAL_FINGERPRINT" \
                    "$PYTHON_BIN" "$ROOT/demo.py" CA1M \
                        --model-path "$MODEL" --clip_path "$CLIP" \
                        --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES" \
                        --config "$config" --output-dir "$output" \
                        --device cuda --seq "$scene" --seed 0 \
                        > "$log" 2>&1
                index=$((index + 1))
            done < "$SCENE_LIST"
        ) &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "$pid" || failures=1; done
    (( failures == 0 )) || { echo "$phase worker failed" >&2; exit 1; }
    count="$(find "$output" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
    [[ "$count" == "10" ]] || { echo "$phase produced $count/10 predictions" >&2; exit 1; }
}

run_phase c0 "$C0_CONFIG" "$C0_ROOT" BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT
run_phase c1 "$C1_CONFIG" "$C1_ROOT" BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT

evaluate() {
    local phase="$1" pred_root="$2"
    (
        cd "$ROOT/evaluation"
        env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
            MPLCONFIGDIR="$LOG_ROOT/mplconfig_eval_$phase" \
            "$PYTHON_BIN" eval_ca1m.py --dataset ca1m \
                --data_path "$EVAL_VIEW" --pred_root "$pred_root" \
                --ap_iou_thresholds 0.15,0.25,0.5 --num_workers 0 \
                --cluster_sampling seed_fps --use_3d_nms --use_cls_nms \
                --per_class_proposal --gpu 0
    ) > "$LOG_ROOT/eval_${phase}.log" 2>&1
    chmod 0444 "$LOG_ROOT/eval_${phase}.log"
}

evaluate c0 "$C0_ROOT"
evaluate c1 "$C1_ROOT"
echo "=== CA-1M C0 original ==="
grep -E 'eval (mAP|APrec|ARecall):|mAP:' "$LOG_ROOT/eval_c0.log" | tail -12
echo "=== CA-1M C1 Selective Boxer G0 ==="
grep -E 'eval (mAP|APrec|ARecall):|mAP:' "$LOG_ROOT/eval_c1.log" | tail -12
echo "CA-1M C0/C1 paired evaluation completed: $LOG_ROOT"
