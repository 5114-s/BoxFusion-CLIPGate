#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
OFFICIAL_ROOT="${BOXFUSION_OFFICIAL_ROOT:-$LIVE_ROOT/upstream_clean/BoxFusion_shallow}"
OFFICIAL_COMMIT="${BOXFUSION_OFFICIAL_COMMIT:-b2e0219a7284249bad4a4a8925066839fe2fa33b}"
SCOREFIX_ROOT="${BOXFUSION_SCOREFIX_ROOT:-$ROOT/vendor/boxfusion_score04_real_score}"
SCOREFIX_DEMO_SHA256="d123349f610fe5f143726209cf766d91b3252e7f35ca390669a2f1dd18ea724d"
DATA_ROOT="${BOXFUSION_CA1M_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON_BIN="${BOXFUSION_PYTHON:-$ENV_ROOT/bin/python}"
TAG="${BOXFUSION_CA1M_RUN_TAG:-c0_score04_real_score_full107_v1}"
CONFIG="${BOXFUSION_CA1M_C0_CONFIG:-$ROOT/config/ca1m_c0_score04_real_score_full107.yaml}"
SCENE_LIST="${BOXFUSION_CA1M_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_full107.txt}"
EXPECTED_SCENES="${BOXFUSION_CA1M_EXPECTED_SCENES:-107}"
PROTOCOL="${BOXFUSION_CA1M_PROTOCOL:-full107}"
ALLOW_UNLISTED_SCENES="${BOXFUSION_CA1M_ALLOW_UNLISTED_SCENES:-0}"
EXPECTED_CONFIG_SHA256="${BOXFUSION_CA1M_EXPECTED_CONFIG_SHA256:-}"
EXPECTED_SCENE_LIST_SHA256="${BOXFUSION_CA1M_EXPECTED_SCENE_LIST_SHA256:-}"
EXCLUDED_SCENE_LIST="${BOXFUSION_CA1M_EXCLUDED_SCENE_LIST:-}"
EXPECTED_EXCLUDED_SCENE_LIST_SHA256="${BOXFUSION_CA1M_EXPECTED_EXCLUDED_SCENE_LIST_SHA256:-}"
OUTPUT_ROOT="${BOXFUSION_CA1M_PRED_ROOT:-$ROOT/results/ca1m_repro/$TAG}"
LOG_ROOT="${BOXFUSION_CA1M_LOG_ROOT:-$ROOT/logs/ca1m_repro/$TAG}"
REPORT_ROOT="${BOXFUSION_CA1M_REPORT_ROOT:-$ROOT/reports/ca1m_repro/$TAG}"
EVAL_VIEW="${BOXFUSION_CA1M_EVAL_VIEW:-$ROOT/data/ca1m_eval_${PROTOCOL}_v1}"
RUNTIME_TMP_ROOT="${BOXFUSION_RUNTIME_TMP_ROOT:-/extra/ZhaoX/boxfusion_runtime_tmp/$TAG}"
EVAL_TMPDIR="${BOXFUSION_CA1M_EVAL_TMPDIR:-$RUNTIME_TMP_ROOT/eval}"
MODEL="${BOXFUSION_CA1M_MODEL:-$LIVE_ROOT/models/cutr_rgbd.pth}"
CLIP="${BOXFUSION_CA1M_CLIP:-$LIVE_ROOT/models/open_clip_pytorch_model.bin}"
CLASS_TXT="${BOXFUSION_CA1M_CLASS_TXT:-$LIVE_ROOT/data/panoptic_categories_nomerge.txt}"
CLASS_FEATURES="$SCOREFIX_ROOT/data/class_features.pt"
PST="$LIVE_ROOT/data/pst_1024_0.tiff"
OFFICIAL_URL_LIST="$OFFICIAL_ROOT/data/val.txt"

for path in "$PYTHON_BIN" "$CONFIG" "$SCENE_LIST" "$MODEL" "$CLIP" "$CLASS_TXT" "$CLASS_FEATURES" "$PST" "$OFFICIAL_URL_LIST" "$SCOREFIX_ROOT/demo.py" "$OFFICIAL_ROOT/evaluation/eval_ca1m.py"; do
    [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
if [[ -n "$EXPECTED_CONFIG_SHA256" ]]; then
    [[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$EXPECTED_CONFIG_SHA256" ]] || {
        echo "CA-1M config differs from the frozen protocol" >&2
        exit 2
    }
fi
if [[ -n "$EXPECTED_SCENE_LIST_SHA256" ]]; then
    [[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == "$EXPECTED_SCENE_LIST_SHA256" ]] || {
        echo "CA-1M scene list differs from the frozen protocol" >&2
        exit 2
    }
fi
if [[ -n "$EXCLUDED_SCENE_LIST" ]]; then
    [[ -f "$EXCLUDED_SCENE_LIST" ]] || {
        echo "Missing excluded-scene manifest: $EXCLUDED_SCENE_LIST" >&2
        exit 2
    }
    if [[ -n "$EXPECTED_EXCLUDED_SCENE_LIST_SHA256" ]]; then
        [[ "$(sha256sum "$EXCLUDED_SCENE_LIST" | awk '{print $1}')" == "$EXPECTED_EXCLUDED_SCENE_LIST_SHA256" ]] || {
            echo "Excluded-scene manifest differs from the frozen protocol" >&2
            exit 2
        }
    fi
fi
[[ "$(git -C "$OFFICIAL_ROOT" rev-parse HEAD)" == "$OFFICIAL_COMMIT" ]] || {
    echo "Official source commit differs from the frozen paper anchor" >&2
    exit 2
}
# Ignore only generated/untracked artifacts; no tracked source modification is allowed.
[[ -z "$(git -C "$OFFICIAL_ROOT" status --short --untracked-files=no)" ]] || {
    echo "Official source has tracked working-tree changes; refusing formal reproduction" >&2
    git -C "$OFFICIAL_ROOT" status --short --untracked-files=no >&2
    exit 2
}
[[ "$(git -C "$SCOREFIX_ROOT" rev-parse HEAD)" == "$OFFICIAL_COMMIT" ]] || {
    echo "Score-fix source is not based on the frozen official commit" >&2
    exit 2
}
[[ "$(git -C "$SCOREFIX_ROOT" diff HEAD --name-only)" == "demo.py" ]] || {
    echo "Score-fix source must differ from official code only in demo.py" >&2
    git -C "$SCOREFIX_ROOT" status --short >&2
    exit 2
}
untracked_imports="$(git -C "$SCOREFIX_ROOT" ls-files --others --exclude-standard -- '*.py' '*.so' '*.pyd')"
[[ -z "$untracked_imports" ]] || {
    echo "Score-fix source contains untracked importable source/binaries:" >&2
    echo "$untracked_imports" >&2
    exit 2
}
[[ "$(sha256sum "$SCOREFIX_ROOT/demo.py" | awk '{print $1}')" == "$SCOREFIX_DEMO_SHA256" ]] || {
    echo "Minimal real-score export patch hash differs from the frozen anchor" >&2
    exit 2
}
[[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217" ]] || {
    echo "CuTR checkpoint hash differs from the frozen reproduction anchor" >&2
    exit 2
}
[[ "$(sha256sum "$CLIP" | awk '{print $1}')" == "9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4" ]] || {
    echo "OpenCLIP checkpoint hash differs from the frozen reproduction anchor" >&2
    exit 2
}
[[ "$(sha256sum "$CLASS_FEATURES" | awk '{print $1}')" == "49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197" ]] || {
    echo "Class-feature hash differs from the frozen reproduction anchor" >&2
    exit 2
}
[[ "$(sha256sum "$CLASS_TXT" | awk '{print $1}')" == "0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9" ]] || {
    echo "Class-name list hash differs from the frozen reproduction anchor" >&2
    exit 2
}
[[ "$(sha256sum "$PST" | awk '{print $1}')" == "867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660" ]] || {
    echo "Perspective scoring table hash differs from the frozen reproduction anchor" >&2
    exit 2
}

mkdir -p "$LOG_ROOT/scenes" "$REPORT_ROOT" "$OUTPUT_ROOT" "$EVAL_VIEW" "$RUNTIME_TMP_ROOT"

CONFIG_OUTPUT="$("$PYTHON_BIN" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["data"]["output_dir"])' "$CONFIG")"
[[ "$(readlink -m "$CONFIG_OUTPUT")" == "$(readlink -m "$OUTPUT_ROOT")" ]] || {
    echo "Config output_dir and formal prediction root disagree:" >&2
    echo "  config: $CONFIG_OUTPUT" >&2
    echo "  runner: $OUTPUT_ROOT" >&2
    exit 2
}
CONFIG_DATADIR="$("$PYTHON_BIN" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["data"]["datadir"])' "$CONFIG")"
CONFIG_DATA_ROOT="$(dirname "$(dirname "$CONFIG_DATADIR")")"
[[ "$(readlink -m "$CONFIG_DATA_ROOT")" == "$(readlink -m "$DATA_ROOT")" ]] || {
    echo "Config datadir and audited CA-1M data root disagree:" >&2
    echo "  config root: $CONFIG_DATA_ROOT" >&2
    echo "  audited root: $DATA_ROOT" >&2
    exit 2
}

OFFICIAL_IDS="$REPORT_ROOT/official_scene_ids_sorted.txt"
awk -F'ca1m-val-' 'NF>1 {sub(/\.tar.*/,"",$2); print $2}' "$OFFICIAL_URL_LIST" | sort > "$OFFICIAL_IDS"
[[ "$(wc -l < "$OFFICIAL_IDS")" == "107" ]] || {
    echo "Official URL list did not resolve to 107 scenes" >&2
    exit 2
}
cmp -s <(sort "$SCENE_LIST") "$OFFICIAL_IDS" || {
    if [[ "$EXPECTED_SCENES" == "107" ]]; then
        echo "Frozen CA-1M scene list differs from the official data/val.txt set" >&2
        exit 2
    fi
}

REQUESTED_IDS="$REPORT_ROOT/requested_scene_ids_sorted.txt"
sort "$SCENE_LIST" > "$REQUESTED_IDS"
[[ "$(wc -l < "$REQUESTED_IDS")" == "$EXPECTED_SCENES" ]] || {
    echo "Requested scene list does not contain $EXPECTED_SCENES rows" >&2
    exit 2
}
[[ "$(sort -u "$REQUESTED_IDS" | wc -l)" == "$EXPECTED_SCENES" ]] || {
    echo "Requested scene list contains duplicate scenes" >&2
    exit 2
}
unofficial="$(comm -23 "$REQUESTED_IDS" "$OFFICIAL_IDS")"
[[ -z "$unofficial" ]] || {
    echo "Requested scene list contains scenes outside official CA-1M val:" >&2
    echo "$unofficial" >&2
    exit 2
}

prepare_args=(
    --data-root "$DATA_ROOT"
    --scene-list "$SCENE_LIST"
    --report-output "$REPORT_ROOT/data_preparation.json"
    --expected-scenes "$EXPECTED_SCENES"
)
if [[ "$ALLOW_UNLISTED_SCENES" == "1" ]]; then
    prepare_args+=(--allow-unlisted-scenes)
fi
"$PYTHON_BIN" "$ROOT/tools/prepare_ca1m_full107.py" "${prepare_args[@]}"

[[ "$(grep -cve '^[[:space:]]*$' "$SCENE_LIST")" == "$EXPECTED_SCENES" ]] || {
    echo "CA-1M scene list does not contain exactly $EXPECTED_SCENES scenes" >&2
    exit 2
}

# Build an exact evaluation view.  This prevents the evaluator from treating
# Hugging Face's .cache directory as a CA-1M scene.
while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -n "$scene" ]] || continue
    source_path="$DATA_ROOT/$scene"
    view_path="$EVAL_VIEW/$scene"
    if [[ -L "$view_path" ]]; then
        [[ "$(readlink -f "$view_path")" == "$(readlink -f "$source_path")" ]] || {
            echo "Evaluation-view symlink points to the wrong scene: $view_path" >&2
            exit 2
        }
    elif [[ -e "$view_path" ]]; then
        echo "Refusing non-symlink evaluation-view entry: $view_path" >&2
        exit 2
    else
        ln -s "$source_path" "$view_path"
    fi
done < "$SCENE_LIST"

unexpected="$(find "$EVAL_VIEW" -mindepth 1 -maxdepth 1 -printf '%f\n' | sort | comm -23 - <(sort "$SCENE_LIST"))"
[[ -z "$unexpected" ]] || {
    echo "Unexpected entries in the exact evaluation view:" >&2
    echo "$unexpected" >&2
    exit 2
}

INPUT_AUDIT="$REPORT_ROOT/rgbd_pose_input_audit.json"
INPUT_AUDIT_TMP="$REPORT_ROOT/.rgbd_pose_input_audit.$$.json"
trap 'rm -f "$INPUT_AUDIT_TMP"' EXIT
"$PYTHON_BIN" "$ROOT/tools/audit_ca1m_rgbd_pose.py" \
    --data-root "$DATA_ROOT" --scene-list "$SCENE_LIST" \
    --output "$INPUT_AUDIT_TMP" --depth-scale 1000.0 --min-frames 1
chmod u+w "$INPUT_AUDIT_TMP"
mv -f "$INPUT_AUDIT_TMP" "$INPUT_AUDIT"

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -ge 1 ]] || { echo "No GPUs specified" >&2; exit 2; }

echo "[$(date '+%F %T')] Starting CA-1M score-preserving C0 reproduction"
echo "[$(date '+%F %T')] protocol: $PROTOCOL"
echo "[$(date '+%F %T')] scenes: $EXPECTED_SCENES from $SCENE_LIST"
echo "[$(date '+%F %T')] GPUs: $GPU_SPEC; workers: ${#GPUS[@]}"
echo "[$(date '+%F %T')] score/gap: 0.40/20"
echo "[$(date '+%F %T')] base commit: $OFFICIAL_COMMIT; exported score: detector score"
echo "[$(date '+%F %T')] output: $OUTPUT_ROOT"

pids=()
failures=0
workers="${#GPUS[@]}"
for shard in "${!GPUS[@]}"; do
    mkdir -p "$RUNTIME_TMP_ROOT/shard_$shard"
    (
        index=0
        while IFS= read -r scene || [[ -n "$scene" ]]; do
            [[ -n "$scene" ]] || continue
            index=$((index + 1))
            if (( (index - 1) % workers != shard )); then
                continue
            fi
            prediction="$OUTPUT_ROOT/${scene}_boxes.pkl"
            if [[ -e "$prediction" ]]; then
                if [[ -s "$prediction" ]] && "$PYTHON_BIN" \
                    "$ROOT/tools/validate_ca1m_prediction_file.py" \
                    --prediction "$prediction" >/dev/null 2>&1; then
                    echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] $scene already complete and valid"
                    continue
                fi
                quarantine="$prediction.incomplete.$(date '+%Y%m%d_%H%M%S_%N').$$"
                echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] Quarantining invalid partial prediction: $quarantine"
                mv "$prediction" "$quarantine"
            fi
            log="$LOG_ROOT/scenes/${scene}.log"
            echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] Running $scene (list index $index/$EXPECTED_SCENES)"
            (
                cd "$SCOREFIX_ROOT"
                env \
                -u BOXFUSION_BOXER_DIAGNOSTICS_ROOT \
                -u BOXFUSION_BOXER_GATE_MAX_CENTER_SHIFT_M \
                -u BOXFUSION_BOXER_GATE_MAX_VOLUME_RATIO \
                -u BOXFUSION_BOXER_GATE_MIN_VOLUME_RATIO \
                -u BOXFUSION_ONLINE_ABLATION_PROFILE \
                -u BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT \
                -u BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT \
                -u BOXFUSION_QUALITY_DETECTOR_BLEND \
                -u BOXFUSION_QUALITY_MODE \
                -u BOXFUSION_RELIABLE_VIEWS \
                -u BOXFUSION_SCANNET_MIN_EXTENT \
                -u BOXFUSION_TR3D_RESPONSE \
                CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
                TMPDIR="$RUNTIME_TMP_ROOT/shard_$shard" \
                PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
                PYTHONPYCACHEPREFIX="$LOG_ROOT/pycache_$shard" \
                LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
                MPLCONFIGDIR="$LOG_ROOT/mplconfig_$shard" \
                XDG_CACHE_HOME="$LOG_ROOT/model_cache_$shard" \
                "$PYTHON_BIN" "$SCOREFIX_ROOT/demo.py" CA1M \
                    --model-path "$MODEL" --clip_path "$CLIP" \
                    --class_txt "$CLASS_TXT" --config "$CONFIG" \
                    --device cuda --seq "$scene" \
                    > "$log" 2>&1
            )
            [[ -s "$prediction" ]] && "$PYTHON_BIN" \
                "$ROOT/tools/validate_ca1m_prediction_file.py" \
                --prediction "$prediction" >/dev/null 2>&1 || {
                echo "Scene finished without a prediction: $scene; inspect $log" >&2
                exit 1
            }
            echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] Completed $scene"
        done < "$SCENE_LIST"
        echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] Worker completed shard $shard/$workers"
    ) &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid" || failures=1
done
(( failures == 0 )) || { echo "At least one CA-1M worker failed; evaluation was not started" >&2; exit 1; }

prediction_count="$(find "$OUTPUT_ROOT" -maxdepth 1 -type f -name '*_boxes.pkl' | wc -l)"
[[ "$prediction_count" == "$EXPECTED_SCENES" ]] || {
    echo "Expected $EXPECTED_SCENES predictions, found $prediction_count; evaluation was not started" >&2
    exit 1
}

"$PYTHON_BIN" "$ROOT/tools/audit_ca1m_c0_predictions.py" \
    --scene-list "$SCENE_LIST" --prediction-root "$OUTPUT_ROOT" \
    --output "$REPORT_ROOT/prediction_audit.json" --require-real-score

echo "[$(date '+%F %T')] Completed all $EXPECTED_SCENES scenes; starting CA-1M evaluation ($PROTOCOL)"
mkdir -p "$EVAL_TMPDIR"
(
    cd "$OFFICIAL_ROOT/evaluation"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
        TMPDIR="$EVAL_TMPDIR" \
        MPLCONFIGDIR="$LOG_ROOT/mplconfig_eval" \
        "$PYTHON_BIN" eval_ca1m.py --dataset ca1m \
            --data_path "$EVAL_VIEW" --pred_root "$OUTPUT_ROOT" \
            --ap_iou_thresholds 0.15,0.25,0.5 \
            --cluster_sampling seed_fps --use_3d_nms --use_cls_nms \
            --per_class_proposal --gpu 0
) 2>&1 | tee "$LOG_ROOT/eval_${PROTOCOL}.log"

{
    echo "schema=boxfusion.ca1m_c0_score04_real_score.v2"
    echo "protocol=$PROTOCOL"
    echo "created_at=$(date --iso-8601=seconds)"
    echo "scene_list=$SCENE_LIST"
    echo "scene_list_sha256=$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
    echo "config=$CONFIG"
    echo "config_sha256=$(sha256sum "$CONFIG" | awk '{print $1}')"
    echo "official_root=$OFFICIAL_ROOT"
    echo "official_commit=$OFFICIAL_COMMIT"
    echo "scorefix_root=$SCOREFIX_ROOT"
    echo "scorefix_demo_sha256=$SCOREFIX_DEMO_SHA256"
    echo "model_sha256=$(sha256sum "$MODEL" | awk '{print $1}')"
    echo "clip_sha256=$(sha256sum "$CLIP" | awk '{print $1}')"
    echo "class_features_sha256=$(sha256sum "$CLASS_FEATURES" | awk '{print $1}')"
    echo "pst_sha256=$(sha256sum "$PST" | awk '{print $1}')"
    echo "predictions=$EXPECTED_SCENES"
    echo "official_public_gt_subset=$EXPECTED_SCENES/107"
    if [[ -n "$EXCLUDED_SCENE_LIST" ]]; then
        echo "excluded_scene_list=$EXCLUDED_SCENE_LIST"
        echo "excluded_scene_list_sha256=$(sha256sum "$EXCLUDED_SCENE_LIST" | awk '{print $1}')"
        echo "excluded_scene_ids=$(paste -sd, "$EXCLUDED_SCENE_LIST")"
    fi
    echo "score_thresh=0.4"
    echo "score_export=detector_score"
    echo "released_export_bug=constant_score_1.0"
    echo "paper_reference_ap15_ap25_ap50=31.19/25.51/8.82"
} > "$REPORT_ROOT/run_manifest.txt"

echo "[$(date '+%F %T')] CA-1M score-preserving C0 reproduction completed"
echo "Evaluation log: $LOG_ROOT/eval_${PROTOCOL}.log"
