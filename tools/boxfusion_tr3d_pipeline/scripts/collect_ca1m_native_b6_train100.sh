#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="run"
GPU_SPEC="0,1"
if [[ "${1:-}" == "--preflight" ]]; then
    MODE="preflight"
    GPU_SPEC="${2:-0,1}"
elif [[ -n "${1:-}" ]]; then
    GPU_SPEC="$1"
fi
[[ "$#" -le 2 ]] || { echo "Usage: $0 [--preflight] [gpu0,gpu1]" >&2; exit 2; }

LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="${BOXFUSION_PYTHON:-$ENV_ROOT/bin/python}"
TAG="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_TAG:-ca1m_native_b6_train100_v1}"
MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_MANIFEST:-$ROOT/manifests/ca1m_native_b6_train100_v1/subset_manifest.json}"
SCENE_LIST="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENE_LIST:-$ROOT/manifests/ca1m_native_b6_train100_v1/scene_ids.txt}"
VAL_URL_LIST="${BOXFUSION_CA1M_VAL_URL_LIST:-$LIVE_ROOT/data/val.txt}"
DATA_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1}"
RECORD_TEMPLATE="${BOXFUSION_CA1M_NATIVE_B6_RECORD_CONFIG:-$ROOT/config/ca1m_native_b6_train100_cutr_record.yaml}"
OBSERVER_TEMPLATE="${BOXFUSION_CA1M_NATIVE_B6_OBSERVER_CONFIG:-$ROOT/config/ca1m_native_b6_train100_g0_observer.yaml}"
NAMESPACE="ca1m-native-b6-train100-score04-gap20-cutr-v1"

RECORD_ROOT="$ROOT/results/$TAG/cutr_record"
OBSERVER_ROOT="$ROOT/results/$TAG/g0_observer"
ANCHOR_ROOT="$ROOT/results/$TAG/g0_observer_same_run_anchor"
CACHE_ROOT="$ROOT/cache/$TAG"
NATIVE_ROOT="$ROOT/diagnostics/$TAG/native_b6"
BOXER_ROOT="$ROOT/diagnostics/$TAG/boxer"
LOG_ROOT="$ROOT/logs/$TAG"
REPORT_ROOT="$ROOT/reports/$TAG"
RECORD_COMPLETIONS="$REPORT_ROOT/completion/cutr_record"
OBSERVER_COMPLETIONS="$REPORT_ROOT/completion/g0_observer"
STAGING_ROOT="$ROOT/staging/$TAG"
LOCK_ROOT="${BOXFUSION_RUN_LOCK_ROOT:-/tmp/boxfusion_ca1m_runlocks}"
LOCK_DIR="$LOCK_ROOT/$TAG.lock"
RUNTIME_TMP="${BOXFUSION_RUNTIME_TMP_ROOT:-/tmp/bfc-$TAG}"

MODEL="$LIVE_ROOT/models/cutr_rgbd.pth"
CLIP="$LIVE_ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$LIVE_ROOT/data/class_features.pt"
PST="$LIVE_ROOT/data/pst_1024_0.tiff"

EXPECTED_MANIFEST_SHA="29a32e92cfece667e9fef4389227eacba2b96c55737569fa6219ca7ab527fd23"
EXPECTED_SCENE_LIST_SHA="35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd"
EXPECTED_RECORD_CONFIG_SHA="c9f9de92218c0b03ee2aca22dc9a457cb122baf95db15e4b616697e9cec117cb"
EXPECTED_OBSERVER_CONFIG_SHA="d6132ee9de6d8f5fcd2b06b9c5cad74ddf31dc4f245df7d874fb1b110f515314"
EXPECTED_MODEL_SHA="856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217"
EXPECTED_CLIP_SHA="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
EXPECTED_CLASS_FEATURES_SHA="49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197"
EXPECTED_CLASS_TXT_SHA="0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9"
EXPECTED_PST_SHA="867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660"

die() { echo "$*" >&2; exit 2; }
file_sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
    local path="$1" expected="$2" actual
    [[ -f "$path" && ! -L "$path" ]] || die "Missing regular frozen input: $path"
    actual="$(file_sha "$path")"
    [[ "$actual" == "$expected" ]] || die "SHA256 mismatch: $path ($actual != $expected)"
}

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
(( ${#GPUS[@]} >= 1 )) || die "No GPUs specified"
for gpu in "${GPUS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU: $gpu"; done

[[ -x "$PYTHON" ]] || die "Missing executable Python: $PYTHON"
for path in "$MANIFEST" "$SCENE_LIST" "$VAL_URL_LIST" \
    "$RECORD_TEMPLATE" "$OBSERVER_TEMPLATE" "$MODEL" "$CLIP" \
    "$CLASS_TXT" "$CLASS_FEATURES" "$PST" "$ROOT/demo.py" \
    "$ROOT/tools/audit_ca1m_native_b6_train_inputs.py" \
    "$ROOT/tools/materialize_ca1m_native_b6_train_config.py" \
    "$ROOT/tools/finalize_ca1m_native_b6_train_artifact.py"; do
    [[ -f "$path" && ! -L "$path" ]] || die "Missing regular input: $path"
done
require_sha "$MANIFEST" "$EXPECTED_MANIFEST_SHA"
require_sha "$SCENE_LIST" "$EXPECTED_SCENE_LIST_SHA"
require_sha "$RECORD_TEMPLATE" "$EXPECTED_RECORD_CONFIG_SHA"
require_sha "$OBSERVER_TEMPLATE" "$EXPECTED_OBSERVER_CONFIG_SHA"
require_sha "$MODEL" "$EXPECTED_MODEL_SHA"
require_sha "$CLIP" "$EXPECTED_CLIP_SHA"
require_sha "$CLASS_FEATURES" "$EXPECTED_CLASS_FEATURES_SHA"
require_sha "$CLASS_TXT" "$EXPECTED_CLASS_TXT_SHA"
require_sha "$PST" "$EXPECTED_PST_SHA"

mapfile -t SCENES < <(sed -e 's/[[:space:]]*$//' -e '/^$/d' "$SCENE_LIST")
[[ "${#SCENES[@]}" == "100" ]] || die "Frozen train collection requires 100 scenes"
[[ "$(printf '%s\n' "${SCENES[@]}" | sort -u | wc -l)" == "100" ]] \
    || die "Frozen train scene list contains duplicates"

PROPOSAL_FINGERPRINT="$(printf '%s\n%s\n%s\n%s\n' \
    "$NAMESPACE" "$EXPECTED_SCENE_LIST_SHA" "$EXPECTED_MODEL_SHA" \
    "$EXPECTED_RECORD_CONFIG_SHA" | sha256sum | awk '{print $1}')"
CODE_SOURCES=(
    demo.py boxfusion/capture_stream.py boxfusion/proposal_cache.py boxfusion/boxer_lifter.py
    boxfusion/ca1m_native_b6_observer.py boxfusion/tr3d_r2_geometry.py
    boxfusion/tr3d_r4_smov_observer.py boxfusion/tr3d_terminal_active.py
    config/ca1m_native_b6_train100_cutr_record.yaml
    config/ca1m_native_b6_train100_g0_observer.yaml
    tools/audit_ca1m_native_b6_train_inputs.py
    tools/materialize_ca1m_native_b6_train_config.py
    tools/finalize_ca1m_native_b6_train_artifact.py
    scripts/collect_ca1m_native_b6_train100.sh
)
CODE_MANIFEST_TMP="$(mktemp /tmp/ca1m-native-b6-code.XXXXXX.tsv)"
for source in "${CODE_SOURCES[@]}"; do
    [[ -f "$ROOT/$source" ]] || die "Missing collection code source: $source"
    printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_MANIFEST_TMP"
done
CODE_FINGERPRINT="$(file_sha "$CODE_MANIFEST_TMP")"

PREFLIGHT_DIR="$(mktemp -d /tmp/ca1m-native-b6-train-preflight.XXXXXX)"
PREFLIGHT_TMP="$PREFLIGHT_DIR/audit.json"
trap 'rm -f "$PREFLIGHT_TMP" "$CODE_MANIFEST_TMP"; rmdir "$PREFLIGHT_DIR" 2>/dev/null || true' EXIT
"$PYTHON" "$ROOT/tools/audit_ca1m_native_b6_train_inputs.py" \
    --manifest "$MANIFEST" --val-url-list "$VAL_URL_LIST" \
    --data-root "$DATA_ROOT" --preflight --output "$PREFLIGHT_TMP"

# Materialize both templates in /tmp to prove phase-specific contracts without
# claiming a formal results/cache namespace.
CONFIG_SMOKE_ROOT="$(mktemp -d /tmp/ca1m-native-b6-config.XXXXXX)"
"$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_train_config.py" \
    --template "$RECORD_TEMPLATE" --phase record --data-root "$DATA_ROOT" \
    --output-root "$CONFIG_SMOKE_ROOT/record_pred" --cache-root "$CONFIG_SMOKE_ROOT/cache" \
    --output "$CONFIG_SMOKE_ROOT/record.yaml" >/dev/null
"$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_train_config.py" \
    --template "$OBSERVER_TEMPLATE" --phase observer --data-root "$DATA_ROOT" \
    --output-root "$CONFIG_SMOKE_ROOT/observer_pred" --cache-root "$CONFIG_SMOKE_ROOT/cache" \
    --baseline-root "$CONFIG_SMOKE_ROOT/record_pred" \
    --native-diagnostics-root "$CONFIG_SMOKE_ROOT/native" \
    --boxer-diagnostics-root "$CONFIG_SMOKE_ROOT/boxer" \
    --output "$CONFIG_SMOKE_ROOT/observer.yaml" >/dev/null

echo "CA-1M native-B6 train100 collection preflight"
echo "  frozen train scenes: 100; validation overlap: 0"
echo "  score/gap: real detector score, threshold=0.4, gap=20"
echo "  record: live CuTR -> isolated immutable cache"
echo "  replay: Selective Boxer G0 -> native final-OBB observer"
echo "  evaluation: forbidden; validation GT access: forbidden"
echo "  future data root: $DATA_ROOT"
echo "  proposal fingerprint: $PROPOSAL_FINGERPRINT"
if [[ "$MODE" == "preflight" ]]; then
    echo "Preflight passed; no data, prediction, cache, diagnostic, or report namespace was created."
    exit 0
fi

[[ -d "$DATA_ROOT" && ! -L "$DATA_ROOT" ]] || die "Train100 data root is not ready: $DATA_ROOT"
mkdir -p "$LOCK_ROOT"
mkdir "$LOCK_DIR" || die "Another process owns train100 collection lock: $LOCK_DIR"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true; rm -f "$PREFLIGHT_TMP" "$CODE_MANIFEST_TMP"; rmdir "$PREFLIGHT_DIR" 2>/dev/null || true' EXIT
[[ ! -e "$RUNTIME_TMP" ]] || die "Refusing stale runtime namespace: $RUNTIME_TMP"
mkdir -p "$RUNTIME_TMP" "$REPORT_ROOT" "$STAGING_ROOT" \
    "$RECORD_ROOT" "$OBSERVER_ROOT" "$ANCHOR_ROOT" "$CACHE_ROOT/$NAMESPACE" \
    "$NATIVE_ROOT" "$BOXER_ROOT" "$LOG_ROOT/cutr_record" "$LOG_ROOT/g0_observer" \
    "$RECORD_COMPLETIONS" "$OBSERVER_COMPLETIONS"

# Full audit reads RGB-D, poses, gravity, and intrinsics only. It deliberately
# does not load after_filter_boxes.npy or any validation artifact.
INPUT_AUDIT_TMP="$RUNTIME_TMP/input_audit.json"
"$PYTHON" "$ROOT/tools/audit_ca1m_native_b6_train_inputs.py" \
    --manifest "$MANIFEST" --val-url-list "$VAL_URL_LIST" \
    --data-root "$DATA_ROOT" --output "$INPUT_AUDIT_TMP"
if [[ -e "$REPORT_ROOT/input_audit.json" ]]; then
    cmp -s "$INPUT_AUDIT_TMP" "$REPORT_ROOT/input_audit.json" \
        || die "Train input audit drifted from resumed collection"
else
    cp "$INPUT_AUDIT_TMP" "$REPORT_ROOT/input_audit.json"
    chmod 0444 "$REPORT_ROOT/input_audit.json"
fi

cat > "$RUNTIME_TMP/protocol.txt" <<EOF
schema=boxfusion.ca1m_native_b6_train_collection_protocol.v1
train_only=true
validation_ground_truth_access=false
evaluation_invoked=false
scene_ids_sha256=$EXPECTED_SCENE_LIST_SHA
subset_manifest_sha256=$EXPECTED_MANIFEST_SHA
score_thresh=0.4
selective_boxer_gate=center0.10_volume0.50_2.00
proposal_cache_namespace=$NAMESPACE
proposal_fingerprint=$PROPOSAL_FINGERPRINT
intrinsics_policy=loader_consumes_optional_K_depth_per_frame_npy_v1
record_config_sha256=$EXPECTED_RECORD_CONFIG_SHA
observer_config_sha256=$EXPECTED_OBSERVER_CONFIG_SHA
code_fingerprint=$CODE_FINGERPRINT
EOF
if [[ -e "$REPORT_ROOT/protocol.txt" ]]; then
    cmp -s "$RUNTIME_TMP/protocol.txt" "$REPORT_ROOT/protocol.txt" \
        || die "Train collection protocol drifted"
else
    cp "$RUNTIME_TMP/protocol.txt" "$REPORT_ROOT/protocol.txt"
    chmod 0444 "$REPORT_ROOT/protocol.txt"
fi

finalize_record() {
    local scene="$1"
    "$PYTHON" "$ROOT/tools/finalize_ca1m_native_b6_train_artifact.py" record \
        --scene "$scene" --prediction "$RECORD_ROOT/${scene}_boxes.pkl" \
        --cache-scene-root "$CACHE_ROOT/$NAMESPACE/$scene" \
        --cache-namespace "$NAMESPACE" --proposal-fingerprint "$PROPOSAL_FINGERPRINT" \
        --log "$LOG_ROOT/cutr_record/${scene}.log" \
        --output "$RECORD_COMPLETIONS/${scene}.json" >/dev/null
}

finalize_observer() {
    local scene="$1"
    "$PYTHON" "$ROOT/tools/finalize_ca1m_native_b6_train_artifact.py" observer \
        --scene "$scene" --prediction "$OBSERVER_ROOT/${scene}_boxes.pkl" \
        --anchor "$ANCHOR_ROOT/${scene}_boxes.pkl" \
        --diagnostic "$NATIVE_ROOT/${scene}_ca1m_native_b6.npz" \
        --boxer "$BOXER_ROOT/${scene}_boxer_lifting.jsonl" \
        --log "$LOG_ROOT/g0_observer/${scene}.log" \
        --output "$OBSERVER_COMPLETIONS/${scene}.json" >/dev/null
}

run_scene() {
    local scene="$1" gpu="$2" stage record_cfg observer_cfg
    local record_pred="$RECORD_ROOT/${scene}_boxes.pkl"
    local cache_scene="$CACHE_ROOT/$NAMESPACE/$scene"
    local record_log="$LOG_ROOT/cutr_record/${scene}.log"
    if [[ -e "$RECORD_COMPLETIONS/${scene}.json" ]]; then
        finalize_record "$scene"
    elif [[ -s "$record_pred" && -d "$cache_scene" && -s "$record_log" ]]; then
        finalize_record "$scene"
    else
        [[ ! -e "$record_pred" && ! -e "$cache_scene" && ! -e "$record_log" ]] \
            || die "$scene: partial permanent CuTR record artifacts; refusing overwrite"
        stage="$STAGING_ROOT/${scene}.record.$BASHPID"
        mkdir "$stage" "$stage/pred" "$stage/cache" || die "$scene: record staging collision"
        record_cfg="$stage/config.yaml"
        "$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_train_config.py" \
            --template "$RECORD_TEMPLATE" --phase record --data-root "$DATA_ROOT" \
            --output-root "$stage/pred" --cache-root "$stage/cache" --output "$record_cfg" >/dev/null
        env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=0 \
            PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
            BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT="$PROPOSAL_FINGERPRINT" \
            LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
            MPLCONFIGDIR="$RUNTIME_TMP/mpl_record_${gpu}" XDG_CACHE_HOME="$RUNTIME_TMP/model_record_${gpu}" \
            "$PYTHON" "$ROOT/demo.py" CA1M --model-path "$MODEL" --clip_path "$CLIP" \
                --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES" \
                --config "$record_cfg" --output-dir "$stage/pred" --device cuda --seq "$scene" --seed 0 \
                > "$stage/run.log" 2>&1
        [[ -s "$stage/pred/${scene}_boxes.pkl" && -d "$stage/cache/$NAMESPACE/$scene" ]] \
            || die "$scene: incomplete staged CuTR record"
        mv "$stage/pred/${scene}_boxes.pkl" "$record_pred"
        mv "$stage/cache/$NAMESPACE/$scene" "$cache_scene"
        mv "$stage/run.log" "$record_log"
        chmod -R a-w "$record_pred" "$cache_scene" "$record_log"
        finalize_record "$scene"
    fi

    local observer_pred="$OBSERVER_ROOT/${scene}_boxes.pkl"
    local anchor="$ANCHOR_ROOT/${scene}_boxes.pkl"
    local native="$NATIVE_ROOT/${scene}_ca1m_native_b6.npz"
    local boxer="$BOXER_ROOT/${scene}_boxer_lifting.jsonl"
    local observer_log="$LOG_ROOT/g0_observer/${scene}.log"
    if [[ -e "$OBSERVER_COMPLETIONS/${scene}.json" ]]; then
        finalize_observer "$scene"
    elif [[ -s "$observer_pred" && -s "$anchor" && -s "$native" && -s "$boxer" && -s "$observer_log" ]]; then
        finalize_observer "$scene"
    else
        for artifact in "$observer_pred" "$anchor" "$native" "$boxer" "$observer_log"; do
            [[ ! -e "$artifact" ]] || die "$scene: partial permanent observer artifacts; refusing overwrite"
        done
        stage="$STAGING_ROOT/${scene}.observer.$BASHPID"
        mkdir "$stage" "$stage/pred" "$stage/anchor" "$stage/native" "$stage/boxer" \
            || die "$scene: observer staging collision"
        observer_cfg="$stage/config.yaml"
        "$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_train_config.py" \
            --template "$OBSERVER_TEMPLATE" --phase observer --data-root "$DATA_ROOT" \
            --output-root "$stage/pred" --cache-root "$CACHE_ROOT" --baseline-root "$RECORD_ROOT" \
            --native-diagnostics-root "$stage/native" --boxer-diagnostics-root "$stage/boxer" \
            --output "$observer_cfg" >/dev/null
        env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=0 \
            PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
            BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$PROPOSAL_FINGERPRINT" \
            LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
            MPLCONFIGDIR="$RUNTIME_TMP/mpl_observer_${gpu}" XDG_CACHE_HOME="$RUNTIME_TMP/model_observer_${gpu}" \
            "$PYTHON" "$ROOT/demo.py" CA1M --model-path "$MODEL" --clip_path "$CLIP" \
                --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES" \
                --config "$observer_cfg" --output-dir "$stage/pred" \
                --ca1m-native-b6-same-run-anchor-root "$stage/anchor" \
                --ca1m-native-b6-diagnostics-root "$stage/native" \
                --boxer-diagnostics-root "$stage/boxer" \
                --device cuda --seq "$scene" --seed 0 > "$stage/run.log" 2>&1
        [[ -s "$stage/pred/${scene}_boxes.pkl" && -s "$stage/anchor/${scene}_boxes.pkl" \
            && -s "$stage/native/${scene}_ca1m_native_b6.npz" \
            && -s "$stage/boxer/${scene}_boxer_lifting.jsonl" ]] \
            || die "$scene: incomplete staged G0/native-B6 observer"
        mv "$stage/pred/${scene}_boxes.pkl" "$observer_pred"
        mv "$stage/anchor/${scene}_boxes.pkl" "$anchor"
        mv "$stage/native/${scene}_ca1m_native_b6.npz" "$native"
        mv "$stage/boxer/${scene}_boxer_lifting.jsonl" "$boxer"
        mv "$stage/run.log" "$observer_log"
        chmod a-w "$observer_pred" "$anchor" "$native" "$boxer" "$observer_log"
        finalize_observer "$scene"
    fi
    echo "[$(date '+%F %T')] [GPU $gpu] train-only collection complete: $scene"
}

workers="${#GPUS[@]}"
pids=()
failures=0
for shard in "${!GPUS[@]}"; do
    (
        for index in "${!SCENES[@]}"; do
            (( index % workers == shard )) || continue
            run_scene "${SCENES[$index]}" "${GPUS[$shard]}"
        done
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || failures=1; done
(( failures == 0 )) || die "At least one train100 collection worker failed"

CODE_MANIFEST_AFTER="$RUNTIME_TMP/code_after.tsv"
for source in "${CODE_SOURCES[@]}"; do
    printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_MANIFEST_AFTER"
done
[[ "$(file_sha "$CODE_MANIFEST_AFTER")" == "$CODE_FINGERPRINT" ]] \
    || die "Collection code changed during train100 inference"

"$PYTHON" "$ROOT/tools/finalize_ca1m_native_b6_train_artifact.py" collection \
    --subset-manifest "$MANIFEST" --record-completion-root "$RECORD_COMPLETIONS" \
    --observer-completion-root "$OBSERVER_COMPLETIONS" \
    --output "$REPORT_ROOT/collection_manifest.json"
echo "CA-1M native-B6 train100 collection complete (no evaluation performed):"
echo "  $REPORT_ROOT/collection_manifest.json"
