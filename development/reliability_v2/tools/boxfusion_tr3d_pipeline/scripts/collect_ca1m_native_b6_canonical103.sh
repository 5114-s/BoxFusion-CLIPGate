#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="preflight"
GPU_SPEC="0,1"
case "${1:-}" in
    ""|--preflight) MODE="preflight"; GPU_SPEC="${2:-0,1}" ;;
    --run) MODE="run"; GPU_SPEC="${2:-0,1}" ;;
    *) echo "Usage: $0 [--preflight|--run] [gpu0,gpu1]" >&2; exit 2 ;;
esac
[[ "$#" -le 2 ]] || { echo "Usage: $0 [--preflight|--run] [gpu0,gpu1]" >&2; exit 2; }

LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="${BOXFUSION_PYTHON:-$ENV_ROOT/bin/python}"
TAG="${BOXFUSION_CA1M_NATIVE_B6_CANONICAL_TAG:-ca1m_c3_native_b6_observer_canonical103_v1}"
SCENE_LIST="${BOXFUSION_CA1M_CANONICAL_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_canonical103.txt}"
EXCLUDED_LIST="${BOXFUSION_CA1M_EXCLUDED_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_missing_canonical_gt4.txt}"
OFFICIAL_URL_LIST="${BOXFUSION_CA1M_OFFICIAL_URL_LIST:-$LIVE_ROOT/data/val.txt}"
DATA_ROOT="${BOXFUSION_CA1M_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
RECORD_TEMPLATE="${BOXFUSION_CA1M_NATIVE_B6_CANONICAL_RECORD_CONFIG:-$ROOT/config/ca1m_native_b6_canonical103_cutr_record.yaml}"
OBSERVER_TEMPLATE="${BOXFUSION_CA1M_NATIVE_B6_CANONICAL_OBSERVER_CONFIG:-$ROOT/config/ca1m_native_b6_canonical103_g0_observer.yaml}"
NAMESPACE="ca1m-native-b6-canonical103-score04-gap20-cutr-v1"

HISTORICAL_C0="$ROOT/results/ca1m_repro/c0_score04_real_score_canonical103_v1"
LEGACY_CACHE="$ROOT/cache/ca1m_cutr_proposals/ca1m-score04-gap20-c0-v2"
RECORD_ROOT="$ROOT/results/ca1m_port/${TAG}_cutr_record"
OBSERVER_ROOT="$ROOT/results/ca1m_port/$TAG"
ANCHOR_ROOT="$ROOT/results/ca1m_port/${TAG}_same_run_anchor"
CACHE_ROOT="$ROOT/cache/ca1m_native_b6_canonical103_v1"
NATIVE_ROOT="$ROOT/diagnostics/ca1m_port/$TAG/native_b6"
BOXER_ROOT="$ROOT/diagnostics/ca1m_port/$TAG/boxer"
LOG_ROOT="$ROOT/logs/ca1m_port/$TAG"
REPORT_ROOT="$ROOT/reports/ca1m_port/$TAG"
RECORD_COMPLETIONS="$REPORT_ROOT/completion/cutr_record"
OBSERVER_COMPLETIONS="$REPORT_ROOT/completion/g0_observer"
STAGING_ROOT="$ROOT/staging/$TAG"
LOCK_ROOT="${BOXFUSION_RUN_LOCK_ROOT:-/tmp/boxfusion_ca1m_runlocks}"
LOCK_DIR="$LOCK_ROOT/$TAG.lock"
RUNTIME_BASE="${BOXFUSION_RUNTIME_TMP_ROOT:-/extra/ZhaoX/boxfusion_runtime_tmp/$TAG}"
RUNTIME_TMP="$RUNTIME_BASE/run.$BASHPID"

MODEL="$LIVE_ROOT/models/cutr_rgbd.pth"
CLIP="$LIVE_ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$LIVE_ROOT/data/class_features.pt"
PST="$LIVE_ROOT/data/pst_1024_0.tiff"
BOXER_OFFICIAL="/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer"
BOXER_CHECKPOINT="$BOXER_OFFICIAL/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt"
DINO_CHECKPOINT="$BOXER_OFFICIAL/ckpts/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

EXPECTED_SCENE_LIST_SHA="c3efbe544c7403acc4183d7e4a799dad2bb40f60cbdba38830863f8712f4648f"
EXPECTED_EXCLUDED_SHA="582ec52e296fa907a79eb01f5778b0adea368b3c7ca61e3a972aca42f32d401b"
EXPECTED_OFFICIAL_URL_SHA="895580c85ac12aa7f8a907fe898e5cf8e7249976f39c41202c2670b97ab60d97"
EXPECTED_RECORD_CONFIG_SHA="b6d7acef3b82e3031b57f4dd52a5226fcde3dc1365a53d18056d1e2e7a995131"
EXPECTED_OBSERVER_CONFIG_SHA="4d0a1cad4d0df33750ce0e31282f1f20dd5d447f3f9e9f2d192a1bd7cb686a96"
EXPECTED_MODEL_SHA="856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217"
EXPECTED_CLIP_SHA="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
EXPECTED_CLASS_FEATURES_SHA="49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197"
EXPECTED_CLASS_TXT_SHA="0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9"
EXPECTED_PST_SHA="867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660"
EXPECTED_BOXER_COMMIT="1f86542dc342a4b1d474c87c97c5d1d6566d9148"
EXPECTED_BOXER_SHA="d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
EXPECTED_DINO_SHA="4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"

die() { echo "$*" >&2; exit 2; }
file_sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
    local path="$1" expected="$2" actual
    [[ -f "$path" && ! -L "$path" ]] || die "Missing regular frozen input: $path"
    actual="$(file_sha "$path")"
    [[ "$actual" == "$expected" ]] || die "SHA256 mismatch: $path ($actual != $expected)"
}
publish_immutable() {
    local source="$1" destination="$2"
    if [[ -e "$destination" ]]; then
        cmp -s "$source" "$destination" || die "Resumed manifest drift: $destination"
    else
        cp "$source" "$destination"
        chmod 0444 "$destination"
    fi
}

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
(( ${#GPUS[@]} >= 1 )) || die "No GPUs specified"
for gpu in "${GPUS[@]}"; do [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU: $gpu"; done

for path in "$PYTHON" "$SCENE_LIST" "$EXCLUDED_LIST" "$OFFICIAL_URL_LIST" \
    "$RECORD_TEMPLATE" "$OBSERVER_TEMPLATE" "$MODEL" "$CLIP" "$CLASS_TXT" \
    "$CLASS_FEATURES" "$PST" "$BOXER_CHECKPOINT" "$DINO_CHECKPOINT" \
    "$ROOT/demo.py" "$ROOT/tools/materialize_ca1m_native_b6_canonical_config.py" \
    "$ROOT/tools/audit_ca1m_native_b6_canonical_inputs.py" \
    "$ROOT/tools/audit_ca1m_native_b6_canonical_assets.py" \
    "$ROOT/tools/finalize_ca1m_native_b6_canonical_artifact.py" \
    "$ROOT/tools/audit_ca1m_native_b6_canonical_identity.py"; do
    [[ -f "$path" ]] || die "Missing input: $path"
done
for path in "$DATA_ROOT" "$HISTORICAL_C0" "$LEGACY_CACHE" "$BOXER_OFFICIAL"; do
    [[ -d "$path" ]] || die "Missing input directory: $path"
done
require_sha "$SCENE_LIST" "$EXPECTED_SCENE_LIST_SHA"
require_sha "$EXCLUDED_LIST" "$EXPECTED_EXCLUDED_SHA"
require_sha "$OFFICIAL_URL_LIST" "$EXPECTED_OFFICIAL_URL_SHA"
require_sha "$RECORD_TEMPLATE" "$EXPECTED_RECORD_CONFIG_SHA"
require_sha "$OBSERVER_TEMPLATE" "$EXPECTED_OBSERVER_CONFIG_SHA"
require_sha "$MODEL" "$EXPECTED_MODEL_SHA"
require_sha "$CLIP" "$EXPECTED_CLIP_SHA"
require_sha "$CLASS_FEATURES" "$EXPECTED_CLASS_FEATURES_SHA"
require_sha "$CLASS_TXT" "$EXPECTED_CLASS_TXT_SHA"
require_sha "$PST" "$EXPECTED_PST_SHA"
require_sha "$BOXER_CHECKPOINT" "$EXPECTED_BOXER_SHA"
[[ "$(file_sha "$DINO_CHECKPOINT")" == "$EXPECTED_DINO_SHA" ]] || die "DINOv3 SHA256 mismatch"
[[ "$(git -C "$BOXER_OFFICIAL" rev-parse HEAD)" == "$EXPECTED_BOXER_COMMIT" ]] || die "Boxer source commit mismatch"

mapfile -t SCENES < <(sed -e 's/[[:space:]]*$//' -e '/^$/d' "$SCENE_LIST")
[[ "${#SCENES[@]}" == "103" ]] || die "canonical collection requires 103 scenes"
[[ "$(printf '%s\n' "${SCENES[@]}" | sort -u | wc -l)" == "103" ]] || die "duplicate canonical scenes"

PROPOSAL_FINGERPRINT="$(printf '%s\n%s\n%s\n%s\n' \
    "$NAMESPACE" "$EXPECTED_SCENE_LIST_SHA" "$EXPECTED_MODEL_SHA" \
    "$EXPECTED_RECORD_CONFIG_SHA" | sha256sum | awk '{print $1}')"
CODE_SOURCES=(
    demo.py boxfusion/capture_stream.py boxfusion/proposal_cache.py boxfusion/boxer_lifter.py
    boxfusion/ca1m_native_b6_observer.py boxfusion/tr3d_r2_geometry.py
    boxfusion/tr3d_r4_smov_observer.py boxfusion/tr3d_terminal_active.py
    config/ca1m_native_b6_canonical103_cutr_record.yaml
    config/ca1m_native_b6_canonical103_g0_observer.yaml
    tools/materialize_ca1m_native_b6_canonical_config.py
    tools/audit_ca1m_native_b6_canonical_inputs.py
    tools/audit_ca1m_native_b6_canonical_assets.py
    tools/finalize_ca1m_native_b6_canonical_artifact.py
    tools/finalize_ca1m_native_b6_train_artifact.py
    tools/audit_ca1m_native_b6_canonical_identity.py
    tools/audit_ca1m_c3_native_b6_observer.py
    tools/audit_ca1m_rgbd_file_manifest.py
    scripts/collect_ca1m_native_b6_canonical103.sh
)
PREFLIGHT="$(mktemp -d /tmp/ca1m-native-b6-canonical103.XXXXXX)"
CODE_MANIFEST_TMP="$PREFLIGHT/code.tsv"
for source in "${CODE_SOURCES[@]}"; do
    [[ -f "$ROOT/$source" ]] || die "Missing code source: $source"
    printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_MANIFEST_TMP"
done
CODE_FINGERPRINT="$(file_sha "$CODE_MANIFEST_TMP")"
trap 'rm -rf "$PREFLIGHT"' EXIT

"$PYTHON" "$ROOT/tools/audit_ca1m_native_b6_canonical_inputs.py" \
    --scene-list "$SCENE_LIST" --excluded-scene-list "$EXCLUDED_LIST" \
    --official-url-list "$OFFICIAL_URL_LIST" --data-root "$DATA_ROOT" \
    --preflight --output "$PREFLIGHT/input_preflight.json" >/dev/null
"$PYTHON" "$ROOT/tools/audit_ca1m_native_b6_canonical_assets.py" \
    --scene-list "$SCENE_LIST" --c0-root "$HISTORICAL_C0" \
    --legacy-cache-root "$LEGACY_CACHE" --output "$PREFLIGHT/existing_assets.json" >/dev/null
"$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_canonical_config.py" \
    --template "$RECORD_TEMPLATE" --phase record --data-root "$DATA_ROOT" \
    --output-root "$PREFLIGHT/record_pred" --cache-root "$PREFLIGHT/cache" \
    --output "$PREFLIGHT/record.yaml" >/dev/null
"$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_canonical_config.py" \
    --template "$OBSERVER_TEMPLATE" --phase observer --data-root "$DATA_ROOT" \
    --output-root "$PREFLIGHT/observer_pred" --cache-root "$PREFLIGHT/cache" \
    --baseline-root "$PREFLIGHT/record_pred" --native-diagnostics-root "$PREFLIGHT/native" \
    --boxer-diagnostics-root "$PREFLIGHT/boxer" --output "$PREFLIGHT/observer.yaml" >/dev/null

echo "CA-1M canonical103 GT-free native-B6 collection preflight"
echo "  official scene list: 103; SHA256=$EXPECTED_SCENE_LIST_SHA"
echo "  score/gap: real detector score, threshold=0.4, gap=20"
echo "  record: live CuTR -> independent immutable cache:$NAMESPACE"
echo "  replay: Selective Boxer G0 -> final-OBB native observer"
echo "  old C0/cache: audited only; never consumed"
echo "  GT/evaluation: forbidden"
echo "  proposal fingerprint: $PROPOSAL_FINGERPRINT"
if [[ "$MODE" == "preflight" ]]; then
    echo "Preflight passed; no GPU, evaluation, or formal namespace was started."
    exit 0
fi

mkdir -p "$LOCK_ROOT"
mkdir "$LOCK_DIR" || die "Another process owns canonical103 collection lock: $LOCK_DIR"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true; rm -rf "$RUNTIME_TMP" "$PREFLIGHT"' EXIT
mkdir -p "$RUNTIME_BASE"
mkdir "$RUNTIME_TMP" || die "Failed to claim exclusive runtime namespace: $RUNTIME_TMP"
mkdir -p "$REPORT_ROOT" "$STAGING_ROOT" "$RECORD_ROOT" \
    "$OBSERVER_ROOT" "$ANCHOR_ROOT" "$CACHE_ROOT/$NAMESPACE" "$NATIVE_ROOT" \
    "$BOXER_ROOT" "$LOG_ROOT/cutr_record" "$LOG_ROOT/g0_observer" \
    "$RECORD_COMPLETIONS" "$OBSERVER_COMPLETIONS"

"$PYTHON" "$ROOT/tools/audit_ca1m_native_b6_canonical_inputs.py" \
    --scene-list "$SCENE_LIST" --excluded-scene-list "$EXCLUDED_LIST" \
    --official-url-list "$OFFICIAL_URL_LIST" --data-root "$DATA_ROOT" \
    --output "$RUNTIME_TMP/input_audit.json" >/dev/null
"$PYTHON" "$ROOT/tools/audit_ca1m_rgbd_file_manifest.py" \
    --data-root "$DATA_ROOT" --scene-list "$SCENE_LIST" \
    --output "$RUNTIME_TMP/rgbd_file_manifest.json" >/dev/null
publish_immutable "$RUNTIME_TMP/input_audit.json" "$REPORT_ROOT/input_audit.json"
publish_immutable "$RUNTIME_TMP/rgbd_file_manifest.json" "$REPORT_ROOT/rgbd_file_manifest.json"
publish_immutable "$PREFLIGHT/existing_assets.json" "$REPORT_ROOT/existing_assets_audit.json"
publish_immutable "$CODE_MANIFEST_TMP" "$REPORT_ROOT/code_manifest.tsv"
INPUT_COLLECTION_SHA="$($PYTHON -c 'import json,sys;print(json.load(open(sys.argv[1]))["collection_sha256"])' "$REPORT_ROOT/rgbd_file_manifest.json")"

cat > "$RUNTIME_TMP/protocol.txt" <<EOF
schema=boxfusion.ca1m_native_b6_canonical103_protocol.v1
dataset_split=official_validation_canonical103
scene_list_sha256=$EXPECTED_SCENE_LIST_SHA
score_thresh=0.4
score_export=real_detector_score
gap=20
proposal_cache_namespace=$NAMESPACE
proposal_fingerprint=$PROPOSAL_FINGERPRINT
input_collection_sha256=$INPUT_COLLECTION_SHA
code_fingerprint=$CODE_FINGERPRINT
same_run_anchor_byte_identity_required=true
ground_truth_access=false
evaluation_invoked=false
training_authorized=false
EOF
publish_immutable "$RUNTIME_TMP/protocol.txt" "$REPORT_ROOT/protocol.txt"

finalize_record() {
    local scene="$1"
    "$PYTHON" "$ROOT/tools/finalize_ca1m_native_b6_canonical_artifact.py" record \
        --scene "$scene" --prediction "$RECORD_ROOT/${scene}_boxes.pkl" \
        --cache-scene-root "$CACHE_ROOT/$NAMESPACE/$scene" \
        --cache-namespace "$NAMESPACE" --proposal-fingerprint "$PROPOSAL_FINGERPRINT" \
        --log "$LOG_ROOT/cutr_record/${scene}.log" \
        --output "$RECORD_COMPLETIONS/${scene}.json" >/dev/null
}

finalize_observer() {
    local scene="$1"
    "$PYTHON" "$ROOT/tools/finalize_ca1m_native_b6_canonical_artifact.py" observer \
        --scene "$scene" --prediction "$OBSERVER_ROOT/${scene}_boxes.pkl" \
        --anchor "$ANCHOR_ROOT/${scene}_boxes.pkl" \
        --diagnostic "$NATIVE_ROOT/${scene}_ca1m_native_b6.npz" \
        --boxer "$BOXER_ROOT/${scene}_boxer_lifting.jsonl" \
        --log "$LOG_ROOT/g0_observer/${scene}.log" \
        --output "$OBSERVER_COMPLETIONS/${scene}.json" >/dev/null
}

run_scene() {
    local scene="$1" gpu="$2" stage cfg
    local record_pred="$RECORD_ROOT/${scene}_boxes.pkl"
    local cache_scene="$CACHE_ROOT/$NAMESPACE/$scene"
    local record_log="$LOG_ROOT/cutr_record/${scene}.log"
    if [[ -e "$RECORD_COMPLETIONS/${scene}.json" ]]; then
        finalize_record "$scene"
    elif [[ -s "$record_pred" && -d "$cache_scene" && -s "$record_log" ]]; then
        finalize_record "$scene"
    else
        [[ ! -e "$record_pred" && ! -e "$cache_scene" && ! -e "$record_log" ]] \
            || die "$scene: partial permanent CuTR artifacts; refusing overwrite"
        stage="$STAGING_ROOT/${scene}.record.$BASHPID"
        mkdir "$stage" "$stage/pred" "$stage/cache"
        cfg="$stage/config.yaml"
        "$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_canonical_config.py" \
            --template "$RECORD_TEMPLATE" --phase record --data-root "$DATA_ROOT" \
            --output-root "$stage/pred" --cache-root "$stage/cache" --output "$cfg" >/dev/null
        env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=0 \
            PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
            BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT="$PROPOSAL_FINGERPRINT" \
            LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
            MPLCONFIGDIR="$RUNTIME_TMP/mpl_record_$gpu" XDG_CACHE_HOME="$RUNTIME_TMP/model_record_$gpu" \
            "$PYTHON" "$ROOT/demo.py" CA1M --model-path "$MODEL" --clip_path "$CLIP" \
                --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES" \
                --config "$cfg" --output-dir "$stage/pred" --device cuda --seq "$scene" --seed 0 \
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
        mkdir "$stage" "$stage/pred" "$stage/anchor" "$stage/native" "$stage/boxer"
        cfg="$stage/config.yaml"
        "$PYTHON" "$ROOT/tools/materialize_ca1m_native_b6_canonical_config.py" \
            --template "$OBSERVER_TEMPLATE" --phase observer --data-root "$DATA_ROOT" \
            --output-root "$stage/pred" --cache-root "$CACHE_ROOT" --baseline-root "$RECORD_ROOT" \
            --native-diagnostics-root "$stage/native" --boxer-diagnostics-root "$stage/boxer" \
            --output "$cfg" >/dev/null
        env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=0 \
            PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
            BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$PROPOSAL_FINGERPRINT" \
            LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
            MPLCONFIGDIR="$RUNTIME_TMP/mpl_observer_$gpu" XDG_CACHE_HOME="$RUNTIME_TMP/model_observer_$gpu" \
            "$PYTHON" "$ROOT/demo.py" CA1M --model-path "$MODEL" --clip_path "$CLIP" \
                --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES" \
                --config "$cfg" --output-dir "$stage/pred" \
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
    echo "[$(date '+%F %T')] [GPU $gpu] canonical observer complete: $scene"
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
(( failures == 0 )) || die "At least one canonical103 collection worker failed"

AFTER="$RUNTIME_TMP/code_after.tsv"
for source in "${CODE_SOURCES[@]}"; do
    printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$AFTER"
done
[[ "$(file_sha "$AFTER")" == "$CODE_FINGERPRINT" ]] || die "Code changed during collection"
"$PYTHON" "$ROOT/tools/finalize_ca1m_native_b6_canonical_artifact.py" collection \
    --scene-list "$SCENE_LIST" --record-completion-root "$RECORD_COMPLETIONS" \
    --observer-completion-root "$OBSERVER_COMPLETIONS" \
    --output "$REPORT_ROOT/collection_manifest.json" >/dev/null
"$PYTHON" "$ROOT/tools/audit_ca1m_native_b6_canonical_identity.py" \
    --scene-list "$SCENE_LIST" --anchor-root "$ANCHOR_ROOT" \
    --observer-root "$OBSERVER_ROOT" --diagnostics-root "$NATIVE_ROOT" \
    --boxer-root "$BOXER_ROOT" --log-root "$LOG_ROOT/g0_observer" \
    --output "$REPORT_ROOT/identity_audit.json"
chmod -R a-w "$RECORD_ROOT" "$OBSERVER_ROOT" "$ANCHOR_ROOT" \
    "$CACHE_ROOT" "$ROOT/diagnostics/ca1m_port/$TAG" "$REPORT_ROOT" "$LOG_ROOT"
echo "CA-1M canonical103 native-B6 observer collection complete; no evaluation was performed."
echo "  $REPORT_ROOT/identity_audit.json"
