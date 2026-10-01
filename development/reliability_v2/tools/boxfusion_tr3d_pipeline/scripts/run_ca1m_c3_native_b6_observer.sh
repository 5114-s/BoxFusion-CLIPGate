#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON_BIN="${BOXFUSION_PYTHON:-$ENV_ROOT/bin/python}"
TAG="${BOXFUSION_CA1M_C3_RUN_TAG:-ca1m_c3_native_b6_observer_fixed10_v1}"
SCENE_LIST="${BOXFUSION_CA1M_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_ablation10_even.txt}"
DATA_ROOT="${BOXFUSION_CA1M_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
CONFIG="${BOXFUSION_CA1M_C3_CONFIG:-$ROOT/config/ca1m_c3_native_b6_observer.yaml}"
P1_CONFIG="${BOXFUSION_CA1M_C1_CONFIG:-$ROOT/config/ca1m_c1_selective_boxer_paired.yaml}"
P0_ROOT="${BOXFUSION_CA1M_C0_PRED_ROOT:-$ROOT/results/ca1m_port/c0_original_fixed10_v2}"
P1_ROOT="${BOXFUSION_CA1M_C1_PRED_ROOT:-$ROOT/results/ca1m_port/c1_selective_boxer_fixed10_v2}"
P1_BOXER_ROOT="${BOXFUSION_CA1M_C1_BOXER_ROOT:-$ROOT/diagnostics/ca1m_port/c1_selective_boxer_fixed10_v2}"
P1_LOG_ROOT="${BOXFUSION_CA1M_C1_LOG_ROOT:-$ROOT/logs/ca1m_port/c01_paired_fixed10_v2/c1}"
CACHE_ROOT="${BOXFUSION_CA1M_CUTR_CACHE_ROOT:-$ROOT/cache/ca1m_cutr_proposals/ca1m-score04-gap20-c0-v2}"
C3_ROOT="${BOXFUSION_CA1M_C3_PRED_ROOT:-$ROOT/results/ca1m_port/$TAG}"
SAME_RUN_ANCHOR_ROOT="${BOXFUSION_CA1M_C3_SAME_RUN_ANCHOR_ROOT:-$ROOT/results/ca1m_port/${TAG}_same_run_anchor}"
DIAGNOSTICS_ROOT="${BOXFUSION_CA1M_C3_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/ca1m_port/$TAG}"
NATIVE_DIAGNOSTICS="$DIAGNOSTICS_ROOT/native_b6"
BOXER_DIAGNOSTICS="$DIAGNOSTICS_ROOT/boxer"
LOG_ROOT="${BOXFUSION_CA1M_C3_LOG_ROOT:-$ROOT/logs/ca1m_port/$TAG}"
REPORT_ROOT="${BOXFUSION_CA1M_C3_REPORT_ROOT:-$ROOT/reports/ca1m_port/$TAG}"
EVAL_VIEW="${BOXFUSION_CA1M_C3_EVAL_VIEW:-$ROOT/data/ca1m_eval_$TAG}"
TAG_SHA="$(printf '%s' "$TAG" | sha256sum | cut -c1-12)"
RUNTIME_TMP="${BOXFUSION_RUNTIME_TMP_ROOT:-/tmp/bfc-ca1m-c3-$TAG_SHA}"
MODEL="$LIVE_ROOT/models/cutr_rgbd.pth"
CLIP="$LIVE_ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
CLASS_FEATURES="$LIVE_ROOT/data/class_features.pt"
PST="$LIVE_ROOT/data/pst_1024_0.tiff"
LOCK_ROOT="${BOXFUSION_RUN_LOCK_ROOT:-/tmp/boxfusion_ca1m_runlocks}"
LOCK_DIR="$LOCK_ROOT/$TAG.lock"

EXPECTED_SCENE_LIST_SHA="b81bd6a2f147f964c6a94f3ed838edc1d0f3e801ae642d8ba30b85c643aebeab"
EXPECTED_CONFIG_SHA="f88249f288ab808340e0972f77308748b5e7b29a9f3d4e5d20edc4022346fa75"
EXPECTED_P1_CONFIG_SHA="0fbe23ce42e7f3ff5ade1ed9bb1cc6632c04b4e7bdc013ea23a6bd3b7fb42d09"
EXPECTED_MODEL_SHA="856b89c62c49d518998eeef52db16eadede5c354c6e2dfb291e16fd2887a4217"
EXPECTED_CLIP_SHA="9a78ef8e8c73fd0df621682e7a8e8eb36c6916cb3c16b291a082ecd52ab79cc4"
EXPECTED_CLASS_FEATURES_SHA="49ab2384fbc01406eb7eb24ce89403bbfa9516bc213e11e8cd2014fa8eeea197"
EXPECTED_CLASS_TXT_SHA="0d628e3140d491acfce107268fe51233e1df44f84581f582fe253842fc6557c9"
EXPECTED_EVALUATOR_SHA="8b6de5410b11978076c3ec1cf3e54bfa58a17d13cbdc6b06d9d7d976cf4e0f77"
EXPECTED_EVAL_DET_SHA="6ef54c395e46716e364547115090bae96643bf346b3e8eb1b859719781a557dd"
EXPECTED_PST_SHA="867f0546addc35a5000a421e9f81af4577470751b7a8ffc28e859cca97376660"
EXPECTED_P1_COLLECTION_SHA="c521652ccb77e489037d8bc9e8d91c5d18bf7fc057f249e18ab6d0977428ce5b"
EXPECTED_PROPOSAL_FINGERPRINT="991d51281617a2731784d09be4d8b2839290b6304e5ca5aa735ad1d4e6f8ccb6"

die() { echo "$*" >&2; exit 2; }
file_sha() { sha256sum "$1" | awk '{print $1}'; }
require_sha() {
    local path="$1" expected="$2" actual
    [[ -f "$path" ]] || die "Missing frozen input: $path"
    actual="$(file_sha "$path")"
    [[ "$actual" == "$expected" ]] || die "SHA256 mismatch: $path ($actual != $expected)"
}

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
(( ${#GPUS[@]} >= 1 )) || die "No GPUs specified"
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "Invalid GPU identifier: $gpu"
done

for path in "$PYTHON_BIN" "$SCENE_LIST" "$CONFIG" "$P1_CONFIG" "$MODEL" "$CLIP" \
    "$CLASS_TXT" "$CLASS_FEATURES" "$PST" \
    "$ROOT/demo.py" "$ROOT/boxfusion/ca1m_native_b6_observer.py" \
    "$ROOT/tools/audit_ca1m_c3_native_b6_observer.py"; do
    [[ -f "$path" ]] || die "Missing input: $path"
done
for path in "$DATA_ROOT" "$P0_ROOT" "$P1_ROOT" "$P1_BOXER_ROOT" "$P1_LOG_ROOT" \
    "$CACHE_ROOT" "$ROOT/evaluation"; do
    [[ -d "$path" ]] || die "Missing input directory: $path"
done
for path in "$C3_ROOT" "$SAME_RUN_ANCHOR_ROOT" "$DIAGNOSTICS_ROOT" "$LOG_ROOT" \
    "$REPORT_ROOT" "$EVAL_VIEW" "$RUNTIME_TMP"; do
    [[ ! -e "$path" ]] || die "Refusing existing C3 namespace: $path"
done

require_sha "$SCENE_LIST" "$EXPECTED_SCENE_LIST_SHA"
require_sha "$CONFIG" "$EXPECTED_CONFIG_SHA"
require_sha "$P1_CONFIG" "$EXPECTED_P1_CONFIG_SHA"
require_sha "$MODEL" "$EXPECTED_MODEL_SHA"
require_sha "$CLIP" "$EXPECTED_CLIP_SHA"
require_sha "$CLASS_FEATURES" "$EXPECTED_CLASS_FEATURES_SHA"
require_sha "$CLASS_TXT" "$EXPECTED_CLASS_TXT_SHA"
require_sha "$ROOT/evaluation/eval_ca1m.py" "$EXPECTED_EVALUATOR_SHA"
require_sha "$ROOT/evaluation/utils/eval_det.py" "$EXPECTED_EVAL_DET_SHA"
require_sha "$PST" "$EXPECTED_PST_SHA"

mapfile -t SCENES < <(sed -e 's/[[:space:]]*$//' -e '/^$/d' "$SCENE_LIST")
EXPECTED_SCENES="${#SCENES[@]}"
(( EXPECTED_SCENES == 10 )) || die "C3 fixed10 requires exactly 10 scenes, found $EXPECTED_SCENES"
[[ "$(printf '%s\n' "${SCENES[@]}" | sort -u | wc -l)" == "$EXPECTED_SCENES" ]] \
    || die "Scene list contains duplicate identifiers"

mkdir -p "$LOCK_ROOT"
mkdir "$LOCK_DIR" || die "Another run owns C3 lock: $LOCK_DIR"
ANCHOR_MANIFEST=""
trap '[[ -z "$ANCHOR_MANIFEST" ]] || rm -f "$ANCHOR_MANIFEST"; rmdir "$RUNTIME_TMP" 2>/dev/null || true; rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT
mkdir -p "$(dirname "$RUNTIME_TMP")"
mkdir "$RUNTIME_TMP" || die "Failed to claim exclusive runtime namespace: $RUNTIME_TMP"
ANCHOR_MANIFEST="$(mktemp "$RUNTIME_TMP/anchor.XXXXXX")"
for scene in "${SCENES[@]}"; do
    prediction="$P1_ROOT/${scene}_boxes.pkl"
    boxer="$P1_BOXER_ROOT/${scene}_boxer_lifting.jsonl"
    cache_manifest="$CACHE_ROOT/${scene}/manifest.json"
    p0_prediction="$P0_ROOT/${scene}_boxes.pkl"
    scene_root="$DATA_ROOT/$scene"
    [[ -d "$scene_root" && ! -L "$scene_root" ]] || die "Missing CA-1M scene: $scene_root"
    for path in "$prediction" "$boxer" "$cache_manifest" "$p0_prediction" "$P1_LOG_ROOT/${scene}.log"; do
        [[ -s "$path" ]] || die "Missing frozen P0/P1/cache artifact: $path"
    done
    "$PYTHON_BIN" "$ROOT/tools/validate_ca1m_prediction_file.py" --prediction "$prediction"
    "$PYTHON_BIN" "$ROOT/tools/validate_ca1m_prediction_file.py" --prediction "$p0_prediction"
    "$PYTHON_BIN" - "$cache_manifest" "$scene" "$EXPECTED_PROPOSAL_FINGERPRINT" <<'PY'
import json, sys
from pathlib import Path
path, scene, fingerprint = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
value = json.loads(path.read_text())
if value.get("namespace") != "ca1m-score04-gap20-c0-v2":
    raise SystemExit(f"cache namespace drift: {path}")
if value.get("producer_fingerprint") != fingerprint:
    raise SystemExit(f"cache fingerprint drift: {path}")
if value.get("prediction_file") != f"{scene}_boxes.pkl":
    raise SystemExit(f"cache scene/prediction drift: {path}")
PY
    printf '%s %s %s %s\n' "$scene" "$(file_sha "$prediction")" \
        "$(file_sha "$boxer")" "$(file_sha "$cache_manifest")" >> "$ANCHOR_MANIFEST"
done
ANCHOR_COLLECTION_SHA="$(sha256sum "$ANCHOR_MANIFEST" | awk '{print $1}')"
[[ "$ANCHOR_COLLECTION_SHA" == "$EXPECTED_P1_COLLECTION_SHA" ]] \
    || die "Frozen P1 collection SHA drifted: $ANCHOR_COLLECTION_SHA"

# Prove that the only algorithmic addition to P1 is the independent C3
# observer.  In particular, the old YOLOE/evolving-track system stays off.
"$PYTHON_BIN" - "$P1_CONFIG" "$CONFIG" "$DATA_ROOT" "$CACHE_ROOT" "$P0_ROOT" "$PST" <<'PY'
import copy, sys, yaml
from pathlib import Path
p1_path, c3_path, data_root, cache_root, p0_root, pst = map(Path, sys.argv[1:])
p1 = yaml.safe_load(p1_path.read_text())
c3 = yaml.safe_load(c3_path.read_text())
left, right = copy.deepcopy(p1), copy.deepcopy(c3)
observer = right.pop("ca1m_native_b6_observer", None)
left["data"].pop("output_dir", None)
right["data"].pop("output_dir", None)
left["lifting"]["boxer"].pop("diagnostics_dir", None)
right["lifting"]["boxer"].pop("diagnostics_dir", None)
if left != right:
    raise SystemExit("C3 changes frozen P1 algorithm fields")
if not isinstance(observer, dict) or observer.get("enabled") is not True:
    raise SystemExit("C3 native observer must be enabled")
required = {
    "observer_only": observer.get("observer_only") is True,
    "top_k_views": int(observer.get("top_k_views", 0)) == 5,
    "pixel_stride": int(observer.get("pixel_stride", 0)) == 4,
    "online_disabled": c3.get("online_refinement") == {"enabled": False},
    "data_root": Path(c3["data"]["datadir"]).resolve().parents[1] == data_root.resolve(),
    "proposal_cache": (
        Path(c3["lifting"]["proposal_cache"]["root"])
        / c3["lifting"]["proposal_cache"]["namespace"]
    ).resolve() == cache_root.resolve(),
    "baseline_prediction_root": Path(
        c3["lifting"]["proposal_cache"]["baseline_prediction_root"]
    ).resolve() == p0_root.resolve(),
    "pst": Path(c3["box_fusion"]["pst_path"]).resolve() == pst.resolve(),
}
failed = [key for key, value in required.items() if not value]
if failed:
    raise SystemExit("C3 configuration contract failed: " + ", ".join(failed))
print("CA-1M C3 configuration identity contract OK")
PY

if [[ "${BOXFUSION_CA1M_C3_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "CA-1M C3 preflight passed; no formal output namespace was created"
    exit 0
fi

mkdir -p "$C3_ROOT" "$SAME_RUN_ANCHOR_ROOT" "$NATIVE_DIAGNOSTICS" \
    "$BOXER_DIAGNOSTICS" "$LOG_ROOT/inference" "$REPORT_ROOT" "$EVAL_VIEW"
for shard in "${!GPUS[@]}"; do
    mkdir -p "$RUNTIME_TMP/mplconfig_$shard" "$RUNTIME_TMP/model_cache_$shard" "$RUNTIME_TMP/tmp_$shard"
done
rm -f "$ANCHOR_MANIFEST"
ANCHOR_MANIFEST=""
for scene in "${SCENES[@]}"; do
    ln -s "$DATA_ROOT/$scene" "$EVAL_VIEW/$scene"
done

INPUT_AUDIT="$REPORT_ROOT/input_audit.json"
"$PYTHON_BIN" "$ROOT/tools/audit_ca1m_rgbd_pose.py" \
    --data-root "$DATA_ROOT" --scene-list "$SCENE_LIST" --min-frames 1 --output "$INPUT_AUDIT"
RGBD_MANIFEST="$REPORT_ROOT/rgbd_file_manifest.json"
"$PYTHON_BIN" "$ROOT/tools/audit_ca1m_rgbd_file_manifest.py" \
    --data-root "$DATA_ROOT" --scene-list "$SCENE_LIST" --output "$RGBD_MANIFEST"
RGBD_COLLECTION_SHA="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["collection_sha256"])' "$RGBD_MANIFEST")"

CODE_MANIFEST="$REPORT_ROOT/code_manifest.tsv"
CODE_SOURCES=(
    demo.py boxfusion/box_fusion.py boxfusion/boxer_lifter.py
    boxfusion/capture_stream.py boxfusion/instances.py
    boxfusion/ca1m_native_b6_observer.py boxfusion/tr3d_r2_geometry.py
    boxfusion/tr3d_r4_smov_observer.py boxfusion/tr3d_terminal_active.py
    evaluation/eval_ca1m.py evaluation/utils/eval_det.py
    config/ca1m_c3_native_b6_observer.yaml
    scripts/run_ca1m_c3_native_b6_observer.sh
    tools/audit_ca1m_c3_native_b6_observer.py
    tools/audit_ca1m_rgbd_pose.py tools/audit_ca1m_rgbd_file_manifest.py
    tools/validate_ca1m_prediction_file.py
)
for source in "${CODE_SOURCES[@]}"; do
    [[ -f "$ROOT/$source" ]] || die "Missing code source: $source"
    printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$CODE_MANIFEST"
done
CODE_FINGERPRINT="$(file_sha "$CODE_MANIFEST")"

cat > "$REPORT_ROOT/protocol_manifest.txt" <<EOF
schema=boxfusion.ca1m_c3_native_b6_protocol.v1
dataset=CA1M
stage=C3
mode=final_obb_native_b6_observer
output_mutation_authorized=false
identity_anchor_contract=same_run_pre_native_b6_finalize
online_refinement_enabled=false
target_dataset_training_used=false
scene_list=$SCENE_LIST
scene_list_sha256=$EXPECTED_SCENE_LIST_SHA
config=$CONFIG
config_sha256=$EXPECTED_CONFIG_SHA
p1_collection_sha256=$ANCHOR_COLLECTION_SHA
proposal_fingerprint=$EXPECTED_PROPOSAL_FINGERPRINT
rgbd_collection_sha256=$RGBD_COLLECTION_SHA
code_fingerprint=$CODE_FINGERPRINT
top_k_views=5
pixel_stride=4
depth_margin_m=0.05
geometry=world_yaw_OBB
EOF
chmod 0444 "$REPORT_ROOT/protocol_manifest.txt" "$INPUT_AUDIT" "$RGBD_MANIFEST" "$CODE_MANIFEST"

echo "CA-1M C3 native B6 final-OBB observer"
echo "  scenes: $EXPECTED_SCENES from $SCENE_LIST"
echo "  GPUs: $GPU_SPEC"
echo "  frozen parent: Selective Boxer G0 P1"
echo "  proposal cache: replay:ca1m-score04-gap20-c0-v2"
echo "  same-run anchor: $SAME_RUN_ANCHOR_ROOT"
echo "  predictions: $C3_ROOT"
echo "  diagnostics: $NATIVE_DIAGNOSTICS"
echo "  mutation: disabled"

workers="${#GPUS[@]}"
pids=()
failures=0
for shard in "${!GPUS[@]}"; do
    (
        for index in "${!SCENES[@]}"; do
            (( index % workers == shard )) || continue
            scene="${SCENES[$index]}"
            log="$LOG_ROOT/inference/${scene}.log"
            echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] Running $scene (list index $((index + 1))/$EXPECTED_SCENES)"
            env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${GPUS[$shard]}" \
                PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
                LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
                MPLCONFIGDIR="$RUNTIME_TMP/mplconfig_$shard" \
                XDG_CACHE_HOME="$RUNTIME_TMP/model_cache_$shard" \
                TMPDIR="$RUNTIME_TMP/tmp_$shard" TMP="$RUNTIME_TMP/tmp_$shard" TEMP="$RUNTIME_TMP/tmp_$shard" \
                BOXFUSION_PROPOSAL_CACHE_EXPECTED_FINGERPRINT="$EXPECTED_PROPOSAL_FINGERPRINT" \
                "$PYTHON_BIN" "$ROOT/demo.py" CA1M \
                    --model-path "$MODEL" --clip_path "$CLIP" \
                    --class_txt "$CLASS_TXT" --class-features "$CLASS_FEATURES" \
                    --config "$CONFIG" --output-dir "$C3_ROOT" \
                    --ca1m-native-b6-same-run-anchor-root "$SAME_RUN_ANCHOR_ROOT" \
                    --ca1m-native-b6-diagnostics-root "$NATIVE_DIAGNOSTICS" \
                    --boxer-diagnostics-root "$BOXER_DIAGNOSTICS" \
                    --device cuda --seq "$scene" --seed 0 \
                    > "$log" 2>&1
            prediction="$C3_ROOT/${scene}_boxes.pkl"
            anchor="$SAME_RUN_ANCHOR_ROOT/${scene}_boxes.pkl"
            diagnostic="$NATIVE_DIAGNOSTICS/${scene}_ca1m_native_b6.npz"
            boxer="$BOXER_DIAGNOSTICS/${scene}_boxer_lifting.jsonl"
            for path in "$prediction" "$anchor" "$diagnostic" "$boxer"; do
                [[ -s "$path" ]] || { echo "Incomplete C3 artifact: $path" >&2; exit 1; }
            done
            "$PYTHON_BIN" "$ROOT/tools/validate_ca1m_prediction_file.py" --prediction "$prediction"
            "$PYTHON_BIN" "$ROOT/tools/validate_ca1m_prediction_file.py" --prediction "$anchor"
            rg -q 'CA-1M native B6 observer summary' "$log" \
                || { echo "Missing native observer completion marker: $log" >&2; exit 1; }
            echo "[$(date '+%F %T')] [GPU ${GPUS[$shard]}] Completed $scene"
        done
    ) &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid" || failures=1
done
(( failures == 0 )) || { echo "At least one C3 worker failed" >&2; exit 1; }

FINAL_CODE_MANIFEST="$RUNTIME_TMP/code_manifest_after.tsv"
for source in "${CODE_SOURCES[@]}"; do
    printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$FINAL_CODE_MANIFEST"
done
[[ "$(file_sha "$FINAL_CODE_MANIFEST")" == "$CODE_FINGERPRINT" ]] \
    || die "Core code changed during C3 inference"
require_sha "$CONFIG" "$EXPECTED_CONFIG_SHA"

for suffix_root in "$C3_ROOT:_boxes.pkl" "$SAME_RUN_ANCHOR_ROOT:_boxes.pkl" \
    "$NATIVE_DIAGNOSTICS:_ca1m_native_b6.npz" "$BOXER_DIAGNOSTICS:_boxer_lifting.jsonl"; do
    artifact_root="${suffix_root%%:*}"
    suffix="${suffix_root#*:}"
    count="$(find "$artifact_root" -maxdepth 1 -type f -name "*${suffix}" | wc -l)"
    [[ "$count" == "$EXPECTED_SCENES" ]] || die "$artifact_root has $count/$EXPECTED_SCENES artifacts"
done

"$PYTHON_BIN" "$ROOT/tools/audit_ca1m_c3_native_b6_observer.py" \
    --scene-list "$SCENE_LIST" \
    --anchor-root "$SAME_RUN_ANCHOR_ROOT" --observer-root "$C3_ROOT" \
    --diagnostics-root "$NATIVE_DIAGNOSTICS" \
    --historical-prediction-root "$P1_ROOT" \
    --historical-boxer-root "$P1_BOXER_ROOT" --observer-boxer-root "$BOXER_DIAGNOSTICS" \
    --historical-log-root "$P1_LOG_ROOT" --observer-log-root "$LOG_ROOT/inference" \
    --output "$REPORT_ROOT/identity_audit.json"

(
    cd "$ROOT/evaluation"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
        MPLCONFIGDIR="$RUNTIME_TMP/mplconfig_eval" \
        "$PYTHON_BIN" eval_ca1m.py --dataset ca1m \
            --data_path "$EVAL_VIEW" --pred_root "$C3_ROOT" \
            --ap_iou_thresholds 0.15,0.25,0.5 --num_workers 0 \
            --cluster_sampling seed_fps --use_3d_nms --use_cls_nms \
            --per_class_proposal --gpu "${GPUS[0]}"
) > "$LOG_ROOT/eval.log" 2>&1

FINAL_CODE_MANIFEST="$RUNTIME_TMP/code_manifest_after_reports.tsv"
: > "$FINAL_CODE_MANIFEST"
for source in "${CODE_SOURCES[@]}"; do
    printf '%s\t%s\n' "$(file_sha "$ROOT/$source")" "$source" >> "$FINAL_CODE_MANIFEST"
done
[[ "$(file_sha "$FINAL_CODE_MANIFEST")" == "$CODE_FINGERPRINT" ]] \
    || die "Core code changed while C3 reports were generated"

chmod -R a-w "$C3_ROOT" "$SAME_RUN_ANCHOR_ROOT" "$DIAGNOSTICS_ROOT" "$REPORT_ROOT" "$LOG_ROOT"
echo "=== CA-1M C3 observer (must equal same-run G0 anchor) ==="
grep -E 'eval (mAP|APrec|ARecall):|mAP:' "$LOG_ROOT/eval.log" | tail -12
echo "CA-1M C3 native B6 observer completed"
echo "  identity/coverage/runtime: $REPORT_ROOT/identity_audit.json"
echo "  standard evaluation: $LOG_ROOT/eval.log"
