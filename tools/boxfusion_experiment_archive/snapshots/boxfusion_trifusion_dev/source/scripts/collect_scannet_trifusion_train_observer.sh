#!/usr/bin/env bash
set -euo pipefail

# Strict train-only TriFusion observer collection.
#
# This launcher intentionally calls demo.py per scene and never invokes
# evaluation/eval_scannet.py.  It is the only supported source of AP50 gate
# supervision.  The default is check-only; GPU work starts only with an
# explicit EXECUTE=1 (or BOXFUSION_TRIFUSION_PROTOCOL_EXECUTE=1).
#
# Required:
#   BOXFUSION_TRIFUSION_TRAIN_TEACHER_CACHE=/isolated/train/cache
#   BOXFUSION_TRIFUSION_TRAIN_TEACHER_METADATA_ROOT=/isolated/train/log/metadata
#   BOXFUSION_TRIFUSION_TRAIN_TEACHER_NAMESPACE=sam3-scannet18-train-...
#
# Check only:
#   bash scripts/collect_scannet_trifusion_train_observer.sh 0,1
#
# Execute after review:
#   EXECUTE=1 \
#   BOXFUSION_TRIFUSION_TRAIN_TEACHER_CACHE=... \
#   BOXFUSION_TRIFUSION_TRAIN_TEACHER_METADATA_ROOT=... \
#   BOXFUSION_TRIFUSION_TRAIN_TEACHER_NAMESPACE=... \
#     bash scripts/collect_scannet_trifusion_train_observer.sh 0,1

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXECUTE="${BOXFUSION_TRIFUSION_PROTOCOL_EXECUTE:-${EXECUTE:-0}}"
ALLOW_RESUME="${BOXFUSION_TRIFUSION_TRAIN_ALLOW_RESUME:-0}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
CONFIG="${BOXFUSION_TRIFUSION_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
YOLOE_CHECKPOINT="${BOXFUSION_TRIFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
QUALITY_CHECKPOINT="${BOXFUSION_TRIFUSION_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"

SCENE_LIST="${BOXFUSION_TRIFUSION_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FORBIDDEN_VAL_LIST="${BOXFUSION_TRIFUSION_FORBIDDEN_VAL_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
FRAMES_ROOT="${BOXFUSION_TRIFUSION_TRAIN_FRAMES_ROOT:-$ROOT/data/scannet_train}"
TRAIN_CACHE="${BOXFUSION_TRIFUSION_TRAIN_TEACHER_CACHE:-}"
TRAIN_METADATA_ROOT="${BOXFUSION_TRIFUSION_TRAIN_TEACHER_METADATA_ROOT:-}"
TRAIN_NAMESPACE="${BOXFUSION_TRIFUSION_TRAIN_TEACHER_NAMESPACE:-}"

RUN_TAG="${BOXFUSION_TRIFUSION_TRAIN_RUN_TAG:-trifusion_plus10_train_observer_v1}"
PRED_ROOT="${BOXFUSION_TRIFUSION_TRAIN_PRED_ROOT:-$ROOT/results/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_TRIFUSION_TRAIN_LOG_ROOT:-$ROOT/logs/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_TRIFUSION_TRAIN_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$RUN_TAG}"

die() {
    echo "TriFusion train collector: $*" >&2
    exit 1
}

canonical() {
    realpath -m -- "$1"
}

directory_nonempty() {
    local directory="$1"
    [[ -d "$directory" ]] \
        && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]
}

require_inside_root() {
    local name="$1"
    local path="$2"
    local root_resolved
    local path_resolved
    root_resolved="$(canonical "$ROOT")"
    path_resolved="$(canonical "$path")"
    case "$path_resolved" in
        "$root_resolved"/*) ;;
        *) die "$name must remain inside the isolated checkout: $path" ;;
    esac
}

validate_scene_list() {
    local role="$1"
    local path="$2"
    [[ -s "$path" ]] || die "missing or empty $role scene list: $path"
    local malformed
    malformed="$(
        awk '
            NF && (NF != 1 || $1 !~ /^scene[0-9][0-9][0-9][0-9]_[0-9][0-9]$/) {
                print NR ":" $0
                exit
            }
        ' "$path"
    )"
    [[ -z "$malformed" ]] \
        || die "malformed $role scene row in $path: $malformed"
    local duplicate
    duplicate="$(
        awk '
            NF {
                count[$1] += 1
                if (count[$1] == 2) { print $1; exit }
            }
        ' "$path"
    )"
    [[ -z "$duplicate" ]] \
        || die "duplicate $role scene in $path: $duplicate"
}

validate_no_overlap() {
    local train_list="$1"
    local forbidden_list="$2"
    local overlap
    overlap="$(
        awk '
            NR == FNR { if (NF) forbidden[$1] = 1; next }
            NF && ($1 in forbidden) { print $1; exit }
        ' "$forbidden_list" "$train_list"
    )"
    [[ -z "$overlap" ]] \
        || die "train/validation leakage; forbidden scene: $overlap"
}

validate_cache_provenance() {
    "$PYTHON" -c '
import glob
import hashlib
import json
import os
import sys

cache, metadata_root, namespace, scene_list, frames_root = sys.argv[1:6]
paths = sorted(glob.glob(os.path.join(metadata_root, "shard*.json")))
if not paths:
    raise SystemExit("missing builder shard*.json metadata")
payloads = [json.load(open(path, encoding="utf-8")) for path in paths]
schemas = {str(item.get("schema", "")) for item in payloads}
if schemas != {"boxfusion_scannet_sam3_teacher_cache_v1"}:
    raise SystemExit(
        "unsupported train cache metadata schema: "
        + repr(sorted(schemas))
    )
if not all(item.get("complete") is True for item in payloads):
    raise SystemExit("train cache metadata contains an incomplete shard")
namespaces = {str(item.get("namespace", "")) for item in payloads}
if namespaces != {namespace}:
    raise SystemExit(
        "train cache namespace mismatch: "
        + repr(sorted(namespaces))
    )
with open(scene_list, "rb") as handle:
    expected_sha = hashlib.sha256(handle.read()).hexdigest()
manifest_sha = {
    str(item.get("scene_list", {}).get("sha256", ""))
    for item in payloads
}
if manifest_sha != {expected_sha}:
    raise SystemExit(
        "train cache scene-list SHA mismatch: "
        + repr(sorted(manifest_sha))
    )
requested = {
    line.strip()
    for line in open(scene_list, encoding="utf-8")
    if line.strip()
}
selected_rows = [
    str(row.get("scene_id", ""))
    for item in payloads
    for row in item.get("scenes", [])
]
selected = set(selected_rows)
if len(selected_rows) != len(selected):
    raise SystemExit("train cache manifests contain duplicate scenes")
if selected != requested:
    missing = sorted(requested - selected)
    extra = sorted(selected - requested)
    raise SystemExit(
        f"train cache scene union mismatch: missing={missing[:4]}, "
        f"extra={extra[:4]}"
    )
counts = {
    int(item.get("scene_list", {}).get("all_scene_count", -1))
    for item in payloads
}
if counts != {len(requested)}:
    raise SystemExit(
        "train cache manifest scene count mismatch: " + repr(sorted(counts))
    )
manifest_frames = {
    os.path.realpath(str(item.get("frames_root", "")))
    for item in payloads
}
if manifest_frames != {os.path.realpath(frames_root)}:
    raise SystemExit(
        "train cache frames-root mismatch: "
        + repr(sorted(manifest_frames))
    )
output_dirs = {
    os.path.realpath(str(item.get("output_dir", "")))
    for item in payloads
}
if output_dirs != {os.path.realpath(cache)}:
    raise SystemExit(
        "train cache output-dir mismatch: " + repr(sorted(output_dirs))
    )
shard_counts = {
    int(item.get("shard", {}).get("count", -1))
    for item in payloads
}
shard_indices = [
    int(item.get("shard", {}).get("index", -1))
    for item in payloads
]
if (
    shard_counts != {len(payloads)}
    or sorted(shard_indices) != list(range(len(payloads)))
):
    raise SystemExit(
        "train cache metadata shard set is incomplete or inconsistent"
    )
seen_cache_keys = set()
cache_real = os.path.realpath(cache)
frame_rows = 0
for item in payloads:
    frames = item.get("frames", [])
    expected_frames = int(
        item.get("summary", {}).get("cache_files_expected", -1)
    )
    if expected_frames != len(frames):
        raise SystemExit("train cache metadata frame count mismatch")
    for row in frames:
        frame_rows += 1
        key = str(row.get("cache_key", ""))
        if not key.startswith(namespace + ":"):
            raise SystemExit("train cache key namespace mismatch")
        if key in seen_cache_keys:
            raise SystemExit("duplicate train cache key in metadata")
        seen_cache_keys.add(key)
        cache_path = os.path.realpath(str(row.get("cache_path", "")))
        try:
            within_cache = os.path.commonpath(
                [cache_real, cache_path]
            ) == cache_real
        except ValueError:
            within_cache = False
        if not within_cache:
            raise SystemExit("metadata cache path escapes train cache")
        if not os.path.isfile(cache_path) or os.path.getsize(cache_path) <= 0:
            raise SystemExit(
                "missing train cache artifact referenced by metadata: "
                + cache_path
            )
if frame_rows <= 0:
    raise SystemExit("train cache metadata contains no frame artifacts")
' "$TRAIN_CACHE" "$TRAIN_METADATA_ROOT" "$TRAIN_NAMESPACE" \
        "$SCENE_LIST" "$FRAMES_ROOT" \
        || die "train cache provenance validation failed"
}

validate_diagnostic() {
    local path="$1"
    local scene="$2"
    "$PYTHON" -c '
import sys
import numpy as np

path, expected_scene = sys.argv[1:3]
with np.load(path, allow_pickle=False) as archive:
    required = {
        "scene_id",
        "c4_diagnostics_schema",
        "c4_enabled",
        "c4_mutation_enabled",
        "c4_applied",
        "trifusion_diagnostics_schema",
        "trifusion_enabled",
        "trifusion_mutation_enabled",
        "trifusion_applied",
        "trifusion_missing_diagnostics_schema",
        "trifusion_missing_enabled",
        "trifusion_missing_mutation_enabled",
        "trifusion_missing_applied",
        "trifusion_gate_enabled",
    }
    missing = required - set(archive.files)
    if missing:
        raise SystemExit(f"{path}: missing observer fields {sorted(missing)}")

    def scalar_text(name):
        value = np.asarray(archive[name])
        if value.shape != ():
            raise SystemExit(f"{path}: {name} must be scalar")
        item = value.item()
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        return str(item)

    def scalar_bool(name):
        value = np.asarray(archive[name])
        if value.shape != () or value.dtype != np.bool_:
            raise SystemExit(f"{path}: {name} must be Boolean scalar")
        return bool(value.item())

    if scalar_text("scene_id") != expected_scene:
        raise SystemExit(f"{path}: scene mismatch")
    expected_schemas = {
        "c4_diagnostics_schema": "generic_mask_rgbd_local_geometry_v2",
        "trifusion_diagnostics_schema":
            "boxfusion.trifusion.occupancy_msr_observer.v1",
        "trifusion_missing_diagnostics_schema":
            "boxfusion.trifusion.missing_graph_observer.v1",
    }
    for name, expected in expected_schemas.items():
        if scalar_text(name) != expected:
            raise SystemExit(f"{path}: unsupported {name}")
    for name in (
        "c4_enabled",
        "trifusion_enabled",
        "trifusion_missing_enabled",
    ):
        if not scalar_bool(name):
            raise SystemExit(f"{path}: {name} is false")
    for name in (
        "c4_mutation_enabled",
        "trifusion_mutation_enabled",
        "trifusion_missing_mutation_enabled",
        "trifusion_gate_enabled",
        "trifusion_gate_mutation_enabled",
    ):
        if scalar_bool(name):
            raise SystemExit(f"{path}: forbidden train observer flag {name}")
    for name in (
        "c4_applied",
        "trifusion_applied",
        "trifusion_missing_applied",
    ):
        value = np.asarray(archive[name])
        if value.dtype != np.bool_ or bool(np.any(value)):
            raise SystemExit(f"{path}: {name} contains applied rows")
' "$path" "$scene" \
        || die "observer diagnostic validation failed for $scene"
}

validate_scene_list "train" "$SCENE_LIST"
validate_scene_list "forbidden validation" "$FORBIDDEN_VAL_LIST"
validate_no_overlap "$SCENE_LIST" "$FORBIDDEN_VAL_LIST"

case "$EXECUTE" in
    0|1) ;;
    *) die "EXECUTE must be 0 or 1" ;;
esac
case "$ALLOW_RESUME" in
    0|1) ;;
    *) die "BOXFUSION_TRIFUSION_TRAIN_ALLOW_RESUME must be 0 or 1" ;;
esac

[[ -x "$PYTHON" ]] || die "missing runtime Python: $PYTHON"
[[ -d "$FRAMES_ROOT" ]] || die "missing train-only frames root: $FRAMES_ROOT"
[[ -n "$TRAIN_CACHE" ]] \
    || die "BOXFUSION_TRIFUSION_TRAIN_TEACHER_CACHE is required; no val-cache fallback exists"
[[ -d "$TRAIN_CACHE" ]] || die "missing train-only teacher cache: $TRAIN_CACHE"
[[ -n "$TRAIN_METADATA_ROOT" ]] \
    || die "BOXFUSION_TRIFUSION_TRAIN_TEACHER_METADATA_ROOT is required; provenance is never inferred from the cache"
[[ -d "$TRAIN_METADATA_ROOT" ]] \
    || die "missing train-only teacher metadata root: $TRAIN_METADATA_ROOT"
[[ -n "$TRAIN_NAMESPACE" ]] \
    || die "BOXFUSION_TRIFUSION_TRAIN_TEACHER_NAMESPACE is required"

namespace_lower="${TRAIN_NAMESPACE,,}"
[[ "$namespace_lower" == *train* ]] \
    || die "train cache namespace must explicitly contain 'train'"
if [[ "$namespace_lower" == *val* || "$namespace_lower" == *full100* ]]; then
    die "refusing validation/full100-labelled train cache namespace: $TRAIN_NAMESPACE"
fi
cache_resolved="$(realpath -e -- "$TRAIN_CACHE")"
if [[ "$cache_resolved" == *"/boxfusion_maskgraph_dev/"* ]]; then
    die "train cache must not resolve inside boxfusion_maskgraph_dev"
fi
metadata_resolved="$(realpath -e -- "$TRAIN_METADATA_ROOT")"
if [[ "$metadata_resolved" == *"/boxfusion_maskgraph_dev/"* ]]; then
    die "train metadata must not resolve inside boxfusion_maskgraph_dev"
fi

required_files=(
    "$CONFIG"
    "$YOLOE_CHECKPOINT"
    "$QUALITY_CHECKPOINT"
    "$LIVE_ROOT/models/cutr_rgbd.pth"
    "$LIVE_ROOT/models/open_clip_pytorch_model.bin"
    "$LIVE_ROOT/data/panoptic_categories_nomerge.txt"
    "$LIVE_ROOT/data/class_features.pt"
    "$LIVE_ROOT/data/pst_1024_0.tiff"
)
for path in "${required_files[@]}"; do
    [[ -f "$path" ]] || die "missing required read-only input: $path"
done

require_inside_root "prediction root" "$PRED_ROOT"
require_inside_root "log root" "$LOG_ROOT"
require_inside_root "diagnostics root" "$DIAGNOSTICS_ROOT"
pred_resolved="$(canonical "$PRED_ROOT")"
log_resolved="$(canonical "$LOG_ROOT")"
diagnostics_resolved="$(canonical "$DIAGNOSTICS_ROOT")"
[[ "$pred_resolved" != "$log_resolved" ]] \
    || die "prediction and log roots must differ"
[[ "$pred_resolved" != "$diagnostics_resolved" ]] \
    || die "prediction and diagnostics roots must differ"
[[ "$log_resolved" != "$diagnostics_resolved" ]] \
    || die "log and diagnostics roots must differ"

validate_cache_provenance

IFS=',' read -r -a GPUS <<< "$GPU_SPEC"
[[ "${#GPUS[@]}" -ge 1 ]] || die "at least one GPU index is required"
for gpu in "${GPUS[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU index: $gpu"
done

while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -z "$scene" ]] && continue
    prediction="$PRED_ROOT/${scene}_boxes.pkl"
    diagnostic="$DIAGNOSTICS_ROOT/${scene}_tracks.npz"
    if [[ -s "$prediction" && ! -s "$diagnostic" ]]; then
        die "incomplete prior pair for $scene: prediction without diagnostic"
    fi
    if [[ -s "$diagnostic" && ! -s "$prediction" ]]; then
        die "incomplete prior pair for $scene: diagnostic without prediction"
    fi
    if [[ -s "$prediction" && -s "$diagnostic" ]]; then
        [[ "$ALLOW_RESUME" == "1" ]] \
            || die "existing pair for $scene requires a fresh tag or explicit resume"
        validate_diagnostic "$diagnostic" "$scene"
    fi
done <"$SCENE_LIST"

if [[ "$ALLOW_RESUME" != "1" ]]; then
    for directory in "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT"; do
        if directory_nonempty "$directory"; then
            die "refusing non-empty output directory: $directory"
        fi
    done
fi

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
echo "TriFusion train-only/no-eval observer"
echo "  mode: $([[ "$EXECUTE" == "1" ]] && echo execute || echo check-only)"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  forbidden validation scenes: $FORBIDDEN_VAL_LIST"
echo "  frames: $FRAMES_ROOT"
echo "  teacher metadata: $TRAIN_METADATA_ROOT"
echo "  teacher namespace: $TRAIN_NAMESPACE"
echo "  predictions: $PRED_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "  logs: $LOG_ROOT"
echo "  output contract: observer-only, no gate checkpoint, no val evaluator"
echo "  GPUs (execute mode only): $GPU_SPEC"

if [[ "$EXECUTE" != "1" ]]; then
    echo "CHECK PASSED. No directories were created and no GPU process was started."
    echo "Re-run with EXECUTE=1 only after reviewing the paths above."
    exit 0
fi

mkdir -p \
    "$PRED_ROOT" \
    "$LOG_ROOT/scenes" \
    "$LOG_ROOT/mplconfig" \
    "$DIAGNOSTICS_ROOT"
exec 9>"$LOG_ROOT/run.lock"
if ! flock -n 9; then
    die "another collector holds $LOG_ROOT/run.lock"
fi
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1
echo "Train scene list: $SCENE_LIST"
echo "Forbidden validation scene list: $FORBIDDEN_VAL_LIST"
echo "Train frames root: $FRAMES_ROOT"
echo "Train teacher cache: $TRAIN_CACHE"
echo "Train teacher metadata root: $TRAIN_METADATA_ROOT"
echo "Train teacher namespace: $TRAIN_NAMESPACE"
echo "Validation evaluator policy: forbidden"

ENV_LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64"
CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
"$PYTHON" -c \
    "import torch, torchvision, open_clip, ultralytics; assert torch.cuda.is_available(), 'runtime cannot see CUDA'" \
    || die "runtime/CUDA preflight failed"

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT
unset BOXFUSION_TRIFUSION_AP50_GATE_CHECKPOINT

run_worker() {
    local gpu="$1"
    local shard="$2"
    local shards="$3"
    local index=0
    local completed=0
    while IFS= read -r scene || [[ -n "$scene" ]]; do
        [[ -z "$scene" ]] && continue
        if (( index % shards != shard )); then
            index=$((index + 1))
            continue
        fi
        local prediction="$PRED_ROOT/${scene}_boxes.pkl"
        local diagnostic="$DIAGNOSTICS_ROOT/${scene}_tracks.npz"
        local scene_log="$LOG_ROOT/scenes/${scene}.log"
        if [[ -s "$prediction" && -s "$diagnostic" ]]; then
            validate_diagnostic "$diagnostic" "$scene"
            completed=$((completed + 1))
            echo "[GPU $gpu] $scene already complete and validated"
            index=$((index + 1))
            continue
        fi

        echo "[GPU $gpu] collecting $scene ($((index + 1))/$scene_count)"
        if ! (
            cd "$ROOT"
            CUDA_VISIBLE_DEVICES="$gpu" \
            PYTHONHASHSEED=0 \
            PYTHONDONTWRITEBYTECODE=1 \
            PYTHONUNBUFFERED=1 \
            MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
            XDG_CACHE_HOME="$LOG_ROOT/model_cache_gpu${gpu}" \
            LD_LIBRARY_PATH="$ENV_LD_LIBRARY_PATH" \
            "$PYTHON" demo.py scannet \
                --model-path "$LIVE_ROOT/models/cutr_rgbd.pth" \
                --clip_path "$LIVE_ROOT/models/open_clip_pytorch_model.bin" \
                --class_txt "$LIVE_ROOT/data/panoptic_categories_nomerge.txt" \
                --class-features "$LIVE_ROOT/data/class_features.pt" \
                --config "$CONFIG" \
                --output-dir "$PRED_ROOT" \
                --diagnostics-root "$DIAGNOSTICS_ROOT" \
                --scannet-frames-root "$FRAMES_ROOT" \
                --online-proposal-every-keyframes 5 \
                --device cuda \
                --seed 0 \
                --seq "$scene" \
                --online-proposal-provider yoloe \
                --online-proposal-checkpoint "$YOLOE_CHECKPOINT" \
                --c4-proposal-cache-directory "$TRAIN_CACHE" \
                --c4-proposal-cache-namespace "$TRAIN_NAMESPACE" \
                --c4-proposal-cache-missing-policy error \
                --online-quality-checkpoint "$QUALITY_CHECKPOINT" \
                --online-quality-mode iou_mlp \
                --online-quality-detector-blend 0.40 \
                --online-ablation-profile trifusion_plus10_observer \
                --online-candidate-ttl-clock provider_call \
                --online-candidate-track-ttl 3 \
                --scannet-min-extent 0.40 \
                --no-online-archive-confirmed-tracks
        ) >"$scene_log" 2>&1; then
            echo "ERROR: GPU $gpu failed on $scene" >&2
            tail -n 40 "$scene_log" >&2 || true
            return 1
        fi
        [[ -s "$prediction" ]] \
            || { echo "missing prediction for $scene" >&2; return 1; }
        [[ -s "$diagnostic" ]] \
            || { echo "missing diagnostic for $scene" >&2; return 1; }
        validate_diagnostic "$diagnostic" "$scene"
        completed=$((completed + 1))
        echo "[GPU $gpu] completed and validated $scene"
        index=$((index + 1))
    done <"$SCENE_LIST"
    echo "[GPU $gpu] completed $completed train scenes"
}

child_pids=()
cleanup() {
    local pid
    for pid in "${child_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM

for shard in "${!GPUS[@]}"; do
    run_worker "${GPUS[$shard]}" "$shard" "${#GPUS[@]}" &
    child_pids+=("$!")
done

worker_status=0
for pid in "${child_pids[@]}"; do
    if ! wait "$pid"; then
        worker_status=1
    fi
done
trap - INT TERM
[[ "$worker_status" -eq 0 ]] || die "at least one train collector failed"

"$PYTHON" -c '
import pathlib
import sys

scene_list, prediction_root, diagnostic_root = sys.argv[1:4]
scenes = {
    line.strip()
    for line in open(scene_list, encoding="utf-8")
    if line.strip()
}
predictions = {
    path.name.removesuffix("_boxes.pkl")
    for path in pathlib.Path(prediction_root).glob("scene*_boxes.pkl")
    if path.is_file() and path.stat().st_size > 0
}
diagnostics = {
    path.name.removesuffix("_tracks.npz")
    for path in pathlib.Path(diagnostic_root).glob("scene*_tracks.npz")
    if path.is_file() and path.stat().st_size > 0
}
if predictions != scenes or diagnostics != scenes:
    raise SystemExit(
        "final artifact set mismatch: "
        f"pred_missing={sorted(scenes-predictions)[:4]}, "
        f"pred_extra={sorted(predictions-scenes)[:4]}, "
        f"diag_missing={sorted(scenes-diagnostics)[:4]}, "
        f"diag_extra={sorted(diagnostics-scenes)[:4]}"
    )
' "$SCENE_LIST" "$PRED_ROOT" "$DIAGNOSTICS_ROOT" \
    || die "final train artifact-set validation failed"

echo "TriFusion train-only observer collection completed."
echo "No validation evaluator was invoked."
