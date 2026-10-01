#!/usr/bin/env bash
set -euo pipefail

# Leak-free TriFusion orchestration.
#
# Usage:
#   bash scripts/run_scannet_trifusion_protocol.sh check
#   bash scripts/run_scannet_trifusion_protocol.sh train-observer 0,1
#   bash scripts/run_scannet_trifusion_protocol.sh train-gate
#   bash scripts/run_scannet_trifusion_protocol.sh fixed10-observer 0,1
#   bash scripts/run_scannet_trifusion_protocol.sh fixed10-report
#
# Every mode is check/print-only by default.  Set EXECUTE=1 (or
# BOXFUSION_TRIFUSION_PROTOCOL_EXECUTE=1) for exactly one reviewed stage.

MODE="${1:-check}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

EXECUTE="${BOXFUSION_TRIFUSION_PROTOCOL_EXECUTE:-${EXECUTE:-0}}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"

TRAIN_SCENES="${BOXFUSION_TRIFUSION_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
FULL_VAL_SCENES="${BOXFUSION_TRIFUSION_FORBIDDEN_VAL_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
FIXED10_SCENES="${BOXFUSION_TRIFUSION_FIXED10_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
TRAIN_FRAMES="${BOXFUSION_TRIFUSION_TRAIN_FRAMES_ROOT:-$ROOT/data/scannet_train}"
FIXED_FRAMES="${BOXFUSION_TRIFUSION_FIXED10_FRAMES_ROOT:-/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames}"
GT_ROOT="${BOXFUSION_TRIFUSION_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCAN_ROOT="${BOXFUSION_TRIFUSION_SCAN_ROOT:-/extra/ZhaoX/scannet_data/scans}"

TRAIN_CACHE="${BOXFUSION_TRIFUSION_TRAIN_TEACHER_CACHE:-}"
TRAIN_METADATA="${BOXFUSION_TRIFUSION_TRAIN_TEACHER_METADATA_ROOT:-}"
TRAIN_NAMESPACE="${BOXFUSION_TRIFUSION_TRAIN_TEACHER_NAMESPACE:-}"
FIXED_CACHE="${BOXFUSION_TRIFUSION_FIXED10_TEACHER_CACHE:-$ROOT/cache/sam3_teacher/sam3_teacher_full100_c050_frozen_v1}"
FIXED_METADATA="${BOXFUSION_TRIFUSION_FIXED10_TEACHER_METADATA_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/logs/sam3_teacher_full100_c050_frozen_v1/metadata}"
FIXED_NAMESPACE="${BOXFUSION_TRIFUSION_FIXED10_TEACHER_NAMESPACE:-sam3-scannet18-val100-c050-frozen-v1}"

TRAIN_TAG="${BOXFUSION_TRIFUSION_TRAIN_RUN_TAG:-trifusion_plus10_train_observer_v1}"
TRAIN_PRED="${BOXFUSION_TRIFUSION_TRAIN_PRED_ROOT:-$ROOT/results/$TRAIN_TAG}"
TRAIN_LOG="${BOXFUSION_TRIFUSION_TRAIN_LOG_ROOT:-$ROOT/logs/$TRAIN_TAG}"
TRAIN_DIAG="${BOXFUSION_TRIFUSION_TRAIN_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$TRAIN_TAG}"
TRAIN_GEOMETRY="${BOXFUSION_TRIFUSION_TRAIN_GEOMETRY_ROOT:-$ROOT/datasets/trifusion_gate_trainonly_geometry_v1}"
TRAIN_GEOMETRY_SUMMARY="${BOXFUSION_TRIFUSION_TRAIN_GEOMETRY_SUMMARY:-$ROOT/reports/trifusion_gate_trainonly_v1/geometry_summary.json}"
GATE_DATASET="${BOXFUSION_TRIFUSION_GATE_DATASET:-$ROOT/datasets/scannet_trifusion_ap50_gate_trainonly_v1.npz}"
GATE_CHECKPOINT="${BOXFUSION_TRIFUSION_GATE_CHECKPOINT:-$ROOT/models/scannet_trifusion_ap50_gate_trainonly_v1.npz}"

FIXED_TAG="${BOXFUSION_TRIFUSION_FIXED10_RUN_TAG:-trifusion_plus10_fixed10_gate_observer_v1}"
FIXED_PRED="${BOXFUSION_TRIFUSION_FIXED10_PRED_ROOT:-$ROOT/results/$FIXED_TAG}"
FIXED_LOG="${BOXFUSION_TRIFUSION_FIXED10_LOG_ROOT:-$ROOT/logs/$FIXED_TAG}"
FIXED_DIAG="${BOXFUSION_TRIFUSION_FIXED10_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/$FIXED_TAG}"
FIXED_EVAL="${BOXFUSION_TRIFUSION_FIXED10_EVAL_ROOT:-$ROOT/evaluation/$FIXED_TAG}"
FIXED_REPORT_ROOT="${BOXFUSION_TRIFUSION_FIXED10_REPORT_ROOT:-$ROOT/reports/trifusion_plus10_fixed10_v1}"
FIXED_GEOMETRY="${BOXFUSION_TRIFUSION_FIXED10_GEOMETRY_ROOT:-$FIXED_REPORT_ROOT/geometry}"
FIXED_SUPPLEMENTAL="${BOXFUSION_TRIFUSION_FIXED10_SUPPLEMENTAL_ROOT:-$FIXED_REPORT_ROOT/supplemental}"
FIXED_GEOMETRY_SUMMARY="${BOXFUSION_TRIFUSION_FIXED10_GEOMETRY_SUMMARY:-$FIXED_REPORT_ROOT/geometry_summary.json}"
FIXED_SUPPLEMENTAL_SUMMARY="${BOXFUSION_TRIFUSION_FIXED10_SUPPLEMENTAL_SUMMARY:-$FIXED_REPORT_ROOT/supplemental_summary.json}"
FIXED_ORACLE_REPORT="${BOXFUSION_TRIFUSION_FIXED10_ORACLE_REPORT:-$FIXED_REPORT_ROOT/oracle_report.json}"
FIXED_COUNTERFACTUAL_REPORT="${BOXFUSION_TRIFUSION_FIXED10_COUNTERFACTUAL_REPORT:-$FIXED_REPORT_ROOT/gate_counterfactual_report.json}"

die() {
    echo "TriFusion protocol: $*" >&2
    exit 1
}

canonical() {
    realpath -m -- "$1"
}

inside_root() {
    local name="$1"
    local path="$2"
    local resolved
    resolved="$(canonical "$path")"
    case "$resolved" in
        "$(canonical "$ROOT")"/*) ;;
        *) die "$name must remain inside the isolated checkout: $path" ;;
    esac
}

print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

require_fresh() {
    local name="$1"
    local path="$2"
    [[ ! -e "$path" ]] || die "$name already exists; choose a fresh path: $path"
}

validate_scene_protocol() {
    "$PYTHON" -c '
import re
import sys

train_path, val_path, fixed_path = sys.argv[1:4]
pattern = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
def read(path, role):
    rows = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
    if not rows:
        raise SystemExit(f"empty {role} scene list: {path}")
    if any(not pattern.fullmatch(row) for row in rows):
        raise SystemExit(f"malformed {role} scene list: {path}")
    if len(rows) != len(set(rows)):
        raise SystemExit(f"duplicate scene in {role} list: {path}")
    return rows
train = read(train_path, "train")
val = read(val_path, "full validation")
fixed = read(fixed_path, "fixed10")
overlap = sorted(set(train) & set(val))
if overlap:
    raise SystemExit(f"train/full-val leakage: {overlap[:4]}")
if len(fixed) != 10:
    raise SystemExit(f"fixed10 must contain exactly 10 scenes, got {len(fixed)}")
outside = sorted(set(fixed) - set(val))
if outside:
    raise SystemExit(f"fixed10 is not a full-val subset: {outside[:4]}")
' "$TRAIN_SCENES" "$FULL_VAL_SCENES" "$FIXED10_SCENES" \
        || die "scene protocol validation failed"
}

validate_fixed_cache() {
    [[ -d "$FIXED_CACHE" ]] || die "missing fixed10 read-only cache: $FIXED_CACHE"
    [[ -d "$FIXED_METADATA" ]] \
        || die "missing fixed10 read-only builder metadata: $FIXED_METADATA"
    "$PYTHON" -c '
import glob
import hashlib
import json
import os
import sys

cache, metadata_root, namespace, val_list, fixed_list, frames_root = sys.argv[1:7]
paths = sorted(glob.glob(os.path.join(metadata_root, "shard*.json")))
if not paths:
    raise SystemExit("missing full-val builder shard*.json metadata")
payloads = [json.load(open(path, encoding="utf-8")) for path in paths]
if {
    str(p.get("schema", "")) for p in payloads
} != {"boxfusion_scannet_sam3_teacher_cache_v1"}:
    raise SystemExit("unsupported full-val cache metadata schema")
if not all(p.get("complete") is True for p in payloads):
    raise SystemExit("full-val cache metadata contains an incomplete shard")
if {str(p.get("namespace", "")) for p in payloads} != {namespace}:
    raise SystemExit("full-val cache namespace mismatch")
expected_sha = hashlib.sha256(open(val_list, "rb").read()).hexdigest()
if {
    str(p.get("scene_list", {}).get("sha256", "")) for p in payloads
} != {expected_sha}:
    raise SystemExit("full-val cache scene-list SHA mismatch")
val = {line.strip() for line in open(val_list, encoding="utf-8") if line.strip()}
fixed = {line.strip() for line in open(fixed_list, encoding="utf-8") if line.strip()}
rows = [
    str(row.get("scene_id", ""))
    for payload in payloads
    for row in payload.get("scenes", [])
]
if len(rows) != len(set(rows)) or set(rows) != val:
    raise SystemExit("full-val cache manifest scene union mismatch")
if {
    int(p.get("scene_list", {}).get("all_scene_count", -1))
    for p in payloads
} != {len(val)}:
    raise SystemExit("full-val cache scene count mismatch")
if not fixed <= set(rows):
    raise SystemExit("fixed10 scenes are absent from full-val cache")
if {
    os.path.realpath(str(p.get("frames_root", ""))) for p in payloads
} != {os.path.realpath(frames_root)}:
    raise SystemExit("full-val cache frames-root mismatch")
if {
    os.path.realpath(str(p.get("output_dir", ""))) for p in payloads
} != {os.path.realpath(cache)}:
    raise SystemExit("full-val cache output-dir mismatch")
counts = {int(p.get("shard", {}).get("count", -1)) for p in payloads}
indices = sorted(int(p.get("shard", {}).get("index", -1)) for p in payloads)
if counts != {len(payloads)} or indices != list(range(len(payloads))):
    raise SystemExit("full-val cache metadata shard set is incomplete")
cache_real = os.path.realpath(cache)
keys = set()
frame_count = 0
for payload in payloads:
    frames = payload.get("frames", [])
    if int(payload.get("summary", {}).get("cache_files_expected", -1)) != len(frames):
        raise SystemExit("full-val cache metadata frame count mismatch")
    for row in frames:
        frame_count += 1
        key = str(row.get("cache_key", ""))
        if not key.startswith(namespace + ":") or key in keys:
            raise SystemExit("full-val cache key provenance mismatch")
        keys.add(key)
        artifact = os.path.realpath(str(row.get("cache_path", "")))
        try:
            inside = os.path.commonpath([cache_real, artifact]) == cache_real
        except ValueError:
            inside = False
        if not inside or not os.path.isfile(artifact) or os.path.getsize(artifact) <= 0:
            raise SystemExit("missing/escaped full-val cache artifact: " + artifact)
if frame_count <= 0:
    raise SystemExit("full-val cache metadata contains no frame artifacts")
' "$FIXED_CACHE" "$FIXED_METADATA" "$FIXED_NAMESPACE" \
        "$FULL_VAL_SCENES" "$FIXED10_SCENES" "$FIXED_FRAMES" \
        || die "fixed10 cache provenance validation failed"
}

validate_complete_pairs() {
    local scene_list="$1"
    local pred_root="$2"
    local diag_root="$3"
    "$PYTHON" -c '
import pathlib
import sys
scene_list, pred_root, diag_root = sys.argv[1:4]
scenes = {line.strip() for line in open(scene_list, encoding="utf-8") if line.strip()}
pred = {
    p.name[:-10] for p in pathlib.Path(pred_root).glob("scene*_boxes.pkl")
    if p.is_file() and p.stat().st_size > 0
}
diag = {
    p.name[:-11] for p in pathlib.Path(diag_root).glob("scene*_tracks.npz")
    if p.is_file() and p.stat().st_size > 0
}
if pred != scenes or diag != scenes:
    raise SystemExit(
        f"artifact mismatch: pred_missing={sorted(scenes-pred)[:4]}, "
        f"pred_extra={sorted(pred-scenes)[:4]}, "
        f"diag_missing={sorted(scenes-diag)[:4]}, "
        f"diag_extra={sorted(diag-scenes)[:4]}"
    )
' "$scene_list" "$pred_root" "$diag_root" \
        || die "prediction/diagnostic artifact validation failed"
}

validate_gate_checkpoint() {
    [[ -s "$GATE_DATASET" ]] || die "missing gate training archive: $GATE_DATASET"
    [[ -s "$GATE_CHECKPOINT" ]] || die "missing gate checkpoint: $GATE_CHECKPOINT"
    (
        cd "$ROOT"
        PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -c '
import json
import sys
import numpy as np
from boxfusion.ap50_safety_gate import load_ap50_safety_gate

checkpoint, dataset, train_list, val_list = sys.argv[1:5]
train = {line.strip() for line in open(train_list, encoding="utf-8") if line.strip()}
val = {line.strip() for line in open(val_list, encoding="utf-8") if line.strip()}
with np.load(dataset, allow_pickle=False) as archive:
    dataset_scenes = {str(item) for item in np.asarray(archive["scene_ids"]).tolist()}
if not dataset_scenes or not dataset_scenes <= train or dataset_scenes & val:
    raise SystemExit("gate archive scene provenance is not train-only")
gate = load_ap50_safety_gate(checkpoint)
metadata = dict(gate.metadata)
trained = set(map(str, metadata.get("training_scenes", [])))
heldout = set(map(str, metadata.get("validation_scenes", [])))
used = trained | heldout
if not trained or not heldout:
    raise SystemExit("gate checkpoint lacks internal train split provenance")
if not used <= train or used & val or used != dataset_scenes:
    raise SystemExit("gate checkpoint scene provenance is not train-only")
' "$GATE_CHECKPOINT" "$GATE_DATASET" "$TRAIN_SCENES" "$FULL_VAL_SCENES"
    ) || die "gate checkpoint provenance validation failed"
}

validate_fixed_diagnostics() {
    "$PYTHON" -c '
import sys
import numpy as np
scene_list, root = sys.argv[1:3]
for scene in (line.strip() for line in open(scene_list, encoding="utf-8")):
    if not scene:
        continue
    path = f"{root}/{scene}_tracks.npz"
    with np.load(path, allow_pickle=False) as archive:
        def flag(name):
            value = np.asarray(archive[name])
            if value.shape != () or value.dtype != np.bool_:
                raise SystemExit(f"{path}: malformed {name}")
            return bool(value.item())
        if not flag("trifusion_gate_enabled"):
            raise SystemExit(f"{path}: gate inference was disabled")
        for name in (
            "c4_mutation_enabled",
            "trifusion_mutation_enabled",
            "trifusion_missing_mutation_enabled",
            "trifusion_gate_mutation_enabled",
        ):
            if flag(name):
                raise SystemExit(f"{path}: forbidden mutation flag {name}")
        for name in ("c4_applied", "trifusion_applied", "trifusion_missing_applied"):
            if bool(np.any(np.asarray(archive[name], dtype=bool))):
                raise SystemExit(f"{path}: observer applied mutations")
' "$FIXED10_SCENES" "$FIXED_DIAG" \
        || die "fixed10 observer diagnostic validation failed"
}

run_cpu() {
    if [[ "$EXECUTE" == "1" ]]; then
        env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$@"
    else
        print_command env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 "$@"
    fi
}

case "$EXECUTE" in
    0|1) ;;
    *) die "EXECUTE must be 0 or 1" ;;
esac
case "$MODE" in
    check|train-observer|train-gate|fixed10-observer|fixed10-report) ;;
    *) die "mode must be check, train-observer, train-gate, fixed10-observer, or fixed10-report" ;;
esac

[[ -x "$PYTHON" ]] || die "missing Python environment: $PYTHON"
for path in "$TRAIN_SCENES" "$FULL_VAL_SCENES" "$FIXED10_SCENES"; do
    [[ -s "$path" ]] || die "missing or empty scene list: $path"
done
[[ -d "$GT_ROOT" ]] || die "missing GT root: $GT_ROOT"
[[ -d "$SCAN_ROOT" ]] || die "missing ScanNet scans root: $SCAN_ROOT"
validate_scene_protocol

for output in \
    "$TRAIN_PRED" "$TRAIN_LOG" "$TRAIN_DIAG" "$TRAIN_GEOMETRY" \
    "$TRAIN_GEOMETRY_SUMMARY" "$GATE_DATASET" "$GATE_CHECKPOINT" \
    "$FIXED_PRED" "$FIXED_LOG" "$FIXED_DIAG" "$FIXED_EVAL" \
    "$FIXED_REPORT_ROOT" "$FIXED_GEOMETRY" "$FIXED_SUPPLEMENTAL" \
    "$FIXED_GEOMETRY_SUMMARY" "$FIXED_SUPPLEMENTAL_SUMMARY" \
    "$FIXED_ORACLE_REPORT" "$FIXED_COUNTERFACTUAL_REPORT"; do
    inside_root "output" "$output"
done

echo "TriFusion leak-free protocol"
echo "  mode: $MODE / $([[ "$EXECUTE" == "1" ]] && echo execute || echo check-only)"
echo "  train scenes: $TRAIN_SCENES"
echo "  forbidden scenes (always full val): $FULL_VAL_SCENES"
echo "  fixed observer scenes: $FIXED10_SCENES"
echo "  train cache: ${TRAIN_CACHE:-UNSET}"
echo "  train metadata: ${TRAIN_METADATA:-UNSET}"
echo "  fixed cache: $FIXED_CACHE (read-only validation use only)"
echo "  fixed metadata: $FIXED_METADATA (read-only validation use only)"
echo "  gate archive: $GATE_DATASET"
echo "  gate checkpoint: $GATE_CHECKPOINT"
echo "  WARNING: fixed10 remains observer identity; standard AP is frozen B6."
echo "  WARNING: oracle output is GT-conditioned analysis, never active AP."

if [[ "$MODE" == "check" ]]; then
    validate_fixed_cache
fi

if [[ "$MODE" == "check" || "$MODE" == "train-observer" ]]; then
    collector=(
        env
        BOXFUSION_TRIFUSION_PROTOCOL_EXECUTE="$([[ "$MODE" == "train-observer" ]] && echo "$EXECUTE" || echo 0)"
        BOXFUSION_TRIFUSION_TRAIN_SCENE_LIST="$TRAIN_SCENES"
        BOXFUSION_TRIFUSION_FORBIDDEN_VAL_SCENE_LIST="$FULL_VAL_SCENES"
        BOXFUSION_TRIFUSION_TRAIN_FRAMES_ROOT="$TRAIN_FRAMES"
        BOXFUSION_TRIFUSION_TRAIN_TEACHER_CACHE="$TRAIN_CACHE"
        BOXFUSION_TRIFUSION_TRAIN_TEACHER_METADATA_ROOT="$TRAIN_METADATA"
        BOXFUSION_TRIFUSION_TRAIN_TEACHER_NAMESPACE="$TRAIN_NAMESPACE"
        BOXFUSION_TRIFUSION_TRAIN_RUN_TAG="$TRAIN_TAG"
        BOXFUSION_TRIFUSION_TRAIN_PRED_ROOT="$TRAIN_PRED"
        BOXFUSION_TRIFUSION_TRAIN_LOG_ROOT="$TRAIN_LOG"
        BOXFUSION_TRIFUSION_TRAIN_DIAGNOSTICS_ROOT="$TRAIN_DIAG"
        bash "$ROOT/scripts/collect_scannet_trifusion_train_observer.sh"
        "$GPU_SPEC"
    )
    print_command "${collector[@]}"
    "${collector[@]}"
fi

if [[ "$MODE" == "check" ]]; then
    echo "CHECK PASSED. No output directory was created and no GPU process was started."
    exit 0
fi

if [[ "$MODE" == "train-observer" ]]; then
    exit 0
fi

if [[ "$MODE" == "train-gate" ]]; then
    validate_complete_pairs "$TRAIN_SCENES" "$TRAIN_PRED" "$TRAIN_DIAG"
    [[ -s "$TRAIN_LOG/driver.log" ]] || die "missing train collector driver.log"
    grep -Fq "No validation evaluator was invoked." "$TRAIN_LOG/driver.log" \
        || die "train collector did not record the no-evaluator contract"
    grep -Fq "Train scene list: $TRAIN_SCENES" "$TRAIN_LOG/driver.log" \
        || die "train collector scene-list provenance mismatch"
    grep -Fq "Forbidden validation scene list: $FULL_VAL_SCENES" "$TRAIN_LOG/driver.log" \
        || die "train collector forbidden-list provenance mismatch"
    grep -Fq "Train teacher cache: $TRAIN_CACHE" "$TRAIN_LOG/driver.log" \
        || die "train collector cache provenance mismatch"
    grep -Fq "Train teacher metadata root: $TRAIN_METADATA" "$TRAIN_LOG/driver.log" \
        || die "train collector metadata provenance mismatch"
    grep -Fq "Train teacher namespace: $TRAIN_NAMESPACE" "$TRAIN_LOG/driver.log" \
        || die "train collector namespace provenance mismatch"
    require_fresh "train geometry root" "$TRAIN_GEOMETRY"
    require_fresh "train geometry summary" "$TRAIN_GEOMETRY_SUMMARY"
    require_fresh "gate training archive" "$GATE_DATASET"
    require_fresh "gate checkpoint" "$GATE_CHECKPOINT"
    run_cpu "$PYTHON" "$ROOT/tools/build_trifusion_geometry_candidates.py" \
        --diagnostics-root "$TRAIN_DIAG" \
        --prediction-root "$TRAIN_PRED" \
        --scene-list "$TRAIN_SCENES" \
        --output-root "$TRAIN_GEOMETRY" \
        --summary-json "$TRAIN_GEOMETRY_SUMMARY"
    run_cpu "$PYTHON" "$ROOT/tools/build_ap50_gate_training_from_trifusion.py" \
        --geometry-root "$TRAIN_GEOMETRY" \
        --prediction-root "$TRAIN_PRED" \
        --scene-list "$TRAIN_SCENES" \
        --forbidden-scene-list "$FULL_VAL_SCENES" \
        --gt-root "$GT_ROOT" \
        --scan-root "$SCAN_ROOT" \
        --output "$GATE_DATASET" \
        --verified-only
    run_cpu "$PYTHON" "$ROOT/tools/train_ap50_safety_gate.py" \
        "$GATE_DATASET" \
        --output "$GATE_CHECKPOINT" \
        --forbidden-scene-list "$FULL_VAL_SCENES"
    if [[ "$EXECUTE" == "1" ]]; then
        validate_gate_checkpoint
    else
        echo "CHECK PASSED. CPU commands were printed but not executed."
    fi
    exit 0
fi

validate_gate_checkpoint
validate_fixed_cache

if [[ "$MODE" == "fixed10-observer" ]]; then
    for path in "$FIXED_PRED" "$FIXED_LOG" "$FIXED_DIAG" "$FIXED_EVAL"; do
        require_fresh "fixed10 observer output" "$path"
    done
    fixed_command=(
        env
        BOXFUSION_TRIFUSION_FULL100=0
        BOXFUSION_TRIFUSION_SCENE_LIST="$FIXED10_SCENES"
        BOXFUSION_SCANNET_FRAMES_ROOT="$FIXED_FRAMES"
        BOXFUSION_TRIFUSION_TEACHER_CACHE="$FIXED_CACHE"
        BOXFUSION_TRIFUSION_TEACHER_NAMESPACE="$FIXED_NAMESPACE"
        BOXFUSION_TRIFUSION_CACHE_MISSING_POLICY=error
        BOXFUSION_TRIFUSION_AP50_GATE_CHECKPOINT="$GATE_CHECKPOINT"
        BOXFUSION_TRIFUSION_RUN_TAG="$FIXED_TAG"
        BOXFUSION_TRIFUSION_PRED_ROOT="$FIXED_PRED"
        BOXFUSION_TRIFUSION_LOG_ROOT="$FIXED_LOG"
        BOXFUSION_TRIFUSION_DIAGNOSTICS_ROOT="$FIXED_DIAG"
        BOXFUSION_TRIFUSION_EVAL_ROOT="$FIXED_EVAL"
        bash "$ROOT/scripts/run_scannet_trifusion_observer.sh"
        "$GPU_SPEC"
    )
    print_command "${fixed_command[@]}"
    if [[ "$EXECUTE" == "1" ]]; then
        "${fixed_command[@]}"
    else
        echo "CHECK PASSED. Fixed10 GPU command was printed but not executed."
    fi
    exit 0
fi

validate_complete_pairs "$FIXED10_SCENES" "$FIXED_PRED" "$FIXED_DIAG"
validate_fixed_diagnostics
[[ -s "$FIXED_LOG/driver.log" ]] || die "missing fixed10 driver.log"
grep -Fq "C4 proposal cache directory: $FIXED_CACHE" "$FIXED_LOG/driver.log" \
    || die "fixed10 log cache path mismatch"
grep -Fq "C4 proposal cache namespace: $FIXED_NAMESPACE" "$FIXED_LOG/driver.log" \
    || die "fixed10 log cache namespace mismatch"
grep -Fq "TriFusion AP50 gate checkpoint: $GATE_CHECKPOINT" "$FIXED_LOG/driver.log" \
    || die "fixed10 log gate checkpoint mismatch"
require_fresh "fixed10 report root" "$FIXED_REPORT_ROOT"
run_cpu "$PYTHON" "$ROOT/tools/build_trifusion_geometry_candidates.py" \
    --diagnostics-root "$FIXED_DIAG" \
    --prediction-root "$FIXED_PRED" \
    --scene-list "$FIXED10_SCENES" \
    --output-root "$FIXED_GEOMETRY" \
    --summary-json "$FIXED_GEOMETRY_SUMMARY"
run_cpu "$PYTHON" "$ROOT/tools/export_trifusion_supplemental_candidates.py" \
    --diagnostics-root "$FIXED_DIAG" \
    --scene-list "$FIXED10_SCENES" \
    --output-root "$FIXED_SUPPLEMENTAL" \
    --summary-json "$FIXED_SUPPLEMENTAL_SUMMARY"
run_cpu "$PYTHON" "$ROOT/tools/report_trifusion_oracles.py" \
    --pred-root "$FIXED_PRED" \
    --scene-list "$FIXED10_SCENES" \
    --gt-root "$GT_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --geometry-root "$FIXED_GEOMETRY" \
    --supplemental-root "$FIXED_SUPPLEMENTAL" \
    --output "$FIXED_ORACLE_REPORT"
run_cpu "$PYTHON" "$ROOT/tools/evaluate_trifusion_counterfactual.py" \
    --pred-root "$FIXED_PRED" \
    --diagnostics-root "$FIXED_DIAG" \
    --scene-list "$FIXED10_SCENES" \
    --gt-root "$GT_ROOT" \
    --scan-root "$SCAN_ROOT" \
    --output "$FIXED_COUNTERFACTUAL_REPORT"
if [[ "$EXECUTE" == "1" ]]; then
    [[ -s "$FIXED_ORACLE_REPORT" ]] || die "fixed10 oracle report was not created"
    [[ -s "$FIXED_COUNTERFACTUAL_REPORT" ]] \
        || die "fixed10 gate counterfactual report was not created"
else
    echo "CHECK PASSED. CPU commands were printed but not executed."
fi
