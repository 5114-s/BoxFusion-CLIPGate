#!/usr/bin/env bash
set -euo pipefail

# Strict one-module-at-a-time YiDu observer protocol.
#
# Usage (fixed ten scenes by default):
#   bash scripts/run_scannet_yidu_ablation.sh B0 0,1
#   bash scripts/run_scannet_yidu_ablation.sh A1 0,1
#   ...
#   bash scripts/run_scannet_yidu_ablation.sh A6 0,1
#
# Full 100 scenes are deliberately opt-in and should only be attempted after
# the previous fixed-10 stage has passed its identity and oracle audit:
#   BOXFUSION_YIDU_FULL100=1 \
#     bash scripts/run_scannet_yidu_ablation.sh A3 0,1

STAGE="${1:-}"
GPU_SPEC="${2:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Do not silently inherit an unrelated active Conda environment.  The
# isolated online environment is the reproducible default; callers may still
# override it explicitly with BOXFUSION_ENV_ROOT.
RUNTIME_ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
RUNTIME_PYTHON="$RUNTIME_ENV_ROOT/bin/python"
export BOXFUSION_ENV_ROOT="$RUNTIME_ENV_ROOT"
if [[ ! -x "$RUNTIME_PYTHON" ]]; then
    echo "Missing YiDu runtime Python: $RUNTIME_PYTHON" >&2
    exit 1
fi
if ! "$RUNTIME_PYTHON" -c \
    'import open_clip, torch, ultralytics' >/dev/null 2>&1; then
    echo "YiDu runtime is missing torch/open_clip/ultralytics: $RUNTIME_PYTHON" >&2
    echo "Set BOXFUSION_ENV_ROOT to a complete BoxFusion environment." >&2
    exit 1
fi
PYTHON_VERSION="$("$RUNTIME_PYTHON" -c 'import platform; print(platform.python_version())')"
TORCH_VERSION="$("$RUNTIME_PYTHON" -c 'import torch; print(torch.__version__)')"

case "${STAGE^^}" in
    B0) STAGE="B0"; PROFILE="yidu_b0_frozen_b6" ;;
    A1) STAGE="A1"; PROFILE="yidu_a1_adaptive_erosion_observer" ;;
    A2) STAGE="A2"; PROFILE="yidu_a2_dfu_filter_observer" ;;
    A3) STAGE="A3"; PROFILE="yidu_a3_voxel_components_observer" ;;
    A4) STAGE="A4"; PROFILE="yidu_a4_occupancy_msr_observer" ;;
    A5) STAGE="A5"; PROFILE="yidu_a5_raw_fused_query_observer" ;;
    A6) STAGE="A6"; PROFILE="yidu_a6_quality_gate_observer" ;;
    *)
        echo "Stage must be one of B0,A1,A2,A3,A4,A5,A6" >&2
        exit 2
        ;;
esac

FULL100="${BOXFUSION_YIDU_FULL100:-0}"
ALLOW_RESUME="${BOXFUSION_YIDU_ALLOW_RESUME:-0}"
DRY_RUN="${BOXFUSION_YIDU_DRY_RUN:-0}"
case "$FULL100" in 0|1) ;; *)
    echo "BOXFUSION_YIDU_FULL100 must be 0 or 1" >&2; exit 2 ;;
esac
case "$ALLOW_RESUME" in 0|1) ;; *)
    echo "BOXFUSION_YIDU_ALLOW_RESUME must be 0 or 1" >&2; exit 2 ;;
esac
case "$DRY_RUN" in 0|1) ;; *)
    echo "BOXFUSION_YIDU_DRY_RUN must be 0 or 1" >&2; exit 2 ;;
esac
if [[ "$DRY_RUN" == "1" && "$ALLOW_RESUME" == "1" ]]; then
    echo "YiDu dry-run and resume cannot be combined" >&2
    exit 2
fi

CONFIG="${BOXFUSION_YIDU_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
FRAMES_ROOT="${BOXFUSION_SCANNET_FRAMES_ROOT:-$LIVE_ROOT/upstream_clean/scannet_readme_frames}"
YOLOE_CHECKPOINT="${BOXFUSION_YIDU_YOLOE_CHECKPOINT:-${BOXFUSION_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}}"
QUALITY_CHECKPOINT="${BOXFUSION_YIDU_B6_CHECKPOINT:-${BOXFUSION_QUALITY_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}}"
TEACHER_CACHE="${BOXFUSION_YIDU_TEACHER_CACHE:-/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/cache/sam3_teacher/sam3_teacher_full100_c050_frozen_v1}"
TEACHER_NAMESPACE="${BOXFUSION_YIDU_TEACHER_NAMESPACE:-sam3-scannet18-val100-c050-frozen-v1}"
CACHE_MISSING_POLICY="${BOXFUSION_YIDU_CACHE_MISSING_POLICY:-error}"
YIDU_GATE_CHECKPOINT="${BOXFUSION_YIDU_GATE_CHECKPOINT:-}"
YIDU_GATE_TRAINING_ARCHIVE="${BOXFUSION_YIDU_GATE_TRAINING_ARCHIVE:-}"
YIDU_GATE_TRAIN_SCENE_LIST="${BOXFUSION_YIDU_GATE_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
YIDU_GATE_FORBIDDEN_SCENE_LIST="${BOXFUSION_YIDU_GATE_FORBIDDEN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"

if [[ -n "${BOXFUSION_YIDU_SCENE_LIST:-}" ]]; then
    SCENE_LIST="$BOXFUSION_YIDU_SCENE_LIST"
    SCOPE_TAG="custom"
elif [[ "$FULL100" == "1" ]]; then
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
    SCOPE_TAG="full100"
else
    SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
    SCOPE_TAG="ablation10"
fi

RUN_TAG="${BOXFUSION_YIDU_RUN_TAG:-yidu_${STAGE,,}_${SCOPE_TAG}_observer_v1}"
PRED_ROOT="${BOXFUSION_YIDU_PRED_ROOT:-$ROOT/results/yidu_ablation/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_YIDU_LOG_ROOT:-$ROOT/logs/yidu_ablation/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_YIDU_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/yidu_ablation/$RUN_TAG}"
EVAL_ROOT="${BOXFUSION_YIDU_EVAL_ROOT:-$ROOT/evaluation/yidu_ablation/$RUN_TAG}"

die() {
    echo "YiDu runner: $*" >&2
    exit 1
}

canonical() {
    realpath -m -- "$1"
}

require_output_inside_checkout() {
    local role="$1"
    local path="$2"
    local root_real
    local path_real
    root_real="$(canonical "$ROOT")"
    path_real="$(canonical "$path")"
    case "$path_real" in
        "$root_real"/*) ;;
        *) die "$role must remain inside the isolated checkout: $path" ;;
    esac
}

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    die "BOXFUSION_YIDU_RUN_TAG contains unsafe characters: $RUN_TAG"
fi
for pair in \
    "prediction:$PRED_ROOT" \
    "log:$LOG_ROOT" \
    "diagnostics:$DIAGNOSTICS_ROOT" \
    "evaluation:$EVAL_ROOT"; do
    require_output_inside_checkout "${pair%%:*} root" "${pair#*:}"
done
output_roots=(
    "$(canonical "$PRED_ROOT")"
    "$(canonical "$LOG_ROOT")"
    "$(canonical "$DIAGNOSTICS_ROOT")"
    "$(canonical "$EVAL_ROOT")"
)
if [[ "$(printf '%s\n' "${output_roots[@]}" | sort -u | wc -l)" -ne 4 ]]; then
    die "prediction/log/diagnostics/evaluation roots must be distinct"
fi

if [[ -n "${BOXFUSION_ONLINE_ABLATION_PROFILE:-}" \
      && "$BOXFUSION_ONLINE_ABLATION_PROFILE" != "$PROFILE" ]]; then
    echo "Refusing conflicting BOXFUSION_ONLINE_ABLATION_PROFILE" >&2
    exit 2
fi
[[ "$CACHE_MISSING_POLICY" == "error" ]] \
    || die "strict YiDu runs require cache missing policy=error"

required_files=(
    "$CONFIG"
    "$SCENE_LIST"
    "$YOLOE_CHECKPOINT"
    "$QUALITY_CHECKPOINT"
)
for path in "${required_files[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing required YiDu input: $path" >&2
        exit 1
    fi
done
if [[ ! -s "$SCENE_LIST" ]]; then
    echo "YiDu scene list is empty: $SCENE_LIST" >&2
    exit 1
fi
"$RUNTIME_PYTHON" - "$SCENE_LIST" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [
    row.strip()
    for row in path.read_text(encoding="utf-8").splitlines()
    if row.strip()
]
pattern = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
if not rows or any(pattern.fullmatch(row) is None for row in rows):
    raise SystemExit(f"invalid YiDu scene list: {path}")
if len(rows) != len(set(rows)):
    raise SystemExit(f"duplicate scene ID in YiDu scene list: {path}")
PY
[[ -d "$FRAMES_ROOT" ]] || die "missing ScanNet frames root: $FRAMES_ROOT"
if [[ "$STAGE" != "B0" ]]; then
    if [[ ! -d "$TEACHER_CACHE" || -z "$TEACHER_NAMESPACE" ]]; then
        echo "A1-A6 require an immutable SAM3 teacher cache and namespace" >&2
        exit 1
    fi
    "$RUNTIME_PYTHON" - \
        "$TEACHER_CACHE" "$TEACHER_NAMESPACE" \
        "$SCENE_LIST" "$FRAMES_ROOT" <<'PY'
import glob
import json
import os
import sys

cache, namespace, scene_list, frames_root = sys.argv[1:5]
paths = sorted(
    glob.glob(os.path.join(cache, "manifests", "provenance*.json"))
)
if not paths:
    raise SystemExit("missing SAM3 cache provenance manifests")
payloads = [json.load(open(path, encoding="utf-8")) for path in paths]
if {str(item.get("namespace", "")) for item in payloads} != {namespace}:
    raise SystemExit("SAM3 cache namespace mismatch")
if not all(
    os.path.realpath(str(item.get("frames_root", "")))
    == os.path.realpath(frames_root)
    for item in payloads
):
    raise SystemExit("SAM3 cache frames-root mismatch")
rows = [
    str(row.get("scene_id", ""))
    for item in payloads
    for row in item.get("scenes", [])
]
if not rows or len(rows) != len(set(rows)):
    raise SystemExit("SAM3 cache scene manifests are empty or duplicated")
requested = {
    row.strip()
    for row in open(scene_list, encoding="utf-8")
    if row.strip()
}
missing = sorted(requested - set(rows))
if missing:
    raise SystemExit(
        "SAM3 cache lacks requested scenes: " + ", ".join(missing[:8])
    )
PY
fi
if [[ "$STAGE" == "A6" ]]; then
    if [[ -z "$YIDU_GATE_CHECKPOINT" ]]; then
        echo "A6 requires BOXFUSION_YIDU_GATE_CHECKPOINT from train-only data" >&2
        exit 1
    fi
    if [[ ! -f "$YIDU_GATE_CHECKPOINT" ]]; then
        echo "Missing YiDu quality-gate checkpoint: $YIDU_GATE_CHECKPOINT" >&2
        exit 1
    fi
    if [[ -z "$YIDU_GATE_TRAINING_ARCHIVE" ]]; then
        echo "A6 requires BOXFUSION_YIDU_GATE_TRAINING_ARCHIVE" >&2
        exit 1
    fi
    for path in \
        "$YIDU_GATE_TRAINING_ARCHIVE" \
        "$YIDU_GATE_TRAIN_SCENE_LIST" \
        "$YIDU_GATE_FORBIDDEN_SCENE_LIST"; do
        if [[ ! -f "$path" ]]; then
            echo "Missing YiDu gate provenance input: $path" >&2
            exit 1
        fi
    done
    "$RUNTIME_PYTHON" "$ROOT/tools/validate_yidu_gate_provenance.py" \
        --checkpoint "$YIDU_GATE_CHECKPOINT" \
        --training-archive "$YIDU_GATE_TRAINING_ARCHIVE" \
        --train-scene-list "$YIDU_GATE_TRAIN_SCENE_LIST" \
        --forbidden-scene-list "$YIDU_GATE_FORBIDDEN_SCENE_LIST"
elif [[ -n "$YIDU_GATE_CHECKPOINT" ]]; then
    echo "Only A6 may receive BOXFUSION_YIDU_GATE_CHECKPOINT" >&2
    exit 2
elif [[ -n "$YIDU_GATE_TRAINING_ARCHIVE" ]]; then
    echo "Only A6 may receive BOXFUSION_YIDU_GATE_TRAINING_ARCHIVE" >&2
    exit 2
fi

validate_artifact_pairs() {
    local require_complete="$1"
    "$RUNTIME_PYTHON" - \
        "$ROOT" "$SCENE_LIST" "$PRED_ROOT" "$DIAGNOSTICS_ROOT" \
        "$STAGE" "$require_complete" <<'PY'
import sys
from pathlib import Path

import numpy as np

root, scene_list, prediction_root, diagnostic_root, stage, complete = (
    sys.argv[1:7]
)
sys.path.insert(0, root)
from tools.export_yidu_geometry_candidates import export_scene

scenes = [
    row.strip()
    for row in Path(scene_list).read_text(encoding="utf-8").splitlines()
    if row.strip()
]
expected = set(scenes)
prediction_root = Path(prediction_root)
diagnostic_root = Path(diagnostic_root)
present_predictions = {
    path.name.removesuffix("_boxes.pkl")
    for path in prediction_root.glob("scene*_boxes.pkl")
    if path.is_file() and path.stat().st_size > 0
}
present_diagnostics = {
    path.name.removesuffix("_tracks.npz")
    for path in diagnostic_root.glob("scene*_tracks.npz")
    if path.is_file() and path.stat().st_size > 0
}
extra_predictions = sorted(present_predictions - expected)
extra_diagnostics = sorted(present_diagnostics - expected)
if extra_predictions or extra_diagnostics:
    raise SystemExit(
        "unexpected YiDu artifacts: "
        f"pred={extra_predictions[:4]}, diag={extra_diagnostics[:4]}"
    )
for scene in scenes:
    prediction = prediction_root / f"{scene}_boxes.pkl"
    diagnostic = diagnostic_root / f"{scene}_tracks.npz"
    has_prediction = prediction.is_file() and prediction.stat().st_size > 0
    has_diagnostic = diagnostic.is_file() and diagnostic.stat().st_size > 0
    if has_prediction != has_diagnostic:
        raise SystemExit(f"incomplete prediction/diagnostic pair: {scene}")
    if complete == "1" and not has_prediction:
        raise SystemExit(f"missing completed YiDu artifact pair: {scene}")
    if not has_prediction:
        continue
    if stage != "B0":
        export_scene(
            scene_id=scene,
            diagnostic_path=diagnostic,
            prediction_path=prediction,
            expected_stage=stage,
        )
        with np.load(diagnostic, allow_pickle=False) as payload:
            required = {
                "yidu_zero_write_check_enabled",
                "yidu_zero_write_verified",
                "yidu_zero_write_pre_sha256",
                "yidu_zero_write_post_sha256",
                "yidu_zero_write_array_names",
                "yidu_zero_write_changed_fields",
            }
            missing = required - set(payload.files)
            if missing:
                raise SystemExit(
                    f"{scene}: missing zero-write diagnostics: "
                    + ", ".join(sorted(missing))
                )
            if not bool(
                np.asarray(
                    payload["yidu_zero_write_check_enabled"]
                ).item()
            ):
                raise SystemExit(
                    f"{scene}: in-process zero-write check was disabled"
                )
            if not bool(
                np.asarray(payload["yidu_zero_write_verified"]).item()
            ):
                raise SystemExit(
                    f"{scene}: in-process zero-write check was not verified"
                )
            before = str(
                np.asarray(
                    payload["yidu_zero_write_pre_sha256"]
                ).item()
            )
            after = str(
                np.asarray(
                    payload["yidu_zero_write_post_sha256"]
                ).item()
            )
            if (
                len(before) != 64
                or any(char not in "0123456789abcdef" for char in before)
                or before != after
            ):
                raise SystemExit(
                    f"{scene}: invalid or mismatched zero-write hashes"
                )
            names = np.asarray(
                payload["yidu_zero_write_array_names"]
            )
            changed = np.asarray(
                payload["yidu_zero_write_changed_fields"]
            )
            if names.size == 0 or changed.size != 0:
                raise SystemExit(
                    f"{scene}: zero-write field audit is incomplete "
                    "or reports output mutation"
                )
    else:
        with np.load(diagnostic, allow_pickle=False) as payload:
            required = {
                "scene_id", "yidu_stage", "yidu_enabled",
                "yidu_mutation_enabled", "yidu_applied_count",
                "yidu_applied",
            }
            if required - set(payload.files):
                raise SystemExit(f"{scene}: incomplete B0 diagnostics")
            if str(np.asarray(payload["scene_id"]).item()) != scene:
                raise SystemExit(f"{scene}: B0 diagnostic scene mismatch")
            if str(np.asarray(payload["yidu_stage"]).item()) != "B0":
                raise SystemExit(f"{scene}: expected B0 diagnostics")
            if bool(np.asarray(payload["yidu_enabled"]).item()):
                raise SystemExit(f"{scene}: B0 unexpectedly enables YiDu")
            if bool(np.asarray(payload["yidu_mutation_enabled"]).item()):
                raise SystemExit(f"{scene}: B0 mutation flag is enabled")
            if int(np.asarray(payload["yidu_applied_count"]).item()) != 0:
                raise SystemExit(f"{scene}: B0 reports applied rows")
            if bool(np.any(np.asarray(payload["yidu_applied"]))):
                raise SystemExit(f"{scene}: B0 contains applied rows")
PY
}

# Never mix predictions from different stages or source states.
if [[ "$ALLOW_RESUME" != "1" ]]; then
    for directory in \
        "$PRED_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT" "$EVAL_ROOT"; do
        if [[ -d "$directory" ]] \
            && [[ -n "$(find "$directory" -mindepth 1 -print -quit 2>/dev/null)" ]]; then
            echo "Refusing non-empty YiDu experiment directory: $directory" >&2
            echo "Choose a fresh BOXFUSION_YIDU_RUN_TAG." >&2
            exit 1
        fi
    done
fi
if [[ "$ALLOW_RESUME" == "1" ]]; then
    validate_artifact_pairs 0
fi

unset BOXFUSION_DISABLE_ONLINE_REFINEMENT
unset BOXFUSION_REFINER_CHECKPOINT
unset BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_JOINT_DETECTOR_BLEND
unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT
unset BOXFUSION_PROPOSAL_CACHE_DIRECTORY
unset BOXFUSION_PROPOSAL_CACHE_NAMESPACE
unset BOXFUSION_PROPOSAL_CACHE_MISSING_POLICY
unset BOXFUSION_SCANNET_POST_MIN_EXTENT

export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_ONLINE_ABLATION_PROFILE="$PROFILE"

# Frozen B6 contract.  Every YiDu stage is an observer child of this anchor.
export BOXFUSION_PROPOSAL_PROVIDER="yoloe"
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
export BOXFUSION_QUALITY_MODE="iou_mlp"
export BOXFUSION_QUALITY_DETECTOR_BLEND="0.40"
export BOXFUSION_SCANNET_MIN_EXTENT="0.40"
export BOXFUSION_PROPOSAL_INTERVAL="5"
export BOXFUSION_CANDIDATE_TTL_CLOCK="provider_call"
export BOXFUSION_CANDIDATE_TRACK_TTL="3"
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS="0"
export BOXFUSION_INFERENCE_SEED="0"
export BOXFUSION_EVAL_SEED="0"
export BOXFUSION_LIVE_ROOT="$LIVE_ROOT"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"

if [[ "$STAGE" != "B0" ]]; then
    export BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY="$TEACHER_CACHE"
    export BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE="$TEACHER_NAMESPACE"
    export BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY="$CACHE_MISSING_POLICY"
else
    unset BOXFUSION_C4_PROPOSAL_CACHE_DIRECTORY
    unset BOXFUSION_C4_PROPOSAL_CACHE_NAMESPACE
    unset BOXFUSION_C4_PROPOSAL_CACHE_MISSING_POLICY
fi
if [[ "$STAGE" == "A6" ]]; then
    export BOXFUSION_YIDU_GATE_CHECKPOINT="$YIDU_GATE_CHECKPOINT"
else
    unset BOXFUSION_YIDU_GATE_CHECKPOINT
fi

export BOXFUSION_ONLINE_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_ONLINE_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_DIAGNOSTICS_ROOT="$DIAGNOSTICS_ROOT"
export BOXFUSION_EVAL_ROOT="$EVAL_ROOT"

DRY_RUN_TEMP=""
if [[ "$DRY_RUN" == "1" ]]; then
    DRY_RUN_TEMP="$(mktemp -d)"
    MANIFEST="$DRY_RUN_TEMP/run_manifest.json"
else
    mkdir -p "$LOG_ROOT"
    MANIFEST="$LOG_ROOT/run_manifest.json"
fi
if [[ "$ALLOW_RESUME" == "1" && ! -f "$MANIFEST" ]]; then
    echo "Resume requires an existing run manifest: $MANIFEST" >&2
    exit 1
fi
MANIFEST_ARGS=(
    --output "$MANIFEST"
    --stage "$STAGE"
    --profile "$PROFILE"
    --config "$CONFIG"
    --scene-list "$SCENE_LIST"
    --b6-checkpoint "$QUALITY_CHECKPOINT"
    --yoloe-checkpoint "$YOLOE_CHECKPOINT"
    --teacher-cache "$TEACHER_CACHE"
    --teacher-namespace "$TEACHER_NAMESPACE"
    --cache-missing-policy "$CACHE_MISSING_POLICY"
    --live-root "$LIVE_ROOT"
    --frames-root "$FRAMES_ROOT"
    --prediction-root "$PRED_ROOT"
    --log-root "$LOG_ROOT"
    --diagnostics-root "$DIAGNOSTICS_ROOT"
    --evaluation-root "$EVAL_ROOT"
    --minimum-extent 0.40
    --post-minimum-extent disabled
    --inference-seed 0
    --evaluation-seed 0
    --python-executable "$RUNTIME_PYTHON"
    --python-version "$PYTHON_VERSION"
    --torch-version "$TORCH_VERSION"
)
if [[ -n "$YIDU_GATE_CHECKPOINT" ]]; then
    MANIFEST_ARGS+=(--gate-checkpoint "$YIDU_GATE_CHECKPOINT")
    MANIFEST_ARGS+=(
        --gate-training-archive "$YIDU_GATE_TRAINING_ARCHIVE"
        --gate-train-scene-list "$YIDU_GATE_TRAIN_SCENE_LIST"
        --gate-forbidden-scene-list "$YIDU_GATE_FORBIDDEN_SCENE_LIST"
    )
fi
if [[ "$ALLOW_RESUME" == "1" ]]; then
    MANIFEST_ARGS+=(--verify-existing)
fi
"$RUNTIME_PYTHON" "$ROOT/tools/build_yidu_run_manifest.py" "${MANIFEST_ARGS[@]}"

scene_count="$(awk 'NF { count += 1 } END { print count + 0 }' "$SCENE_LIST")"
echo "YiDu strict incremental observer"
echo "  stage/profile: $STAGE / $PROFILE"
echo "  scenes: $scene_count from $SCENE_LIST"
echo "  frozen B6: blend=0.40 / minimum extent=0.40 / seeds=0,0"
echo "  SAM3 cache: $([[ "$STAGE" == "B0" ]] && echo disabled || echo "$TEACHER_CACHE")"
echo "  gate checkpoint: ${YIDU_GATE_CHECKPOINT:-disabled}"
echo "  gate training archive: ${YIDU_GATE_TRAINING_ARCHIVE:-disabled}"
echo "  output contract: boxes/scores/count/order/IDs identical to frozen B6"
echo "  manifest: $MANIFEST"
echo "  GPUs: $GPU_SPEC"

if [[ "$DRY_RUN" == "1" ]]; then
    rm -f "$MANIFEST"
    rmdir "$DRY_RUN_TEMP"
    echo "DRY RUN PASSED. No output was kept and no GPU process was started."
    exit 0
fi

bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
validate_artifact_pairs 1
echo "YiDu $STAGE artifact-set validation completed."
