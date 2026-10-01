#!/usr/bin/env bash
set -euo pipefail

# P1G: frozen B6 + frozen P1S + detached multi-view occupancy/MSR geometry.
#
# This entry point is observer-only.  It cannot activate refined boxes or
# change formal boxes, scores, labels, counts, or ordering.  The safe default
# is a two-scene train-only smoke run.  Every larger protocol scope must be
# selected explicitly with BOXFUSION_P1G_SCOPE.
#
# Safe smoke:
#   bash scripts/run_scannet_p1g_ablation.sh 0,1
#
# Train-only parameter development:
#   BOXFUSION_P1G_SCOPE=train_fit60 \
#   BOXFUSION_P1G_RUN_TAG=p1g_fit60_cfg01 \
#     bash scripts/run_scannet_p1g_ablation.sh 0,1
#
# Supported scopes:
#   train_smoke2 (default), train_fit60, train_cal20, train_audit20,
#   train_fresh50, fixed_val10, custom.
#
# `fixed_val10` is quarantined and requires an explicit acknowledgement.
# full100 is deliberately unsupported by this observer-development script.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VAL_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SCOPE="${BOXFUSION_P1G_SCOPE:-train_smoke2}"
SCENE_ROLE=train_only

if [[ "${BOXFUSION_P1G_FULL100:-0}" != "0" ]]; then
    echo "P1G full100 is forbidden by the observer-development protocol." >&2
    echo "Audit a frozen train-only configuration first; this script never auto-runs full100." >&2
    exit 2
fi

case "$SCOPE" in
    train_smoke2)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1_smoke2.txt"
        ;;
    train_fit60)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_fit60.txt"
        ;;
    train_cal20)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_cal20.txt"
        ;;
    train_audit20)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_audit20.txt"
        ;;
    train_fresh50)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_audit50_fresh_v1.txt"
        ;;
    fixed_val10)
        if [[ "${BOXFUSION_P1G_ALLOW_TOUCHED_VAL10:-0}" != "1" ]]; then
            echo "fixed_val10 is quarantined; set BOXFUSION_P1G_ALLOW_TOUCHED_VAL10=1 only after the frozen train-only GO decision." >&2
            exit 2
        fi
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
        SCENE_ROLE=touched_validation
        ;;
    custom)
        if [[ -z "${BOXFUSION_P1G_SCENE_LIST:-}" ]]; then
            echo "BOXFUSION_P1G_SCOPE=custom requires BOXFUSION_P1G_SCENE_LIST" >&2
            exit 2
        fi
        SCENE_LIST="$BOXFUSION_P1G_SCENE_LIST"
        SCENE_ROLE="${BOXFUSION_P1G_SCENE_ROLE:-train_only}"
        case "$SCENE_ROLE" in
            train_only) ;;
            touched_validation)
                if [[ "${BOXFUSION_P1G_ALLOW_TOUCHED_VAL10:-0}" != "1" ]]; then
                    echo "A validation custom list requires BOXFUSION_P1G_ALLOW_TOUCHED_VAL10=1" >&2
                    exit 2
                fi
                ;;
            *)
                echo "BOXFUSION_P1G_SCENE_ROLE must be train_only or touched_validation" >&2
                exit 2
                ;;
        esac
        ;;
    full100|val100)
        echo "P1G scope '$SCOPE' is forbidden; no automatic full100 path exists." >&2
        exit 2
        ;;
    *)
        echo "Unsupported BOXFUSION_P1G_SCOPE: $SCOPE" >&2
        exit 2
        ;;
esac

P1S_CHECKPOINT="${BOXFUSION_P1G_P1S_CHECKPOINT:-$ROOT/models/scannet_p1s_native_sparse.pt}"
QUALITY_CHECKPOINT="${BOXFUSION_P_B6_CHECKPOINT:-$ROOT/models/scannet_b6_iou_mlp.npz}"
YOLOE_CHECKPOINT="${BOXFUSION_P_YOLOE_CHECKPOINT:-$ROOT/models/yoloe-11s-seg-pf.pt}"
CONFIG="${BOXFUSION_P_CONFIG:-$ROOT/config/scannet_online_refinement.yaml}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
if [[ "$SCENE_ROLE" == "train_only" ]]; then
    FRAMES_ROOT="${BOXFUSION_P1G_FRAMES_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_dev/data/scannet_train}"
else
    FRAMES_ROOT="${BOXFUSION_P1G_FRAMES_ROOT:-/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames}"
fi

for path in \
    "$PYTHON" "$SCENE_LIST" "$VAL_LIST" "$P1S_CHECKPOINT" \
    "$QUALITY_CHECKPOINT" "$YOLOE_CHECKPOINT" "$CONFIG"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P1G input: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$FRAMES_ROOT" ]]; then
    echo "Missing P1G frames root: $FRAMES_ROOT" >&2
    exit 1
fi

# Fail closed on empty/duplicated lists and on any train/validation overlap.
"$PYTHON" -c '
from pathlib import Path
import sys

scene_path, val_path, role = map(str, sys.argv[1:4])
def read(path):
    rows = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise SystemExit(f"empty scene list: {path}")
    if len(rows) != len(set(rows)):
        raise SystemExit(f"duplicate scene IDs in: {path}")
    return rows

scenes = read(scene_path)
validation = set(read(val_path))
overlap = sorted(set(scenes) & validation)
if role == "train_only" and overlap:
    raise SystemExit(
        "train-only P1G list overlaps ScanNet validation: "
        + ", ".join(overlap[:8])
    )
print(f"P1G scene-list preflight OK: {len(scenes)} scenes, role={role}")
' "$SCENE_LIST" "$VAL_LIST" "$SCENE_ROLE"

RUN_TAG="${BOXFUSION_P1G_RUN_TAG:-p1g_${SCOPE}_b6_p1s_frozen_v1}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "BOXFUSION_P1G_RUN_TAG contains unsafe path characters" >&2
    exit 2
fi

# Standard AP is not informative in an observer-only stage and the train
# scopes do not use the validation evaluator.  The dedicated offline audit is
# the only supported measurement path.
if [[ "${BOXFUSION_P1G_SKIP_EVALUATION:-1}" != "1" ]]; then
    echo "P1G observer runs require BOXFUSION_P1G_SKIP_EVALUATION=1; use audit_scannet_p1g.sh afterwards." >&2
    exit 2
fi

# Downstream proposal/ranking heads are forbidden in this geometry-only stage.
unset BOXFUSION_P2_OCCUPANCY_CHECKPOINT
unset BOXFUSION_REFINER_CHECKPOINT BOXFUSION_JOINT_CHECKPOINT
unset BOXFUSION_TRIFUSION_GATE_CHECKPOINT BOXFUSION_YIDU_GATE_CHECKPOINT
unset BOXFUSION_QUALITY_APPLY_TO_SUPPLEMENTAL
export BOXFUSION_ENV_ROOT="$ENV_ROOT"
export BOXFUSION_ONLINE_CONFIG="$CONFIG"
export BOXFUSION_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_SCANNET_FRAMES_ROOT="$FRAMES_ROOT"
export BOXFUSION_ONLINE_ABLATION_PROFILE=p1g_multiview_occupancy_msr_observer
export BOXFUSION_P_STAGE=P1G
export BOXFUSION_P1_RESIDUAL_MODE=infer
export BOXFUSION_P1_COLLECT_VOXELS=0
export BOXFUSION_P1_RESIDUAL_CHECKPOINT="$P1S_CHECKPOINT"
export BOXFUSION_PROPOSAL_PROVIDER=yoloe
export BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT"
export BOXFUSION_QUALITY_MODE=iou_mlp
export BOXFUSION_QUALITY_CHECKPOINT="$QUALITY_CHECKPOINT"
export BOXFUSION_QUALITY_DETECTOR_BLEND="${BOXFUSION_P_B6_DETECTOR_BLEND:-0.40}"
export BOXFUSION_SCANNET_MIN_EXTENT="${BOXFUSION_P_B6_MIN_EXTENT:-0.40}"
export BOXFUSION_PROPOSAL_INTERVAL="${BOXFUSION_P_PROPOSAL_INTERVAL:-5}"
export BOXFUSION_CANDIDATE_TTL_CLOCK=provider_call
export BOXFUSION_CANDIDATE_TRACK_TTL=3
export BOXFUSION_ARCHIVE_CONFIRMED_TRACKS=0
export BOXFUSION_INFERENCE_SEED=0
export BOXFUSION_EVAL_SEED=0
export BOXFUSION_SKIP_EVALUATION=1
export BOXFUSION_ONLINE_PRED_ROOT="$ROOT/results/p1g_ablation/$RUN_TAG"
export BOXFUSION_ONLINE_LOG_ROOT="$ROOT/logs/p1g_ablation/$RUN_TAG"
export BOXFUSION_DIAGNOSTICS_ROOT="$ROOT/diagnostics/p1g_ablation/$RUN_TAG"
export BOXFUSION_EVAL_ROOT="$ROOT/evaluation/p1g_ablation/$RUN_TAG"
export BOXFUSION_P_MANIFEST="$ROOT/logs/p1g_ablation/$RUN_TAG/run_manifest.json"

echo "P1G observer: scope=$SCOPE, role=$SCENE_ROLE, tag=$RUN_TAG, GPUs=$GPU_SPEC"
echo "P1G parent: frozen P1S $P1S_CHECKPOINT"
echo "P1G config: $CONFIG"
echo "P1G frames: $FRAMES_ROOT"
echo "P1G safety: observer-only; formal prediction mutation is impossible"
exec bash "$ROOT/scripts/run_scannet_online_refinement.sh" "$GPU_SPEC"
