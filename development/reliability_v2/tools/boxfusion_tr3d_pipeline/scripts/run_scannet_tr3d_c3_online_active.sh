#!/usr/bin/env bash
set -euo pipefail

# Train-authorized online C3 append branch.  The terminal R3 tree remains an
# immutable paired anchor; C3 output is written during the same scene process.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
TAG="${BOXFUSION_C3_ACTIVE_RUN_TAG:-c3_online_active_smoke1_v1}"
SCENE_LIST="${BOXFUSION_C3_ACTIVE_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_smoke_scene0277_00.txt}"
POLICY="${BOXFUSION_C3_ACTIVE_POLICY:-}"
ARTIFACT_BASE="${BOXFUSION_TR3D_TERMINAL_ARTIFACT_BASE:-$ROOT}"
PYTHON_BIN="$(readlink -f "${BOXFUSION_C3_ACTIVE_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}")"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
SCANS_ROOT="${BOXFUSION_C3_ACTIVE_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_C3_ACTIVE_GT_ROOT:-$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data}"

[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,95}$ ]] || {
    echo "Invalid BOXFUSION_C3_ACTIVE_RUN_TAG: $TAG" >&2
    exit 2
}
if [[ -z "$POLICY" ]]; then
    echo "Set BOXFUSION_C3_ACTIVE_POLICY to a train-only authorized checkpoint" >&2
    exit 2
fi
[[ -x "$PYTHON_BIN" ]] || { echo "Missing C3 active Python: $PYTHON_BIN" >&2; exit 2; }
for path in "$SCENE_LIST" "$POLICY"; do
    [[ -f "$path" && ! -L "$path" ]] || {
        echo "Missing/non-regular C3 active input: $path" >&2
        exit 2
    }
done
[[ ! -w "$POLICY" ]] || {
    echo "C3 active policy must be immutable (chmod 0444): $POLICY" >&2
    exit 2
}
for path in "$SCANS_ROOT" "$GT_ROOT"; do
    [[ -d "$path" ]] || { echo "Missing C3 active evaluation root: $path" >&2; exit 2; }
done

LIST_SHA="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
LIST_SCOPE="$(basename "$SCENE_LIST" .txt)-${LIST_SHA:0:12}"
ANCHOR_ROOT="$ARTIFACT_BASE/results/b6_g0_tr3d_terminal/$TAG/$LIST_SCOPE"
ACTIVE_ROOT="$ARTIFACT_BASE/results/b6_g0_tr3d_c3_online_active/$TAG/$LIST_SCOPE"
IDENTITY_ROOT="$ARTIFACT_BASE/diagnostics/b6_g0_tr3d_terminal/$TAG/$LIST_SCOPE/online/tr3d_c3_online_identity"
ACTIVE_DIAGNOSTICS_ROOT="$ARTIFACT_BASE/diagnostics/b6_g0_tr3d_c3_online_active/$TAG/$LIST_SCOPE"
REPORT_ROOT="$ARTIFACT_BASE/reports/b6_g0_tr3d_c3_online_active/$TAG/$LIST_SCOPE"
EVAL_ROOT="$ARTIFACT_BASE/evaluation/b6_g0_tr3d_c3_online_active/$TAG/$LIST_SCOPE"
LOG_ROOT="$ARTIFACT_BASE/logs/b6_g0_tr3d_c3_online_active/$TAG/$LIST_SCOPE"
AUDIT_REPORT="$REPORT_ROOT/identity_audit.json"
RESUME="${BOXFUSION_C3_ACTIVE_RESUME:-0}"

if [[ "$RESUME" == "0" ]]; then
    for path in "$ACTIVE_ROOT" "$ACTIVE_DIAGNOSTICS_ROOT" "$REPORT_ROOT" "$EVAL_ROOT" "$LOG_ROOT"; do
        [[ ! -e "$path" ]] || {
            echo "Refusing existing immutable C3 active namespace: $path" >&2
            echo "Choose a new BOXFUSION_C3_ACTIVE_RUN_TAG." >&2
            exit 2
        }
    done
elif [[ "$RESUME" != "1" ]]; then
    echo "BOXFUSION_C3_ACTIVE_RESUME must be 0 or 1" >&2
    exit 2
fi
mkdir -p "$REPORT_ROOT" "$EVAL_ROOT" "$LOG_ROOT/mplconfig"

export BOXFUSION_TR3D_TERMINAL_RUN_TAG="$TAG"
export BOXFUSION_C3_ONLINE_RUN_TAG="$TAG"
export BOXFUSION_C3_ONLINE_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_B6_BOXER_SCENE_LIST="$SCENE_LIST"
export BOXFUSION_C3_ONLINE_ENABLED=1
export BOXFUSION_C3_ONLINE_CANDIDATE_SOURCE=parent_score
export BOXFUSION_C3_ONLINE_ACTIVE_POLICY="$POLICY"
export BOXFUSION_C3_ONLINE_ACTIVE_OUTPUT_ROOT="$ACTIVE_ROOT"
export BOXFUSION_C3_ONLINE_ACTIVE_DIAGNOSTICS_ROOT="$ACTIVE_DIAGNOSTICS_ROOT"
export BOXFUSION_YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev/models/yoloe-11s-seg-pf.pt}"

echo "Train-authorized online C3 active run"
echo "  tag: $TAG"
echo "  scenes: $SCENE_LIST"
echo "  policy: $POLICY"
echo "  paired terminal R3 anchor: $ANCHOR_ROOT"
echo "  online C3 active output: $ACTIVE_ROOT"
echo "  note: TR3D parent proposals remain p100 cache replay"

bash "$ROOT/scripts/run_scannet_tr3d_c3_online_identity.sh" "$GPU_SPEC"

"$PYTHON_BIN" "$ROOT/tools/audit_tr3d_c3_online_active.py" \
    --scene-list "$SCENE_LIST" --policy "$POLICY" \
    --anchor-root "$ANCHOR_ROOT" --active-root "$ACTIVE_ROOT" \
    --diagnostics-root "$ACTIVE_DIAGNOSTICS_ROOT" --report "$AUDIT_REPORT" \
    > "$LOG_ROOT/identity_audit_stdout.json"
chmod 0444 "$LOG_ROOT/identity_audit_stdout.json"

EVAL_LOG="$LOG_ROOT/eval_active.log"
(
    cd "$ROOT/evaluation"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 \
        MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
        "$PYTHON_BIN" eval_scannet.py \
        --dataset scannet --data_path "$SCANS_ROOT" --gt_root "$GT_ROOT" \
        --dump_dir "$EVAL_ROOT" --num_point 40000 \
        --cluster_sampling seed_fps --use_3d_nms --use_cls_nms \
        --per_class_proposal --num_workers 0 --gpu 0 --seed 0 \
        --scene_list "$SCENE_LIST" --pred_root "$ACTIVE_ROOT"
) > "$EVAL_LOG" 2>&1
chmod 0444 "$EVAL_LOG"

echo "=== terminal R3 anchor ==="
grep -E '^eval (mAP|APrec|ARecall):' \
    "$ARTIFACT_BASE/logs/b6_g0_tr3d_terminal/$TAG/$LIST_SCOPE/eval_active_stdout.log"
echo "=== terminal R3 + train-authorized online C3 ==="
grep -E '^eval (mAP|APrec|ARecall):' "$EVAL_LOG"
echo "Online C3 active evaluation completed"
echo "  audit: $AUDIT_REPORT"
