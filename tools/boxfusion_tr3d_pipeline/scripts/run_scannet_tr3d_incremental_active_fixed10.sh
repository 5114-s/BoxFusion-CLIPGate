#!/usr/bin/env bash
set -euo pipefail

# Train-authorized causal incremental TR3D append-only ablation.  A fresh
# terminal-R3 anchor and observer diagnostic are produced in the same run;
# only the separate active tree receives supplemental candidates.

GPU_SPEC="${1:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
TAG="${BOXFUSION_INCREMENTAL_ACTIVE_TAG:-incremental_tr3d_active_fixed10_v1}"
SCENE_LIST="${BOXFUSION_INCREMENTAL_ACTIVE_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
POLICY="${BOXFUSION_INCREMENTAL_ACTIVE_POLICY:-$ROOT/models/tr3d_incremental_novelty_gate_train100_v1.json}"
YOLOE_CHECKPOINT="${BOXFUSION_YOLOE_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev/models/yoloe-11s-seg-pf.pt}"
PYTHON_BIN="${BOXFUSION_INCREMENTAL_ACTIVE_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCANS_ROOT="${BOXFUSION_INCREMENTAL_ACTIVE_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_INCREMENTAL_ACTIVE_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"

[[ "$TAG" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{2,95}$ ]] || { echo "Invalid tag: $TAG" >&2; exit 2; }
for path in "$SCENE_LIST" "$POLICY" "$YOLOE_CHECKPOINT"; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }
done
[[ ! -w "$POLICY" ]] || { echo "Policy must be immutable: $POLICY" >&2; exit 2; }
for path in "$SCANS_ROOT" "$GT_ROOT"; do
    [[ -d "$path" ]] || { echo "Missing evaluation root: $path" >&2; exit 2; }
done

LIST_SHA="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
SCOPE="$(basename "$SCENE_LIST" .txt)-${LIST_SHA:0:12}"
ANCHOR_ROOT="$ROOT/results/b6_g0_tr3d_terminal/$TAG/$SCOPE"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/b6_g0_tr3d_terminal/$TAG/$SCOPE/online/tr3d_incremental"
ACTIVE_ROOT="$ROOT/results/b6_g0_tr3d_incremental_active/$TAG/$SCOPE"
REPORT_ROOT="$ROOT/reports/b6_g0_tr3d_incremental_active/$TAG/$SCOPE"
EVAL_ROOT="$ROOT/evaluation/b6_g0_tr3d_incremental_active/$TAG/$SCOPE"
LOG_ROOT="$ROOT/logs/b6_g0_tr3d_incremental_active/$TAG/$SCOPE"
MANIFEST="$REPORT_ROOT/materialize_manifest.json"
AUDIT="$REPORT_ROOT/identity_audit.json"

for path in "$ANCHOR_ROOT" "$DIAGNOSTICS_ROOT" "$ACTIVE_ROOT" "$REPORT_ROOT" "$EVAL_ROOT" "$LOG_ROOT"; do
    [[ ! -e "$path" ]] || { echo "Refusing existing namespace: $path" >&2; exit 2; }
done
mkdir -p "$REPORT_ROOT" "$EVAL_ROOT" "$LOG_ROOT/mplconfig"

echo "Train-authorized incremental TR3D fixed10 active ablation"
echo "  tag/scenes: $TAG / $SCENE_LIST"
echo "  policy: $POLICY"
echo "  GPUs/interval: $GPU_SPEC / every 5 keyframes"
echo "  active safety: anchor IoU<=0.10, candidate NMS=0.25, max=6/scene"

BOXFUSION_B6_BOXER_SCENE_LIST="$SCENE_LIST" \
BOXFUSION_TR3D_TERMINAL_RUN_TAG="$TAG" \
BOXFUSION_YOLOE_CHECKPOINT="$YOLOE_CHECKPOINT" \
BOXFUSION_INCREMENTAL_TR3D_OBSERVER=1 \
BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES=5 \
    bash "$ROOT/scripts/run_scannet_b6_g0_tr3d_terminal_active.sh" "$GPU_SPEC"

"$PYTHON_BIN" "$ROOT/tools/materialize_tr3d_incremental_novelty_active.py" \
    --scene-list "$SCENE_LIST" --policy "$POLICY" \
    --anchor-root "$ANCHOR_ROOT" --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --output-root "$ACTIVE_ROOT" --manifest "$MANIFEST" \
    --candidate-nms-iou 0.25 > "$LOG_ROOT/materialize_stdout.json"
chmod 0444 "$LOG_ROOT/materialize_stdout.json"

"$PYTHON_BIN" "$ROOT/tools/audit_tr3d_incremental_novelty_active.py" \
    --scene-list "$SCENE_LIST" --policy "$POLICY" --manifest "$MANIFEST" \
    --anchor-root "$ANCHOR_ROOT" --active-root "$ACTIVE_ROOT" \
    --output "$AUDIT" > "$LOG_ROOT/identity_audit_stdout.json"
chmod 0444 "$LOG_ROOT/identity_audit_stdout.json"

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
) > "$LOG_ROOT/eval_active.log" 2>&1
chmod 0444 "$LOG_ROOT/eval_active.log"

echo "=== terminal R3 anchor ==="
grep -E '^eval (mAP|APrec|ARecall):' \
    "$ROOT/logs/b6_g0_tr3d_terminal/$TAG/$SCOPE/eval_active_stdout.log"
echo "=== terminal R3 + incremental novelty gate ==="
grep -E '^eval (mAP|APrec|ARecall):' "$LOG_ROOT/eval_active.log"
"$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("supplemental candidates:",d["applied_count"],"eligible before NMS:",d["eligible_before_nms"])' "$MANIFEST"
echo "Incremental novelty fixed10 active evaluation completed: $AUDIT"
