#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-6}"
GPU_SPEC="${2:-0,1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[[ "$STAGE" =~ ^[1-6]$ ]] || { echo "stage must be 1..6" >&2; exit 2; }
TAG="${BOXFUSION_LIGHTWEIGHT_ACTIVE_TAG:-tr3d_lightweight_l${STAGE}_fixed10_v1}"
SCENE_LIST="${BOXFUSION_LIGHTWEIGHT_ACTIVE_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
POLICY="${BOXFUSION_LIGHTWEIGHT_ACTIVE_POLICY:-$ROOT/models/tr3d_lightweight_l${STAGE}_gate_train100_v1.json}"
YOLOE="${BOXFUSION_YOLOE_CHECKPOINT:-/data/ZhaoX/OVM3D-Dett/boxfusion_stage3_dev/models/yoloe-11s-seg-pf.pt}"
PYTHON_BIN="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCANS_ROOT="${BOXFUSION_LIGHTWEIGHT_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
GT_ROOT="${BOXFUSION_LIGHTWEIGHT_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"

for path in "$SCENE_LIST" "$POLICY" "$YOLOE"; do [[ -f "$path" && ! -L "$path" ]] || { echo "Missing input: $path" >&2; exit 2; }; done
[[ ! -w "$POLICY" ]] || { echo "Policy must be immutable: $POLICY" >&2; exit 2; }
list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
ANCHOR_ROOT="$ROOT/results/b6_g0_tr3d_terminal/$TAG/$scope"
DIAGNOSTICS_ROOT="$ROOT/diagnostics/b6_g0_tr3d_terminal/$TAG/$scope/online/tr3d_incremental"
ACTIVE_ROOT="$ROOT/results/b6_g0_tr3d_lightweight_active/$TAG/$scope"
REPORT_ROOT="$ROOT/reports/b6_g0_tr3d_lightweight_active/$TAG/$scope"
EVAL_ROOT="$ROOT/evaluation/b6_g0_tr3d_lightweight_active/$TAG/$scope"
LOG_ROOT="$ROOT/logs/b6_g0_tr3d_lightweight_active/$TAG/$scope"
MANIFEST="$REPORT_ROOT/materialize_manifest.json"
AUDIT="$REPORT_ROOT/identity_audit.json"
for path in "$ANCHOR_ROOT" "$DIAGNOSTICS_ROOT" "$ACTIVE_ROOT" "$REPORT_ROOT" "$EVAL_ROOT" "$LOG_ROOT"; do [[ ! -e "$path" ]] || { echo "Refusing existing namespace: $path" >&2; exit 2; }; done
mkdir -p "$REPORT_ROOT" "$EVAL_ROOT" "$LOG_ROOT/mplconfig"

echo "Lightweight L$STAGE active ablation: tag=$TAG, scenes=$SCENE_LIST, GPUs=$GPU_SPEC"
BOXFUSION_B6_BOXER_SCENE_LIST="$SCENE_LIST" \
BOXFUSION_TR3D_TERMINAL_RUN_TAG="$TAG" \
BOXFUSION_YOLOE_CHECKPOINT="$YOLOE" \
BOXFUSION_INCREMENTAL_TR3D_OBSERVER=1 \
BOXFUSION_TR3D_LIGHTWEIGHT_FUSION=1 \
BOXFUSION_TR3D_LIGHTWEIGHT_STAGE="$STAGE" \
BOXFUSION_TR3D_INCREMENTAL_EVERY_KEYFRAMES=5 \
    bash "$ROOT/scripts/run_scannet_b6_g0_tr3d_terminal_active.sh" "$GPU_SPEC"

"$PYTHON_BIN" "$ROOT/tools/materialize_tr3d_lightweight_active.py" \
    --scene-list "$SCENE_LIST" --policy "$POLICY" --stage "$STAGE" \
    --anchor-root "$ANCHOR_ROOT" --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --output-root "$ACTIVE_ROOT" --manifest "$MANIFEST" --candidate-nms-iou 0.25 \
    > "$LOG_ROOT/materialize_stdout.json"
chmod 0444 "$LOG_ROOT/materialize_stdout.json"
"$PYTHON_BIN" "$ROOT/tools/audit_tr3d_lightweight_active.py" \
    --scene-list "$SCENE_LIST" --policy "$POLICY" --stage "$STAGE" \
    --manifest "$MANIFEST" --anchor-root "$ANCHOR_ROOT" \
    --active-root "$ACTIVE_ROOT" --output "$AUDIT" \
    > "$LOG_ROOT/identity_audit_stdout.json"
chmod 0444 "$LOG_ROOT/identity_audit_stdout.json"
(
    cd "$ROOT/evaluation"
    env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$LOG_ROOT/mplconfig" \
        "$PYTHON_BIN" eval_scannet.py --dataset scannet --data_path "$SCANS_ROOT" \
        --gt_root "$GT_ROOT" --dump_dir "$EVAL_ROOT" --num_point 40000 \
        --cluster_sampling seed_fps --use_3d_nms --use_cls_nms \
        --per_class_proposal --num_workers 0 --gpu 0 --seed 0 \
        --scene_list "$SCENE_LIST" --pred_root "$ACTIVE_ROOT"
) > "$LOG_ROOT/eval_active.log" 2>&1
chmod 0444 "$LOG_ROOT/eval_active.log"
echo "=== terminal R3 anchor ==="
grep -E '^eval (mAP|APrec|ARecall):' "$ROOT/logs/b6_g0_tr3d_terminal/$TAG/$scope/eval_active_stdout.log"
echo "=== terminal R3 + lightweight L$STAGE ==="
grep -E '^eval (mAP|APrec|ARecall):' "$LOG_ROOT/eval_active.log"
"$PYTHON_BIN" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("supplemental:",d["applied_count"],"eligible:",d["eligible_before_nms"])' "$MANIFEST"
echo "Lightweight L$STAGE active evaluation completed: $AUDIT"
