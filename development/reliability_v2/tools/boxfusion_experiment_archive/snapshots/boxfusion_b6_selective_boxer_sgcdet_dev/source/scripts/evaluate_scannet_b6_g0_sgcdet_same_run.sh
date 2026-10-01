#!/usr/bin/env bash
set -euo pipefail

# Evaluate the exact pre-geometry counterfactual of one combined active run.

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
SCENE_LIST="${BOXFUSION_COMBO_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
ACTIVE_TAG="${BOXFUSION_COMBO_ACTIVE_TAG:-g0_sgcdet_active_fixed10_v1}"
IDENTITY_TAG="${BOXFUSION_COMBO_COUNTERFACTUAL_TAG:-${ACTIVE_TAG}_same_run_identity_v1}"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "GPU must be one non-negative integer index" >&2
    exit 2
fi

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
active_pred="$ROOT/results/b6_g0_sgcdet/$ACTIVE_TAG/$list_scope"
active_diagnostics="$ROOT/diagnostics/b6_g0_sgcdet/$ACTIVE_TAG/$list_scope/online"
active_log="$ROOT/logs/b6_g0_sgcdet/$ACTIVE_TAG/$list_scope/eval_stdout.log"
identity_pred="$ROOT/results/b6_g0_sgcdet_counterfactual/$IDENTITY_TAG/$list_scope"
identity_eval="$ROOT/evaluation/b6_g0_sgcdet_counterfactual/$IDENTITY_TAG/$list_scope"
identity_log_root="$ROOT/logs/b6_g0_sgcdet_counterfactual/$IDENTITY_TAG/$list_scope"
identity_log="$identity_log_root/eval_stdout.log"
report="$identity_log_root/paired_report.json"

for dependency in "$PYTHON" "$SCENE_LIST" "$active_log"; do
    if [[ ! -f "$dependency" ]]; then
        echo "Missing paired-evaluation dependency: $dependency" >&2
        exit 1
    fi
done
for dependency in "$active_pred" "$active_diagnostics"; do
    if [[ ! -d "$dependency" ]]; then
        echo "Missing paired-evaluation directory: $dependency" >&2
        exit 1
    fi
done
if [[ -e "$identity_pred" || -e "$identity_log_root" ]]; then
    echo "Refusing to overwrite an existing paired evaluation: $IDENTITY_TAG" >&2
    echo "Set BOXFUSION_COMBO_COUNTERFACTUAL_TAG to a fresh tag." >&2
    exit 1
fi

mkdir -p "$identity_log_root/mplconfig" "$identity_eval"
"$PYTHON" "$ROOT/tools/build_sgcdet_same_run_identity.py" \
    --active-pred-root "$active_pred" \
    --diagnostics-root "$active_diagnostics" \
    --scene-list "$SCENE_LIST" \
    --output-root "$identity_pred" \
    > "$identity_log_root/build_stdout.json"

echo "Evaluating same-run identity counterfactual: $identity_pred"
CUDA_VISIBLE_DEVICES="$GPU" \
PYTHONDONTWRITEBYTECODE=1 \
MPLCONFIGDIR="$identity_log_root/mplconfig" \
LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
"$PYTHON" "$ROOT/evaluation/eval_scannet.py" \
    --dataset scannet \
    --data_path /extra/ZhaoX/scannet_data/scans \
    --gt_root "$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data" \
    --dump_dir "$identity_eval" \
    --num_point 40000 \
    --cluster_sampling seed_fps \
    --use_3d_nms \
    --use_cls_nms \
    --per_class_proposal \
    --num_workers 0 \
    --gpu 0 \
    --seed 0 \
    --scene_list "$SCENE_LIST" \
    --pred_root "$identity_pred" \
    > "$identity_log" 2>&1

"$PYTHON" - "$active_log" "$identity_log" "$identity_pred/manifest.json" "$report" <<'PY'
import json
import re
import sys

active_log, identity_log, manifest_path, report_path = sys.argv[1:]

def metrics(path):
    with open(path, "r", encoding="utf-8") as handle:
        values = [float(value) for value in re.findall(r"eval mAP:\s*([0-9.]+)", handle.read())]
    if len(values) != 3:
        raise SystemExit(f"Expected exactly three mAP values in {path}, got {values}")
    return values

active = metrics(active_log)
identity = metrics(identity_log)
with open(manifest_path, "r", encoding="utf-8") as handle:
    manifest = json.load(handle)

report = {
    "schema": "boxfusion.g0_sgcdet_same_run_evaluation.v1",
    "thresholds": [0.15, 0.25, 0.50],
    "identity_map": identity,
    "active_map": active,
    "delta_map": [right - left for left, right in zip(identity, active)],
    "geometry_manifest": manifest,
}
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "Paired report: $report"
