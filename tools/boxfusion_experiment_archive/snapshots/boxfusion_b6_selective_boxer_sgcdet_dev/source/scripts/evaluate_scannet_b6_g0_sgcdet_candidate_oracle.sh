#!/usr/bin/env bash
set -euo pipefail

# Diagnose whether the frozen sparse candidate geometry or its runtime gate is
# responsible for the missing AP25/AP50 gain. This command is GT-only offline
# analysis and never changes online inference outputs.

GPU="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
LIVE_ROOT="${BOXFUSION_LIVE_ROOT:-/data/ZhaoX/BoxFusion}"
SCENE_LIST="${BOXFUSION_COMBO_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
ACTIVE_TAG="${BOXFUSION_COMBO_ACTIVE_TAG:-g0_sgcdet_active_fixed10_v1}"
ORACLE_TAG="${BOXFUSION_SGCDET_ORACLE_TAG:-g0_sgcdet_candidate_oracle_fixed10_v1}"
CHECKPOINT="${BOXFUSION_COMBO_SPARSE_CHECKPOINT:-$ROOT/models/scannet_sgcdet_sparse_refiner.pt}"
EXPECTED_SHA="beda774fc3b8f384b408a14388d6b115704e5039b7a110a187760ac9cfd6d182"

if [[ ! "$GPU" =~ ^[0-9]+$ ]]; then
    echo "GPU must be one non-negative integer index" >&2
    exit 2
fi
for dependency in "$PYTHON" "$SCENE_LIST" "$CHECKPOINT"; do
    if [[ ! -f "$dependency" ]]; then
        echo "Missing candidate-oracle dependency: $dependency" >&2
        exit 1
    fi
done

list_sha="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
list_scope="$(basename "$SCENE_LIST" .txt)-${list_sha:0:12}"
prediction_root="$ROOT/results/b6_g0_sgcdet/$ACTIVE_TAG/$list_scope"
diagnostics_root="$ROOT/diagnostics/b6_g0_sgcdet/$ACTIVE_TAG/$list_scope/online"
active_log="$ROOT/logs/b6_g0_sgcdet/$ACTIVE_TAG/$list_scope/eval_stdout.log"
report_root="$ROOT/reports/b6_g0_sgcdet_candidate_oracle/$ORACLE_TAG/$list_scope"
counterfactual_root="$ROOT/results/b6_g0_sgcdet_candidate_oracle/$ORACLE_TAG/$list_scope"
fast_report="$report_root/candidate_oracle.json"
official_report="$report_root/official_evaluation.json"
eval_root="$ROOT/evaluation/b6_g0_sgcdet_candidate_oracle/$ORACLE_TAG/$list_scope"

for dependency in "$prediction_root" "$diagnostics_root"; do
    if [[ ! -d "$dependency" ]]; then
        echo "Missing candidate-oracle input directory: $dependency" >&2
        exit 1
    fi
done
if [[ ! -f "$active_log" ]]; then
    echo "Missing active evaluation log: $active_log" >&2
    exit 1
fi
for destination in "$report_root" "$counterfactual_root" "$eval_root"; do
    if [[ -e "$destination" ]]; then
        echo "Refusing to overwrite candidate-oracle artifact: $destination" >&2
        echo "Set BOXFUSION_SGCDET_ORACLE_TAG to a fresh tag." >&2
        exit 1
    fi
done

mkdir -p "$report_root" "$eval_root"
echo "Reconstructing every valid sparse candidate from the frozen checkpoint"
"$PYTHON" "$ROOT/tools/evaluate_sgcdet_candidate_oracle.py" \
    --prediction-root "$prediction_root" \
    --diagnostics-root "$diagnostics_root" \
    --gt-root "$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data" \
    --scan-root /extra/ZhaoX/scannet_data/scans \
    --scene-list "$SCENE_LIST" \
    --checkpoint "$CHECKPOINT" \
    --expected-checkpoint-sha256 "$EXPECTED_SHA" \
    --counterfactual-root "$counterfactual_root" \
    --output "$fast_report" \
    > "$report_root/build_stdout.json"

methods=(
    identity
    active_replay
    all_valid_candidates
    rowwise_best_iou_oracle
    forward_ap_oracle_0.15
    forward_ap_oracle_0.25
    forward_ap_oracle_0.50
)
for method in "${methods[@]}"; do
    method_eval="$eval_root/$method"
    method_log="$report_root/${method}_official_eval.log"
    mkdir -p "$method_eval" "$report_root/mplconfig"
    echo "Official ScanNet evaluation: $method"
    CUDA_VISIBLE_DEVICES="$GPU" \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR="$report_root/mplconfig" \
    LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
    "$PYTHON" "$ROOT/evaluation/eval_scannet.py" \
        --dataset scannet \
        --data_path /extra/ZhaoX/scannet_data/scans \
        --gt_root "$LIVE_ROOT/evaluation/data_util/scannet_train_detection_data" \
        --dump_dir "$method_eval" \
        --num_point 40000 \
        --cluster_sampling seed_fps \
        --use_3d_nms \
        --use_cls_nms \
        --per_class_proposal \
        --num_workers 0 \
        --gpu 0 \
        --seed 0 \
        --scene_list "$SCENE_LIST" \
        --pred_root "$counterfactual_root/$method" \
        > "$method_log" 2>&1
done

"$PYTHON" - \
    "$fast_report" "$active_log" "$official_report" "$report_root" \
    "${methods[@]}" <<'PY'
import json
import re
import sys
from pathlib import Path

fast_path, active_log, output_path, report_root, *methods = sys.argv[1:]

def metrics(path):
    text = Path(path).read_text(encoding="utf-8")
    maps = [float(value) for value in re.findall(r"eval mAP:\s*([0-9.]+)", text)]
    precision = [float(value) for value in re.findall(r"eval APrec:\s*([0-9.]+)", text)]
    recall = [float(value) for value in re.findall(r"eval ARecall:\s*([0-9.]+)", text)]
    if len(maps) != 3 or len(precision) != 3 or len(recall) != 3:
        raise SystemExit(f"invalid evaluation log {path}")
    return {"mAP": maps, "APrec": precision, "ARecall": recall}

fast = json.loads(Path(fast_path).read_text(encoding="utf-8"))
official = {
    method: metrics(Path(report_root) / f"{method}_official_eval.log")
    for method in methods
}
active_source = metrics(active_log)
if official["active_replay"] != active_source:
    raise SystemExit(
        "active replay does not reproduce the source active evaluation: "
        f"{official['active_replay']} != {active_source}"
    )

mapping = {
    "identity": "identity",
    "active_replay": "active",
    "all_valid_candidates": "all_valid_candidates",
    "rowwise_best_iou_oracle": "rowwise_best_iou_oracle",
    "forward_ap_oracle_0.15": "forward_ap_oracle_0.15",
    "forward_ap_oracle_0.25": "forward_ap_oracle_0.25",
    "forward_ap_oracle_0.50": "forward_ap_oracle_0.50",
}
thresholds = ("0.15", "0.25", "0.50")
for export_name, report_name in mapping.items():
    expected = [
        fast["methods"][report_name]["thresholds"][threshold]["ap"]
        for threshold in thresholds
    ]
    observed = official[export_name]["mAP"]
    if any(abs(left - right) > 1.5e-6 for left, right in zip(expected, observed)):
        raise SystemExit(
            f"fast/official AP mismatch for {export_name}: "
            f"{expected} != {observed}"
        )

baseline = official["identity"]["mAP"]
delta = {
    method: [
        100.0 * (value - base)
        for value, base in zip(values["mAP"], baseline)
    ]
    for method, values in official.items()
}
report = {
    "schema": "boxfusion.sgcdet_candidate_oracle_official_eval.v1",
    "thresholds": [0.15, 0.25, 0.50],
    "warning": "GT-only oracle diagnostic; not a deployable model result.",
    "source_active_evaluation": active_source,
    "official": official,
    "delta_ap_percentage_points_from_identity": delta,
    "candidate_summary": fast["candidate_diagnostics"],
    "checkpoint": fast["checkpoint"],
}
Path(output_path).write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, indent=2, sort_keys=True))
PY

echo "Candidate-oracle report: $official_report"
