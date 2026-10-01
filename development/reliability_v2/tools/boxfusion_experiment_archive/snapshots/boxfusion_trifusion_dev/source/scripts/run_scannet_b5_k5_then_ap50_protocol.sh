#!/usr/bin/env bash
set -euo pipefail

# End-to-end fixed-10 decision protocol requested for B5:
#   1. collect train-only diagnostics under the exact K=5 runtime contract;
#   2. retrain the paired B5-v2 improvement control;
#   3. compare it with a same-contract identity run on the fixed 10 scenes;
#   4. only when score-locked AP50 does not improve, train and evaluate the
#      AP50-aware objective.
#
# This launcher deliberately refuses to start while any CUDA compute process
# is present. It is therefore safe to prepare now while another experiment is
# running, but must be invoked only after that experiment has finished.
#
# Usage:
#   bash scripts/run_scannet_b5_k5_then_ap50_protocol.sh 0,1

GPU_SPEC="${1:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"
EPSILON="${BOXFUSION_B5_AP50_BRANCH_EPSILON:-0.000001}"

IDENTITY_TAG="${BOXFUSION_B5V3_IDENTITY_RUN_TAG:-b5v3_gatealigned_k5_identity_extent040_ablation10_v2}"
K5_TAG="${BOXFUSION_B5V2_K5_RUN_TAG:-b5v2_k5_gatealigned_refiner_only_extent040_ablation10_v2}"
AP50_TAG="${BOXFUSION_B5V3_RUN_TAG:-b5v3_ap50_gatealigned_refiner_only_extent040_ablation10_v2}"
REPORT_ROOT="${BOXFUSION_B5_PROTOCOL_REPORT_ROOT:-$ROOT/reports/b5_k5_ap50_protocol_v2}"
K5_REPORT="$REPORT_ROOT/k5_control_vs_identity.json"
AP50_REPORT="$REPORT_ROOT/ap50_aware_vs_identity.json"

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing protocol Python environment: $PYTHON" >&2
    exit 1
fi
if ! "$PYTHON" -c \
    "value=float('$EPSILON'); assert value >= 0.0" >/dev/null 2>&1; then
    echo "BOXFUSION_B5_AP50_BRANCH_EPSILON must be non-negative" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for the no-interference GPU guard" >&2
    exit 1
fi
active_cuda_pids="$(
    nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        2>/dev/null | awk 'NF { print $1 }'
)"
if [[ -n "$active_cuda_pids" ]]; then
    echo "Refusing to start while CUDA compute processes are active:" >&2
    echo "$active_cuda_pids" >&2
    echo "Wait for the current experiment to finish, then rerun this command." >&2
    exit 1
fi

mkdir -p "$REPORT_ROOT"

echo "[1/6] Collecting exact K=5 train-only diagnostics"
bash "$ROOT/scripts/collect_scannet_b5v3_k5_train.sh" "$GPU_SPEC"

echo "[2/6] Training the paired K=5 improvement control (CPU only)"
bash "$ROOT/scripts/train_scannet_b5v2_k5_refiner.sh"

echo "[3/6] Running the exact-contract fixed-10 identity"
bash "$ROOT/scripts/run_scannet_b5v3_gatealigned_identity.sh" "$GPU_SPEC"

echo "[4/6] Running the K=5 improvement control on the same fixed 10 scenes"
bash "$ROOT/scripts/run_scannet_b5v2_k5_refiner.sh" "$GPU_SPEC"

echo "[5/6] Producing the paired, identity-score-locked K=5 report"
BOXFUSION_B5_REPORT_JSON="$K5_REPORT" \
    bash "$ROOT/scripts/report_scannet_b5_ap50.sh" \
    "$ROOT/results/$IDENTITY_TAG" \
    "$ROOT/results/$K5_TAG" \
    "$ROOT/diagnostics/$K5_TAG" >/dev/null

if "$PYTHON" -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
identity = float(report["metrics"]["identity"]["0.50"]["ap"])
candidate = float(
    report["metrics"]["candidate_identity_scores"]["0.50"]["ap"]
)
epsilon = float(sys.argv[2])
print(
    "K=5 fixed-10 score-locked AP50: "
    f"identity={identity:.6f}, candidate={candidate:.6f}, "
    f"delta={candidate - identity:+.6f}"
)
raise SystemExit(0 if candidate > identity + epsilon else 1)
' "$K5_REPORT" "$EPSILON"; then
    echo "K=5 retraining improved AP50; AP50-aware fallback was not run."
    echo "Paired report: $K5_REPORT"
    exit 0
fi

echo "K=5 AP50 did not improve; activating the AP50-aware fallback."
echo "[6/6] Training and evaluating the AP50-aware objective"
bash "$ROOT/scripts/train_scannet_b5v3_ap50_refiner.sh"
bash "$ROOT/scripts/run_scannet_b5v3_ap50_refiner.sh" "$GPU_SPEC"
BOXFUSION_B5_REPORT_JSON="$AP50_REPORT" \
    bash "$ROOT/scripts/report_scannet_b5_ap50.sh" \
    "$ROOT/results/$IDENTITY_TAG" \
    "$ROOT/results/$AP50_TAG" \
    "$ROOT/diagnostics/$AP50_TAG" >/dev/null

"$PYTHON" -c '
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
identity = float(report["metrics"]["identity"]["0.50"]["ap"])
candidate = float(
    report["metrics"]["candidate_identity_scores"]["0.50"]["ap"]
)
print(
    "AP50-aware fixed-10 score-locked AP50: "
    f"identity={identity:.6f}, candidate={candidate:.6f}, "
    f"delta={candidate - identity:+.6f}"
)
' "$AP50_REPORT"

echo "K=5 report: $K5_REPORT"
echo "AP50-aware report: $AP50_REPORT"
