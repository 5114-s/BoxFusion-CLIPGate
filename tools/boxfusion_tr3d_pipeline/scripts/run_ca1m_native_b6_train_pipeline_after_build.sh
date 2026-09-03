#!/usr/bin/env bash
set -euo pipefail

# Continue the CA-1M-native B6 train-only route after the exact-100 scene
# builder finishes.  This wrapper never evaluates CA-1M validation data and
# refuses to start GPU collection when another compute process is present.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
BUILD_SESSION="${BOXFUSION_CA1M_NATIVE_B6_BUILD_SESSION:-ca1m_native_b6_train100_build_queue}"
BUILD_LOG="${BOXFUSION_CA1M_NATIVE_B6_BUILD_LOG:-/extra/ZhaoX/ca1m_recovery_logs/ca1m_native_b6_train100_build_v1.log}"
BUILD_REPORT="${BOXFUSION_CA1M_NATIVE_B6_BUILD_REPORT:-$ROOT/reports/ca1m_native_b6_train100_v1/exact100_completion.json}"
GPU_SPEC="${BOXFUSION_CA1M_NATIVE_B6_GPUS:-0,1}"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"

die() { echo "$*" >&2; exit 2; }

[[ -x "$PYTHON" ]] || die "Missing Python: $PYTHON"
[[ -f "$ROOT/scripts/collect_ca1m_native_b6_train100.sh" ]] \
    || die "Missing collection runner"
[[ -f "$ROOT/scripts/train_ca1m_native_b6_quality.sh" ]] \
    || die "Missing native-B6 training runner"

echo "[$(date '+%F %T')] Waiting for exact-100 CA-1M train-scene build: $BUILD_SESSION"
while tmux has-session -t "$BUILD_SESSION" 2>/dev/null; do
    sleep 30
done

[[ -f "$BUILD_LOG" && ! -L "$BUILD_LOG" ]] || die "Missing regular build log: $BUILD_LOG"
grep -q '^EXIT=0$' "$BUILD_LOG" || die "Train-scene builder did not finish with EXIT=0"
[[ -f "$BUILD_REPORT" && ! -L "$BUILD_REPORT" ]] \
    || die "Missing exact-100 completion report: $BUILD_REPORT"

"$PYTHON" -c '
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
assert d.get("schema") == "boxfusion.ca1m_native_b6_train100_completion.v1"
assert d.get("ok") is True
assert d.get("train_only") is True
assert d.get("validation_ground_truth_access") is False
assert d.get("counts", {}).get("exact_scenes") == 100
assert len(d.get("scenes", [])) == 100
' "$BUILD_REPORT"

compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
    | sed -e '/^[[:space:]]*$/d' || true)"
[[ -z "$compute_pids" ]] || die "GPU compute processes appeared; refusing automatic collection: $compute_pids"

echo "[$(date '+%F %T')] Exact-100 build passed; starting GT-free dual-GPU collection"
bash "$ROOT/scripts/collect_ca1m_native_b6_train100.sh" "$GPU_SPEC"

echo "[$(date '+%F %T')] Collection passed; joining train-only native-B6 dataset"
bash "$ROOT/scripts/train_ca1m_native_b6_quality.sh" --build-dataset
bash "$ROOT/scripts/train_ca1m_native_b6_quality.sh" --preflight

echo "[$(date '+%F %T')] Starting train-only OOF/dev fitting and gate"
set +e
bash "$ROOT/scripts/train_ca1m_native_b6_quality.sh" --train
train_status=$?
set -e
if [[ "$train_status" == "0" ]]; then
    echo "[$(date '+%F %T')] Native-B6 train gate PASS; validation was not run"
elif [[ "$train_status" == "3" ]]; then
    echo "[$(date '+%F %T')] Native-B6 train gate FAIL; active validation remains unauthorized"
else
    echo "[$(date '+%F %T')] Native-B6 training failed unexpectedly: $train_status" >&2
    exit "$train_status"
fi
echo "PIPELINE_EXIT=0"
echo "TRAIN_GATE_EXIT=$train_status"
