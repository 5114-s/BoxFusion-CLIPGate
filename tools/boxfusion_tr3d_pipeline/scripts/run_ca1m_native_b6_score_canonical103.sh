#!/usr/bin/env bash
set -euo pipefail

# Score-only canonical-103 skeleton.  It consumes an already completed
# same-run native-B6 observer collection and never invokes eval_ca1m.py.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
MODE="${1:---preflight}"
case "$MODE" in
    --preflight) TOOL_MODE="preflight" ;;
    --observer) TOOL_MODE="observer" ;;
    --active) TOOL_MODE="active" ;;
    *) echo "Usage: $0 [--preflight|--observer|--active]" >&2; exit 2 ;;
esac
[[ "$#" -le 1 ]] || { echo "Usage: $0 [--preflight|--observer|--active]" >&2; exit 2; }

SCENE_LIST="${BOXFUSION_CA1M_NATIVE_B6_SCORE_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_canonical103.txt}"
SOURCE_TAG="${BOXFUSION_CA1M_NATIVE_B6_SCORE_SOURCE_TAG:-ca1m_c3_native_b6_observer_canonical103_v1}"
ANCHOR_ROOT="${BOXFUSION_CA1M_NATIVE_B6_SCORE_ANCHOR_ROOT:-$ROOT/results/ca1m_port/${SOURCE_TAG}_same_run_anchor}"
OBSERVER_ROOT="${BOXFUSION_CA1M_NATIVE_B6_SCORE_OBSERVER_ROOT:-$ROOT/results/ca1m_port/$SOURCE_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_CA1M_NATIVE_B6_SCORE_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/ca1m_port/$SOURCE_TAG/native_b6}"
CHECKPOINT="${BOXFUSION_CA1M_NATIVE_B6_SCORE_CHECKPOINT:-$ROOT/models/ca1m_native_b6_iou_mlp_v1.npz}"
CHECKPOINT_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_SCORE_CHECKPOINT_MANIFEST:-$ROOT/models/ca1m_native_b6_iou_mlp_v1.manifest.json}"
TAG="${BOXFUSION_CA1M_NATIVE_B6_SCORE_TAG:-ca1m_native_b6_score_canonical103_v1}"
ACTIVE_ROOT="${BOXFUSION_CA1M_NATIVE_B6_SCORE_ACTIVE_ROOT:-$ROOT/results/ca1m_native_b6_score/$TAG}"
REPORT="${BOXFUSION_CA1M_NATIVE_B6_SCORE_REPORT:-$ROOT/reports/ca1m_native_b6_score/$TAG/${TOOL_MODE}.json}"
EXPECTED_LIST_SHA="c3efbe544c7403acc4183d7e4a799dad2bb40f60cbdba38830863f8712f4648f"

die() { echo "$*" >&2; exit 2; }
[[ -x "$PYTHON" ]] || die "Missing Python: $PYTHON"
for path in "$SCENE_LIST" "$CHECKPOINT" "$CHECKPOINT_MANIFEST" \
    "$ROOT/tools/apply_ca1m_native_b6_counterfactual.py"; do
    [[ -f "$path" && ! -L "$path" ]] || die "Missing regular input: $path"
done
for path in "$ANCHOR_ROOT" "$OBSERVER_ROOT" "$DIAGNOSTICS_ROOT"; do
    [[ -d "$path" && ! -L "$path" ]] || die "Missing artifact root: $path"
done
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == "$EXPECTED_LIST_SHA" ]] \
    || die "canonical103 scene-list SHA256 drifted"
[[ "$(sed -e 's/[[:space:]]*$//' -e '/^$/d' "$SCENE_LIST" | wc -l)" == "103" ]] \
    || die "canonical103 runner requires exactly 103 scenes"

arguments=(
    "$PYTHON" "$ROOT/tools/apply_ca1m_native_b6_counterfactual.py"
    --mode "$TOOL_MODE"
    --scene-list "$SCENE_LIST"
    --anchor-root "$ANCHOR_ROOT"
    --observer-root "$OBSERVER_ROOT"
    --diagnostics-root "$DIAGNOSTICS_ROOT"
    --checkpoint "$CHECKPOINT"
    --checkpoint-manifest "$CHECKPOINT_MANIFEST"
)
if [[ "$TOOL_MODE" == "observer" ]]; then
    arguments+=(--output "$REPORT")
elif [[ "$TOOL_MODE" == "active" ]]; then
    arguments+=(--prediction-output-root "$ACTIVE_ROOT" --output "$REPORT")
fi

echo "CA-1M native-B6 canonical103 score-only ${TOOL_MODE}"
echo "  source same-run anchor: $ANCHOR_ROOT"
echo "  14-D observer diagnostics: $DIAGNOSTICS_ROOT"
echo "  evaluation/GT access: disabled"
"${arguments[@]}"
echo "Canonical103 score stage complete; standard evaluation was not invoked."
