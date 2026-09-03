#!/usr/bin/env bash
set -euo pipefail

# Strict offline audit for a completed observer-only P1G run.
#
# All numerical gates are environment-controlled and printed before use.
# A STOP decision is an experimental result and exits zero by default.  Set
# BOXFUSION_P1G_REQUIRE_GO=1 for CI to propagate the reporter's exit code 3.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VAL_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SCOPE="${BOXFUSION_P1G_SCOPE:-train_smoke2}"
SCENE_ROLE=train_only

if [[ "${BOXFUSION_P1G_FULL100:-0}" != "0" ]]; then
    echo "P1G full100 audit is forbidden by this train-only protocol." >&2
    exit 2
fi

case "$SCOPE" in
    train_smoke2)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1_smoke2.txt"
        DEFAULT_MIN_TP50=1
        ;;
    train_fit60)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_fit60.txt"
        DEFAULT_MIN_TP50=2
        ;;
    train_cal20)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_cal20.txt"
        DEFAULT_MIN_TP50=2
        ;;
    train_audit20)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_audit20.txt"
        DEFAULT_MIN_TP50=2
        ;;
    train_fresh50)
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_p1g_audit50_fresh_v1.txt"
        DEFAULT_MIN_TP50=5
        ;;
    fixed_val10)
        if [[ "${BOXFUSION_P1G_ALLOW_TOUCHED_VAL10:-0}" != "1" ]]; then
            echo "fixed_val10 audit requires BOXFUSION_P1G_ALLOW_TOUCHED_VAL10=1 after a frozen train-only GO." >&2
            exit 2
        fi
        SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt"
        SCENE_ROLE=touched_validation
        DEFAULT_MIN_TP50=2
        ;;
    custom)
        if [[ -z "${BOXFUSION_P1G_SCENE_LIST:-}" ]]; then
            echo "BOXFUSION_P1G_SCOPE=custom requires BOXFUSION_P1G_SCENE_LIST" >&2
            exit 2
        fi
        SCENE_LIST="$BOXFUSION_P1G_SCENE_LIST"
        SCENE_ROLE="${BOXFUSION_P1G_SCENE_ROLE:-train_only}"
        DEFAULT_MIN_TP50="${BOXFUSION_P1G_DEFAULT_MIN_NOVEL_TP50:-2}"
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
        echo "P1G scope '$SCOPE' is forbidden; no automatic full100 audit exists." >&2
        exit 2
        ;;
    *)
        echo "Unsupported BOXFUSION_P1G_SCOPE: $SCOPE" >&2
        exit 2
        ;;
esac

RUN_TAG="${BOXFUSION_P1G_RUN_TAG:-p1g_${SCOPE}_b6_p1s_frozen_v1}"
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "BOXFUSION_P1G_RUN_TAG contains unsafe path characters" >&2
    exit 2
fi

PRED_ROOT="${BOXFUSION_P1G_PRED_ROOT:-$ROOT/results/p1g_ablation/$RUN_TAG}"
DIAGNOSTICS_ROOT="${BOXFUSION_P1G_DIAGNOSTICS_ROOT:-$ROOT/diagnostics/p1g_ablation/$RUN_TAG}"
REPORT_ROOT="${BOXFUSION_P1G_REPORT_ROOT:-$ROOT/reports/p1g_ablation/$RUN_TAG}"
GT_ROOT="${BOXFUSION_P1G_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_P1G_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
ENV_ROOT="${BOXFUSION_ENV_ROOT:-/home/admin1/miniconda3/envs/boxfusion-online}"
PYTHON="$ENV_ROOT/bin/python"

THRESHOLDS_TEXT="${BOXFUSION_P1G_THRESHOLDS:-0.15 0.25 0.50}"
read -r -a THRESHOLDS <<< "$THRESHOLDS_TEXT"
MIN_NOVEL_TP50="${BOXFUSION_P1G_MIN_NOVEL_TP50:-$DEFAULT_MIN_TP50}"
MIN_PARENT_TP25_DELTA="${BOXFUSION_P1G_MIN_PARENT_TP25_DELTA:-0}"
MAX_P1G_SECONDS="${BOXFUSION_P1G_MAX_SECONDS_PER_SCENE:-0.18}"
MAX_TOTAL_SECONDS="${BOXFUSION_P1G_MAX_TOTAL_SECONDS_PER_SCENE:-0.80}"
MAX_CANDIDATES="${BOXFUSION_P1G_MAX_CANDIDATES_PER_SCENE:-256}"
REQUIRE_GO="${BOXFUSION_P1G_REQUIRE_GO:-0}"

case "$REQUIRE_GO" in
    0|1) ;;
    *)
        echo "BOXFUSION_P1G_REQUIRE_GO must be 0 or 1" >&2
        exit 2
        ;;
esac

for path in "$PYTHON" "$SCENE_LIST" "$VAL_LIST" "$ROOT/tools/report_p1g_geometry.py"; do
    if [[ ! -f "$path" ]]; then
        echo "Missing P1G audit input: $path" >&2
        exit 1
    fi
done
for directory in "$PRED_ROOT" "$DIAGNOSTICS_ROOT" "$GT_ROOT" "$SCANS_ROOT"; do
    if [[ ! -d "$directory" ]]; then
        echo "Missing P1G audit directory: $directory" >&2
        exit 1
    fi
done

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
overlap = sorted(set(scenes) & set(read(val_path)))
if role == "train_only" and overlap:
    raise SystemExit(
        "train-only P1G list overlaps ScanNet validation: "
        + ", ".join(overlap[:8])
    )
print(f"P1G audit scene-list preflight OK: {len(scenes)} scenes, role={role}")
' "$SCENE_LIST" "$VAL_LIST" "$SCENE_ROLE"

mkdir -p "$REPORT_ROOT"
REPORT="$REPORT_ROOT/geometry.json"

echo "P1G audit: scope=$SCOPE, role=$SCENE_ROLE, tag=$RUN_TAG"
echo "  predictions: $PRED_ROOT"
echo "  diagnostics: $DIAGNOSTICS_ROOT"
echo "  ground truth: $GT_ROOT"
echo "  scans: $SCANS_ROOT"
echo "  thresholds: ${THRESHOLDS[*]}"
echo "  gates: novel TP50 >= $MIN_NOVEL_TP50; parent TP25 delta >= $MIN_PARENT_TP25_DELTA"
echo "  budgets: P1G <= ${MAX_P1G_SECONDS}s/scene; P1S+P1G <= ${MAX_TOTAL_SECONDS}s/scene; candidates <= $MAX_CANDIDATES/scene"

set +e
"$PYTHON" "$ROOT/tools/report_p1g_geometry.py" \
    --scene-list "$SCENE_LIST" \
    --prediction-root "$PRED_ROOT" \
    --diagnostics-root "$DIAGNOSTICS_ROOT" \
    --gt-root "$GT_ROOT" \
    --scans-root "$SCANS_ROOT" \
    --thresholds "${THRESHOLDS[@]}" \
    --minimum-novel-tp50 "$MIN_NOVEL_TP50" \
    --minimum-parent-tp25-delta "$MIN_PARENT_TP25_DELTA" \
    --maximum-p1g-runtime-seconds-per-scene "$MAX_P1G_SECONDS" \
    --maximum-total-runtime-seconds-per-scene "$MAX_TOTAL_SECONDS" \
    --maximum-candidates-per-scene "$MAX_CANDIDATES" \
    --output "$REPORT" \
    > "$REPORT_ROOT/geometry.stdout.json"
REPORT_STATUS=$?
set -e

if [[ "$REPORT_STATUS" -ne 0 && "$REPORT_STATUS" -ne 3 ]]; then
    echo "P1G reporter failed with status $REPORT_STATUS" >&2
    exit "$REPORT_STATUS"
fi

"$PYTHON" -c '
import json
import sys

path, scope = sys.argv[1:3]
report = json.load(open(path, encoding="utf-8"))
gate = report["go_no_go"]
diagnosis = report["diagnosis"]
if not gate["passes"]:
    protocol = "STOP_P1G"
elif scope in {"train_smoke2", "train_fit60", "train_cal20"}:
    protocol = "EXPLORATORY_PASS_ONLY"
elif scope == "train_audit20":
    protocol = "GO_FRESH50_AUDIT"
elif scope == "train_fresh50":
    protocol = "GO_ONE_SHOT_VAL10_OBSERVER"
elif scope == "fixed_val10":
    protocol = "GO_DESIGN_P1Q_OBSERVER_ONLY"
else:
    protocol = "PASS_REQUIRES_REGISTERED_SCOPE_REVIEW"
print("P1G reporter decision:", gate["decision"])
print("P1G protocol decision:", protocol)
print("P1G diagnosis:", diagnosis["classification"])
print("P1G interpretation:", diagnosis["interpretation"])
' "$REPORT" "$SCOPE"

echo "P1G audit report: $REPORT"
if [[ "$REQUIRE_GO" == "1" ]]; then
    exit "$REPORT_STATUS"
fi
exit 0
