#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

# This is the reproducible subset for which the authors published canonical
# after_filter_boxes.npy files.  The four missing-GT scenes remain outside the
# evaluation view and are never silently ignored by the evaluator.
export BOXFUSION_CA1M_RUN_TAG="c0_score04_real_score_canonical103_v1"
export BOXFUSION_CA1M_PROTOCOL="canonical103"
export BOXFUSION_CA1M_EXPECTED_SCENES="103"
export BOXFUSION_CA1M_ALLOW_UNLISTED_SCENES="1"
export BOXFUSION_CA1M_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/ca1m_val_canonical103.txt"
export BOXFUSION_CA1M_C0_CONFIG="$ROOT/config/ca1m_c0_score04_real_score_canonical103.yaml"
export BOXFUSION_CA1M_EXPECTED_CONFIG_SHA256="aa16c70410d731fcc1d8ee985f168f814f3bfd183fb7d5b99e6b814c60d9b368"
export BOXFUSION_CA1M_EXPECTED_SCENE_LIST_SHA256="c3efbe544c7403acc4183d7e4a799dad2bb40f60cbdba38830863f8712f4648f"
export BOXFUSION_CA1M_EXCLUDED_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/ca1m_missing_canonical_gt4.txt"
export BOXFUSION_CA1M_EXPECTED_EXCLUDED_SCENE_LIST_SHA256="582ec52e296fa907a79eb01f5778b0adea368b3c7ca61e3a972aca42f32d401b"
export BOXFUSION_CA1M_PRED_ROOT="$ROOT/results/ca1m_repro/c0_score04_real_score_canonical103_v1"
export BOXFUSION_CA1M_LOG_ROOT="$ROOT/logs/ca1m_repro/c0_score04_real_score_canonical103_v1"
export BOXFUSION_CA1M_REPORT_ROOT="$ROOT/reports/ca1m_repro/c0_score04_real_score_canonical103_v1"
export BOXFUSION_CA1M_EVAL_VIEW="$ROOT/data/ca1m_eval_canonical103_v1"
export BOXFUSION_RUNTIME_TMP_ROOT="/extra/ZhaoX/boxfusion_runtime_tmp/c0_score04_real_score_canonical103_v1"

if (( $# == 0 )); then
    set -- 0,1
fi
exec bash "$ROOT/scripts/run_ca1m_c0_score04_real_score_full107.sh" "$@"
