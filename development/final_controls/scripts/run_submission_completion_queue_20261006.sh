#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
DEV="$ROOT/development/final_controls"
PYTHON=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
QUEUE_ROOT="$ROOT/reports/submission_completion_queue_20261006"
EFFICIENCY="$ROOT/reports/original_boxfusion_vs_final_recar_efficiency100_prewarmed_20261006"
CA_RUN="$DEV/experiments/ca1m_novis107_20261006"
TEMPORAL_RUN="$DEV/experiments/mvsr_novis_temporal_20261006"
MULTISCAN_RUN="$DEV/experiments/multiscan50_novis_20261006"
SCANNET_FACTORIAL="$DEV/experiments/mvsr_design_20261005/factorial_no_visibility"
CA_FACTORIAL="$CA_RUN/factorial_no_visibility"

mkdir -p "$QUEUE_ROOT"
exec 9>"$QUEUE_ROOT/run.lock"
flock -n 9 || { echo "submission completion queue already active" >&2; exit 1; }
exec > >(tee -a "$QUEUE_ROOT/queue.log") 2>&1

stage() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" | tee "$QUEUE_ROOT/status.tsv"
}

run_with_retries() {
  local label="$1" attempt status
  shift
  for attempt in 1 2 3; do
    stage "${label}_attempt_${attempt}"
    if "$@"; then
      stage "${label}_complete"
      return 0
    else
      status=$?
    fi
    stage "${label}_retry_after_exit_${status}"
    sleep 60
  done
  stage "${label}_failed_after_3_attempts"
  return 1
}

stage "waiting_for_prewarmed_efficiency"
while [[ ! -s "$EFFICIENCY/summary.json" || ! -s "$EFFICIENCY/REPORT.md" ]]; do
  if ! pgrep -f '[r]un_official100_prewarm.sh' >/dev/null \
      && ! pgrep -f '[b]enchmark_scene_prewarm.py' >/dev/null \
      && ! pgrep -f '[s]ummarize_prewarm.py' >/dev/null; then
    run_with_retries "resuming_prewarmed_efficiency" \
      bash "$ROOT/development/final_efficiency_original_vs_recar_20261005/run_official100_prewarm.sh"
  fi
  sleep 60
done

if [[ ! -s "$CA_RUN/report_no_visibility/results.json" ]]; then
  run_with_retries "running_ca1m107_final_factorial" \
    env RUN_TAG=ca1m_novis107_20261006 \
    bash "$DEV/scripts/run_ca1m_novis107.sh"
else
  stage "reusing_ca1m107_final_factorial"
fi

if [[ ! -s "$TEMPORAL_RUN/report/results.json" ]]; then
  run_with_retries "running_scannet100_final_mvsr_temporal" \
    env RUN_TAG=mvsr_novis_temporal_20261006 \
    bash "$DEV/scripts/run_mvsr_novis_temporal_official100.sh"
else
  stage "reusing_scannet100_final_mvsr_temporal"
fi

if [[ ! -s "$MULTISCAN_RUN/report/metrics.json" \
   || ! -s "$MULTISCAN_RUN/budget_report/results.json" ]]; then
  run_with_retries "running_multiscan50_final_no_visibility" \
    env RUN_TAG=multiscan50_novis_20261006 \
    bash "$DEV/scripts/run_multiscan50_novis.sh"
else
  stage "reusing_multiscan50_final_no_visibility"
fi

if [[ ! -s "$ROOT/reports/final_bootstrap_scannet_novis_20261006/results.json" ]]; then
  stage "running_scannet_final_bootstrap"
  "$PYTHON" "$ROOT/tools/bootstrap_final_recar_pairs.py" \
    --dataset scannet \
    --scene-list "$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt" \
    --base-root "$SCANNET_FACTORIAL/base" \
    --treatment "full=$SCANNET_FACTORIAL/full" \
    --treatment "mvsr=$SCANNET_FACTORIAL/mvsr" \
    --replicates 2000 --seed 20261006 \
    --output "$ROOT/reports/final_bootstrap_scannet_novis_20261006"
fi

if [[ ! -s "$ROOT/reports/final_bootstrap_ca1m_novis_20261006/results.json" ]]; then
  stage "running_ca1m_final_bootstrap"
  "$PYTHON" "$ROOT/tools/bootstrap_final_recar_pairs.py" \
    --dataset ca1m \
    --scene-list "$ROOT/tools/boxfusion_tr3d_pipeline/evaluation/data_util/meta_data/ca1m_val_full107.txt" \
    --ca1m-data-root /extra/ZhaoX/boxfusion_ca1m \
    --base-root "$CA_FACTORIAL/base" \
    --treatment "full=$CA_FACTORIAL/p_a_m2" \
    --replicates 2000 --seed 20261006 \
    --output "$ROOT/reports/final_bootstrap_ca1m_novis_20261006"
fi

stage "refreshing_final_mvsr_mechanism_statistics"
MPLCONFIGDIR="$QUEUE_ROOT/mplconfig" \
  "$PYTHON" "$ROOT/tools/plot_mvsr_support_ranking.py"

stage "writing_canonical_submission_ledger"
"$PYTHON" "$ROOT/tools/summarize_submission_completion.py"
stage "complete"
touch "$QUEUE_ROOT/DONE"
