#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/ZhaoX/BoxFusion
ENV_ROOT=/home/admin1/miniconda3/envs/boxfusion-online
PYTHON="$ENV_ROOT/bin/python"
SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
MODEL="$ROOT/models/cutr_rgbd.pth"
CLIP_MODEL="$ROOT/models/open_clip_pytorch_model.bin"
CLASS_TXT="$ROOT/data/panoptic_categories_nomerge.txt"
WRAPPER="$ROOT/tools/run_plr_online_ablation_variant.py"
REFERENCE_MANIFEST="$ROOT/results/scannet_strict_causal_nochild_official100_v2_factorial/manifest.json"
CANONICAL_NATIVE="$ROOT/results/scannet_strict_causal_nochild_official100_v2_native"
CANONICAL_FULL="$ROOT/results/scannet_strict_causal_nochild_official100_v2"
CANONICAL_DIAG="$ROOT/diagnostics/strict_causal_nochild_official100_v2"

RUN_ROOT="$ROOT/results/scannet_plr_online_key_controls_20260925"
DIAG_ROOT="$ROOT/diagnostics/scannet_plr_online_key_controls_20260925"
LOG_ROOT="$ROOT/logs/scannet_plr_online_key_controls_20260925"
SCENE_LOG_ROOT="$LOG_ROOT/scenes"
OUTPUT_ROOT="$RUN_ROOT/materialized"
EVIDENCE_ROOT="$RUN_ROOT/direct_matched_budget_anchor_evidence"
REPORT_ROOT="$ROOT/reports/plr_online_key_controls_official100_20260925"
MPL_ROOT="$LOG_ROOT/mplconfig"
EXPERIMENT_PREFIX=plr_online_key_controls_20260925

mkdir -p "$SCENE_LOG_ROOT" "$MPL_ROOT" "$REPORT_ROOT"
exec 9>"$LOG_ROOT/run.lock"
flock -n 9 || { echo "PLR online key-control runner is already active" >&2; exit 1; }
exec > >(tee -a "$LOG_ROOT/driver.log") 2>&1

[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5 ]] || {
  echo "Official100 scene-list hash mismatch" >&2
  exit 1
}
mapfile -t SCENES < <(awk 'NF && $1 !~ /^#/ {print $1}' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 100 ]] || { echo "Expected 100 scenes" >&2; exit 1; }
[[ -s "$REFERENCE_MANIFEST" ]] || { echo "Missing reference manifest" >&2; exit 1; }

run_variant() {
  local mode="$1"
  local tag="$2"
  local config="$3"
  local native_root="$RUN_ROOT/${tag}_native"
  local online_root="$RUN_ROOT/${tag}_full"
  local diagnostics_root="$DIAG_ROOT/$tag"
  local variant_log_root="$SCENE_LOG_ROOT/$tag"
  local fingerprint_file="$LOG_ROOT/${tag}.fingerprint"
  mkdir -p "$native_root" "$online_root" "$diagnostics_root" "$variant_log_root"

  local run_fingerprint
  run_fingerprint="$(sha256sum "$config" "$WRAPPER" "$ROOT/demo.py" \
    "$ROOT/boxfusion/online_candidate_map.py" \
    "$ROOT/boxfusion/online_candidate_runtime.py" \
    "$ROOT/boxfusion/box_manager.py" "$ROOT/boxfusion/instances.py" \
    "$MODEL" "$CLIP_MODEL" "$CLASS_TXT" | sha256sum | awk '{print $1}')"
  if [[ -s "$fingerprint_file" ]]; then
    [[ "$(tr -d '\n' < "$fingerprint_file")" == "$run_fingerprint" ]] || {
      echo "Fingerprint differs for $tag" >&2
      exit 1
    }
  else
    printf '%s\n' "$run_fingerprint" > "$fingerprint_file"
  fi

  validate_scene() {
    local scene="$1"
    local log="$variant_log_root/$scene.log"
    [[ -s "$native_root/${scene}_boxes.pkl" ]] || return 1
    [[ -s "$online_root/${scene}_boxes.pkl" ]] || return 1
    [[ -s "$diagnostics_root/${scene}.json" ]] || return 1
    [[ -s "$log" ]] || return 1
    [[ "$(grep -Fc 'Online runtime receipt |' "$log")" -eq 1 ]] || return 1
    ! grep -Eq 'Traceback|RuntimeError|CUDA out of memory' "$log" || return 1
    if [[ "$mode" == "direct_matched_budget" ]]; then
      [[ -s "$EVIDENCE_ROOT/${scene}.npz" ]] || return 1
    fi
    "$PYTHON" - "$diagnostics_root/${scene}.json" "$mode" \
      "$REFERENCE_MANIFEST" "$scene" <<'PY' >/dev/null
import json, sys
row = json.load(open(sys.argv[1], encoding="utf-8"))
mode, manifest_path, scene = sys.argv[2:]
assert row["strictly_causal"] is True
assert row["online_incremental"] is True
assert row["scene_end_inference"] is False
assert row["state"]["m1p"]["use_children"] is False
assert row["state"]["m1p"]["ablation_mode"] == mode
if mode == "direct_matched_budget":
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    expected = int(manifest["per_scene"][scene]["m1p"])
    assert row["state"]["terminal_counts"]["m1p"] == expected
    assert row["state"]["m1p"]["matched_output_quota"] == expected
else:
    assert row["state"]["m1p"]["native_dedup_disabled"] is True
PY
  }

  echo "[$(date '+%F %T')] variant=$tag mode=$mode fingerprint=$run_fingerprint"
  for index in "${!SCENES[@]}"; do
    local scene="${SCENES[$index]}"
    local log="$variant_log_root/$scene.log"
    if validate_scene "$scene"; then
      echo "[$(date '+%F %T')] $tag [$((index + 1))/100] Reusing $scene"
      continue
    fi
    for stale in "$native_root/${scene}_boxes.pkl" "$online_root/${scene}_boxes.pkl" \
      "$diagnostics_root/${scene}.json" "$log"; do
      [[ ! -e "$stale" ]] || { echo "Invalid partial artifact: $stale" >&2; exit 1; }
    done
    echo "[$(date '+%F %T')] $tag [$((index + 1))/100] Running $scene"
    (
      cd "$ROOT"
      CUDA_VISIBLE_DEVICES=0,1 \
      CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      PYTHONHASHSEED=0 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 \
      MPLCONFIGDIR="$MPL_ROOT" \
      LD_LIBRARY_PATH="$ENV_ROOT/lib:$ENV_ROOT/opt/rviz_ogre_vendor/lib:/usr/local/cuda-12.1/lib64" \
      PYTHONPATH="$ROOT:$ROOT/third_party/WeDetect" \
      PLR_ABLATION_MODE="$mode" PLR_REFERENCE_MANIFEST="$REFERENCE_MANIFEST" \
      PLR_EVIDENCE_CACHE_ROOT="$EVIDENCE_ROOT" \
      "$PYTHON" "$WRAPPER" scannet \
        --model-path "$MODEL" --clip_path "$CLIP_MODEL" \
        --class_txt "$CLASS_TXT" --config "$config" \
        --device cuda:0 --seq "$scene"
    ) > "$log" 2>&1
    validate_scene "$scene" || {
      echo "Scene validation failed: $tag/$scene; see $log" >&2
      exit 1
    }
    echo "[$(date '+%F %T')] $tag [$((index + 1))/100] Completed $scene"
  done
}

run_variant direct_matched_budget direct_matched_budget \
  "$ROOT/config/scannet_t05_boxer_strict_causal_nochild_plr_direct_matched_official100.yaml"
run_variant no_native_dedup without_native_dedup \
  "$ROOT/config/scannet_t05_boxer_strict_causal_nochild_plr_no_dedup_official100.yaml"

if [[ -e "$OUTPUT_ROOT/manifest.json" ]]; then
  echo "Materialized manifest already exists; refusing to overwrite" >&2
  exit 1
fi
"$PYTHON" "$ROOT/tools/materialize_plr_online_key_controls.py" \
  --scene-list "$SCENE_LIST" \
  --canonical-native "$CANONICAL_NATIVE" \
  --canonical-full "$CANONICAL_FULL" \
  --canonical-diagnostics "$CANONICAL_DIAG" \
  --direct-native "$RUN_ROOT/direct_matched_budget_native" \
  --direct-full "$RUN_ROOT/direct_matched_budget_full" \
  --direct-diagnostics "$DIAG_ROOT/direct_matched_budget" \
  --no-dedup-native "$RUN_ROOT/without_native_dedup_native" \
  --no-dedup-full "$RUN_ROOT/without_native_dedup_full" \
  --no-dedup-diagnostics "$DIAG_ROOT/without_native_dedup" \
  --output "$OUTPUT_ROOT"

for arm in direct_matched_budget without_native_dedup full_plr; do
  bash "$ROOT/scripts/eval_scannet_official100_real_score.sh" \
    "${EXPERIMENT_PREFIX}_${arm}" "$OUTPUT_ROOT/$arm"
done

"$PYTHON" "$ROOT/tools/summarize_plr_online_key_controls.py" \
  --manifest "$OUTPUT_ROOT/manifest.json" \
  --log-root "$ROOT/logs/scannet_official100_real_score" \
  --experiment-prefix "$EXPERIMENT_PREFIX" \
  --output "$REPORT_ROOT"

CALR_OUTPUT="$RUN_ROOT/materialized_calr"
CALR_PREFIX=candidate_recovery_table3_20260925_calr
if [[ -e "$CALR_OUTPUT/manifest.json" ]]; then
  echo "CALR materialized manifest already exists; refusing to overwrite" >&2
  exit 1
fi
"$PYTHON" "$ROOT/tools/materialize_calr_current_key_controls.py" \
  --scene-list "$SCENE_LIST" \
  --baseline "$ROOT/results/scannet_strict_causal_nochild_official100_v2_factorial/p_m2" \
  --full "$ROOT/results/scannet_strict_causal_nochild_official100_v2_factorial/p_a_m2" \
  --anchor-cache "$EVIDENCE_ROOT" \
  --output "$CALR_OUTPUT"

for arm in lower_threshold strongest_matched_budget full_calr; do
  bash "$ROOT/scripts/eval_scannet_official100_real_score.sh" \
    "${CALR_PREFIX}_${arm}" "$CALR_OUTPUT/$arm"
done

"$PYTHON" "$ROOT/tools/summarize_candidate_recovery_table3.py" \
  --plr-results "$REPORT_ROOT/results.json" \
  --calr-manifest "$CALR_OUTPUT/manifest.json" \
  --log-root "$ROOT/logs/scannet_official100_real_score" \
  --calr-prefix "$CALR_PREFIX" \
  --output "$REPORT_ROOT"

EVIDENCE_REPORT="$ROOT/reports/current_candidate_evidence_nochild_official100_20260925"
if [[ -e "$EVIDENCE_REPORT/results.json" ]]; then
  echo "Current candidate-evidence audit already exists; refusing to overwrite" >&2
  exit 1
fi
"$PYTHON" "$ROOT/tools/audit_current_candidate_evidence.py" \
  --scene-list "$SCENE_LIST" \
  --native "$CANONICAL_NATIVE" \
  --full "$CANONICAL_FULL" \
  --diagnostics "$CANONICAL_DIAG" \
  --evidence-cache "$EVIDENCE_ROOT" \
  --factorial "$ROOT/results/scannet_strict_causal_nochild_official100_v2_factorial" \
  --output "$EVIDENCE_REPORT"
echo "[$(date '+%F %T')] ScanNet six-row recovery table complete: $REPORT_ROOT/REPORT.md"
echo "[$(date '+%F %T')] Candidate evidence audit complete: $EVIDENCE_REPORT/REPORT.md"
