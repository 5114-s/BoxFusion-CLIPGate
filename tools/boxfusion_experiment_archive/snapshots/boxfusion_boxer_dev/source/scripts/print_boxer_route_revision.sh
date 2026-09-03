#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"

{
  sha256sum \
    "$CODE_ROOT/demo.py" \
    "$CODE_ROOT/boxfusion/boxer_lifter.py" \
    "$CODE_ROOT/boxfusion/proposal_cache.py" \
    "$CODE_ROOT/boxfusion/cubify_transformer.py" \
    "$CODE_ROOT/boxfusion/instances.py" \
    "$CODE_ROOT/boxfusion/boxes.py" \
    "$CODE_ROOT/boxfusion/box_manager.py" \
    "$CODE_ROOT/boxfusion/box_fusion.py" \
    "$CODE_ROOT/boxfusion/capture_stream.py" \
    "$CODE_ROOT/boxfusion/preprocessor.py" \
    "$CODE_ROOT/config/scannet_cutr_paired_scorefix.yaml" \
    "$CODE_ROOT/config/scannet_cutr_replay_scorefix.yaml" \
    "$CODE_ROOT/config/scannet_boxer_observer_scorefix.yaml" \
    "$CODE_ROOT/config/scannet_boxer_active_scorefix.yaml" \
    "$CODE_ROOT/config/scannet_boxer_pre_observer_scorefix.yaml" \
    "$CODE_ROOT/config/scannet_boxer_pre_active_scorefix.yaml" \
    "$CODE_ROOT/evaluation/eval_scannet.py" \
    "$CODE_ROOT/scripts/run_scannet_boxer_profile.sh" \
    "$CODE_ROOT/scripts/run_scannet_boxer_smoke.sh" \
    "$CODE_ROOT/scripts/run_scannet_boxer_fixed10.sh" \
    "$CODE_ROOT/scripts/run_scannet_boxer_full100.sh" \
    "$CODE_ROOT/tools/audit_boxer_lifting_contract.py" \
    "$CODE_ROOT/tools/summarize_boxer_lifting_ablation.py"
  git -C "$CODE_ROOT/third_party/boxer" rev-parse HEAD
  printf '%s\n' \
    "python=$(readlink -f "$PYTHON")" \
    "python_version=$("$PYTHON" --version 2>&1)"
} | sha256sum | awk '{print $1}'
