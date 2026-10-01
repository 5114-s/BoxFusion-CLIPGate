#!/usr/bin/env bash
set -euo pipefail

# Fixed-10 paired control: original improvement objective retrained on the
# exact K=5, gate-aligned runtime diagnostics.

GPU_SPEC="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

export BOXFUSION_B5V2_REFINER_CHECKPOINT="${BOXFUSION_B5V2_K5_CHECKPOINT:-$ROOT/models/scannet_b5v2_k5_gatealigned_extent040_refiner_v2.pt}"
export BOXFUSION_B5V2_RUN_TAG="${BOXFUSION_B5V2_K5_RUN_TAG:-b5v2_k5_gatealigned_refiner_only_extent040_ablation10_v2}"
export BOXFUSION_B5V2_SCENE_LIST="${BOXFUSION_B5V3_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
export BOXFUSION_B5V2_MIN_EXTENT="${BOXFUSION_B5V3_RUNTIME_MIN_EXTENT:-0.40}"
export BOXFUSION_B5V2_PROPOSAL_INTERVAL="${BOXFUSION_B5V3_PROPOSAL_INTERVAL:-5}"

exec bash "$ROOT/scripts/run_scannet_b5v2_refiner.sh" "$GPU_SPEC"
