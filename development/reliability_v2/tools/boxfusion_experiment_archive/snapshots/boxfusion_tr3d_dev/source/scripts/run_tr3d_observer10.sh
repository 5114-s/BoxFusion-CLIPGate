#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TR3D_ROOT="$ROOT"
# shellcheck source=scripts/lib/tr3d_launch_common.sh
source "$ROOT/scripts/lib/tr3d_launch_common.sh"
cd "$ROOT"

GPU_SPEC="${1:-0}"
RUN_TAG="${2:-${BOXFUSION_TR3D_RUN_TAG:-}}"
[[ -n "$RUN_TAG" ]] || tr3d_die \
  "provide a unique run tag: $0 0,1 tr3d_t1_observer10_v1"
tr3d_require_tag "$RUN_TAG"
tr3d_parse_gpus "$GPU_SPEC"
tr3d_select_env "${BOXFUSION_TR3D_ENV:-$ROOT/.conda/boxfusion-tr3d}"

CONFIG="${BOXFUSION_TR3D_CONFIG:-$ROOT/config/tr3d/tr3d_scannet_foreground_official_val.py}"
CHECKPOINT="${BOXFUSION_TR3D_CHECKPOINT:-}"
MANIFEST="${BOXFUSION_TR3D_INPUT_MANIFEST:-$ROOT/data/tr3d_scannet/scene_manifest.jsonl}"
SCENE_LIST="${BOXFUSION_TR3D_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
PREFIX_ID="${BOXFUSION_TR3D_PREFIX_ID:-full}"
FROZEN_MANIFEST="${BOXFUSION_TR3D_FROZEN_MANIFEST:-$ROOT/manifests/frozen_b6_full100.json}"
CACHE_ROOT="${BOXFUSION_TR3D_CACHE_ROOT:-$ROOT/cache/tr3d_residual/$RUN_TAG}"
ATTEMPT_TAG="${BOXFUSION_TR3D_ATTEMPT_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
LOG_ROOT="${BOXFUSION_TR3D_LOG_ROOT:-$ROOT/logs/tr3d/$RUN_TAG/$ATTEMPT_TAG}"
RESUME="${BOXFUSION_TR3D_RESUME:-0}"

[[ -n "$CHECKPOINT" ]] || tr3d_die \
  "set BOXFUSION_TR3D_CHECKPOINT to a frozen T1 one-class checkpoint"
tr3d_require_file "$CONFIG"
tr3d_require_file "$CHECKPOINT"
tr3d_require_file "$MANIFEST"
tr3d_require_file "$SCENE_LIST"
tr3d_require_file "$FROZEN_MANIFEST"
scene_count="$(awk 'NF {count += 1} END {print count + 0}' "$SCENE_LIST")"
[[ "$scene_count" == "10" ]] || tr3d_die \
  "observer10 requires exactly 10 non-empty scene rows: $SCENE_LIST"
[[ "$RESUME" == "0" || "$RESUME" == "1" ]] \
  || tr3d_die "BOXFUSION_TR3D_RESUME must be 0 or 1"
tr3d_prepare_unique_root "$CACHE_ROOT" "$RESUME"
tr3d_require_new_root "$LOG_ROOT"
python "$ROOT/tools/verify_frozen_b6_manifest.py" \
  --manifest "$FROZEN_MANIFEST"
tr3d_check_environment "$CONFIG" "${TR3D_GPUS[0]}"

echo "Genuine TR3D T1 frozen-B6 observer (10 scenes)"
echo "  run/attempt: $RUN_TAG / $ATTEMPT_TAG"
echo "  scenes: $SCENE_LIST"
echo "  checkpoint: $CHECKPOINT"
echo "  cache: $CACHE_ROOT"
echo "  logs: $LOG_ROOT"
echo "  observer_only=true; mutation_enabled=false; applied_count=0"

BOXFUSION_TR3D_ENV="$TR3D_ENV_REF" \
BOXFUSION_TR3D_CONFIG="$CONFIG" \
BOXFUSION_TR3D_CHECKPOINT="$CHECKPOINT" \
BOXFUSION_TR3D_INPUT_MANIFEST="$MANIFEST" \
BOXFUSION_TR3D_SCENE_LIST="$SCENE_LIST" \
BOXFUSION_TR3D_PREFIX_ID="$PREFIX_ID" \
BOXFUSION_TR3D_CACHE_ROOT="$CACHE_ROOT" \
BOXFUSION_TR3D_LOG_ROOT="$LOG_ROOT" \
BOXFUSION_TR3D_RUN_TAG="$RUN_TAG" \
  bash "$ROOT/scripts/run_tr3d_cache_inference.sh" "$TR3D_GPU_SPEC"

checkpoint_sha="$(tr3d_sha256 "$CHECKPOINT")"
config_sha="$(tr3d_sha256 "$CONFIG")"
python "$ROOT/tools/validate_tr3d_residual_cache.py" \
  --cache-root "$CACHE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --checkpoint-sha256 "$checkpoint_sha" \
  --config-sha256 "$config_sha"

echo "Observer cache complete. Audit it with:"
echo "  BOXFUSION_TR3D_CHECKPOINT='$CHECKPOINT' \\"
echo "  BOXFUSION_TR3D_CACHE_ROOT='$CACHE_ROOT' \\"
echo "  BOXFUSION_TR3D_SCENE_LIST='$SCENE_LIST' \\"
echo "  bash scripts/audit_tr3d_observer.sh '$RUN_TAG'"
