#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TR3D_ROOT="$ROOT"
# shellcheck source=scripts/lib/tr3d_launch_common.sh
source "$ROOT/scripts/lib/tr3d_launch_common.sh"
cd "$ROOT"

GPU="${1:-0}"
RUN_TAG="${2:-${BOXFUSION_TR3D_RUN_TAG:-}}"
[[ -n "$RUN_TAG" ]] || tr3d_die \
  "provide a unique run tag: $0 0 tr3d_t1_smoke_v1"
tr3d_require_tag "$RUN_TAG"
tr3d_parse_gpus "$GPU"
(( ${#TR3D_GPUS[@]} == 1 )) || tr3d_die "single-scene smoke uses one GPU"
tr3d_select_env "${BOXFUSION_TR3D_ENV:-$ROOT/.conda/boxfusion-tr3d}"

CONFIG="${BOXFUSION_TR3D_CONFIG:-$ROOT/config/tr3d/tr3d_scannet_foreground_official_val.py}"
CHECKPOINT="${BOXFUSION_TR3D_CHECKPOINT:-}"
MANIFEST="${BOXFUSION_TR3D_INPUT_MANIFEST:-$ROOT/data/tr3d_scannet/scene_manifest.jsonl}"
SCENE_LIST="${BOXFUSION_TR3D_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_smoke1.txt}"
PREFIX_ID="${BOXFUSION_TR3D_PREFIX_ID:-full}"
CACHE_ROOT="${BOXFUSION_TR3D_CACHE_ROOT:-$ROOT/cache/tr3d_residual/$RUN_TAG}"
LOG_ROOT="${BOXFUSION_TR3D_LOG_ROOT:-$ROOT/logs/tr3d/$RUN_TAG}"

[[ -n "$CHECKPOINT" ]] || tr3d_die \
  "set BOXFUSION_TR3D_CHECKPOINT to a frozen T1 one-class checkpoint"
tr3d_require_file "$CONFIG"
tr3d_require_file "$CHECKPOINT"
tr3d_require_file "$MANIFEST"
tr3d_require_file "$SCENE_LIST"
scene_count="$(awk 'NF {count += 1} END {print count + 0}' "$SCENE_LIST")"
[[ "$scene_count" == "1" ]] || tr3d_die \
  "single-scene smoke requires exactly one non-empty scene row: $SCENE_LIST"
SCENE_ID="$(awk 'NF {print; exit}' "$SCENE_LIST")"
[[ "$SCENE_ID" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] \
  || tr3d_die "invalid smoke scene: $SCENE_ID"
tr3d_require_new_root "$CACHE_ROOT"
tr3d_require_new_root "$LOG_ROOT"

tr3d_check_environment "$CONFIG" "$GPU"

echo "Genuine TR3D T1 one-scene immutable-cache smoke"
echo "  scene/prefix: $SCENE_ID/$PREFIX_ID"
echo "  checkpoint: $CHECKPOINT"
echo "  cache: $CACHE_ROOT"
echo "  logs: $LOG_ROOT"

BOXFUSION_TR3D_ENV="$TR3D_ENV_REF" \
BOXFUSION_TR3D_CONFIG="$CONFIG" \
BOXFUSION_TR3D_CHECKPOINT="$CHECKPOINT" \
BOXFUSION_TR3D_INPUT_MANIFEST="$MANIFEST" \
BOXFUSION_TR3D_SCENE_LIST="$SCENE_LIST" \
BOXFUSION_TR3D_PREFIX_ID="$PREFIX_ID" \
BOXFUSION_TR3D_CACHE_ROOT="$CACHE_ROOT" \
BOXFUSION_TR3D_LOG_ROOT="$LOG_ROOT/export" \
BOXFUSION_TR3D_RUN_TAG="$RUN_TAG" \
  bash "$ROOT/scripts/run_tr3d_cache_inference.sh" "$GPU"

checkpoint_sha="$(tr3d_sha256 "$CHECKPOINT")"
config_sha="$(tr3d_sha256 "$CONFIG")"
python "$ROOT/tools/validate_tr3d_residual_cache.py" \
  --cache-root "$CACHE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --checkpoint-sha256 "$checkpoint_sha" \
  --config-sha256 "$config_sha"
