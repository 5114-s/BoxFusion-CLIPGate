#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
TR3D_ROOT="$ROOT"
# shellcheck source=scripts/lib/tr3d_launch_common.sh
source "$ROOT/scripts/lib/tr3d_launch_common.sh"
cd "$ROOT"

RUN_TAG="${1:-${BOXFUSION_TR3D_RUN_TAG:-}}"
[[ -n "$RUN_TAG" ]] || tr3d_die \
  "provide the observer run tag: $0 tr3d_t1_observer10_v1"
tr3d_require_tag "$RUN_TAG"

CONFIG="${BOXFUSION_TR3D_CONFIG:-$ROOT/config/tr3d/tr3d_scannet_foreground_official_val.py}"
CHECKPOINT="${BOXFUSION_TR3D_CHECKPOINT:-}"
CACHE_ROOT="${BOXFUSION_TR3D_CACHE_ROOT:-$ROOT/cache/tr3d_residual/$RUN_TAG}"
SCENE_LIST="${BOXFUSION_TR3D_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val_ablation10_even.txt}"
PREFIX_ID="${BOXFUSION_TR3D_PREFIX_ID:-full}"
MANIFEST="${BOXFUSION_TR3D_FROZEN_MANIFEST:-$ROOT/manifests/frozen_b6_full100.json}"
GT_ROOT="${BOXFUSION_TR3D_GT_ROOT:-/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data}"
SCANS_ROOT="${BOXFUSION_TR3D_SCANS_ROOT:-/extra/ZhaoX/scannet_data/scans}"
AUDIT_TAG="${BOXFUSION_TR3D_AUDIT_TAG:-$(date +%Y%m%d_%H%M%S)_$$}"
REPORT_ROOT="${BOXFUSION_TR3D_REPORT_ROOT:-$ROOT/reports/tr3d/$RUN_TAG/$AUDIT_TAG}"
REPORT="$REPORT_ROOT/union_oracle.json"

[[ -n "$CHECKPOINT" ]] || tr3d_die \
  "set BOXFUSION_TR3D_CHECKPOINT to the exact cache-producing checkpoint"
tr3d_require_file "$CONFIG"
tr3d_require_file "$CHECKPOINT"
tr3d_require_file "$SCENE_LIST"
tr3d_require_file "$MANIFEST"
[[ -d "$CACHE_ROOT" ]] || tr3d_die "cache root does not exist: $CACHE_ROOT"
[[ -d "$GT_ROOT" ]] || tr3d_die "GT root does not exist: $GT_ROOT"
[[ -d "$SCANS_ROOT" ]] || tr3d_die "ScanNet scans root does not exist: $SCANS_ROOT"
tr3d_require_new_root "$REPORT_ROOT"

checkpoint_sha="$(tr3d_sha256 "$CHECKPOINT")"
config_sha="$(tr3d_sha256 "$CONFIG")"
python "$ROOT/tools/validate_tr3d_residual_cache.py" \
  --cache-root "$CACHE_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --checkpoint-sha256 "$checkpoint_sha" \
  --config-sha256 "$config_sha"

python "$ROOT/tools/audit_tr3d_residual_observer.py" \
  --manifest "$MANIFEST" \
  --cache-root "$CACHE_ROOT" \
  --gt-root "$GT_ROOT" \
  --scans-root "$SCANS_ROOT" \
  --scene-list "$SCENE_LIST" \
  --prefix-id "$PREFIX_ID" \
  --checkpoint-sha256 "$checkpoint_sha" \
  --config-sha256 "$config_sha" \
  --report "$REPORT"

echo "Immutable anchor observer audit written to $REPORT"
