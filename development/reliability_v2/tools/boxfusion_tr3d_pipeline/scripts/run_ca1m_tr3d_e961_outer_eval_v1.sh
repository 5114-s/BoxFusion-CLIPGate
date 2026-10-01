#!/usr/bin/env bash
set -euo pipefail

# Immutable protocol V1 bound a training implementation that failed independent
# review. Keep this legacy command as an explicit tombstone: it must never
# forward to the V2 runner or create any runtime/evaluation artifact.
PIPELINE_ROOT="/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline"
INVALID_V1="$PIPELINE_ROOT/manifests/ca1m_tr3d_e961_outer_dev_eval_v1/PREREGISTRATION_PROTOCOL_V1_INVALID.json"
INVALID_V1_SHA256="31d39340015df4101725d475310ec09b5daa19751c677ae0d2e51f75ad5ad3d8"

[[ -f "$INVALID_V1" && ! -L "$INVALID_V1" ]] || {
  echo "E961 outer evaluation protocol V1 is invalid; its invalidation receipt is missing" >&2
  exit 66
}
actual_sha256="$(sha256sum "$INVALID_V1" | awk '{print $1}')"
[[ "$actual_sha256" == "$INVALID_V1_SHA256" ]] || {
  echo "E961 outer evaluation protocol V1 invalidation receipt drifted" >&2
  exit 66
}

echo "E961 outer evaluation protocol V1 is INVALID/SUPERSEDED and cannot run." >&2
echo "Only the reviewed V2 command is eligible after a real R2 receipt exists:" >&2
echo "  bash scripts/run_ca1m_tr3d_e961_outer_eval_v2.sh --run OUTER_R2_RUN_TAG GPU_ID" >&2
exit 66
