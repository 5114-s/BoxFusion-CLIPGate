#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

MODE="${1:---static-preflight}"
case "$MODE" in
  --static-preflight|--preflight|--run) ;;
  *) echo "usage: $0 [--static-preflight|--preflight|--run]" >&2; exit 2 ;;
esac

PY=/home/admin1/miniconda3/envs/boxfusion-online/bin/python
CONFIG="$ROOT/config/ca1m_tr3d_terminal_train100_v3.json"
CHECKPOINT_BINDING="${BOXFUSION_CA1M_TR3D_V3_CHECKPOINT_BINDING:-}"
CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export CUBLAS_WORKSPACE_CONFIG

[[ -x "$PY" ]] || { echo "missing pipeline Python: $PY" >&2; exit 2; }
[[ "$CUBLAS_WORKSPACE_CONFIG" == ":4096:8" ]] || {
  echo "v3 requires CUBLAS_WORKSPACE_CONFIG=:4096:8" >&2
  exit 2
}
[[ -z "${BOXFUSION_TR3D_CHECKPOINT:-}" ]] || {
  echo "raw BOXFUSION_TR3D_CHECKPOINT is forbidden; use the v3 binding" >&2
  exit 2
}
[[ -z "${BOXFUSION_TR3D_CONFIG:-}" ]] || {
  echo "raw BOXFUSION_TR3D_CONFIG is forbidden; use the v3 binding" >&2
  exit 2
}

if [[ "$MODE" == "--static-preflight" ]]; then
  exec "$PY" tools/preflight_ca1m_tr3d_terminal_train100_v3.py \
    --config "$CONFIG"
fi

[[ -n "$CHECKPOINT_BINDING" ]] || {
  echo "set BOXFUSION_CA1M_TR3D_V3_CHECKPOINT_BINDING to the sealed CA checkpoint manifest" >&2
  exit 2
}
[[ -f "$CHECKPOINT_BINDING" && ! -L "$CHECKPOINT_BINDING" ]] || {
  echo "missing regular checkpoint binding: $CHECKPOINT_BINDING" >&2
  exit 2
}

"$PY" tools/preflight_ca1m_tr3d_terminal_train100_v3.py \
  --config "$CONFIG" \
  --checkpoint-binding "$CHECKPOINT_BINDING"
if [[ "$MODE" == "--preflight" ]]; then
  exit 0
fi

echo "v3 run is not authorized: final frame lineage, G0+CLIP+reliable TopK3 anchors," >&2
echo "and retrained native-B6 artifacts must be sealed before proposal/overlay collection" >&2
exit 2
