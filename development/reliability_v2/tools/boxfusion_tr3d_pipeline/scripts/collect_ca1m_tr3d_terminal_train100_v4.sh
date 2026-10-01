#!/usr/bin/env bash
set -euo pipefail

# Static-only launcher for the split terminal-v4 route.  The checked-in
# contract allows only the separately sealed --run-proposals stage.  Full
# --run remains blocked until stage O receives final-base/native-B6-v2 bindings.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${1:---static-preflight}"
[[ "$#" -le 1 ]] || { echo "Usage: $0 [--static-preflight|--preflight|--run-proposals|--run]" >&2; exit 2; }
case "$MODE" in
  --static-preflight|--preflight|--run-proposals|--run) ;;
  *) echo "Usage: $0 [--static-preflight|--preflight|--run-proposals|--run]" >&2; exit 2 ;;
esac

PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
CONFIG="$ROOT/config/ca1m_tr3d_terminal_train100_v4_p5.json"
PREFLIGHT="$ROOT/tools/preflight_ca1m_tr3d_terminal_train100_v4.py"

[[ -x "$PYTHON" ]] || { echo "Missing pipeline Python: $PYTHON" >&2; exit 2; }
for path in "$CONFIG" "$PREFLIGHT" \
  "$ROOT/tools/run_ca1m_tr3d_proposal_cache_v4.py" \
  "$ROOT/tools/overlay_ca1m_tr3d_terminal_v4.py" \
  "$ROOT/boxfusion/ca1m_tr3d_terminal_v4.py"; do
  [[ -f "$path" && ! -L "$path" ]] || { echo "Missing regular v4 source: $path" >&2; exit 2; }
done

[[ -z "${BOXFUSION_TR3D_CHECKPOINT:-}" ]] || {
  echo "Raw BOXFUSION_TR3D_CHECKPOINT is forbidden; v4 uses the sealed CA binding" >&2
  exit 2
}
[[ -z "${BOXFUSION_TR3D_CONFIG:-}" ]] || {
  echo "Raw BOXFUSION_TR3D_CONFIG is forbidden; v4 uses the sealed CA binding" >&2
  exit 2
}
[[ -z "${BOXFUSION_NATIVE_B6_CHECKPOINT:-}" ]] || {
  echo "Raw/legacy B6 override is forbidden; stage O requires the sealed final-base v2 binding" >&2
  exit 2
}

if [[ "$MODE" == "--run" ]]; then
  # This returns 2 before any worker/model/evaluator is opened while bindings
  # and explicit per-stage authorizations remain pending.
  exec "$PYTHON" "$PREFLIGHT" --config "$CONFIG" --require-run
fi

if [[ "$MODE" == "--run-proposals" ]]; then
  # Read-only authorization/code/parity/binding validation happens first.
  # The Python runner then recomputes and checks every pending point hash on
  # CPU before constructing the GPU worker.
  "$PYTHON" "$PREFLIGHT" --config "$CONFIG" --require-proposal-run >/dev/null
  exec "$PYTHON" "$ROOT/tools/run_ca1m_tr3d_proposal_cache_v4.py" \
    --collection-config "$CONFIG" \
    --device "${BOXFUSION_CA1M_TR3D_V4_DEVICE:-cuda:0}"
fi

exec "$PYTHON" "$PREFLIGHT" --config "$CONFIG"
