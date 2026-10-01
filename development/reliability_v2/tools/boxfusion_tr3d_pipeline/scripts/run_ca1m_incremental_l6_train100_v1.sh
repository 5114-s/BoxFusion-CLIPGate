#!/usr/bin/env bash
set -euo pipefail

# Static-only CA-1M incremental/L6 entry point.  It intentionally exposes no
# GPU, GT, model, cache, or policy override.  --run remains fail-closed until
# the new final-base -> native-B6-v2 -> terminal-v4 -> benefit-v2 chain is
# sealed and a separate execution driver is audited.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
MODE="${1:---static-preflight}"
[[ "$#" -le 1 ]] || {
  echo "Usage: $0 [--static-preflight|--preflight|--run]" >&2
  exit 2
}
case "$MODE" in
  --static-preflight|--preflight|--run) ;;
  *)
    echo "Usage: $0 [--static-preflight|--preflight|--run]" >&2
    exit 2
    ;;
esac

PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
CONFIG="$ROOT/config/ca1m_incremental_l6_train100_v1.json"
PREFLIGHT="$ROOT/tools/preflight_ca1m_incremental_l6_train100_v1.py"

[[ -x "$PYTHON" ]] || { echo "Missing pipeline Python: $PYTHON" >&2; exit 2; }
for path in \
  "$CONFIG" \
  "$PREFLIGHT" \
  "$ROOT/boxfusion/ca1m_incremental_l6.py"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "Missing regular CA L6 static source: $path" >&2
    exit 2
  }
done

for name in \
  BOXFUSION_TR3D_CHECKPOINT \
  BOXFUSION_TR3D_CONFIG \
  BOXFUSION_INCREMENTAL_POLICY \
  BOXFUSION_LIGHTWEIGHT_POLICY \
  BOXFUSION_NATIVE_B6_CHECKPOINT \
  BOXFUSION_TERMINAL_CACHE_ROOT \
  BOXFUSION_VALIDATION_GT_ROOT; do
  [[ -z "${!name:-}" ]] || {
    echo "Raw/legacy override is forbidden for CA L6: $name" >&2
    exit 2
  }
done

if [[ "$MODE" == "--run" ]]; then
  exec "$PYTHON" "$PREFLIGHT" --config "$CONFIG" --require-run
fi
exec "$PYTHON" "$PREFLIGHT" --config "$CONFIG"
