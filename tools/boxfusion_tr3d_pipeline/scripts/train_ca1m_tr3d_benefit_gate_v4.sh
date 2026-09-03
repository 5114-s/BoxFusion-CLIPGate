#!/usr/bin/env bash
set -euo pipefail

# Static v4 benefit-gate binding launcher.  --preflight and --run are
# deliberately blocked until all final-base/B6-OOF/P/O/evidence bindings are
# populated and the config is sealed.  Neither mode creates an output before
# that full validation succeeds.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
CONFIG="${BOXFUSION_CA1M_TR3D_BENEFIT_V4_CONFIG:-$ROOT/config/ca1m_tr3d_benefit_gate_train100_v4.json}"
PREFLIGHT="$ROOT/tools/preflight_ca1m_tr3d_benefit_gate_v4.py"

usage() {
  echo "Usage: $0 [--static-contract|--preflight|--run]" >&2
}

MODE="${1:---preflight}"
[[ "$#" -le 1 ]] || { usage; exit 2; }
case "$MODE" in
  --static-contract|--preflight|--run) ;;
  --help|-h) usage; exit 0 ;;
  *) usage; exit 2 ;;
esac

[[ -x "$PYTHON" ]] || { echo "Missing pipeline Python: $PYTHON" >&2; exit 2; }
for path in "$CONFIG" "$PREFLIGHT" \
  "$ROOT/boxfusion/ca1m_tr3d_terminal_gate_v4.py"; do
  [[ -f "$path" && ! -L "$path" ]] || {
    echo "Missing regular gate-v4 source: $path" >&2
    exit 2
  }
done

for name in \
  BOXFUSION_TR3D_CHECKPOINT BOXFUSION_TR3D_CONFIG \
  BOXFUSION_NATIVE_B6_CHECKPOINT BOXFUSION_CA1M_TR3D_TERMINAL_ROOT \
  BOXFUSION_CA1M_NATIVE_B6_OBSERVER_ROOT; do
  [[ -z "${!name:-}" ]] || {
    echo "Raw/legacy override is forbidden by gate-v4: $name" >&2
    exit 2
  }
done

exec "$PYTHON" "$PREFLIGHT" --config "$CONFIG" "$MODE"

