#!/usr/bin/env bash
set -euo pipefail

GPU_SPEC="${1:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

for profile in u0_control u1_observer u2_active; do
    bash "$ROOT/scripts/run_scannet_b6_boxer_uncertainty.sh" \
        "$profile" "$GPU_SPEC"
done

bash "$ROOT/scripts/audit_scannet_b6_boxer_uncertainty.sh"
bash "$ROOT/scripts/report_scannet_b6_boxer_uncertainty.sh"
