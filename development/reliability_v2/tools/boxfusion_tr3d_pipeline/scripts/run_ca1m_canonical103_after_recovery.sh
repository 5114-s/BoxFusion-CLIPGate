#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
RECOVERY_SESSION="ca1m_canonical36_recovery"
RECOVERY_LOG="/extra/ZhaoX/ca1m_recovery_logs/canonical36_driver.log"
SUCCESS_LINE="CA-1M recovery completed; exact 103-scene audit passed"

echo "[$(date '+%F %T')] Waiting for canonical-36 recovery and audit"
while tmux has-session -t "$RECOVERY_SESSION" 2>/dev/null; do
    sleep 30
done

[[ -f "$RECOVERY_LOG" ]] || {
    echo "Missing recovery driver log: $RECOVERY_LOG" >&2
    exit 2
}
grep -Fq "$SUCCESS_LINE" "$RECOVERY_LOG" || {
    echo "Recovery did not finish with the required canonical-103 audit; inference will not start" >&2
    tail -n 80 "$RECOVERY_LOG" >&2
    exit 1
}

echo "[$(date '+%F %T')] Recovery audit passed; starting corrected canonical-103 C0 reproduction"
exec bash "$ROOT/scripts/run_ca1m_c0_score04_real_score_canonical103.sh" 0,1
