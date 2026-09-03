#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

export BOXFUSION_CA1M_RECOVERY_LIST="${BOXFUSION_CA1M_RECOVERY_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_recovery_canonical36.txt}"
export BOXFUSION_CA1M_RECOVERY_EXPECTED_SCENES=36
export BOXFUSION_CA1M_FINAL_SCENE_LIST="${BOXFUSION_CA1M_FINAL_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_canonical103.txt}"
export BOXFUSION_CA1M_FINAL_EXPECTED_SCENES=103
export BOXFUSION_CA1M_FINAL_ALLOW_UNLISTED_SCENES=1
export BOXFUSION_CA1M_STAGING_ROOT="${BOXFUSION_CA1M_STAGING_ROOT:-/extra/ZhaoX/ca1m_apple_staging_canonical36_v1}"
export BOXFUSION_CA1M_BACKUP_ROOT="${BOXFUSION_CA1M_BACKUP_ROOT:-/extra/ZhaoX/boxfusion_ca1m_partial_backup_canonical36_v1}"
export BOXFUSION_CA1M_RECOVERY_REPORT_ROOT="${BOXFUSION_CA1M_RECOVERY_REPORT_ROOT:-$ROOT/reports/ca1m_repro/apple_recovery_canonical36_v1}"

echo "Recovering 36 scenes with author-published GT to form canonical-103"
echo "Missing-GT scenes remain isolated: 45663164 47115469 47331311 47332000"
exec bash "$ROOT/scripts/recover_ca1m_processed_recovery40.sh"
