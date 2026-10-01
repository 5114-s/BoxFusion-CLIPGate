#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
SCENE_LIST="${BOXFUSION_CA1M_RECOVERY_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_recovery40.txt}"
RECOVERY_EXPECTED="${BOXFUSION_CA1M_RECOVERY_EXPECTED_SCENES:-40}"
FINAL_SCENE_LIST="${BOXFUSION_CA1M_FINAL_SCENE_LIST:-${BOXFUSION_CA1M_FULL107_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_val_full107.txt}}"
FINAL_EXPECTED="${BOXFUSION_CA1M_FINAL_EXPECTED_SCENES:-107}"
FINAL_ALLOW_UNLISTED="${BOXFUSION_CA1M_FINAL_ALLOW_UNLISTED_SCENES:-0}"
TAR_ROOT="${BOXFUSION_CA1M_APPLE_TAR_ROOT:-/extra/ZhaoX/ca1m_apple_tars}"
METADATA_ROOT="${BOXFUSION_CA1M_METADATA_ROOT:-/extra/ZhaoX/ca1m_metadata_frozen_4ac849e7}"
STAGING_ROOT="${BOXFUSION_CA1M_STAGING_ROOT:-/extra/ZhaoX/ca1m_apple_staging_v1}"
LIVE_ROOT="${BOXFUSION_CA1M_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
BACKUP_ROOT="${BOXFUSION_CA1M_BACKUP_ROOT:-/extra/ZhaoX/boxfusion_ca1m_partial_backup_before_apple_v1}"
REPORT_ROOT="${BOXFUSION_CA1M_RECOVERY_REPORT_ROOT:-$ROOT/reports/ca1m_repro/apple_recovery40_v1}"
ORIENTATION_POLICY="${BOXFUSION_CA1M_ORIENTATION_POLICY:-$ROOT/config/ca1m_orientation_policy_canonical36_v1.json}"
ORIENTATION_POLICY_SHA256="${BOXFUSION_CA1M_ORIENTATION_POLICY_SHA256:-3ced901cc2a090c3485d310420025c54377d06de4a87d967910e38fbc9df348f}"
ORIENTATION_EVIDENCE="${BOXFUSION_CA1M_ORIENTATION_EVIDENCE:-$ROOT/reports/ca1m_repro/apple_recovery_canonical36_v1/47331651_partial_orientation_aspect_v1.json}"
ORIENTATION_EVIDENCE_SHA256="${BOXFUSION_CA1M_ORIENTATION_EVIDENCE_SHA256:-ec5a71ce8f4e9371feee3bdeb182cf946c550f2586221e3c5a98c06c255d3e76}"

for path in "$PYTHON_BIN" "$SCENE_LIST" "$FINAL_SCENE_LIST" "$ORIENTATION_POLICY" "$ORIENTATION_EVIDENCE"; do
    [[ -f "$path" ]] || { echo "Missing required input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$ORIENTATION_POLICY" | awk '{print $1}')" == "$ORIENTATION_POLICY_SHA256" ]] || {
    echo "Orientation policy hash differs from the frozen recovery protocol" >&2
    exit 2
}
[[ "$(sha256sum "$ORIENTATION_EVIDENCE" | awk '{print $1}')" == "$ORIENTATION_EVIDENCE_SHA256" ]] || {
    echo "47331651 orientation evidence hash differs from the frozen recovery protocol" >&2
    exit 2
}
[[ "$(grep -cve '^[[:space:]]*$' "$SCENE_LIST")" == "$RECOVERY_EXPECTED" ]] || {
    echo "Recovery list must contain exactly $RECOVERY_EXPECTED scenes" >&2
    exit 2
}
mkdir -p "$STAGING_ROOT" "$LIVE_ROOT" "$BACKUP_ROOT" "$REPORT_ROOT/scenes"
[[ "$(stat -c '%d' "$STAGING_ROOT")" == "$(stat -c '%d' "$LIVE_ROOT")" ]] || {
    echo "Staging and live roots must share one filesystem for atomic promotion" >&2
    exit 2
}

scene_complete() {
    "$PYTHON_BIN" - "$1" <<'PY'
import sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1])
try:
    rgb=list((p/'rgb').glob('*.png'))
    depth=list((p/'depth').glob('*.png'))
    poses=np.load(p/'all_poses.npy', mmap_mode='r').reshape(-1,4,4)
    required=('K_depth.txt','K_rgb.txt','all_poses.npy','T_gravity.npy','after_filter_boxes.npy')
    ok=all((p/x).is_file() for x in required) and len(rgb)>0 and len(rgb)==len(depth)==len(poses)
except Exception:
    ok=False
raise SystemExit(0 if ok else 1)
PY
}

while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -n "$scene" ]] || continue
    live="$LIVE_ROOT/$scene"
    tar_path="$TAR_ROOT/ca1m-val-$scene.tar"
    metadata="$METADATA_ROOT/$scene"
    stage="$STAGING_ROOT/$scene"
    backup="$BACKUP_ROOT/$scene"
    report="$REPORT_ROOT/scenes/$scene.json"

    if [[ -d "$live" ]] && scene_complete "$live"; then
        echo "[$(date '+%F %T')] $scene already complete in live root"
        continue
    fi
    [[ -s "$tar_path" ]] || { echo "Missing Apple tar: $tar_path" >&2; exit 2; }
    tar -tf "$tar_path" >/dev/null || { echo "Invalid Apple tar: $tar_path" >&2; exit 2; }
    for name in K_depth.txt K_rgb.txt all_poses.npy T_gravity.npy after_filter_boxes.npy; do
        [[ -s "$metadata/$name" ]] || { echo "Missing frozen metadata: $metadata/$name" >&2; exit 2; }
    done

    if [[ ! -d "$stage" ]]; then
        "$PYTHON_BIN" "$ROOT/tools/convert_ca1m_apple_tar.py" \
            --tar "$tar_path" --metadata-scene "$metadata" \
            --staging-root "$STAGING_ROOT" --orientation-policy "$ORIENTATION_POLICY" \
            > "$REPORT_ROOT/scenes/$scene.convert.log"
    fi
    tmp_report="$report.tmp.$$"
    "$PYTHON_BIN" "$ROOT/tools/audit_ca1m_apple_conversion.py" \
        --scene-dir "$stage" --tar "$tar_path" --metadata-scene "$metadata" \
        --pixel-check all --orientation-policy "$ORIENTATION_POLICY" \
        --output "$tmp_report" >/dev/null
    mv -f "$tmp_report" "$report"

    if [[ -e "$live" || -L "$live" ]]; then
        [[ ! -e "$backup" && ! -L "$backup" ]] || {
            echo "Backup target already exists; refusing ambiguous recovery: $backup" >&2
            exit 2
        }
        mv "$live" "$backup"
        echo "[$(date '+%F %T')] Backed up partial scene $scene -> $backup"
    fi
    mv "$stage" "$live"
    scene_complete "$live" || {
        echo "Promoted scene failed completeness check: $live" >&2
        exit 1
    }
    echo "[$(date '+%F %T')] Promoted audited scene $scene -> $live"
done < "$SCENE_LIST"

final_audit_args=(
    --data-root "$LIVE_ROOT"
    --scene-list "$FINAL_SCENE_LIST"
    --report-output "$REPORT_ROOT/final_after_recovery.json"
    --expected-scenes "$FINAL_EXPECTED"
)
if [[ "$FINAL_ALLOW_UNLISTED" == "1" ]]; then
    final_audit_args+=(--allow-unlisted-scenes)
fi
"$PYTHON_BIN" "$ROOT/tools/prepare_ca1m_full107.py" "${final_audit_args[@]}"

echo "[$(date '+%F %T')] CA-1M recovery completed; exact $FINAL_EXPECTED-scene audit passed"
