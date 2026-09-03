#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCENE_LIST="${BOXFUSION_CA1M_RECOVERY_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_recovery40.txt}"
OUTPUT_ROOT="${BOXFUSION_CA1M_METADATA_ROOT:-/extra/ZhaoX/ca1m_metadata_frozen_4ac849e7}"
HF_HOME="${HF_HOME:-/extra/ZhaoX/hf_cache}"
TMPDIR="${TMPDIR:-/extra/ZhaoX/hf_tmp}"
REPO="${BOXFUSION_CA1M_HF_REPO:-Kevin1804/BoxFusion}"
REVISION="${BOXFUSION_CA1M_HF_REVISION:-4ac849e7953be3a60b146165c5f37cecd2997a16}"
HF_PYTHON="${BOXFUSION_HF_PYTHON:-/home/admin1/miniconda3/envs/temp/bin/python}"

[[ -f "$SCENE_LIST" ]] || { echo "Missing recovery list: $SCENE_LIST" >&2; exit 2; }
[[ "$(grep -cve '^[[:space:]]*$' "$SCENE_LIST")" == "40" ]] || {
    echo "Recovery list must contain the frozen 40 incomplete scene IDs" >&2
    exit 2
}
mkdir -p "$OUTPUT_ROOT" "$HF_HOME" "$TMPDIR"
export HF_HOME TMPDIR
[[ -x "$HF_PYTHON" ]] || { echo "Missing Hugging Face Python: $HF_PYTHON" >&2; exit 2; }

"$HF_PYTHON" "$ROOT/tools/download_ca1m_frozen_metadata.py" \
    --repo "$REPO" --revision "$REVISION" \
    --scene-list "$SCENE_LIST" --output-root "$OUTPUT_ROOT"

echo "[$(date '+%F %T')] Frozen CA-1M recovery metadata complete: $OUTPUT_ROOT"
