#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SCENE_LIST="${BOXFUSION_CA1M_RECOVERY_LIST:-$ROOT/evaluation/data_util/meta_data/ca1m_recovery40.txt}"
URL_LIST="${BOXFUSION_CA1M_OFFICIAL_URL_LIST:-/data/ZhaoX/BoxFusion/data/val.txt}"
OUTPUT_ROOT="${BOXFUSION_CA1M_APPLE_TAR_ROOT:-/extra/ZhaoX/ca1m_apple_tars}"
TMPDIR="${TMPDIR:-/extra/ZhaoX/ca1m_download_tmp}"
export TMPDIR

for path in "$SCENE_LIST" "$URL_LIST"; do
    [[ -f "$path" ]] || { echo "Missing required list: $path" >&2; exit 2; }
done
[[ "$(grep -cve '^[[:space:]]*$' "$SCENE_LIST")" == "40" ]] || {
    echo "Recovery list must contain the frozen 40 incomplete scene IDs" >&2
    exit 2
}
mkdir -p "$OUTPUT_ROOT" "$TMPDIR"

while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -n "$scene" ]] || continue
    [[ "$scene" =~ ^[0-9]{8}$ ]] || { echo "Invalid scene ID: $scene" >&2; exit 2; }
    matches="$(awk -v scene="$scene" \
        '$0 ~ ("/ca1m-val-" scene "\\.tar$") { print }' "$URL_LIST")"
    match_count="$(printf '%s\n' "$matches" | awk 'NF {count++} END {print count+0}')"
    [[ "$match_count" == "1" ]] || {
        echo "Expected exactly one official Apple URL for $scene" >&2
        exit 2
    }
    url="$matches"
    final="$OUTPUT_ROOT/ca1m-val-$scene.tar"
    partial="$final.part"
    if [[ -s "$final" ]]; then
        tar -tf "$final" >/dev/null || {
            echo "Existing tar fails integrity check; refusing overwrite: $final" >&2
            exit 1
        }
        echo "[$(date '+%F %T')] $scene already downloaded and valid"
        continue
    fi
    [[ ! -e "$final" ]] || { echo "Refusing empty output: $final" >&2; exit 1; }
    echo "[$(date '+%F %T')] Downloading official Apple CA-1M scene $scene"
    wget --continue --tries=20 --timeout=60 --waitretry=10 \
        --progress=dot:giga --output-document="$partial" "$url"
    tar -tf "$partial" >/dev/null || {
        echo "Downloaded tar fails integrity check: $partial" >&2
        exit 1
    }
    mv "$partial" "$final"
    echo "[$(date '+%F %T')] Completed $scene: $final"
done < "$SCENE_LIST"

echo "[$(date '+%F %T')] Official Apple CA-1M recovery tars complete: $OUTPUT_ROOT"
