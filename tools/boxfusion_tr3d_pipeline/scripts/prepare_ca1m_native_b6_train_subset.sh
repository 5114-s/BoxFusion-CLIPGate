#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
TRAIN_URL_LIST="${BOXFUSION_CA1M_TRAIN_URL_LIST:-/data/ZhaoX/BoxFusion/data/train.txt}"
VAL_URL_LIST="${BOXFUSION_CA1M_VAL_URL_LIST:-/data/ZhaoX/BoxFusion/data/val.txt}"
SUBSET_SIZE="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENES:-100}"
NAMESPACE="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_NAMESPACE:-boxfusion.ca1m-native-b6.train100.v1}"
MANIFEST_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_MANIFEST_ROOT:-$ROOT/manifests/ca1m_native_b6_train100_v1}"
DOWNLOAD_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_TAR_ROOT:-/extra/ZhaoX/ca1m_apple_train_tars}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/prepare_ca1m_native_b6_train_subset.sh [--preflight|--download]

Default: --preflight. It freezes/audits the train-only subset and reports local
readiness; it does not download, extract, inspect validation GT, or train.

Explicit --download fetches only the frozen subset from the official Apple URLs
with wget --continue, validates each tar, then records local SHA256 hashes.
EOF
}

mode="preflight"
case "${1:---preflight}" in
    --preflight) mode="preflight" ;;
    --download) mode="download" ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
[[ "$#" -le 1 ]] || { usage >&2; exit 2; }

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
for path in "$TRAIN_URL_LIST" "$VAL_URL_LIST"; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "Missing regular URL list: $path" >&2; exit 2; }
done
[[ "$SUBSET_SIZE" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid subset size: $SUBSET_SIZE" >&2; exit 2; }

prepare=(
    "$PYTHON" "$ROOT/tools/prepare_ca1m_native_b6_train_subset.py"
    --train-url-list "$TRAIN_URL_LIST"
    --val-url-list "$VAL_URL_LIST"
    --subset-size "$SUBSET_SIZE"
    --namespace "$NAMESPACE"
    --output-dir "$MANIFEST_ROOT"
    --download-root "$DOWNLOAD_ROOT"
)
"${prepare[@]}"

if [[ "$mode" == "preflight" ]]; then
    echo "Preflight only: no CA-1M training data was downloaded."
    echo "Explicit download command:"
    echo "  bash scripts/prepare_ca1m_native_b6_train_subset.sh --download"
    exit 0
fi

[[ ! -L "$DOWNLOAD_ROOT" ]] || { echo "Refusing symlink download root: $DOWNLOAD_ROOT" >&2; exit 2; }
mkdir -p "$DOWNLOAD_ROOT"

while IFS= read -r url || [[ -n "$url" ]]; do
    [[ -n "$url" ]] || continue
    [[ "$url" =~ ^https://ml-site\.cdn-apple\.com/datasets/ca1m/train/ca1m-train-([0-9]{8})\.tar$ ]] || {
        echo "Frozen manifest contains a non-official URL: $url" >&2
        exit 2
    }
    scene="${BASH_REMATCH[1]}"
    final="$DOWNLOAD_ROOT/ca1m-train-$scene.tar"
    partial="$final.part"
    [[ ! -L "$final" && ! -L "$partial" ]] || {
        echo "Refusing symlink local artifact for $scene" >&2
        exit 2
    }
    if [[ -s "$final" ]]; then
        tar -tf "$final" >/dev/null || {
            echo "Existing tar fails full integrity scan; refusing overwrite: $final" >&2
            exit 1
        }
        echo "[$(date '+%F %T')] $scene already downloaded and valid"
        continue
    fi
    [[ ! -e "$final" ]] || { echo "Refusing empty final tar: $final" >&2; exit 1; }
    echo "[$(date '+%F %T')] Downloading official CA-1M train scene $scene"
    wget --continue --tries=20 --timeout=60 --waitretry=10 \
        --progress=dot:giga --output-document="$partial" "$url"
    tar -tf "$partial" >/dev/null || {
        echo "Downloaded tar fails full integrity scan: $partial" >&2
        exit 1
    }
    mv "$partial" "$final"
done < "$MANIFEST_ROOT/urls.txt"

"${prepare[@]}" --hash-existing --require-complete
echo "CA-1M-native B6 frozen train subset is downloaded and SHA256-audited: $DOWNLOAD_ROOT"
