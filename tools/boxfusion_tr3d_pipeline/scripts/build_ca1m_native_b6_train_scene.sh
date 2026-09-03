#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
MANIFEST_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_MANIFEST_ROOT:-$ROOT/manifests/ca1m_native_b6_train100_v1}"
SUBSET_MANIFEST="${BOXFUSION_CA1M_NATIVE_B6_SUBSET_MANIFEST:-$MANIFEST_ROOT/subset_manifest.json}"
VAL_URL_LIST="${BOXFUSION_CA1M_VAL_URL_LIST:-/data/ZhaoX/BoxFusion/data/val.txt}"
TAR_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_TAR_ROOT:-/extra/ZhaoX/ca1m_apple_train_tars}"
OUTPUT_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENE_ROOT:-/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1}"
REPORT_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENE_REPORT_ROOT:-$ROOT/reports/ca1m_native_b6_train_scenes_v1}"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/build_ca1m_native_b6_train_scene.sh [--preflight|--single-scene] [SCENE_ID]

Default is --preflight and one scene only.  With no SCENE_ID, the first frozen
scene having a complete .tar (not .part) is selected.  --single-scene is the
only mode that creates output.  There is intentionally no batch mode.
EOF
}

mode="preflight"
case "${1:---preflight}" in
    --preflight) mode="preflight"; shift || true ;;
    --single-scene) mode="single"; shift || true ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
[[ "$#" -le 1 ]] || { usage >&2; exit 2; }
scene="${1:-}"

[[ -x "$PYTHON" ]] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
for path in "$SUBSET_MANIFEST" "$VAL_URL_LIST" "$MANIFEST_ROOT/scene_ids.txt"; do
    [[ -f "$path" && ! -L "$path" ]] || { echo "Missing regular frozen input: $path" >&2; exit 2; }
done
[[ ! -L "$TAR_ROOT" && ! -L "$OUTPUT_ROOT" ]] || {
    echo "Refusing symlink tar/output root" >&2
    exit 2
}

if [[ -z "$scene" ]]; then
    while IFS= read -r candidate || [[ -n "$candidate" ]]; do
        [[ "$candidate" =~ ^[0-9]{8}$ ]] || { echo "Invalid frozen scene ID: $candidate" >&2; exit 2; }
        if [[ -s "$TAR_ROOT/ca1m-train-$candidate.tar" && ! -e "$TAR_ROOT/ca1m-train-$candidate.tar.part" ]]; then
            scene="$candidate"
            break
        fi
    done < "$MANIFEST_ROOT/scene_ids.txt"
fi
[[ "$scene" =~ ^[0-9]{8}$ ]] || {
    echo "No complete frozen train tar is available for a single-scene check" >&2
    exit 3
}
grep -qx "$scene" "$MANIFEST_ROOT/scene_ids.txt" || {
    echo "Scene is not in frozen train100 manifest: $scene" >&2
    exit 2
}
tar_path="$TAR_ROOT/ca1m-train-$scene.tar"
[[ -s "$tar_path" && ! -L "$tar_path" ]] || { echo "Missing complete train tar: $tar_path" >&2; exit 3; }
[[ ! -e "$tar_path.part" ]] || { echo "Refusing scene with partial sibling: $tar_path.part" >&2; exit 3; }

builder=(
    "$PYTHON" "$ROOT/tools/build_ca1m_native_b6_train_scene.py"
    --tar "$tar_path"
    --scene-id "$scene"
    --subset-manifest "$SUBSET_MANIFEST"
    --val-url-list "$VAL_URL_LIST"
    --output-root "$OUTPUT_ROOT"
)

if [[ "$mode" == "preflight" ]]; then
    "${builder[@]}" --mode preflight
    echo "Preflight passed for train-only scene $scene; no output was created."
    echo "Explicit single-scene build:"
    echo "  bash scripts/build_ca1m_native_b6_train_scene.sh --single-scene $scene"
    exit 0
fi

"${builder[@]}" --mode build
mkdir -p "$REPORT_ROOT"
audit="$REPORT_ROOT/${scene}_audit.json"
"$PYTHON" "$ROOT/tools/audit_ca1m_native_b6_train_scene.py" \
    --scene-dir "$OUTPUT_ROOT/$scene" \
    --geometry-check full \
    --pixel-check sample \
    --output "$audit"
echo "CA-1M-native B6 single train scene built and audited: $OUTPUT_ROOT/$scene"
echo "Audit: $audit"
