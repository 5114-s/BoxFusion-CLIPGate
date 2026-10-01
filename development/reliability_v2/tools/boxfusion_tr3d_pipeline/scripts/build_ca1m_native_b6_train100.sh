#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
MANIFEST_ROOT="${BOXFUSION_CA1M_NATIVE_B6_TRAIN_MANIFEST_ROOT:-$ROOT/manifests/ca1m_native_b6_train100_v1}"
ORIENTATION_POLICY="$ROOT/config/ca1m_orientation_policy_train100_v1.json"
EXPECTED_ORIENTATION_POLICY_SHA256="e17c6388dea34ecb4f774d75d58ee8f59e997e75f1b2dfa0622acf80447bcb4f"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/build_ca1m_native_b6_train100.sh [--preflight|--run]

Default is --preflight.  It verifies exact100 frozen IDs and local tar headers,
but never starts conversion.  Explicit --run is resume-safe: existing numeric
scenes are fully audited and skipped; absent numeric scenes are built then
fully audited. Hidden/quarantine/partial artifacts are preserved and ignored.
EOF
}

mode="preflight"
case "${1:---preflight}" in
    --preflight) mode="preflight" ;;
    --run) mode="run" ;;
    --help|-h) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac
[[ "$#" -le 1 ]] || { usage >&2; exit 2; }

observed_orientation_policy_sha256="$(sha256sum "$ORIENTATION_POLICY" | awk '{print $1}')"
[[ "$observed_orientation_policy_sha256" == "$EXPECTED_ORIENTATION_POLICY_SHA256" ]] || {
    echo "CA-1M train100 orientation policy hash mismatch" >&2
    exit 2
}

exec "$PYTHON" "$ROOT/tools/build_ca1m_native_b6_train100.py" \
    --mode "$mode" \
    --subset-manifest "$MANIFEST_ROOT/subset_manifest.json" \
    --scene-ids "$MANIFEST_ROOT/scene_ids.txt" \
    --val-url-list "${BOXFUSION_CA1M_VAL_URL_LIST:-/data/ZhaoX/BoxFusion/data/val.txt}" \
    --tar-root "${BOXFUSION_CA1M_NATIVE_B6_TRAIN_TAR_ROOT:-/extra/ZhaoX/ca1m_apple_train_tars}" \
    --output-root "${BOXFUSION_CA1M_NATIVE_B6_TRAIN_SCENE_ROOT:-/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1}" \
    --report-root "${BOXFUSION_CA1M_NATIVE_B6_TRAIN100_REPORT_ROOT:-$ROOT/reports/ca1m_native_b6_train100_v1}" \
    --lock "${BOXFUSION_CA1M_NATIVE_B6_TRAIN100_LOCK:-/tmp/boxfusion_ca1m_native_b6_train100_v1.lock}" \
    --python "$PYTHON" \
    --builder "$ROOT/tools/build_ca1m_native_b6_train_scene.py" \
    --auditor "$ROOT/tools/audit_ca1m_native_b6_train_scene.py" \
    --orientation-policy "$ORIENTATION_POLICY"
