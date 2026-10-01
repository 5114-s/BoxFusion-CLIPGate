#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PY="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"
SCENE_LIST="$ROOT/manifests/ca1m_native_b6_train100_v1/scene_ids.txt"
DATA_ROOT="/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1"
PROPOSAL_ROOT="$ROOT/diagnostics/ca1m_tr3d_terminal_ca_native_train100_v4/proposals"
OVERLAY_ROOT="$ROOT/diagnostics/ca1m_tr3d_terminal_ca_native_train100_v4/overlays"
OVERLAY_MANIFEST="$ROOT/reports/ca1m_tr3d_terminal_ca_native_train100_v4/overlay_collection_manifest_v2.json"
EVIDENCE_ROOT="$ROOT/diagnostics/ca1m_tr3d_benefit_gate_final_base_v4/candidate_evidence"
MANIFEST="$ROOT/reports/ca1m_tr3d_benefit_gate_final_base_v4/candidate_evidence_manifest.json"
TOOL="$ROOT/tools/build_ca1m_tr3d_candidate_evidence_v4.py"

[[ -x "$PY" ]] || { echo "missing Python: $PY" >&2; exit 2; }
[[ -f "$SCENE_LIST" && ! -L "$SCENE_LIST" ]] || exit 2
[[ "$(sha256sum "$SCENE_LIST" | awk '{print $1}')" == \
  "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd" ]] || exit 2
[[ "$(sed '/^[[:space:]]*$/d' "$SCENE_LIST" | wc -l)" == "100" ]] || exit 2
[[ -d "$PROPOSAL_ROOT" && -d "$OVERLAY_ROOT" && -d "$DATA_ROOT" ]] || exit 2
[[ -f "$OVERLAY_MANIFEST" && ! -L "$OVERLAY_MANIFEST" ]] || exit 2
[[ "$(sha256sum "$OVERLAY_MANIFEST" | awk '{print $1}')" == \
  "a34c820f8338c80fa974933fe53435e28f87936fce3efad5630f0f3afc18cd1d" ]] || exit 2
[[ ! -e "$MANIFEST" ]] || { echo "refusing existing manifest: $MANIFEST" >&2; exit 2; }

mkdir -p "$EVIDENCE_ROOT"
while IFS= read -r scene; do
  [[ -n "$scene" ]] || continue
  target="$EVIDENCE_ROOT/${scene}_ca1m_tr3d_candidate_evidence_v4.npz"
  if [[ -e "$target" ]]; then
    continue
  fi
  env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" "$TOOL" --collect \
    --scene-list "$SCENE_LIST" --scene "$scene" --data-root "$DATA_ROOT" \
    --proposal-root "$PROPOSAL_ROOT" --overlay-root "$OVERLAY_ROOT" \
    --output-root "$EVIDENCE_ROOT"
done < "$SCENE_LIST"

env -u PYTHONPATH PYTHONDONTWRITEBYTECODE=1 "$PY" "$TOOL" --seal \
  --scene-list "$SCENE_LIST" --proposal-root "$PROPOSAL_ROOT" \
  --overlay-root "$OVERLAY_ROOT" --output-root "$EVIDENCE_ROOT" \
  --overlay-collection-manifest "$OVERLAY_MANIFEST" \
  --manifest "$MANIFEST"
echo "CANDIDATE_EVIDENCE_V4_EXIT=0"
