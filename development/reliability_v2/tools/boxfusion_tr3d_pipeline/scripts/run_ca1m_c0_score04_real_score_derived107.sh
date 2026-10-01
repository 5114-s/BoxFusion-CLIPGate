#!/usr/bin/env bash
set -euo pipefail

# Evaluate a deliberately non-canonical 107-scene CA-1M overlay.  This entry
# point reuses the 103 corrected-C0 predictions byte-for-byte and runs BoxFusion
# only for the four scenes whose evaluation GT had to be derived locally.
#
# IMPORTANT: this protocol is useful as an internal 107-scene diagnostic only.
# Its metrics are not comparable to the paper's official 107-scene result.

GPU_SPEC="${1:-0,1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
PYTHON_BIN="${BOXFUSION_PYTHON:-/home/admin1/miniconda3/envs/boxfusion-online/bin/python}"

FULL_LIST="$ROOT/evaluation/data_util/meta_data/ca1m_val_full107.txt"
CANONICAL_LIST="$ROOT/evaluation/data_util/meta_data/ca1m_val_canonical103.txt"
DERIVED_LIST="$ROOT/evaluation/data_util/meta_data/ca1m_missing_canonical_gt4.txt"
CONFIG="$ROOT/config/ca1m_c0_score04_real_score_derived107.yaml"

CANONICAL_DATA_ROOT="${BOXFUSION_CA1M_CANONICAL_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m}"
DERIVED_DATA_ROOT="${BOXFUSION_CA1M_DERIVED107_DATA_ROOT:-/extra/ZhaoX/boxfusion_ca1m_derived107_v1}"
DERIVED_GT_MANIFEST="${BOXFUSION_CA1M_DERIVED107_GT_MANIFEST:-$DERIVED_DATA_ROOT/derived_gt_manifest.json}"
CANONICAL_PRED_ROOT="${BOXFUSION_CA1M_CANONICAL103_PRED_ROOT:-$ROOT/results/ca1m_repro/c0_score04_real_score_canonical103_v1}"

TAG="${BOXFUSION_CA1M_DERIVED107_RUN_TAG:-c0_score04_real_score_derived107_v1}"
PRED_ROOT="${BOXFUSION_CA1M_DERIVED107_PRED_ROOT:-$ROOT/results/ca1m_repro/$TAG}"
LOG_ROOT="${BOXFUSION_CA1M_DERIVED107_LOG_ROOT:-$ROOT/logs/ca1m_repro/$TAG}"
REPORT_ROOT="${BOXFUSION_CA1M_DERIVED107_REPORT_ROOT:-$ROOT/reports/ca1m_repro/$TAG}"
EVAL_VIEW="${BOXFUSION_CA1M_DERIVED107_EVAL_VIEW:-$ROOT/data/ca1m_eval_derived107_v1}"
RUNTIME_TMP_ROOT="${BOXFUSION_CA1M_DERIVED107_RUNTIME_TMP_ROOT:-/extra/ZhaoX/boxfusion_runtime_tmp/$TAG}"
# Keep this short: the CA-1M evaluator/Open3D stack can create AF_UNIX paths.
EVAL_TMPDIR="${BOXFUSION_CA1M_DERIVED107_EVAL_TMPDIR:-/tmp/bf_ca1m_d107_eval}"

FULL_LIST_SHA256="bd5f3fc66168114048a1b12addc45949c8f54f9c016b921bacfb6fe9e3e7dc2f"
CANONICAL_LIST_SHA256="c3efbe544c7403acc4183d7e4a799dad2bb40f60cbdba38830863f8712f4648f"
DERIVED_LIST_SHA256="582ec52e296fa907a79eb01f5778b0adea368b3c7ca61e3a972aca42f32d401b"
CONFIG_SHA256="4ebb0ebea8d2f99f2fdfa0b5493be812268700575e3887dc0e62509bb043a93f"

for path in "$PYTHON_BIN" "$FULL_LIST" "$CANONICAL_LIST" "$DERIVED_LIST" \
    "$CONFIG" "$DERIVED_GT_MANIFEST"; do
    [[ -f "$path" ]] || { echo "Missing derived107 input: $path" >&2; exit 2; }
done
[[ "$(sha256sum "$FULL_LIST" | awk '{print $1}')" == "$FULL_LIST_SHA256" ]] || {
    echo "Frozen CA-1M full107 list differs" >&2; exit 2;
}
[[ "$(sha256sum "$CANONICAL_LIST" | awk '{print $1}')" == "$CANONICAL_LIST_SHA256" ]] || {
    echo "Frozen CA-1M canonical103 list differs" >&2; exit 2;
}
[[ "$(sha256sum "$DERIVED_LIST" | awk '{print $1}')" == "$DERIVED_LIST_SHA256" ]] || {
    echo "Frozen CA-1M derived-four list differs" >&2; exit 2;
}
[[ "$(sha256sum "$CONFIG" | awk '{print $1}')" == "$CONFIG_SHA256" ]] || {
    echo "Derived107 config differs from its frozen protocol" >&2; exit 2;
}

mkdir -p "$PRED_ROOT" "$LOG_ROOT" "$REPORT_ROOT" "$RUNTIME_TMP_ROOT" "$EVAL_TMPDIR"

# Validate the overlay and its non-canonical provenance before predictions are
# linked or inference starts.  Numeric scene entries must be the exact official
# 107 set: 103 symlinks into the canonical live root and four physical derived
# directories.  The aggregate manifest and every derived scene must explicitly
# deny official comparability and paper claims.
"$PYTHON_BIN" - \
    "$DERIVED_DATA_ROOT" "$CANONICAL_DATA_ROOT" \
    "$FULL_LIST" "$CANONICAL_LIST" "$DERIVED_LIST" \
    "$DERIVED_GT_MANIFEST" "$REPORT_ROOT/derived_overlay_audit.json" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

(
    overlay_root, canonical_root, full_list, canonical_list, derived_list,
    manifest_path, report_path,
) = map(Path, sys.argv[1:])

def rows(path: Path) -> tuple[str, ...]:
    value = tuple(x.strip() for x in path.read_text().splitlines() if x.strip())
    if len(value) != len(set(value)):
        raise ValueError(f"duplicate rows in {path}")
    return value

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

full = rows(full_list)
canonical = rows(canonical_list)
derived = rows(derived_list)
if len(full) != 107 or len(canonical) != 103 or len(derived) != 4:
    raise ValueError("derived107 scene partitions must be exactly 107=103+4")
if set(full) != set(canonical) | set(derived) or set(canonical) & set(derived):
    raise ValueError("derived107 scene partitions overlap or do not cover full107")
if not overlay_root.is_dir() or overlay_root.is_symlink():
    raise ValueError(f"invalid derived107 overlay root: {overlay_root}")

numeric = {
    entry.name for entry in overlay_root.iterdir()
    if entry.name.isdigit() and (entry.is_dir() or entry.is_symlink())
}
if numeric != set(full):
    raise ValueError(
        f"derived107 overlay exact-set mismatch: missing={sorted(set(full)-numeric)}, "
        f"unexpected={sorted(numeric-set(full))}"
    )

required = ("K_depth.txt", "K_rgb.txt", "all_poses.npy", "after_filter_boxes.npy")
canonical_rows = []
for scene in canonical:
    entry = overlay_root / scene
    target = canonical_root / scene
    if not entry.is_symlink() or entry.resolve() != target.resolve():
        raise ValueError(f"{scene}: canonical overlay entry is not the exact canonical symlink")
    for name in required:
        if not (entry / name).is_file():
            raise ValueError(f"{scene}: missing canonical overlay file {name}")
    canonical_rows.append({"scene_id": scene, "target": str(target.resolve())})

derived_rows = []
for scene in derived:
    entry = overlay_root / scene
    if entry.is_symlink() or not entry.is_dir():
        raise ValueError(f"{scene}: derived scene must be a physical directory")
    for name in required:
        path = entry / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"{scene}: missing/non-regular derived file {name}")
    scene_manifest = entry / "derived_gt_manifest.json"
    if not scene_manifest.is_file() or scene_manifest.is_symlink():
        raise ValueError(f"{scene}: missing regular per-scene provenance manifest")
    payload = json.loads(scene_manifest.read_text())
    if payload.get("scene_id") != scene:
        raise ValueError(f"{scene}: per-scene manifest ID mismatch")
    if payload.get("derived") is not True:
        raise ValueError(f"{scene}: per-scene manifest must state derived=true")
    if payload.get("official_comparable") is not False:
        raise ValueError(f"{scene}: per-scene manifest must deny official comparability")
    if payload.get("paper_claim_permitted") is not False:
        raise ValueError(f"{scene}: per-scene manifest must deny paper claims")
    derived_rows.append({
        "scene_id": scene,
        "manifest": str(scene_manifest.resolve()),
        "manifest_sha256": sha256(scene_manifest),
        "artifact_sha256": {name: sha256(entry / name) for name in required},
    })

aggregate = json.loads(manifest_path.read_text())
if aggregate.get("schema") != "boxfusion.ca1m_derived107_overlay.v1":
    raise ValueError("aggregate manifest schema is not the frozen derived107 schema")
if aggregate.get("derived") is not True:
    raise ValueError("aggregate manifest must state derived=true")
if aggregate.get("official_comparable") is not False:
    raise ValueError("aggregate manifest must deny official comparability")
if aggregate.get("paper_claim_permitted") is not False:
    raise ValueError("aggregate manifest must deny paper claims")
declared = aggregate.get("derived_gt_scenes")
if isinstance(declared, int):
    if declared != 4:
        raise ValueError("aggregate manifest derived_gt_scenes must equal 4")
elif set(str(x) for x in (declared or ())) != set(derived):
    raise ValueError("aggregate manifest derived_gt_scenes does not match frozen four")
if int(aggregate.get("canonical_scene_count", -1)) != 103:
    raise ValueError("aggregate manifest canonical_scene_count must equal 103")
if int(aggregate.get("derived_scene_count", -1)) != 4:
    raise ValueError("aggregate manifest derived_scene_count must equal 4")
scene_manifests = aggregate.get("scene_manifests")
if not isinstance(scene_manifests, dict) or set(scene_manifests) != set(derived):
    raise ValueError("aggregate manifest scene_manifests must cover the frozen four")

report = {
    "schema": "boxfusion.ca1m_derived107_overlay_audit.v1",
    "ok": True,
    "derived": True,
    "official_comparable": False,
    "paper_claim_permitted": False,
    "overlay_root": str(overlay_root.resolve()),
    "aggregate_manifest": str(manifest_path.resolve()),
    "aggregate_manifest_sha256": sha256(manifest_path),
    "scene_count": 107,
    "canonical_scene_symlinks": 103,
    "derived_gt_scenes": list(derived),
    "canonical_scenes": canonical_rows,
    "derived_scene_provenance": derived_rows,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=report_path.parent, delete=False) as handle:
    json.dump(report, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(report_path)
print(json.dumps({
    "ok": True, "scene_count": 107, "canonical_scene_symlinks": 103,
    "derived_gt_scenes": list(derived), "official_comparable": False,
}, indent=2))
PY

# The canonical source must itself remain an exact, valid real-score set.
"$PYTHON_BIN" "$ROOT/tools/audit_ca1m_c0_predictions.py" \
    --scene-list "$CANONICAL_LIST" --prediction-root "$CANONICAL_PRED_ROOT" \
    --output "$REPORT_ROOT/canonical103_source_prediction_audit.json" \
    --require-real-score

# Fail on unrelated prediction files.  Missing derived-four files are expected
# on a first run and will be produced by the base runner below.
unexpected_predictions="$({
    find "$PRED_ROOT" -maxdepth 1 -type f -name '*_boxes.pkl' -printf '%f\n' | sort
    sed 's/$/_boxes.pkl/' "$FULL_LIST" | sort
} | sort | uniq -u | head -n 1)"
if [[ -n "$unexpected_predictions" ]]; then
    # The symmetric-difference expression also contains expected-but-missing
    # names, so independently reject only files outside full107.
    actual_list="$REPORT_ROOT/.actual_predictions.$$"
    expected_list="$REPORT_ROOT/.expected_predictions.$$"
    trap 'rm -f "$actual_list" "$expected_list"' EXIT
    find "$PRED_ROOT" -maxdepth 1 -type f -name '*_boxes.pkl' -printf '%f\n' | sort > "$actual_list"
    sed 's/$/_boxes.pkl/' "$FULL_LIST" | sort > "$expected_list"
    outside="$(comm -23 "$actual_list" "$expected_list")"
    [[ -z "$outside" ]] || {
        echo "Unexpected predictions in derived107 root:" >&2
        echo "$outside" >&2
        exit 2
    }
    rm -f "$actual_list" "$expected_list"
    trap - EXIT
fi

LINK_AUDIT="$REPORT_ROOT/canonical103_prediction_hardlinks.tsv"
: > "$LINK_AUDIT"
while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -n "$scene" ]] || continue
    source="$CANONICAL_PRED_ROOT/${scene}_boxes.pkl"
    target="$PRED_ROOT/${scene}_boxes.pkl"
    [[ -f "$source" && ! -L "$source" ]] || {
        echo "$scene: canonical prediction is missing/non-regular" >&2; exit 2;
    }
    if [[ ! -e "$target" && ! -L "$target" ]]; then
        ln "$source" "$target"
    fi
    [[ -f "$target" && ! -L "$target" ]] || {
        echo "$scene: derived107 reused prediction is missing/non-regular" >&2; exit 2;
    }
    [[ "$source" -ef "$target" ]] || {
        echo "$scene: reused canonical prediction is not a hard link" >&2; exit 2;
    }
    source_sha="$(sha256sum "$source" | awk '{print $1}')"
    target_sha="$(sha256sum "$target" | awk '{print $1}')"
    [[ "$source_sha" == "$target_sha" ]] || {
        echo "$scene: canonical/reused prediction SHA256 mismatch" >&2; exit 2;
    }
    "$PYTHON_BIN" "$ROOT/tools/validate_ca1m_prediction_file.py" \
        --prediction "$target"
    printf '%s\t%s\t%s\n' "$scene" "$source_sha" "hardlink" >> "$LINK_AUDIT"
done < "$CANONICAL_LIST"
[[ "$(wc -l < "$LINK_AUDIT")" == "103" ]] || {
    echo "Canonical prediction hard-link audit did not cover 103 scenes" >&2; exit 2;
}

if [[ "${BOXFUSION_CA1M_DERIVED107_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    echo "[$(date '+%F %T')] Derived107 preflight passed; inference/evaluation skipped"
    echo "Overlay audit: $REPORT_ROOT/derived_overlay_audit.json"
    echo "Prediction reuse audit: $LINK_AUDIT"
    exit 0
fi

echo "[$(date '+%F %T')] Starting non-canonical derived107 evaluation"
echo "[$(date '+%F %T')] Reused canonical predictions: 103 hard links"
echo "[$(date '+%F %T')] Derived GT scenes: $(paste -sd, "$DERIVED_LIST")"
echo "[$(date '+%F %T')] official_comparable=false; paper_claim_permitted=false"

export BOXFUSION_CA1M_RUN_TAG="$TAG"
export BOXFUSION_CA1M_PROTOCOL="derived107_noncanonical"
export BOXFUSION_CA1M_EXPECTED_SCENES="107"
export BOXFUSION_CA1M_ALLOW_UNLISTED_SCENES="0"
export BOXFUSION_CA1M_DATA_ROOT="$DERIVED_DATA_ROOT"
export BOXFUSION_CA1M_SCENE_LIST="$FULL_LIST"
export BOXFUSION_CA1M_C0_CONFIG="$CONFIG"
export BOXFUSION_CA1M_EXPECTED_CONFIG_SHA256="$CONFIG_SHA256"
export BOXFUSION_CA1M_EXPECTED_SCENE_LIST_SHA256="$FULL_LIST_SHA256"
export BOXFUSION_CA1M_PRED_ROOT="$PRED_ROOT"
export BOXFUSION_CA1M_LOG_ROOT="$LOG_ROOT"
export BOXFUSION_CA1M_REPORT_ROOT="$REPORT_ROOT"
export BOXFUSION_CA1M_EVAL_VIEW="$EVAL_VIEW"
export BOXFUSION_RUNTIME_TMP_ROOT="$RUNTIME_TMP_ROOT"
export BOXFUSION_CA1M_EVAL_TMPDIR="$EVAL_TMPDIR"

bash "$ROOT/scripts/run_ca1m_c0_score04_real_score_full107.sh" "$GPU_SPEC"

# Re-audit the 103 hard links after inference/evaluation, then make the formal
# run manifest unambiguous.  Never leave the base runner's generic 107/107
# field looking like an official-GT claim.
while IFS= read -r scene || [[ -n "$scene" ]]; do
    [[ -n "$scene" ]] || continue
    [[ "$CANONICAL_PRED_ROOT/${scene}_boxes.pkl" -ef "$PRED_ROOT/${scene}_boxes.pkl" ]] || {
        echo "$scene: canonical prediction hard link changed during run" >&2; exit 2;
    }
done < "$CANONICAL_LIST"

"$PYTHON_BIN" - "$REPORT_ROOT/run_manifest.txt" "$DERIVED_GT_MANIFEST" "$LINK_AUDIT" <<'PY'
import hashlib
import os
import sys
import tempfile
from pathlib import Path

manifest, gt_manifest, link_audit = map(Path, sys.argv[1:])
if not manifest.is_file():
    raise FileNotFoundError(manifest)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

rows = []
for line in manifest.read_text().splitlines():
    if line.startswith("schema="):
        rows.append("schema=boxfusion.ca1m_c0_score04_real_score.derived107.v1")
    elif line.startswith("official_public_gt_subset="):
        rows.append("official_public_gt_subset=103/107")
    else:
        rows.append(line)
rows.extend([
    "derived=true",
    "official_comparable=false",
    "paper_claim_permitted=false",
    "derived_gt_scenes=4",
    "derived_gt_scene_ids=45663164,47115469,47331311,47332000",
    f"derived_gt_manifest={gt_manifest.resolve()}",
    f"derived_gt_manifest_sha256={sha256(gt_manifest)}",
    "canonical_predictions_reused=103",
    "canonical_prediction_reuse=hardlink_byte_exact_sha256_verified",
    f"canonical_prediction_link_audit={link_audit.resolve()}",
    f"canonical_prediction_link_audit_sha256={sha256(link_audit)}",
    "metric_label=derived107_noncanonical_internal_diagnostic",
])
with tempfile.NamedTemporaryFile("w", dir=manifest.parent, delete=False) as handle:
    handle.write("\n".join(rows) + "\n")
    temporary = Path(handle.name)
os.replace(temporary, manifest)
PY

grep -qx 'official_comparable=false' "$REPORT_ROOT/run_manifest.txt"
grep -qx 'paper_claim_permitted=false' "$REPORT_ROOT/run_manifest.txt"
grep -qx 'derived_gt_scenes=4' "$REPORT_ROOT/run_manifest.txt"

echo "[$(date '+%F %T')] Derived107 non-canonical evaluation completed"
echo "Evaluation log: $LOG_ROOT/eval_derived107_noncanonical.log"
echo "Protocol manifest: $REPORT_ROOT/run_manifest.txt"
