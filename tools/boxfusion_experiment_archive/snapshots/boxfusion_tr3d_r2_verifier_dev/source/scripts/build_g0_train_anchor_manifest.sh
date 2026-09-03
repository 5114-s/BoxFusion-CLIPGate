#!/usr/bin/env bash
set -euo pipefail

# Freeze the audited train-only G0 + Selective-Boxer prediction tree.  The
# source collection was deliberately never evaluated, so the numeric metric
# slots required by frozen_anchor_manifest.v1 are placeholders and are
# explicitly marked as non-metrics in metadata.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE_ROOT="${BOXFUSION_G0_TRAIN_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_sgcdet_dev}"
RESULT_ROOT="${BOXFUSION_G0_TRAIN_RESULT_ROOT:-$SOURCE_ROOT/results/b6_g0_sgcdet_train/g0_sgcdet_sparse_observer_train_v1/scannetv2_train_b6_100-c83575e05df2}"
COLLECTION_MANIFEST="${BOXFUSION_G0_TRAIN_COLLECTION_MANIFEST:-$SOURCE_ROOT/manifests/b6_g0_sgcdet_train/g0_sgcdet_sparse_observer_train_v1/scannetv2_train_b6_100-c83575e05df2/collection_manifest.json}"
SCENE_LIST="${BOXFUSION_G0_TRAIN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt}"
VAL_SCENE_LIST="${BOXFUSION_G0_FORBIDDEN_SCENE_LIST:-$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt}"
OUTPUT="${BOXFUSION_G0_TRAIN_MANIFEST:-$ROOT/manifests/frozen_g0_selective_boxer_train100.json}"
PYTHON_BIN="${BOXFUSION_TR3D_CONTROL_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"

for path in \
  "$PYTHON_BIN" "$RESULT_ROOT" "$COLLECTION_MANIFEST" \
  "$SCENE_LIST" "$VAL_SCENE_LIST"; do
  [[ -e "$path" ]] || {
    echo "Missing train-anchor input: $path" >&2
    exit 2
  }
done

# Bind the result directory to the already audited collection manifest before
# signing a second immutable manifest.  This also proves exact train/val
# disjointness and that evaluation was skipped.
"$PYTHON_BIN" - \
  "$RESULT_ROOT" "$COLLECTION_MANIFEST" "$SCENE_LIST" "$VAL_SCENE_LIST" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import sys


def fail(message: str) -> None:
    raise SystemExit(f"train-anchor preflight refused: {message}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenes(path: Path) -> list[str]:
    values = [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(values) != 100 or len(set(values)) != 100:
        fail(f"{path} must contain exactly 100 unique scenes")
    if any(re.fullmatch(r"scene\d{4}_\d{2}", value) is None for value in values):
        fail(f"{path} contains an invalid ScanNet scene id")
    return values


result_root = Path(sys.argv[1]).resolve()
collection_path = Path(sys.argv[2]).resolve()
scene_list = Path(sys.argv[3]).resolve()
val_scene_list = Path(sys.argv[4]).resolve()
train = scenes(scene_list)
validation = {
    line.strip().split()[0]
    for line in val_scene_list.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
}
overlap = sorted(set(train) & validation)
if overlap:
    fail(f"train/validation overlap: {overlap[:8]}")

collection = json.loads(collection_path.read_text(encoding="utf-8"))
expected = {
    "schema": "boxfusion.g0_sgcdet_train_collection_manifest.v1",
    "scene_count": 100,
    "skip_evaluation": True,
    "profile": "sgcdet_sparse_observer",
    "proposal_cache_mode": "disabled",
    "full_output_identity_scenes": 100,
}
for key, value in expected.items():
    if collection.get(key) != value:
        fail(f"collection {key}={collection.get(key)!r}, expected {value!r}")
if collection.get("prediction_rows") != collection.get("full_output_identity_rows"):
    fail("collection does not prove full prediction-output identity")
if collection.get("scene_list_sha256") != sha256_file(scene_list):
    fail("collection scene-list hash disagrees with the requested train100 list")
if collection.get("forbidden_scene_list_sha256") != sha256_file(val_scene_list):
    fail("collection forbidden-list hash disagrees with the frozen val100 list")

expected_names = {f"{scene}_boxes.pkl" for scene in train}
actual_paths = sorted(result_root.glob("scene*_boxes.pkl"))
if {path.name for path in actual_paths} != expected_names:
    fail("prediction directory is missing or has extra scene box files")
digest = hashlib.sha256()
for path in actual_paths:
    digest.update(str(path.relative_to(result_root)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(sha256_file(path).encode("ascii"))
    digest.update(b"\n")
if digest.hexdigest() != collection.get("prediction_bundle_sha256"):
    fail("prediction bundle hash disagrees with the collection manifest")
PY

"$PYTHON_BIN" "$ROOT/tools/build_frozen_anchor_manifest.py" \
  --anchor-name G0-Selective-Boxer-Train100 \
  --reference-root "$RESULT_ROOT" \
  --scene-list "$SCENE_LIST" \
  --artifact "collection_manifest=$COLLECTION_MANIFEST" \
  --ap15 0 \
  --ap25 0 \
  --ap50 0 \
  --metadata-json '{"profile":"g0_selective_boxer/train100","class_agnostic":true,"train_only":true,"ap_evaluation_status":"not_evaluated_train_only","anchor_metrics_percent_semantics":"required_schema_placeholders_not_measurements","source_collection_schema":"boxfusion.g0_sgcdet_train_collection_manifest.v1"}' \
  --required-scene-count 100 \
  --output "$OUTPUT"

"$PYTHON_BIN" "$ROOT/tools/verify_frozen_anchor_manifest.py" \
  --manifest "$OUTPUT"
"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json
from pathlib import Path
import sys

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metadata = payload.get("metadata", {})
if metadata.get("ap_evaluation_status") != "not_evaluated_train_only":
    raise SystemExit("train anchor AP provenance is not explicitly train-only")
if any(float(value) != 0.0 for value in payload["anchor_metrics_percent"].values()):
    raise SystemExit("train anchor schema placeholders changed unexpectedly")
PY

echo "Frozen train100 G0 anchor: $OUTPUT"
echo "AP status: not_evaluated_train_only (numeric fields are schema placeholders)"
