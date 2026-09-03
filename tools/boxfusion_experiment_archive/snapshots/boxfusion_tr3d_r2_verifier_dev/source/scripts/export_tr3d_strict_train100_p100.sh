#!/usr/bin/env bash
set -euo pipefail

# Create one immutable p100 trajectory-prefix export for the audited train100
# split.  This launcher has no resume/overwrite mode: mkdir atomically claims
# the namespace, and a failed export remains visibly quarantined.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT"

PYTHON_BIN="${BOXFUSION_TR3D_EXPORT_PYTHON:-/home/admin1/miniconda3/envs/boxfusion2/bin/python}"
TRAIN_ASSET_ROOT="${BOXFUSION_G0_TRAIN_ROOT:-/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_sgcdet_dev}"
OUTPUT_ROOT="$ROOT/data/tr3d_prefix_train100_boxfusion_causal_p100_v1"
FRAMES_ROOT="${BOXFUSION_SCANNET_TRAIN_FRAMES_ROOT:-$TRAIN_ASSET_ROOT/data/scannet_train}"
TRAIN_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_train_b6_100.txt"
VAL_SCENE_LIST="$ROOT/evaluation/data_util/meta_data/scannetv2_val.txt"
SOURCE_INFO="/extra/ZhaoX/scannet_data/scannet_infos_train.pkl"
SOURCE_POINTS="/extra/ZhaoX/scannet_data/points"
STEM="trajectory_prefix_train100_boxfusion_causal_p100_v1"
MANIFEST="$OUTPUT_ROOT/manifests/$STEM.jsonl"
SUMMARY="$OUTPUT_ROOT/manifests/$STEM.summary.json"
INFO="$OUTPUT_ROOT/annotations/scannet_infos_prefix_train100_boxfusion_causal_p100_v1.pkl"

for path in \
  "$PYTHON_BIN" "$FRAMES_ROOT" "$TRAIN_SCENE_LIST" "$VAL_SCENE_LIST" \
  "$SOURCE_INFO" "$SOURCE_POINTS"; do
  [[ -e "$path" ]] || {
    echo "Missing strict train-export input: $path" >&2
    exit 2
  }
done

audit_train_inputs() {
  "$PYTHON_BIN" - \
    "$ROOT" "$FRAMES_ROOT" "$TRAIN_SCENE_LIST" "$VAL_SCENE_LIST" \
    "$SOURCE_INFO" <<'PY'
from pathlib import Path
import sys

root, frames, train_path, val_path, source_info = map(Path, sys.argv[1:])
sys.path.insert(0, str(root.resolve()))
from tools.tr3d_data import index_info_rows, load_info, read_scene_list

train = read_scene_list(train_path.resolve())
validation = set(read_scene_list(val_path.resolve()))
if len(train) != 100:
    raise SystemExit(f"strict train export requires exactly 100 scenes, found {len(train)}")
overlap = sorted(set(train) & validation)
if overlap:
    raise SystemExit(f"strict train export overlaps validation: {overlap[:8]}")
missing_frames = [scene for scene in train if not (frames / scene).is_dir()]
if missing_frames:
    raise SystemExit(f"train RGB-D root is missing scenes: {missing_frames[:8]}")
_, rows = load_info(source_info.resolve())
source_scenes = set(index_info_rows(rows))
missing_info = sorted(set(train) - source_scenes)
if missing_info:
    raise SystemExit(f"scannet_infos_train.pkl is missing scenes: {missing_info[:8]}")
PY
}

audit_complete_export() {
  for path in "$MANIFEST" "$SUMMARY" "$INFO"; do
    [[ -f "$path" ]] || return 1
  done
  "$PYTHON_BIN" - \
    "$ROOT" "$OUTPUT_ROOT" "$FRAMES_ROOT" "$TRAIN_SCENE_LIST" \
    "$VAL_SCENE_LIST" "$MANIFEST" "$SUMMARY" "$INFO" <<'PY'
from pathlib import Path
import json
import sys

root, output, frames, train_path, val_path, manifest_path, summary_path, info_path = (
    map(Path, sys.argv[1:])
)
sys.path.insert(0, str(root.resolve()))
from tools.tr3d_data import load_info, read_scene_list, scene_id_from_info

output = output.resolve()
frames = frames.resolve()
train = read_scene_list(train_path.resolve())
validation = set(read_scene_list(val_path.resolve()))
if len(train) != 100 or set(train) & validation:
    raise SystemExit("train100 scene scope/count contract failed")

rows = [
    json.loads(line)
    for line in manifest_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != 100 or [row.get("scene_id") for row in rows] != train:
    raise SystemExit("prefix manifest must contain one ordered row per train100 scene")
expected_points = set()
for scene, row in zip(train, rows):
    expected = output / "points" / "prefixes" / scene / f"{scene}__p100.bin"
    if row.get("schema") != "boxfusion.tr3d.trajectory_prefix.v1":
        raise SystemExit(f"{scene}: wrong prefix schema")
    if row.get("status") != "exported" or row.get("tag") != "p100":
        raise SystemExit(f"{scene}: prefix is not an exported p100 row")
    if float(row.get("fraction", -1)) != 1.0:
        raise SystemExit(f"{scene}: prefix fraction is not 1.0")
    if row.get("clock_policy") != "g0_post_frame_tail_guard_v1":
        raise SystemExit(f"{scene}: BoxFusion clock policy drifted")
    if row.get("pose_policy") != "previous_valid_inf_only_v1":
        raise SystemExit(f"{scene}: pose policy drifted")
    if Path(row.get("source_frames_root", "")).resolve() != frames:
        raise SystemExit(f"{scene}: manifest did not use the locked train frame root")
    point = Path(row.get("point_path", "")).resolve()
    if point != expected or not point.is_file() or point.stat().st_size <= 0:
        raise SystemExit(f"{scene}: missing/misdirected p100 point artifact")
    if not isinstance(row.get("point_count"), int) or row["point_count"] <= 0:
        raise SystemExit(f"{scene}: invalid point count")
    expected_points.add(expected)
actual_points = {path.resolve() for path in (output / "points").rglob("*.bin")}
if actual_points != expected_points:
    raise SystemExit("strict export point-file set is missing or has extras")

_, info_rows = load_info(info_path.resolve())
if len(info_rows) != 100 or [scene_id_from_info(row) for row in info_rows] != train:
    raise SystemExit("strict export info rows disagree with ordered train100 scenes")
for scene, row in zip(train, info_rows):
    relative = row.get("lidar_points", {}).get("lidar_path")
    if relative != f"prefixes/{scene}/{scene}__p100.bin":
        raise SystemExit(f"{scene}: unexpected info lidar path {relative!r}")

summary = json.loads(summary_path.read_text(encoding="utf-8"))
required = {
    "scene_count_requested": 100,
    "scene_count_exported": 100,
    "prefix_count": 100,
    "annotation_row_count": 100,
    "manifest_only": False,
    "clock_policy": "g0_post_frame_tail_guard_v1",
    "pose_policy": "previous_valid_inf_only_v1",
}
for key, value in required.items():
    if summary.get(key) != value:
        raise SystemExit(f"summary {key}={summary.get(key)!r}, expected {value!r}")
if summary.get("errors") != []:
    raise SystemExit("strict train export recorded scene errors")
if Path(summary.get("scene_list", "")).resolve() != train_path.resolve():
    raise SystemExit("summary scene list is not the frozen train100 list")
PY
}

audit_train_inputs
if [[ -e "$OUTPUT_ROOT" ]]; then
  if audit_complete_export; then
    echo "Strict train100 p100 export already complete and content-audited."
    echo "No RGB-D decoding or artifact write was performed."
    exit 0
  fi
  echo "Refusing partial/external strict train-export root: $OUTPUT_ROOT" >&2
  echo "This create-only launcher never overwrites or resumes a visible namespace." >&2
  exit 2
fi

# Atomic namespace claim.  A concurrent launcher can only win or lose this
# mkdir; it can never share files with this process.
mkdir "$OUTPUT_ROOT"

echo "Creating strict BoxFusion-clock train100 p100 export"
echo "  output: $OUTPUT_ROOT"
echo "  scenes: 100 train-only scenes; validation overlap=0"
echo "  frames: $FRAMES_ROOT"
echo "  source info: $SOURCE_INFO"
echo "  clock: gap=25, post-frame tail guard"
echo "  pose: previous-valid carry-forward for infinity only"
echo "  RGB-D: pixel_stride=4, depth=[0.1,6.0]m, voxel=0.01m"
echo "  policy: atomic namespace claim, create-only, partial failures quarantined"

"$PYTHON_BIN" "$ROOT/tools/export_tr3d_trajectory_prefixes.py" \
  --prepared-root "$OUTPUT_ROOT" \
  --frames-root "$FRAMES_ROOT" \
  --scene-list "$TRAIN_SCENE_LIST" \
  --source-info "$SOURCE_INFO" \
  --source-points "$SOURCE_POINTS" \
  --output-info-name "$(basename "$INFO")" \
  --manifest-name "$(basename "$MANIFEST")" \
  --fractions 1.0 \
  --frame-stride 25 \
  --pixel-stride 4 \
  --voxel-size 0.01 \
  --depth-scale 1000 \
  --min-depth 0.1 \
  --max-depth 6.0 \
  --min-observed-points 20 \
  --min-visibility-fraction 0.0

audit_complete_export
echo "Strict train100 p100 export complete and content-audited."
