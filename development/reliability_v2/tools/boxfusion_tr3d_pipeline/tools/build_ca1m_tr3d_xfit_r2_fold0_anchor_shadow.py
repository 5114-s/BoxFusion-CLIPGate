#!/usr/bin/env python3
"""Create the GT-free fold0 final-base + B6-v2 OOF anchor shadow.

The source benefit dataset contains folds 0/2/3/4 and was already sealed by
the prior terminal-v4 experiment.  This extractor opens only its baseline
identity fields; candidate and GT-target members are never requested.  The
result is an exact fold0-only input for the later R2 oracle evaluation and has
no fold1 row or official-validation dependency.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal import write_npz_create_only  # noqa: E402
from boxfusion.ca1m_tr3d_xfit_r2_eval import (  # noqa: E402
    FOLD0_SHA256,
    create_or_verify_json,
    regular_file,
    sha256_file,
)


SOURCE = ROOT / "datasets/ca1m_tr3d_benefit_final_base_v4.npz"
SOURCE_SHA256 = "8e234d727d25c5448924814d0a04d87af2acb9963af5c39b700dacde03d19b32"
SOURCE_MANIFEST = ROOT / "datasets/ca1m_tr3d_benefit_final_base_v4.manifest.json"
SOURCE_MANIFEST_SHA256 = "9873e2be71fd4f1f24675042af54543b47e48bdbd15617fb6b16b0c14786a76b"
OOF = ROOT / "models/ca1m_native_b6_final_base_oof_row_scores_v2.npz"
OOF_SHA256 = "82b5e70c635958398c04b0e3ba5dbf25203b61bcabd725330bd68812d156e5ed"
FINAL_BASE = ROOT / "reports/ca1m_native_final_base_train100_v1/collection_manifest.json"
FINAL_BASE_SHA256 = "110dac18eb436eb6735299141d77fa00d4fed6042f93de81e5288e6b5d0e2f52"
SCENE_LIST = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/"
    "tr3d_ca1m_visible_xfit_v2_formal/splits/predict_fold0.txt"
)
OUTPUT = ROOT / "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/fold0_final_base_b6_v2_oof.npz"
MANIFEST = OUTPUT.with_suffix(".manifest.json")
SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_fold0_final_base_b6_oof.v1"


def _bound(path: Path, expected: str, name: str) -> Path:
    source = regular_file(path, name)
    if sha256_file(source) != expected:
        raise ValueError(f"{name} SHA256 drift")
    return source


def build() -> dict:
    source = _bound(SOURCE, SOURCE_SHA256, "terminal-v4 benefit dataset")
    source_manifest = _bound(
        SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256, "terminal-v4 dataset manifest"
    )
    oof = _bound(OOF, OOF_SHA256, "native-B6-v2 OOF sidecar")
    final_base = _bound(FINAL_BASE, FINAL_BASE_SHA256, "final-base manifest")
    scene_list = _bound(SCENE_LIST, FOLD0_SHA256, "fold0 scene list")
    scenes = tuple(row.strip() for row in scene_list.read_text().splitlines() if row.strip())
    if len(scenes) != 20 or len(set(scenes)) != 20:
        raise ValueError("fold0 scene list differs")

    # np.load is lazy for NPZ.  Do not request any candidate, target, best-GT,
    # best-IoU, or GT-count member from this source archive.
    opened_members = (
        "schema", "complete", "baseline_scene_ids", "baseline_fold_ids",
        "baseline_row_indices", "baseline_corners", "baseline_scores",
    )
    with np.load(source, allow_pickle=False) as archive:
        if str(np.asarray(archive["schema"]).item()) != (
            "boxfusion.ca1m_tr3d_benefit_gate_dataset.v4"
        ) or bool(np.asarray(archive["complete"]).item()) is not True:
            raise ValueError("source terminal-v4 dataset schema differs")
        scene_ids = np.asarray(archive["baseline_scene_ids"]).astype(str)
        folds = np.asarray(archive["baseline_fold_ids"], dtype=np.int8)
        row_indices = np.asarray(archive["baseline_row_indices"], dtype=np.int64)
        corners = np.asarray(archive["baseline_corners"], dtype=np.float32)
        scores = np.asarray(archive["baseline_scores"], dtype=np.float32)
    if (
        set(np.unique(folds).tolist()) != {0, 2, 3, 4}
        or 1 in set(folds.tolist())
        or scene_ids.shape != folds.shape
        or row_indices.shape != folds.shape
        or corners.shape != (len(folds), 8, 3)
        or scores.shape != folds.shape
        or not np.isfinite(corners).all()
        or not np.isfinite(scores).all()
    ):
        raise ValueError("source baseline projection differs")
    keep = folds == 0
    output_scenes = scene_ids[keep]
    output_rows = row_indices[keep]
    output_corners = np.ascontiguousarray(corners[keep])
    output_scores = np.ascontiguousarray(scores[keep])
    if (
        len(output_scenes) != 1505
        or set(output_scenes.tolist()) != set(scenes)
        or np.any(output_scores < 0.0)
        or np.any(output_scores > 1.0)
    ):
        raise ValueError("fold0 final-base/B6 projection differs")
    for scene in scenes:
        rows = output_rows[output_scenes == scene]
        if not np.array_equal(rows, np.arange(len(rows), dtype=np.int64)):
            raise ValueError(f"{scene}: final-base row identity differs")
    payload = {
        "schema": np.asarray(SCHEMA),
        "complete": np.asarray(True, dtype=np.bool_),
        "train_only": np.asarray(True, dtype=np.bool_),
        "fold0_only": np.asarray(True, dtype=np.bool_),
        "fold_id": np.asarray(0, dtype=np.int8),
        "fold1_access": np.asarray(False, dtype=np.bool_),
        "official_validation_access": np.asarray(False, dtype=np.bool_),
        "ground_truth_access": np.asarray(False, dtype=np.bool_),
        "candidate_access": np.asarray(False, dtype=np.bool_),
        "each_score_model_excludes_scene": np.asarray(True, dtype=np.bool_),
        "scene_ids": np.asarray(output_scenes, dtype=np.str_),
        "row_indices": output_rows,
        "anchor_corners": output_corners,
        "b6_oof_scores": output_scores,
        "source_dataset_sha256": np.asarray(SOURCE_SHA256),
        "source_dataset_manifest_sha256": np.asarray(SOURCE_MANIFEST_SHA256),
        "oof_sidecar_sha256": np.asarray(OOF_SHA256),
        "final_base_manifest_sha256": np.asarray(FINAL_BASE_SHA256),
        "fold0_scene_list_sha256": np.asarray(FOLD0_SHA256),
    }
    if OUTPUT.exists() or OUTPUT.is_symlink():
        with np.load(regular_file(OUTPUT, "fold0 anchor shadow", immutable=True), allow_pickle=False) as archive:
            current = {name: np.array(archive[name], copy=True) for name in archive.files}
        if set(current) != set(payload) or any(
            not np.array_equal(current[name], value) for name, value in payload.items()
        ):
            raise FileExistsError("refusing to replace differing fold0 anchor shadow")
    else:
        write_npz_create_only(OUTPUT, payload)
    manifest = {
        "schema": f"{SCHEMA}.manifest",
        "complete": True,
        "create_only": True,
        "train_only": True,
        "fold0_only": True,
        "fold_id": 0,
        "scene_count": 20,
        "row_count": len(output_scenes),
        "fold1_access": False,
        "official_validation_access": False,
        "ground_truth_access": False,
        "candidate_access": False,
        "source_npz_members_opened": list(opened_members),
        "source_gt_or_candidate_members_opened": False,
        "score_source": "ca1m_native_b6_final_base_all_fold_oof_v2_fold0_rows",
        "each_score_model_excludes_scene": True,
        "artifact": {
            "path": str(OUTPUT.resolve()), "sha256": sha256_file(OUTPUT),
            "schema": SCHEMA,
        },
        "sources": {
            "terminal_v4_dataset": {"path": str(source), "sha256": SOURCE_SHA256},
            "terminal_v4_dataset_manifest": {
                "path": str(source_manifest), "sha256": SOURCE_MANIFEST_SHA256,
            },
            "native_b6_v2_oof": {"path": str(oof), "sha256": OOF_SHA256},
            "final_base": {"path": str(final_base), "sha256": FINAL_BASE_SHA256},
            "fold0_scene_list": {"path": str(scene_list), "sha256": FOLD0_SHA256},
        },
    }
    create_or_verify_json(MANIFEST, manifest, "fold0 anchor shadow manifest")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
