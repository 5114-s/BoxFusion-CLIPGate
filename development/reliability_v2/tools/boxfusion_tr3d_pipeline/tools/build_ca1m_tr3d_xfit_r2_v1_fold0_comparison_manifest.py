#!/usr/bin/env python3
"""Seal the exact fold0 subset of the v1 terminal proposal manifest.

Only immutable proposal metadata are projected.  No proposal array, anchor,
B6 artifact, GT artifact, fold1 artifact, or official-validation artifact is
opened.  The output is comparison-only and cannot authorize R2 activation.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_xfit_r2_eval import (  # noqa: E402
    FOLD0_SHA256,
    create_or_verify_json,
    regular_file,
    sha256_file,
)


SOURCE = ROOT / "reports/ca1m_tr3d_terminal_ca_native_train100_v4/proposal_collection_manifest_v5.json"
SOURCE_SHA256 = "a8a9bcbccb8212e6a346b60e3657859f06751b1d2309204919b0de725babc349"
SCENE_LIST = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/"
    "tr3d_ca1m_visible_xfit_v2_formal/splits/predict_fold0.txt"
)
OUTPUT = ROOT / "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/v1_fold0_proposal_comparison_manifest.json"
SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_v1_fold0_proposal_comparison.v1"


def build() -> dict:
    source = regular_file(SOURCE, "v1 proposal collection", immutable=True)
    scene_list = regular_file(SCENE_LIST, "fold0 scene list")
    if sha256_file(source) != SOURCE_SHA256 or sha256_file(scene_list) != FOLD0_SHA256:
        raise ValueError("v1 collection or fold0 scene-list SHA256 drift")
    scenes = tuple(row.strip() for row in scene_list.read_text().splitlines() if row.strip())
    value = json.loads(source.read_text(encoding="utf-8"))
    if (
        value.get("schema") != "boxfusion.ca1m_tr3d_proposal_collection.v4"
        or value.get("complete") is not True
        or value.get("ground_truth_access") is not False
        or value.get("scene_count") != 100
    ):
        raise ValueError("v1 proposal collection contract differs")
    rows = {str(row.get("scene_id")): row for row in value.get("scenes", [])}
    if len(rows) != 100 or not set(scenes).issubset(rows):
        raise ValueError("v1 proposal collection does not cover exact fold0")
    selected = [rows[scene] for scene in scenes]
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "create_only": True,
        "train_only": True,
        "comparison_only": True,
        "activation_authorized": False,
        "fold0_only": True,
        "fold_id": 0,
        "scene_count": 20,
        "fold1_proposal_artifact_access": False,
        "official_validation_access": False,
        "ground_truth_access": False,
        "source_manifest": {
            "path": str(source), "sha256": SOURCE_SHA256,
            "metadata_projection_only": True,
        },
        "scene_list": {"path": str(scene_list), "sha256": FOLD0_SHA256},
        "checkpoint_binding": value["checkpoint_binding"],
        "point_inference_config": value["point_inference_config"],
        "scenes": selected,
    }
    create_or_verify_json(OUTPUT, payload, "v1 fold0 comparison manifest")
    return payload


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
