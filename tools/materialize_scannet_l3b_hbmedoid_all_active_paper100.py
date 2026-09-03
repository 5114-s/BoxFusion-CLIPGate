#!/usr/bin/env python3
"""Materialize the L3B all-track active paper100 stress test.

Every sealed L3B T1 track contributes its single fixed HBMedoid geometry.  No
admission, novelty or NMS gate is applied: this intentionally measures the
actual AP consequence of activating the complete L3B pool.  Native B05 rows
remain an exact prefix.  The tool is create-only and has no GT/evaluator input.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from tools.audit_scannet_l0_f3_f4_perview_paper100_oracle import (  # noqa: E402
    _json,
    _sha,
    _source_map,
)
from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    BirthMaterializationError,
    _assert_native_prefix,
    _load_native_prediction,
    _regular_file,
    _scene_list,
    _sha256,
    _write_json,
    _write_pickle,
)
from tools.run_scannet_l3b_hbmedoid_t1_selector_paper100 import (  # noqa: E402
    PROTOCOL_ID as L3B_PROTOCOL_ID,
    SCHEMA as L3B_SCHEMA,
)


SCHEMA = "boxfusion.scannet_l3b_hbmedoid_all_active_paper100.v1"
DEFAULT_SCENE_LIST = ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
DEFAULT_BASELINE = ROOT / "results/scannet_t05_boxer_replay_active_score05"
DEFAULT_SHADOW = ROOT / "logs/scannet_l3b_hbmedoid_t1_selector_paper100_score05/final/L3B_HBMEDOID_T1_SELECTOR_PAPER100.json"
DEFAULT_OUTPUT = ROOT / "results/scannet_l3b_hbmedoid_all_active_score05"
MANIFEST_NAME = "L3B_HBMEDOID_ALL_ACTIVE_PAPER100.json"


class L3BActiveError(BirthMaterializationError):
    pass


def _augmented_payload(native: Any, corners: list[np.ndarray]) -> list[Any] | tuple[Any, ...]:
    suffix = [(0, np.ascontiguousarray(box, dtype=np.float32), 1.0) for box in corners]
    rows: list[Any] | tuple[Any, ...]
    if isinstance(native.rows, tuple):
        rows = tuple(native.rows) + tuple(suffix)
    else:
        rows = list(native.rows) + suffix
    output: list[Any] | tuple[Any, ...] = (rows,) if isinstance(native.payload, tuple) else [rows]
    _assert_native_prefix(native.rows, output[0], "in-memory L3B all-active output")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise L3BActiveError(f"refusing to overwrite output root: {output_root}")
    scenes = _scene_list(args.scene_list, 100)
    shadow = _json(args.shadow, "L3B shadow")
    scene_rows = shadow.get("scenes")
    if (
        shadow.get("schema") != L3B_SCHEMA
        or shadow.get("protocol_id") != L3B_PROTOCOL_ID
        or shadow.get("complete") is not True
        or shadow.get("overall_pass") is not True
        or shadow.get("contracts", {}).get("ground_truth_access") is not False
        or shadow.get("contracts", {}).get("birth_enabled") is not False
        or not isinstance(scene_rows, list)
        or len(scene_rows) != 100
    ):
        raise L3BActiveError("L3B shadow contract differs")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    scene_reports: dict[str, Any] = {}
    native_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    total_native = total_births = 0
    try:
        for expected_index, (scene, scene_row) in enumerate(zip(scenes, scene_rows)):
            if not isinstance(scene_row, Mapping) or scene_row.get("scene_id") != scene or scene_row.get("scene_index") != expected_index:
                raise L3BActiveError(f"L3B scene order differs: {scene}")
            selections = scene_row.get("selections")
            f4_receipt = scene_row.get("f4")
            if not isinstance(selections, list) or not isinstance(f4_receipt, Mapping):
                raise L3BActiveError(f"L3B scene ledger differs: {scene}")
            f4_path = Path(str(f4_receipt.get("path", "")))
            if _sha(f4_path) != f4_receipt.get("sha256"):
                raise L3BActiveError(f"sealed F4 hash differs: {scene}")
            source_map = _source_map(_json(f4_path, f"F4 scene {scene}"), scene)
            birth_corners: list[np.ndarray] = []
            suffix_rows: list[dict[str, Any]] = []
            for track_id, selection in enumerate(selections):
                if (
                    not isinstance(selection, Mapping)
                    or selection.get("track_id") != track_id
                    or selection.get("hypothesis") != "HB"
                    or selection.get("past_only_at_decision") is not True
                ):
                    raise L3BActiveError(f"L3B selection differs: {scene}:{track_id}")
                source_id = str(selection.get("source_id", ""))
                source = source_map.get(source_id)
                hypotheses = source.get("hypotheses") if isinstance(source, Mapping) else None
                hb = hypotheses.get("HB") if isinstance(hypotheses, Mapping) else None
                if not isinstance(hb, Mapping) or hb.get("valid") is not True:
                    raise L3BActiveError(f"chosen HB is invalid: {source_id}")
                corners = np.asarray(hb.get("world_corners"), dtype=np.float64)
                if corners.shape != (8, 3) or not np.isfinite(corners).all() or np.any(corners.max(axis=0) <= corners.min(axis=0)):
                    raise L3BActiveError(f"chosen HB corners differ: {source_id}")
                birth_corners.append(corners)
                suffix_rows.append(
                    {
                        "suffix_index": track_id,
                        "track_id": track_id,
                        "source_id": source_id,
                        "hypothesis": "HB",
                        "class_id": 0,
                        "score": 1.0,
                        "decision_frame_id": int(selection["decision_frame_id"]),
                        "emit_event": str(selection["emit_event"]),
                    }
                )

            native_path = _regular_file(args.baseline_root / f"{scene}_boxes.pkl", "native B05 prediction")
            native_digest = _sha256(native_path)
            native_hashes[scene] = native_digest
            native = _load_native_prediction(native_path)
            output_path = stage / f"{scene}_boxes.pkl"
            _write_pickle(output_path, _augmented_payload(native, birth_corners))
            reloaded = _load_native_prediction(output_path)
            _assert_native_prefix(native.rows, reloaded.rows, scene)
            if len(reloaded.rows) != len(native.rows) + len(birth_corners):
                raise L3BActiveError(f"suffix count differs: {scene}")
            if _sha256(native_path) != native_digest:
                raise L3BActiveError(f"native input changed: {scene}")
            output_hashes[scene] = _sha256(output_path)
            scene_reports[scene] = {
                "native_count": len(native.rows),
                "birth_count": len(birth_corners),
                "output_count": len(reloaded.rows),
                "native_prefix_row_identity_verified": True,
                "suffix": suffix_rows,
            }
            total_native += len(native.rows)
            total_births += len(birth_corners)

        if total_births != shadow.get("counts", {}).get("track_count"):
            raise L3BActiveError("L3B all-active global birth census differs")
        manifest = {
            "schema": SCHEMA,
            "mode": "active_all_tracks_stress_test",
            "complete": True,
            "scene_count": 100,
            "native_count": total_native,
            "birth_count": total_births,
            "output_count": total_native + total_births,
            "score_mode": "constant_1.0",
            "class_mode": "inert_0_scannet_class_agnostic_evaluator",
            "native_rows_are_unchanged_prefix": True,
            "native_clip_unchanged": True,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "training": False,
            "online_learning": False,
            "past_only": True,
            "admission_gate": "none_all_L3B_tracks_are_appended",
            "expected_use": "stress_test_not_Cbest",
            "inputs": {
                "scene_list": os.fspath(args.scene_list.resolve()),
                "scene_list_sha256": _sha256(args.scene_list),
                "baseline_root": os.fspath(args.baseline_root.resolve()),
                "l3b_shadow": os.fspath(args.shadow.resolve()),
                "l3b_shadow_sha256": _sha256(args.shadow),
            },
            "native_prediction_sha256": native_hashes,
            "output_prediction_sha256": output_hashes,
            "scenes": scene_reports,
        }
        _write_json(stage / MANIFEST_NAME, manifest)
        os.replace(stage, output_root)
    except Exception:
        # Leave an exact hidden stage for diagnosis; never replace or delete a
        # user-visible result after a failed create-only materialization.
        raise
    print(json.dumps({"output_root": os.fspath(output_root), "native_count": total_native, "birth_count": total_births, "output_count": total_native + total_births}, sort_keys=True))


if __name__ == "__main__":
    main()
