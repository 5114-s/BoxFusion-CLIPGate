from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from boxfusion.tr3d_c3_online_identity import ROUTE, SCHEMA
from tools.materialize_tr3d_c3_online_shadow import (
    COMPLETION_SCHEMA,
    _load_completion_marker,
    _load_identity_diagnostic,
    _selected_candidates,
)


def _sha(char: str) -> str:
    return char * 64


def test_completion_marker_binds_active_and_c3_hashes(tmp_path: Path) -> None:
    path = tmp_path / "scene.run_fingerprint"
    values = {
        "schema": COMPLETION_SCHEMA,
        "scene_fingerprint": _sha("1"),
        "active_prediction_sha256": _sha("2"),
        "same_run_baseline_sha256": _sha("3"),
        "r3_diagnostic_sha256": _sha("4"),
        "boxer_diagnostic_sha256": _sha("5"),
        "c3_online_diagnostic_sha256": _sha("6"),
    }
    path.write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    path.chmod(0o444)
    assert _load_completion_marker(path) == values

    path.chmod(0o644)
    path.write_text(path.read_text() + "active_prediction_sha256=" + _sha("7") + "\n")
    path.chmod(0o444)
    with pytest.raises(ValueError, match="unsupported|duplicate"):
        _load_completion_marker(path)


def test_online_selection_never_uses_frozen_teacher_bit() -> None:
    scene_id, prefix_id, parent_sha = "scene0000_00", "p100", _sha("a")
    parent = SimpleNamespace(
        proposal_ids=np.asarray([10, 11, 12, 13], dtype=np.int64),
        corners_world=np.arange(4 * 8 * 3, dtype=np.float32).reshape(4, 8, 3),
    )
    candidates = []
    for row, (frozen, online) in enumerate(
        ((True, False), (False, True), (True, True), (False, False))
    ):
        proposal_id = int(parent.proposal_ids[row])
        candidates.append(
            {
                "source_rank": row + 1,
                "parent_row": row,
                "proposal_id": proposal_id,
                "identity_key": f"{scene_id}:{prefix_id}:{parent_sha}:{proposal_id}",
                "frozen_sam3_mask2_depth": frozen,
                "online_yoloe_mask2_depth": online,
                "c1_depth_dino_track_score": 1.0 - row * 0.1,
            }
        )
    diagnostic = {"candidates": candidates, "online_selected_count": 2}
    rows, proposal_ids, scores, corners = _selected_candidates(
        diagnostic,
        parent,
        scene_id=scene_id,
        prefix_id=prefix_id,
        parent_sha256=parent_sha,
    )
    assert rows.tolist() == [1, 2]
    assert proposal_ids.tolist() == [11, 12]
    assert scores.tolist() == pytest.approx([0.9, 0.8])
    assert np.array_equal(corners, parent.corners_world[[1, 2]])


def test_identity_diagnostic_fails_closed_on_mutation(tmp_path: Path) -> None:
    path = tmp_path / "scene0000_00_c3_online_identity.json"
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "observer_only": True,
        "enabled": True,
        "applied_count": 0,
        "mutation_enabled": False,
        "ground_truth_access": False,
        "clip_access": False,
        "teacher_labels_used_for_gate": False,
        "online_sam3_forward": False,
        "online_dino_forward": False,
        "scene_id": "scene0000_00",
        "route": ROUTE,
        "gate_name": "mask2_depth",
        "source_rank_max": 5,
        "identity_coverage": 1.0,
        "missing_identity_count": 0,
        "out_of_universe_selected_count": 0,
        "candidate_generation_is_live": False,
        "online_confirmation_provider": "runtime_yoloe_mask_real_depth",
        "candidate_count": 0,
        "exact_identity_joined_count": 0,
        "candidates": [],
    }
    path.write_text(json.dumps(payload))
    path.chmod(0o444)
    assert _load_identity_diagnostic(path, "scene0000_00")["observer_only"]

    path.chmod(0o644)
    payload["mutation_enabled"] = True
    path.write_text(json.dumps(payload))
    path.chmod(0o444)
    with pytest.raises(ValueError, match="contract failed"):
        _load_identity_diagnostic(path, "scene0000_00")
