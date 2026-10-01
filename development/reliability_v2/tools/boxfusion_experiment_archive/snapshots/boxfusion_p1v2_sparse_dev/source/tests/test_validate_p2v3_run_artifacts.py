"""Safety, ancestry, and geometry checks for P2-v3 run artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import tools.validate_p2v3_run_artifacts as validator
from boxfusion.p2_local_mask_geometry import (
    P2V2_DIAGNOSTIC_SCHEMA,
    P2V2_SOURCE,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corners(boxes: np.ndarray) -> np.ndarray:
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    return (
        boxes[:, None, :3]
        + 0.5 * signs[None] * boxes[:, None, 3:]
    ).astype(np.float32)


def _payload(scene: str, p2_sha: str) -> dict[str, np.ndarray]:
    component = np.asarray(
        [[0.1, 0.0, 0.0, 0.4, 0.6, 0.8]], dtype=np.float32
    )
    parent = np.asarray(
        [[0.0, 0.0, 0.0, 0.6, 0.6, 0.6]], dtype=np.float32
    )
    fused = 0.75 * component + 0.25 * parent
    config = {
        "enabled": True,
        "observer_only": True,
        "mutate": False,
        "collect_diagnostics": True,
        "minimum_component_weight": 0.35,
        "maximum_component_weight": 0.85,
        "max_candidates_per_step": 16,
        "max_scene_candidates": 64,
    }
    return {
        "scene_id": np.asarray(scene),
        "p2_checkpoint_sha256": np.asarray(p2_sha),
        "p2v2_schema": np.asarray(P2V2_DIAGNOSTIC_SCHEMA),
        "p2v2_source": np.asarray(P2V2_SOURCE),
        "p2v2_parent_p2_checkpoint_sha256": np.asarray(p2_sha),
        "p2v2_step_frame_ids": np.asarray([5], dtype=np.int64),
        "p2v2_step_provider_steps": np.asarray([1], dtype=np.int64),
        "p2v2_step_candidate_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v2_candidate_ids": np.asarray(
            ["p2v2:fedcba9876543210"], dtype=np.str_
        ),
        "p2v2_parent_p2_candidate_ids": np.asarray(
            ["p1:5:1:2:3"], dtype=np.str_
        ),
        "p2v2_mask_source_ids": np.asarray(
            ["scene0001_00:5:0"], dtype=np.str_
        ),
        "p2v2_candidate_boxes": component,
        "p2v2_candidate_corners": _corners(component),
        "p2v2_candidate_scores": np.asarray(
            [0.8], dtype=np.float32
        ),
        "p2v3_schema": np.asarray(
            validator.P2V3_DIAGNOSTIC_SCHEMA
        ),
        "p2v3_stage": np.asarray("P2V3"),
        "p2v3_profile": np.asarray(validator.P2V3_PROFILE),
        "p2v3_source": np.asarray(validator.P2V3_SOURCE),
        "p2v3_parent_p2v2_schema": np.asarray(
            P2V2_DIAGNOSTIC_SCHEMA
        ),
        "p2v3_reliability_contract": np.asarray(
            validator.P2V3_RELIABILITY_CONTRACT
        ),
        "p2v3_parent_p2_checkpoint_sha256": np.asarray(p2_sha),
        "p2v3_enabled": np.asarray(True, dtype=bool),
        "p2v3_observer_only": np.asarray(True, dtype=bool),
        "p2v3_uses_ground_truth": np.asarray(False, dtype=bool),
        "p2v3_reads_semantic_labels": np.asarray(False, dtype=bool),
        "p2v3_mutation_enabled": np.asarray(False, dtype=bool),
        "p2v3_applied_count": np.asarray(0, dtype=np.int64),
        "p2v3_complete": np.asarray(True, dtype=bool),
        "p2v3_config_json": np.asarray(
            json.dumps(config, sort_keys=True)
        ),
        "p2v3_step_frame_ids": np.asarray([5], dtype=np.int64),
        "p2v3_step_provider_steps": np.asarray([1], dtype=np.int64),
        "p2v3_step_input_candidate_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v3_step_eligible_candidate_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v3_step_candidate_counts": np.asarray(
            [1], dtype=np.int64
        ),
        "p2v3_step_seconds": np.asarray([0.01], dtype=np.float64),
        "p2v3_step_failed": np.asarray([False], dtype=bool),
        "p2v3_step_errors": np.asarray([""], dtype=np.str_),
        "p2v3_candidate_ids": np.asarray(
            ["p2v3:0123456789abcdef"], dtype=np.str_
        ),
        "p2v3_parent_p2v2_candidate_ids": np.asarray(
            ["p2v2:fedcba9876543210"], dtype=np.str_
        ),
        "p2v3_parent_p2_candidate_ids": np.asarray(
            ["p1:5:1:2:3"], dtype=np.str_
        ),
        "p2v3_mask_source_ids": np.asarray(
            ["scene0001_00:5:0"], dtype=np.str_
        ),
        "p2v3_candidate_component_boxes": component,
        "p2v3_candidate_component_corners": _corners(component),
        "p2v3_candidate_parent_boxes": parent,
        "p2v3_candidate_parent_corners": _corners(parent),
        "p2v3_candidate_fused_boxes": fused,
        "p2v3_candidate_fused_corners": _corners(fused),
        "p2v3_candidate_scores": np.asarray(
            [0.8], dtype=np.float32
        ),
        "p2v3_candidate_component_weights": np.asarray(
            [0.75], dtype=np.float32
        ),
        "p2v3_candidate_center_component_weights": np.asarray(
            [[0.75, 0.75, 0.75]], dtype=np.float32
        ),
        "p2v3_candidate_extent_component_weights": np.asarray(
            [[0.75, 0.75, 0.75]], dtype=np.float32
        ),
        "p2v3_candidate_component_reliabilities": np.asarray(
            [0.8], dtype=np.float32
        ),
        "p2v3_candidate_parent_reliabilities": np.asarray(
            [0.6], dtype=np.float32
        ),
        "p2v3_candidate_mask_reliabilities": np.asarray(
            [0.9], dtype=np.float32
        ),
        "p2v3_candidate_depth_reliabilities": np.asarray(
            [0.8], dtype=np.float32
        ),
        "p2v3_candidate_support_reliabilities": np.asarray(
            [0.7], dtype=np.float32
        ),
        "p2v3_candidate_agreement_reliabilities": np.asarray(
            [0.6], dtype=np.float32
        ),
        "p2v3_candidate_applied": np.asarray([False], dtype=bool),
    }


def _artifacts(tmp_path: Path):
    scene = "scene0001_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    predictions = tmp_path / "predictions"
    diagnostics = tmp_path / "diagnostics"
    predictions.mkdir()
    diagnostics.mkdir()
    (predictions / f"{scene}_boxes.pkl").write_bytes(b"formal output")
    p1_checkpoint = tmp_path / "p1.pt"
    p2_checkpoint = tmp_path / "p2.pt"
    p1_checkpoint.write_bytes(b"p1")
    p2_checkpoint.write_bytes(b"p2")
    payload = _payload(scene, _sha(p2_checkpoint))
    diagnostic = diagnostics / f"{scene}_tracks.npz"
    np.savez_compressed(diagnostic, **payload)
    return {
        "scene": scene,
        "scene_list": scene_list,
        "predictions": predictions,
        "diagnostics": diagnostics,
        "p1": p1_checkpoint,
        "p2": p2_checkpoint,
        "payload": payload,
        "diagnostic": diagnostic,
    }


def _rewrite(artifacts, **updates) -> None:
    payload = dict(artifacts["payload"])
    payload.update(updates)
    np.savez_compressed(artifacts["diagnostic"], **payload)


def _patch_parent(monkeypatch, artifacts) -> None:
    monkeypatch.setattr(
        validator,
        "validate_p2v2",
        lambda **_: {
            "p1_checkpoint_sha256": _sha(artifacts["p1"]),
            "p2_checkpoint_sha256": _sha(artifacts["p2"]),
        },
    )


def _validate(monkeypatch, artifacts):
    _patch_parent(monkeypatch, artifacts)
    return validator.validate(
        scene_list=artifacts["scene_list"],
        prediction_root=artifacts["predictions"],
        diagnostics_root=artifacts["diagnostics"],
        expected_p1_checkpoint=artifacts["p1"],
        expected_p2_checkpoint=artifacts["p2"],
    )


def test_valid_p2v3_artifacts_and_public_loader(
    tmp_path: Path,
    monkeypatch,
) -> None:
    artifacts = _artifacts(tmp_path)
    loaded = validator.load_p2v3_diagnostic(
        artifacts["diagnostic"],
        expected_scene_id=artifacts["scene"],
        expected_p2_checkpoint_sha256=_sha(artifacts["p2"]),
    )
    assert loaded.scene_id == artifacts["scene"]
    assert loaded.fused_boxes.shape == (1, 6)
    assert loaded.component_weights.tolist() == pytest.approx([0.75])
    assert loaded.runtime_seconds == pytest.approx(0.01)
    assert not loaded.fused_boxes.flags.writeable

    report = _validate(monkeypatch, artifacts)
    assert report["scene_count"] == 1
    assert report["p2v3_step_count"] == 1
    assert report["p2v3_input_candidate_count"] == 1
    assert report["p2v3_scene_candidate_count"] == 1
    assert (
        report["formal_output_safety"][
            "cross_run_pickle_byte_equality_required"
        ]
        is False
    )


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "p2v3_uses_ground_truth",
            np.asarray(True, dtype=bool),
            "unsafe p2v3_uses_ground_truth",
        ),
        (
            "p2v3_reads_semantic_labels",
            np.asarray(True, dtype=bool),
            "unsafe p2v3_reads_semantic_labels",
        ),
        (
            "p2v3_mutation_enabled",
            np.asarray(True, dtype=bool),
            "unsafe p2v3_mutation_enabled",
        ),
        (
            "p2v3_applied_count",
            np.asarray(1, dtype=np.int64),
            "mutated formal output",
        ),
        (
            "p2v3_candidate_applied",
            np.asarray([True], dtype=bool),
            "unsafe P2-v3 candidate flags",
        ),
    ],
)
def test_unsafe_observer_contract_is_rejected(
    tmp_path: Path,
    key: str,
    value: np.ndarray,
    message: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    _rewrite(artifacts, **{key: value})
    with pytest.raises(ValueError, match=message):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])


def test_parent_steps_and_input_counts_must_align(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    _rewrite(
        artifacts,
        p2v3_step_frame_ids=np.asarray([6], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="scheduling is not aligned"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])

    _rewrite(
        artifacts,
        p2v3_step_input_candidate_counts=np.asarray(
            [0], dtype=np.int64
        ),
    )
    with pytest.raises(ValueError, match="inputs disagree"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])


def test_checkpoint_parent_chain_is_enforced(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    _rewrite(
        artifacts,
        p2v3_parent_p2_checkpoint_sha256=np.asarray("0" * 64),
    )
    with pytest.raises(ValueError, match="parent checkpoint mismatch"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        (
            "p2v3_candidate_component_weights",
            np.asarray([0.9], dtype=np.float32),
            "component weight violates configured bounds",
        ),
        (
            "p2v3_candidate_depth_reliabilities",
            np.asarray([1.1], dtype=np.float32),
            "invalid p2v3_candidate_depth_reliabilities",
        ),
        (
            "p2v3_candidate_center_component_weights",
            np.asarray([[0.9, 0.75, 0.75]], dtype=np.float32),
            "invalid p2v3_candidate_center_component_weights",
        ),
    ],
)
def test_weight_and_reliability_ranges_are_enforced(
    tmp_path: Path,
    key: str,
    value: np.ndarray,
    message: str,
) -> None:
    artifacts = _artifacts(tmp_path)
    _rewrite(artifacts, **{key: value})
    with pytest.raises(ValueError, match=message):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])


def test_all_three_box_corner_pairs_are_checked(tmp_path: Path) -> None:
    artifacts = _artifacts(tmp_path)
    bad = np.array(
        artifacts["payload"]["p2v3_candidate_fused_corners"],
        copy=True,
    )
    bad[0, 0, 0] += 0.1
    _rewrite(artifacts, p2v3_candidate_fused_corners=bad)
    with pytest.raises(ValueError, match="fused corners disagree"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])


def test_candidate_ids_and_semantic_fields_are_rejected(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    _rewrite(
        artifacts,
        p2v3_parent_p2v2_candidate_ids=np.asarray(
            ["p2:wrong-parent"], dtype=np.str_
        ),
    )
    with pytest.raises(ValueError, match="candidate IDs"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])

    _rewrite(
        artifacts,
        p2v3_parent_p2_candidate_ids=np.asarray([""], dtype=np.str_),
    )
    with pytest.raises(ValueError, match="candidate IDs"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])

    _rewrite(
        artifacts,
        p2v3_candidate_labels=np.asarray(["chair"], dtype=np.str_),
    )
    with pytest.raises(ValueError, match="semantic P2-v3"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])


def test_parent_p2v2_schema_provenance_is_enforced(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    _rewrite(
        artifacts,
        p2v3_parent_p2v2_schema=np.asarray("wrong.schema"),
    )
    with pytest.raises(ValueError, match="parent_p2v2_schema"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])


def test_object_arrays_and_scene_nms_counts_fail_closed(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(tmp_path)
    _rewrite(
        artifacts,
        p2v3_extra=np.asarray([{"unsafe": True}], dtype=object),
    )
    with pytest.raises(ValueError, match="object dtype"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])

    _rewrite(
        artifacts,
        p2v3_step_candidate_counts=np.asarray([0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="scene candidate count"):
        validator.load_p2v3_diagnostic(artifacts["diagnostic"])
