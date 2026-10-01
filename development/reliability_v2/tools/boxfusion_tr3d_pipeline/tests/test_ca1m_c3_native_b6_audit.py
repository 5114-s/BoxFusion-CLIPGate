from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxfusion.ca1m_native_b6_observer import build_ca1m_native_b6_observer
from boxfusion.tr3d_r2_geometry import yaw_obb_corners_world
from tools.audit_ca1m_c3_native_b6_observer import (
    audit_diagnostic,
    build_train_readiness,
)


SCENE = "42898811"


def _config(root: Path) -> dict:
    return {
        "dataset": "CA1M",
        "online_refinement": {"enabled": False},
        "ca1m_native_b6_observer": {
            "enabled": True,
            "observer_only": True,
            "top_k_views": 2,
            "pixel_stride": 1,
            "depth_margin_m": 0.0,
            "min_depth_m": 0.1,
            "max_depth_m": 8.0,
            "near_clip_m": 1e-3,
            "max_cached_keyframes": 8,
            "diagnostics": {"enabled": True, "root": str(root)},
        },
    }


def _diagnostic(tmp_path: Path) -> tuple[Path, list[tuple[int, np.ndarray, float]]]:
    observer = build_ca1m_native_b6_observer(_config(tmp_path))
    shape = (17, 17)
    intrinsics = np.asarray(
        [[12.0, 0.0, 8.0], [0.0, 12.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    for frame_id in (0, 20):
        observer.record_keyframe(
            scene_id=SCENE,
            frame_id=frame_id,
            source_frame_id=str(frame_id),
            depth_meters=np.full(shape, 5.0, dtype=np.float32),
            intrinsics=intrinsics,
            camera_to_world=np.eye(4, dtype=np.float64),
        )
    corners = np.stack(
        (
            yaw_obb_corners_world([0, 0, 5, 2, 2, 2, 0.4]),
            yaw_obb_corners_world([100, 0, 5, 1, 1, 1, 0.0]),
        )
    ).astype(np.float32)
    scores = np.asarray([0.8, 0.4], dtype=np.float32)
    observer.finalize(
        scene_id=SCENE,
        corners=corners,
        scores=scores,
        stable_ids=np.asarray([3, 9], dtype=np.int64),
    )
    prediction = [(0, corners[index].copy(), float(scores[index])) for index in range(2)]
    return tmp_path / f"{SCENE}_ca1m_native_b6.npz", prediction


def _rewrite(source: Path, target: Path, mutate) -> Path:
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    mutate(arrays)
    np.savez_compressed(target, **arrays)
    return target


def test_audit_fully_recomputes_native_b6_diagnostic(tmp_path: Path) -> None:
    path, prediction = _diagnostic(tmp_path)
    report = audit_diagnostic(SCENE, path, prediction)
    assert report["mapping_coverage"] == 1.0
    assert report["yaw_and_features_recomputed"]
    assert report["depth_redundancy_recomputed"]
    assert len(report["row_evidence"]) == 2
    assert report["row_evidence"][0]["classified_samples"] > 0
    assert report["row_evidence"][0]["valid_views"] == 2
    assert report["row_evidence"][1]["classified_samples"] == 0


@pytest.mark.parametrize(
    ("name", "mutation", "message"),
    (
        (
            "aggregate",
            lambda arrays: arrays["aggregate_depth_counts"].__setitem__((0, 0), arrays["aggregate_depth_counts"][0, 0] + 1),
            "aggregate counts",
        ),
        (
            "aggregate_evidence",
            lambda arrays: arrays["aggregate_depth_evidence"].__setitem__((0, 0), 0.123),
            "redundant depth evidence",
        ),
        (
            "view_count",
            lambda arrays: arrays["aggregate_view_count"].__setitem__(0, 0),
            "aggregate_view_count",
        ),
        (
            "sample_count",
            lambda arrays: arrays["aggregate_sample_count"].__setitem__(0, 0),
            "aggregate_sample_count",
        ),
        (
            "yaw",
            lambda arrays: arrays["yaw_boxes"].__setitem__((0, 0), arrays["yaw_boxes"][0, 0] + 0.1),
            "yaw_boxes",
        ),
        (
            "feature",
            lambda arrays: arrays["features"].__setitem__((0, 1), 0.0),
            "14-column features",
        ),
        (
            "valid_evidence",
            lambda arrays: arrays["valid_evidence"].__setitem__(0, False),
            "valid_evidence",
        ),
    ),
)
def test_audit_rejects_redundant_field_tampering(
    tmp_path: Path,
    name: str,
    mutation,
    message: str,
) -> None:
    source, prediction = _diagnostic(tmp_path / "source")
    target = _rewrite(source, tmp_path / f"tampered-{name}.npz", mutation)
    with pytest.raises(ValueError, match=message):
        audit_diagnostic(SCENE, target, prediction)


def test_fixed10_readiness_never_authorizes_training() -> None:
    readiness = build_train_readiness(
        engineering_identity_ok=True,
        mapping_coverage=1.0,
        valid_evidence_coverage=1.0,
        feature_integrity_ok=True,
    )
    assert readiness["prerequisites_passed"]
    assert not readiness["authorized"]
    assert readiness["fixed10_validation_only"]
    assert readiness["status"] == "NOT_AUTHORIZED_FIXED10_VALIDATION_ONLY"
