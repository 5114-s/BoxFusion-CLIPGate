from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_r2_cache import (
    DEPTH_EVIDENCE_NAMES,
    depth_evidence_fractions,
    load_tr3d_r2_cache,
    make_tr3d_r2_cache,
    sha256_file,
    tr3d_r2_cache_path,
    validate_tr3d_r2_payload,
    write_tr3d_r2_cache,
)
from boxfusion.tr3d_residual_cache import (
    TR3DResidualCache,
    transform_sha256,
    write_tr3d_residual_cache,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


PROVENANCE = {
    "prefix_manifest_row_sha256": _sha("prefix-row"),
    "frame_artifact_tree_sha256": _sha("frame-tree"),
    "r2_config_sha256": _sha("r2-config"),
    "r2_code_sha256": _sha("r2-code"),
}


def _parent(path: Path, *, score_delta: float = 0.0) -> Path:
    boxes = np.asarray(
        [
            [1, 2, 3, 2, 2, 2, 0],
            [4, 5, 6, 1, 2, 3, 0],
            [7, 8, 9, 3, 2, 1, 0],
        ],
        dtype=np.float32,
    )
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
    corners = boxes[:, None, :3] + signs[None] * boxes[:, None, 3:6] / 2
    cache = TR3DResidualCache(
        scene_id="scene0001_00",
        sample_idx="scene0001_00:p050",
        prefix_id="p050",
        prefix_fraction=0.5,
        boxes_world=boxes,
        corners_world=corners,
        aligned_to_unaligned=np.eye(4, dtype=np.float64),
        axis_alignment_sha256=transform_sha256(np.eye(4)),
        scores_3d=np.asarray(
            [0.9 + score_delta, 0.8, 0.7], dtype=np.float32
        ),
        labels_3d=np.zeros(3, dtype=np.int64),
        proposal_ids=np.asarray([10, 20, 30], dtype=np.int64),
        point_count=np.asarray([100, 80, 60], dtype=np.int32),
        voxel_size=0.01,
        runtime_s=0.2,
        num_input_points=1000,
        checkpoint_sha256=_sha("checkpoint"),
        config_sha256=_sha("config"),
        source_scene_sha256=_sha("source"),
    )
    write_tr3d_residual_cache(path, cache)
    return path


def _r2(parent: Path):
    per_view_counts = np.asarray(
        [
            [
                [21, 3, 4, 2],
                [10, 5, 3, 2],
                [0, 0, 0, 0],
            ],
            [
                [30, 4, 4, 2],
                [18, 4, 5, 3],
                [10, 3, 5, 2],
            ],
        ],
        dtype=np.int32,
    )
    aggregate_counts = per_view_counts.sum(axis=1, dtype=np.int64)
    return make_tr3d_r2_cache(
        parent_cache_path=parent,
        **PROVENANCE,
        proposal_ids=np.asarray([10, 30], dtype=np.int64),
        lineage_ids=np.asarray([100, 300], dtype=np.int64),
        topk_frame_ids=np.asarray(
            [[0, 25, -1], [25, 50, 75]], dtype=np.int64
        ),
        topk_view_valid=np.asarray(
            [[True, True, False], [True, True, True]], dtype=np.bool_
        ),
        topk_projected_area_pixels=np.asarray(
            [[120.0, 80.0, 0.0], [100.0, 90.0, 70.0]],
            dtype=np.float32,
        ),
        topk_projected_area_fraction=np.asarray(
            [[0.12, 0.08, 0.0], [0.10, 0.09, 0.07]],
            dtype=np.float32,
        ),
        per_view_depth_evidence=depth_evidence_fractions(per_view_counts),
        per_view_depth_counts=per_view_counts,
        per_view_point_count=np.asarray(
            [[30, 20, 0], [40, 30, 20]], dtype=np.int32
        ),
        aggregate_depth_evidence=depth_evidence_fractions(
            aggregate_counts
        ),
        aggregate_depth_counts=aggregate_counts,
        aggregate_view_count=np.asarray([2, 3], dtype=np.int32),
        aggregate_point_count=np.asarray([50, 90], dtype=np.int64),
        runtime_s=0.05,
        expected_allowed_frame_ids=(0, 25, 50, 75),
    )


def _load(
    path: Path,
    parent: Path,
    *,
    allowed_frame_ids=(0, 25, 50, 75),
    **overrides,
):
    values = dict(PROVENANCE)
    values.update(overrides)
    return load_tr3d_r2_cache(
        path,
        parent_cache_path=parent,
        expected_scene_id="scene0001_00",
        expected_prefix_id="p050",
        expected_prefix_fraction=0.5,
        expected_allowed_frame_ids=allowed_frame_ids,
        **{
            f"expected_{name}": value for name, value in values.items()
        },
    )


def test_roundtrip_is_parent_bound_readonly_and_non_overwriting(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path / "parent.npz")
    cache = _r2(parent)
    path = tr3d_r2_cache_path(tmp_path / "r2", cache.scene_id, cache.prefix_id)
    write_tr3d_r2_cache(path, cache, parent_cache_path=parent)
    loaded = _load(path, parent)

    assert DEPTH_EVIDENCE_NAMES == (
        "support_fraction",
        "occluded_fraction",
        "free_space_fraction",
        "invalid_fraction",
    )
    assert loaded.parent_cache_sha256 == sha256_file(parent)
    assert loaded.proposal_count == 2
    assert loaded.topk == 3
    np.testing.assert_array_equal(loaded.proposal_ids, [10, 30])
    assert not loaded.per_view_depth_evidence.flags.writeable
    assert not loaded.topk_projected_area_pixels.flags.writeable
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError):
        loaded.aggregate_depth_evidence[0, 0] = 0.0
    with pytest.raises(FileExistsError, match="immutable"):
        write_tr3d_r2_cache(path, cache, parent_cache_path=parent)


def test_load_requires_exact_parent_bytes_and_external_provenance(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path / "parent.npz")
    cache = _r2(parent)
    path = tr3d_r2_cache_path(tmp_path / "r2", cache.scene_id, cache.prefix_id)
    write_tr3d_r2_cache(path, cache, parent_cache_path=parent)

    other = _parent(tmp_path / "other.npz", score_delta=-0.01)
    with pytest.raises(ValueError, match="parent cache provenance"):
        _load(path, other)
    with pytest.raises(ValueError, match="prefix_manifest_row_sha256"):
        _load(path, parent, prefix_manifest_row_sha256=_sha("wrong-row"))
    with pytest.raises(ValueError, match="frame_artifact_tree_sha256"):
        _load(path, parent, frame_artifact_tree_sha256=_sha("wrong-tree"))
    with pytest.raises(ValueError, match="r2_config_sha256"):
        _load(path, parent, r2_config_sha256=_sha("wrong-config"))
    with pytest.raises(ValueError, match="r2_code_sha256"):
        _load(path, parent, r2_code_sha256=_sha("wrong-code"))


def _validate_payload(
    payload,
    parent: Path,
    *,
    allowed_frame_ids=(0, 25, 50, 75),
):
    return validate_tr3d_r2_payload(
        payload,
        parent_cache_path=parent,
        expected_allowed_frame_ids=allowed_frame_ids,
        **{
            f"expected_{name}": value for name, value in PROVENANCE.items()
        },
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "topk_frame_ids",
            np.asarray([[0, 25, 99], [25, 50, 75]], dtype=np.int64),
            "invalid Top-K slots",
        ),
        (
            "topk_view_valid",
            np.asarray(
                [[True, False, True], [True, True, True]], dtype=np.bool_
            ),
            "valid Top-K views",
        ),
        (
            "per_view_point_count",
            np.asarray([[30, 20, 1], [40, 30, 20]], dtype=np.int32),
            "zero point count",
        ),
        (
            "aggregate_view_count",
            np.asarray([3, 3], dtype=np.int32),
            "disagrees with valid views",
        ),
        (
            "topk_projected_area_pixels",
            np.asarray(
                [[120.0, 80.0, 1.0], [100.0, 90.0, 70.0]],
                dtype=np.float32,
            ),
            "zero projected area",
        ),
        (
            "topk_projected_area_pixels",
            np.asarray(
                [[0.0, 80.0, 0.0], [100.0, 90.0, 70.0]],
                dtype=np.float32,
            ),
            "positive projected area",
        ),
        (
            "topk_projected_area_fraction",
            np.asarray(
                [[1.01, 0.08, 0.0], [0.10, 0.09, 0.07]],
                dtype=np.float32,
            ),
            "finite in",
        ),
        (
            "topk_projected_area_pixels",
            np.asarray(
                [[np.nan, 80.0, 0.0], [100.0, 90.0, 70.0]],
                dtype=np.float32,
            ),
            "finite and nonnegative",
        ),
    ],
)
def test_topk_shape_and_sentinel_contract_fails_closed(
    tmp_path: Path, field: str, value: np.ndarray, message: str
) -> None:
    parent = _parent(tmp_path / "parent.npz")
    payload = _r2(parent).as_npz_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_payload(payload, parent)


def test_topk_frames_must_be_subset_of_expected_causal_frames(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path / "parent.npz")
    cache = _r2(parent)
    path = tr3d_r2_cache_path(tmp_path / "r2", cache.scene_id, cache.prefix_id)
    write_tr3d_r2_cache(path, cache, parent_cache_path=parent)

    with pytest.raises(ValueError, match="expected causal frame set"):
        _load(path, parent, allowed_frame_ids=(0, 25, 50))
    with pytest.raises(ValueError, match="unique and nonnegative"):
        _validate_payload(
            cache.as_npz_payload(),
            parent,
            allowed_frame_ids=(0, 25, 25, 50, 75),
        )
    with pytest.raises(ValueError, match="integer sequence"):
        _validate_payload(
            cache.as_npz_payload(),
            parent,
            allowed_frame_ids=(0.0, 25.0, 50.0, 75.0),
        )


def test_evidence_dtype_finite_range_and_parent_lineage_fail_closed(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path / "parent.npz")
    base = _r2(parent).as_npz_payload()

    payload = dict(base)
    payload["per_view_depth_evidence"] = np.asarray(
        payload["per_view_depth_evidence"], dtype=np.float64
    )
    with pytest.raises(ValueError, match="dtype float32"):
        _validate_payload(payload, parent)

    payload = dict(base)
    evidence = payload["per_view_depth_evidence"].copy()
    evidence[0, 0, 0] = np.nan
    payload["per_view_depth_evidence"] = evidence
    with pytest.raises(ValueError, match="finite in"):
        _validate_payload(payload, parent)

    payload = dict(base)
    aggregate = payload["aggregate_depth_evidence"].copy()
    aggregate[0, 0] = 1.01
    payload["aggregate_depth_evidence"] = aggregate
    with pytest.raises(ValueError, match="finite in"):
        _validate_payload(payload, parent)

    payload = dict(base)
    counts = payload["per_view_depth_counts"].copy()
    counts[0, 0, 0] += 1
    payload["per_view_depth_counts"] = counts
    with pytest.raises(ValueError, match="per_view_point_count"):
        _validate_payload(payload, parent)

    payload = dict(base)
    aggregate_counts = payload["aggregate_depth_counts"].copy()
    aggregate_counts[0, 0] += 1
    payload["aggregate_depth_counts"] = aggregate_counts
    with pytest.raises(ValueError, match="aggregate depth counts"):
        _validate_payload(payload, parent)

    payload = dict(base)
    payload["proposal_ids"] = np.asarray([10, 999], dtype=np.int64)
    with pytest.raises(ValueError, match="absent from parent"):
        _validate_payload(payload, parent)

    payload = dict(base)
    payload["lineage_ids"] = np.asarray([100, 100], dtype=np.int64)
    with pytest.raises(ValueError, match="lineage_ids"):
        _validate_payload(payload, parent)


def test_unknown_object_and_tampered_parent_provenance_are_rejected(
    tmp_path: Path,
) -> None:
    parent = _parent(tmp_path / "parent.npz")
    base = _r2(parent).as_npz_payload()

    payload = dict(base)
    payload["unknown"] = np.asarray(1)
    with pytest.raises(ValueError, match="unknown"):
        _validate_payload(payload, parent)

    payload = dict(base)
    payload["proposal_ids"] = np.asarray([object(), object()], dtype=object)
    with pytest.raises(ValueError, match="object arrays"):
        _validate_payload(payload, parent)

    payload = dict(base)
    payload["parent_checkpoint_sha256"] = np.asarray(_sha("tampered"))
    with pytest.raises(ValueError, match="parent checkpoint provenance"):
        _validate_payload(payload, parent)
