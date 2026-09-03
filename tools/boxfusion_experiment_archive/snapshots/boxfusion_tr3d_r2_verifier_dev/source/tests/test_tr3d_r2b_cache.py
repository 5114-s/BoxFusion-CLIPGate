from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from boxfusion.tr3d_r2_cache import (
    depth_evidence_fractions,
    make_tr3d_r2_cache,
    write_tr3d_r2_cache,
)
from boxfusion.tr3d_r2b_cache import (
    PAIRWISE_COSINE_STATISTIC_NAMES,
    derive_feature_aggregates,
    load_tr3d_r2b_cache,
    make_tr3d_r2b_cache,
    sha256_file,
    tr3d_r2b_cache_path,
    validate_tr3d_r2b_payload,
    write_tr3d_r2b_cache,
)
from boxfusion.tr3d_residual_cache import (
    TR3DResidualCache,
    transform_sha256,
    write_tr3d_residual_cache,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


R2_PROVENANCE = {
    "prefix_manifest_row_sha256": _sha("prefix-row"),
    "frame_artifact_tree_sha256": _sha("frame-tree"),
    "r2_config_sha256": _sha("r2-config"),
    "r2_code_sha256": _sha("r2-code"),
}
FEATURE_PROVENANCE = {
    "feature_checkpoint_sha256": _sha("dino-checkpoint"),
    "feature_config_sha256": _sha("dino-config"),
    "feature_code_sha256": _sha("dino-code"),
}


def _tr3d_parent(path: Path) -> Path:
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
    transform = np.eye(4, dtype=np.float64)
    cache = TR3DResidualCache(
        scene_id="scene0001_00",
        sample_idx="scene0001_00:p050",
        prefix_id="p050",
        prefix_fraction=0.5,
        boxes_world=boxes,
        corners_world=corners,
        aligned_to_unaligned=transform,
        axis_alignment_sha256=transform_sha256(transform),
        scores_3d=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        labels_3d=np.zeros(3, dtype=np.int64),
        proposal_ids=np.asarray([10, 20, 30], dtype=np.int64),
        point_count=np.asarray([100, 80, 60], dtype=np.int32),
        voxel_size=0.01,
        runtime_s=0.2,
        num_input_points=1000,
        checkpoint_sha256=_sha("tr3d-checkpoint"),
        config_sha256=_sha("tr3d-config"),
        source_scene_sha256=_sha("source-scene"),
    )
    write_tr3d_residual_cache(path, cache)
    return path


def _r2a(path: Path, tr3d_parent: Path, *, runtime_s: float = 0.1) -> Path:
    per_view_counts = np.asarray(
        [
            [[20, 4, 4, 2], [10, 4, 4, 2], [0, 0, 0, 0]],
            [[30, 4, 4, 2], [18, 4, 5, 3], [10, 3, 5, 2]],
        ],
        dtype=np.int32,
    )
    aggregate_counts = per_view_counts.sum(axis=1, dtype=np.int64)
    cache = make_tr3d_r2_cache(
        parent_cache_path=tr3d_parent,
        **R2_PROVENANCE,
        proposal_ids=np.asarray([10, 30], dtype=np.int64),
        lineage_ids=np.asarray([100, 300], dtype=np.int64),
        topk_frame_ids=np.asarray(
            [[0, 25, -1], [25, 50, 75]], dtype=np.int64
        ),
        topk_view_valid=np.asarray(
            [[True, True, False], [True, True, True]], dtype=np.bool_
        ),
        topk_projected_area_pixels=np.asarray(
            [[120, 80, 0], [100, 90, 70]], dtype=np.float32
        ),
        topk_projected_area_fraction=np.asarray(
            [[0.12, 0.08, 0], [0.10, 0.09, 0.07]], dtype=np.float32
        ),
        per_view_depth_evidence=depth_evidence_fractions(per_view_counts),
        per_view_depth_counts=per_view_counts,
        per_view_point_count=per_view_counts.sum(axis=2).astype(np.int32),
        aggregate_depth_evidence=depth_evidence_fractions(aggregate_counts),
        aggregate_depth_counts=aggregate_counts,
        aggregate_view_count=np.asarray([2, 3], dtype=np.int32),
        aggregate_point_count=aggregate_counts.sum(axis=1),
        runtime_s=runtime_s,
    )
    write_tr3d_r2_cache(path, cache, parent_cache_path=tr3d_parent)
    return path


def _feature_inputs(dtype=np.float16):
    vectors = np.asarray(
        [
            [[1, 0, 0, 0], [0.8, 0.6, 0, 0], [0, 0, 0, 0]],
            [[0, 1, 0, 0], [0, 0, 0, 0], [0, 0.8, 0.6, 0]],
        ],
        dtype=dtype,
    )
    valid = np.asarray(
        [[True, True, False], [True, False, True]], dtype=np.bool_
    )
    counts = np.asarray([[5, 4, 0], [3, 0, 2]], dtype=np.int32)
    return vectors, valid, counts


def _r2b(r2a: Path, tr3d_parent: Path, dtype=np.float16):
    vectors, valid, counts = _feature_inputs(dtype)
    return make_tr3d_r2b_cache(
        parent_r2a_cache_path=r2a,
        parent_tr3d_cache_path=tr3d_parent,
        **{
            f"parent_{name}": value for name, value in R2_PROVENANCE.items()
        },
        **FEATURE_PROVENANCE,
        per_view_feature_valid=valid,
        per_view_feature_count=counts,
        per_view_feature_vector=vectors,
        runtime_s=0.03,
    )


def _load(path: Path, r2a: Path, tr3d_parent: Path, **overrides):
    expected = {
        **{
            f"parent_{name}": value for name, value in R2_PROVENANCE.items()
        },
        **FEATURE_PROVENANCE,
    }
    expected.update(overrides)
    return load_tr3d_r2b_cache(
        path,
        parent_r2a_cache_path=r2a,
        parent_tr3d_cache_path=tr3d_parent,
        expected_scene_id="scene0001_00",
        expected_prefix_id="p050",
        expected_prefix_fraction=0.5,
        **{f"expected_{name}": value for name, value in expected.items()},
    )


def _validate(payload, r2a: Path, tr3d_parent: Path):
    return validate_tr3d_r2b_payload(
        payload,
        parent_r2a_cache_path=r2a,
        parent_tr3d_cache_path=tr3d_parent,
        **{
            f"expected_parent_{name}": value
            for name, value in R2_PROVENANCE.items()
        },
        **{
            f"expected_{name}": value
            for name, value in FEATURE_PROVENANCE.items()
        },
    )


def test_roundtrip_is_observer_only_readonly_and_non_overwriting(
    tmp_path: Path,
) -> None:
    tr3d = _tr3d_parent(tmp_path / "parent.npz")
    r2a = _r2a(tmp_path / "parent.r2a.npz", tr3d)
    cache = _r2b(r2a, tr3d)
    path = tr3d_r2b_cache_path(tmp_path / "r2b", cache.scene_id, "p050")
    write_tr3d_r2b_cache(
        path,
        cache,
        parent_r2a_cache_path=r2a,
        parent_tr3d_cache_path=tr3d,
    )
    loaded = _load(path, r2a, tr3d)

    assert PAIRWISE_COSINE_STATISTIC_NAMES == (
        "mean",
        "median",
        "min",
        "max",
        "std",
    )
    assert loaded.parent_r2a_cache_sha256 == sha256_file(r2a)
    assert loaded.proposal_count == 2
    assert loaded.topk == 3
    assert loaded.feature_dim == 4
    assert loaded.per_view_feature_vector.dtype == np.float16
    assert not loaded.per_view_feature_vector.flags.writeable
    assert not loaded.pairwise_cosine_median.flags.writeable
    assert path.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError):
        loaded.aggregate_feature_vector[0, 0] = 0
    with pytest.raises(FileExistsError, match="immutable R2b"):
        write_tr3d_r2b_cache(
            path,
            cache,
            parent_r2a_cache_path=r2a,
            parent_tr3d_cache_path=tr3d,
        )


def test_float32_and_zero_or_single_view_statistics_are_canonical() -> None:
    vectors = np.asarray(
        [[[0, 0], [0, 0]], [[1, 0], [0, 0]]], dtype=np.float32
    )
    valid = np.asarray([[False, False], [True, False]], dtype=np.bool_)
    counts = np.asarray([[0, 0], [7, 0]], dtype=np.int32)
    result = derive_feature_aggregates(vectors, valid, counts)
    np.testing.assert_array_equal(result["aggregate_view_count"], [0, 1])
    np.testing.assert_array_equal(result["pairwise_cosine_count"], [0, 0])
    np.testing.assert_array_equal(
        result["aggregate_feature_vector"], [[0, 0], [1, 0]]
    )
    for name in PAIRWISE_COSINE_STATISTIC_NAMES:
        np.testing.assert_array_equal(
            result[f"pairwise_cosine_{name}"], [0, 0]
        )


def test_exact_parent_bytes_and_external_hashes_fail_closed(
    tmp_path: Path,
) -> None:
    tr3d = _tr3d_parent(tmp_path / "parent.npz")
    r2a = _r2a(tmp_path / "parent.r2a.npz", tr3d)
    cache = _r2b(r2a, tr3d)
    path = tmp_path / "feature.r2b.npz"
    write_tr3d_r2b_cache(
        path,
        cache,
        parent_r2a_cache_path=r2a,
        parent_tr3d_cache_path=tr3d,
    )
    alternate = _r2a(tmp_path / "alternate.r2a.npz", tr3d, runtime_s=0.2)
    with pytest.raises(ValueError, match="parent R2a cache provenance"):
        _load(path, alternate, tr3d)
    with pytest.raises(ValueError, match="feature_checkpoint_sha256"):
        _load(path, r2a, tr3d, feature_checkpoint_sha256=_sha("wrong"))
    with pytest.raises(ValueError, match="parent_r2_code_sha256"):
        _load(path, r2a, tr3d, parent_r2_code_sha256=_sha("wrong-code"))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "proposal_ids",
            np.asarray([30, 10], dtype=np.int64),
            "proposal_ids disagrees",
        ),
        (
            "lineage_ids",
            np.asarray([100, 999], dtype=np.int64),
            "lineage_ids disagrees",
        ),
        (
            "topk_frame_ids",
            np.asarray([[0, 50, -1], [25, 50, 75]], dtype=np.int64),
            "topk_frame_ids disagrees",
        ),
        (
            "topk_view_valid",
            np.asarray(
                [[True, True, True], [True, True, True]], dtype=np.bool_
            ),
            "topk_view_valid disagrees",
        ),
    ],
)
def test_exact_r2a_row_identity_is_required(
    tmp_path: Path, field: str, value: np.ndarray, message: str
) -> None:
    tr3d = _tr3d_parent(tmp_path / "parent.npz")
    r2a = _r2a(tmp_path / "parent.r2a.npz", tr3d)
    payload = _r2b(r2a, tr3d).as_npz_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        _validate(payload, r2a, tr3d)


def test_feature_sentinel_and_derived_statistics_fail_closed(
    tmp_path: Path,
) -> None:
    tr3d = _tr3d_parent(tmp_path / "parent.npz")
    r2a = _r2a(tmp_path / "parent.r2a.npz", tr3d)
    base = _r2b(r2a, tr3d).as_npz_payload()

    payload = dict(base)
    vectors = payload["per_view_feature_vector"].copy()
    vectors[0, 2, 0] = 1
    payload["per_view_feature_vector"] = vectors
    with pytest.raises(ValueError, match="zero vectors"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    valid = payload["per_view_feature_valid"].copy()
    valid[0, 2] = True
    payload["per_view_feature_valid"] = valid
    with pytest.raises(ValueError, match="feature validity"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    counts = payload["per_view_feature_count"].copy()
    counts[0, 2] = 1
    payload["per_view_feature_count"] = counts
    valid = payload["per_view_feature_valid"].copy()
    valid[0, 2] = True
    payload["per_view_feature_valid"] = valid
    with pytest.raises(ValueError, match="feature-valid slots"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    aggregate = payload["aggregate_feature_vector"].copy()
    aggregate[0, 0] += np.float32(0.01)
    payload["aggregate_feature_vector"] = aggregate
    with pytest.raises(ValueError, match="aggregate_feature_vector"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    median = payload["pairwise_cosine_median"].copy()
    median[0] += np.float32(0.01)
    payload["pairwise_cosine_median"] = median
    with pytest.raises(ValueError, match="pairwise_cosine_median"):
        _validate(payload, r2a, tr3d)


def test_dtype_nonfinite_observer_contract_and_unknown_fields(
    tmp_path: Path,
) -> None:
    tr3d = _tr3d_parent(tmp_path / "parent.npz")
    r2a = _r2a(tmp_path / "parent.r2a.npz", tr3d)
    base = _r2b(r2a, tr3d).as_npz_payload()

    payload = dict(base)
    payload["per_view_feature_vector"] = np.asarray(
        payload["per_view_feature_vector"], dtype=np.float64
    )
    with pytest.raises(ValueError, match="float16 or float32"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    vectors = payload["per_view_feature_vector"].copy()
    vectors[0, 0, 0] = np.nan
    payload["per_view_feature_vector"] = vectors
    with pytest.raises(ValueError, match="finite"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    payload["mutation_enabled"] = np.asarray(True, dtype=np.bool_)
    with pytest.raises(ValueError, match="enables mutation"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    payload["applied_count"] = np.asarray(1, dtype=np.int64)
    with pytest.raises(ValueError, match="applied_count"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    payload["unknown"] = np.asarray(1)
    with pytest.raises(ValueError, match="unknown"):
        _validate(payload, r2a, tr3d)

    payload = dict(base)
    payload["proposal_ids"] = np.asarray([object(), object()], dtype=object)
    with pytest.raises(ValueError, match="object arrays"):
        _validate(payload, r2a, tr3d)
