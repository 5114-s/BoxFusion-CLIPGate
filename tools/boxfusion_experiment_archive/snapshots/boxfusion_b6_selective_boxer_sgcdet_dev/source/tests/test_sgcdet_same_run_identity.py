import pickle

import numpy as np
import pytest

from tools.build_sgcdet_same_run_identity import build


def _corners(center):
    center = np.asarray(center, dtype=np.float32)
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
    return center[None] + 0.5 * signs


def _fixture(tmp_path, *, accepted=True):
    active = tmp_path / "active"
    diagnostics = tmp_path / "diagnostics"
    output = tmp_path / "identity"
    active.mkdir()
    diagnostics.mkdir()
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("scene_test\n", encoding="utf-8")

    pre = np.stack([_corners([0, 0, 0]), _corners([2, 0, 0])])
    post = pre.copy()
    post[1] += np.asarray([0.1, 0.0, 0.0], dtype=np.float32)
    with (active / "scene_test_boxes.pkl").open("wb") as handle:
        pickle.dump(
            [[(0, post[0].copy(), 0.8), (0, post[1].copy(), 0.7)]],
            handle,
        )
    np.savez_compressed(
        diagnostics / "scene_test_tracks.npz",
        output_geometry_schema=np.asarray(
            "boxfusion.full_output_geometry_prepost.v1"
        ),
        output_pre_geometry_corners=pre.astype(np.float32),
        output_post_geometry_corners=post.astype(np.float32),
        output_refit_applied=np.asarray([False, True]),
        result_indices=np.asarray([1], dtype=np.int64),
        sparse_accepted=np.asarray([accepted]),
    )
    return active, diagnostics, scene_list, output, pre


def test_builds_exact_same_run_identity_and_manifest(tmp_path):
    active, diagnostics, scene_list, output, pre = _fixture(tmp_path)
    report = build(active, diagnostics, scene_list, output)

    assert report["total_rows"] == 2
    assert report["changed_rows"] == 1
    assert report["changed_scenes"] == 1
    with (output / "scene_test_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    np.testing.assert_array_equal(rows[0][1], pre[0])
    np.testing.assert_array_equal(rows[1][1], pre[1])
    assert [row[2] for row in rows] == [0.8, 0.7]


def test_rejects_geometry_change_without_sparse_acceptance(tmp_path):
    active, diagnostics, scene_list, output, _ = _fixture(
        tmp_path, accepted=False
    )
    with pytest.raises(ValueError, match="mapped sparse_accepted"):
        build(active, diagnostics, scene_list, output)

