import numpy as np
import pytest
from PIL import Image
from scipy.spatial import cKDTree

from tools.ca1m_seedless_trigger_core import (
    add_clusters, anchor_world_points, recurrence_counts, top_m,
    update_coverage, validate_grids,
)
from tools.validate_ca1m_prenms_query import load_frame


def cube(center):
    return np.asarray([[x, y, z] for x in (-.05, .05) for y in (-.05, .05)
                       for z in (-.05, .05)]) + center


def test_load_frame_metres_and_rgb_to_depth_registration(tmp_path):
    (tmp_path / 'rgb').mkdir()
    (tmp_path / 'depth').mkdir()
    Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8)).save(tmp_path / 'rgb/0.png')
    # The desired RGB centre (30, 10) maps to depth (15, 5). All other pixels
    # are 9m: using/clipping RGB coordinates on this grid would pick 9m.
    depth_mm = np.full((10, 20), 9000, dtype=np.uint16)
    depth_mm[3:8, 13:18] = 2000
    Image.fromarray(depth_mm).save(tmp_path / 'depth/0.png')
    _, rgb, depth_m = load_frame(tmp_path, 0)
    k_rgb = np.array([[20., 0, 20], [0, 20, 10], [0, 0, 1]])
    k_depth = np.diag([.5, .5, 1.]) @ k_rgb
    pose = np.eye(4)
    pose[:3, 3] = [3, 4, 5]
    ids, worlds, depths, stats = anchor_world_points(
        np.array([[30., 10.]]), [0], depth_m, pose, k_rgb, k_depth, rgb.shape)
    assert ids.tolist() == [0]
    np.testing.assert_allclose(depths, [2.])
    np.testing.assert_allclose(worlds, [[4., 4., 7.]])
    assert stats['valid_world_points'] == 1
    assert stats['out_of_range'] == 0


@pytest.mark.parametrize('bad_depth', [np.ones((10, 20, 1)), np.ones((10, 20), dtype=np.uint16)])
def test_depth_contract_rejects_channels_and_implicit_millimetres(bad_depth):
    with pytest.raises(ValueError, match='HxW floating-point'):
        validate_grids((20, 40, 3), bad_depth, np.eye(3), np.diag([.5, .5, 1.]))


def test_invalid_depth_does_not_make_world_points():
    ids, worlds, depths, stats = anchor_world_points(
        np.array([[4., 4.]]), [0], np.full((10, 10), np.nan),
        np.eye(4), np.eye(3), np.eye(3), (10, 10, 3))
    assert ids.size == depths.size == 0 and worlds.shape == (0, 3)
    assert stats['no_valid_depth'] == 1


def test_recurrence_unique_frames_radius_and_camera_translation():
    k = np.array([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    worlds = np.array([[0., 0, 2.], [0., 0, -2.]])
    frames = [0, 20, 40, 60]
    transforms = {f: np.eye(4) for f in frames}
    transforms[40][0, 3] = -.2  # projects the valid point at x=40 in frame 40
    trees = {0: cKDTree([[50, 50]]), 20: cKDTree([[80, 50]] * 100),
             40: cKDTree([[70, 50]]), 60: cKDTree([[80.001, 50]])}
    hits = recurrence_counts(worlds, 0, frames, transforms, k,
                             {f: (100, 100) for f in frames}, trees, 30.)
    assert hits.tolist() == [2, 0]  # exact boundary accepted; own frame excluded


def test_top_m_nested_prefix_and_stable_score_ties():
    ids = np.array([1, 3, 4, 5])
    scores = np.array([1., .6, 1., .9, .9, .7])
    ranked = top_m(ids, scores, 300)
    assert ranked.tolist() == [3, 4, 5, 1]
    np.testing.assert_array_equal(top_m(ids, scores, 2), ranked[:2])
    assert top_m([], scores, 150).size == 0


def test_cluster_support_accumulates_across_frames_not_duplicate_anchors():
    clusters = {}
    box = cube(np.array([.15, .15, 2.55]))
    add_clusters(clusters, 0, [4, 5], np.array([box, box]), [.1, .2])
    state = next(iter(clusters.values()))
    assert state['frames'] == {0} and state['observations'] == 2
    add_clusters(clusters, 20, [7], np.array([box]), [.3])
    add_clusters(clusters, 40, [8], np.array([box]), [.2])
    assert len(clusters) == 1 and state['frames'] == {0, 20, 40}
    assert state['rank'] == (-.3, 20, 7)


def test_coverage_uses_global_missed_gt_id_not_position_in_subset():
    gt = np.array([cube(np.array([x, 0., 2.])) for x in (0., 3., 6.)])
    support, best = {2: set()}, {2: (0., None)}
    update_coverage(support, best, [2], gt[0:1], gt, 0)
    assert support[2] == set()  # overlap with GT 0 cannot be attributed to missed GT 2
    update_coverage(support, best, [2], gt[2:3], gt, 20)
    assert support[2] == {20}
    np.testing.assert_allclose(best[2][1], gt[2])
