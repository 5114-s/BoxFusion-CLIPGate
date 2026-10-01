import numpy as np
import pytest
from PIL import Image

from tools.run_scannet_seedless_full import anchor_points, load_frame, forbid_gt


def test_calibrated_rgb_depth_mapping_and_metres(tmp_path):
    (tmp_path / 'color').mkdir()
    (tmp_path / 'depth').mkdir()
    Image.fromarray(np.zeros((40, 80, 3), np.uint8)).save(tmp_path / 'color/0.jpg')
    depth = np.full((20, 40), 9000, np.uint16)
    # RGB ray (x=.5,y=0) maps to depth x=24,y=8, not resize x=30,y=10.
    depth[6:11, 22:27] = 2000
    Image.fromarray(depth).save(tmp_path / 'depth/0.png')
    _, rgb, dm = load_frame(tmp_path, 0)
    kr = np.array([[40., 0, 40], [0, 40, 20], [0, 0, 1]])
    kd = np.array([[20., 0, 14], [0, 20, 8], [0, 0, 1]])
    pose = np.eye(4)
    pose[:3, 3] = [3, 4, 5]
    ids, worlds, depths, counts = anchor_points(np.array([[60., 20.]]), [0], dm, pose, kr, kd, rgb.shape)
    assert ids.tolist() == [0]
    np.testing.assert_allclose(depths, [2.])
    np.testing.assert_allclose(worlds, [[4., 4., 7.]])
    assert counts['valid_world_points'] == 1
    assert rgb.shape == (40, 80, 3)


def test_outside_depth_grid_is_rejected_instead_of_clipped():
    kr = np.eye(3)
    kd = np.array([[1., 0, 100], [0, 1, 0], [0, 0, 1]])
    ids, _, _, counts = anchor_points(np.array([[4., 4.]]), [0], np.ones((10, 10)),
                                     np.eye(4), kr, kd, (10, 10, 3))
    assert len(ids) == 0 and counts['outside_image'] == 1


def test_depth_contract():
    with pytest.raises(ValueError, match='floating point'):
        anchor_points(np.zeros((1, 2)), [0], np.ones((10, 10), np.uint16),
                      np.eye(4), np.eye(3), np.eye(3), (10, 10, 3))


@pytest.mark.parametrize('path', ['scene_bbox.npy', 'scene.aggregation.json', 'scene_vert.npy'])
def test_worker_gt_guard(path):
    with pytest.raises(RuntimeError, match='GT access forbidden'):
        forbid_gt('open', (path, 'r'))
    forbid_gt('open', ('scene/depth/0.png', 'r'))
