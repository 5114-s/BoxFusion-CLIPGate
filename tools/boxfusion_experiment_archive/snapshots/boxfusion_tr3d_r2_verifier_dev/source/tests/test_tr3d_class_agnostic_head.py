import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("MinkowskiEngine")
pytest.importorskip("mmdet3d")

from mmdet3d.structures import DepthInstance3DBoxes

from tr3d_plugin.class_agnostic_head import TR3DClassAgnosticHead


def _head():
    return TR3DClassAgnosticHead(
        in_channels=8,
        num_reg_outs=6,
        voxel_size=0.01,
        pts_center_threshold=1,
    )


def test_each_box_receives_positive_on_every_feature_level():
    head = _head()
    points = [
        torch.tensor([[0.0, 0.0, 0.0], [4.0, 4.0, 4.0]]),
        torch.tensor([[0.1, 0.0, 0.0], [5.0, 5.0, 5.0]]),
    ]
    boxes = DepthInstance3DBoxes(
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        box_dim=6,
        with_yaw=False,
        origin=(0.5, 0.5, 0.5),
    )
    _, labels = head.get_targets(
        points, boxes, torch.tensor([0], dtype=torch.long), num_classes=1)

    assert labels.tolist() == [0, 1, 0, 1]


def test_nonzero_semantic_labels_fail_closed():
    head = _head()
    points = [torch.tensor([[0.0, 0.0, 0.0]])]
    boxes = DepthInstance3DBoxes(
        torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]),
        box_dim=6,
        with_yaw=False,
        origin=(0.5, 0.5, 0.5),
    )
    with pytest.raises(ValueError, match="all gt_labels"):
        head.get_targets(
            points, boxes, torch.tensor([1], dtype=torch.long), num_classes=1)
