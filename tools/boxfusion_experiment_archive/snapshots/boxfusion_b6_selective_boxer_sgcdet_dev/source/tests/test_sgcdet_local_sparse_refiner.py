from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from boxfusion.sgcdet_local_sparse_refiner import (
    SGCDET_SPARSE_REFINER_REFERENCE,
    SGCDetInspiredLocalSparseRefiner,
    SGCDetLocalSparseRefinerConfig,
    apply_sgcdet_sparse_residual_numpy,
    build_sgcdet_sparse_refiner,
    load_sgcdet_sparse_refiner_checkpoint,
    make_sgcdet_sparse_refiner_checkpoint,
    save_sgcdet_sparse_refiner_checkpoint,
    stable_hard_topk,
)


def _inputs(batch=2, views=3, points=40):
    values = torch.linspace(-0.8, 0.8, batch * views * points * 3)
    points_local = values.reshape(batch, views, points, 3)
    point_mask = torch.ones(batch, views, points, dtype=torch.bool)
    point_mask[:, -1, -5:] = False
    local_boxes = torch.tensor(
        [[0.0, 0.0, 0.0, 2.0, 1.5, 1.0]] * batch,
        dtype=torch.float32,
    )
    quality = torch.linspace(0.05, 0.95, batch * 12).reshape(batch, 12)
    view_features = torch.linspace(
        -1.0, 1.0, batch * views * 9
    ).reshape(batch, views, 9)
    view_mask = torch.ones(batch, views, dtype=torch.bool)
    return (
        points_local,
        point_mask,
        local_boxes,
        quality,
        view_features,
        view_mask,
    )


def test_config_is_exact_coarse_to_fine_top25():
    config = SGCDetLocalSparseRefinerConfig().validated()
    assert config.coarse_grid_size == (8, 8, 4)
    assert config.fine_grid_size == (16, 16, 8)
    assert config.fine_voxel_count == 2048
    assert config.selected_token_count == 512
    assert config.view_feature_dim == 9
    with pytest.raises(ValueError, match="Top-25"):
        SGCDetLocalSparseRefinerConfig(topk_fraction=0.5).validated()
    with pytest.raises(ValueError, match="exactly twice"):
        SGCDetLocalSparseRefinerConfig(
            fine_grid_size=(16, 16, 4)
        ).validated()


def test_stable_hard_topk_is_detached_and_tie_stable():
    scores = torch.tensor(
        [[1.0, 2.0, 2.0, 0.0], [4.0, 4.0, 3.0, 4.0]],
        requires_grad=True,
    )
    indices, mask = stable_hard_topk(scores, 2)
    assert indices.tolist() == [[1, 2], [0, 1]]
    assert mask.tolist() == [
        [False, True, True, False],
        [True, True, False, False],
    ]
    assert not indices.requires_grad
    assert not mask.requires_grad
    with pytest.raises(ValueError, match="cannot exceed"):
        stable_hard_topk(scores, 5)


def test_forward_identity_shapes_sparse_stats_and_defaults():
    torch.manual_seed(4)
    model = SGCDetInspiredLocalSparseRefiner().eval()
    inputs = _inputs()
    with torch.no_grad():
        output = model(*inputs)

    assert output["center_residual"].shape == (2, 3)
    assert output["center_residual_fraction"].shape == (2, 3)
    assert output["log_dimension_residual"].shape == (2, 3)
    assert torch.equal(
        output["center_residual"], torch.zeros(2, 3)
    )
    assert torch.equal(
        output["log_dimension_residual"], torch.zeros(2, 3)
    )
    assert torch.allclose(
        output["candidate_iou"], torch.full((2,), 0.10), atol=1e-7
    )
    assert torch.allclose(
        output["improvement_probability"],
        torch.full((2,), 0.01),
        atol=1e-7,
    )
    assert torch.allclose(
        output["uncertainty"], torch.full((2,), 0.50), atol=1e-7
    )
    assert output["coarse_occupancy_logits"].shape == (2, 256)
    assert output["coarse_occupancy_targets"].shape == (2, 256)
    assert output["occupancy_logits"].shape == (2, 2048)
    assert output["occupancy_targets"].shape == (2, 2048)
    assert output["selected_indices"].shape == (2, 512)
    assert output["selected_mask"].shape == (2, 2048)
    assert output["selected_mask"].dtype == torch.bool
    assert output["selected_mask"].sum(dim=1).tolist() == [512, 512]
    stats = output["selected_stats"]
    assert stats["count"].tolist() == [512, 512]
    assert torch.equal(stats["fraction"], torch.full((2,), 0.25))
    assert torch.all(stats["valid_point_count"] > 0)


def test_forward_accepts_single_cloud_and_validates_inputs():
    model = SGCDetInspiredLocalSparseRefiner().eval()
    points, mask, boxes, quality, features, _ = _inputs(
        batch=1, views=1, points=20
    )
    with torch.no_grad():
        output = model(
            points[:, 0],
            mask[:, 0],
            boxes,
            quality,
            features[:, 0],
            torch.ones(1, dtype=torch.bool),
        )
    assert output["occupancy_logits"].shape == (1, 2048)

    with pytest.raises(ValueError, match="quality_features"):
        model(points[:, 0], mask[:, 0], boxes, quality[:, :-1])
    with pytest.raises(TypeError, match="Boolean"):
        model(points, mask.float(), boxes, quality, features)
    with pytest.raises(ValueError, match="valid point"):
        model(points, torch.zeros_like(mask), boxes, quality, features)
    with pytest.raises(ValueError, match="positive"):
        bad_boxes = boxes.clone()
        bad_boxes[:, 3] = 0.0
        model(points, mask, bad_boxes, quality, features)


def test_view_mask_removes_masked_view_from_occupancy():
    torch.manual_seed(8)
    model = SGCDetInspiredLocalSparseRefiner().eval()
    points, mask, boxes, quality, features, _ = _inputs(
        batch=1, views=2, points=24
    )
    view_mask = torch.tensor([[True, False]])
    altered = points.clone()
    altered[:, 1] = torch.flip(altered[:, 1], dims=(1,)) * 50.0
    with torch.no_grad():
        first = model(
            points, mask, boxes, quality, features, view_mask
        )["occupancy_logits"]
        second = model(
            altered, mask, boxes, quality, features, view_mask
        )["occupancy_logits"]
    assert torch.equal(first, second)


def test_outputs_remain_inside_hard_bounds():
    config = SGCDetLocalSparseRefinerConfig()
    model = SGCDetInspiredLocalSparseRefiner(config).eval()
    with torch.no_grad():
        model.output_layer.bias[:6] = torch.tensor(
            [100.0, -100.0, 100.0, -100.0, 100.0, -100.0]
        )
        model.output_layer.bias[6:] = torch.tensor([100.0, -100.0, 100.0])
    inputs = _inputs(batch=1, views=2, points=20)
    with torch.no_grad():
        output = model(*(value[:1] for value in inputs))
    assert torch.all(
        output["center_residual_fraction"].abs()
        <= config.max_center_fraction + 1e-7
    )
    assert torch.all(
        output["log_dimension_residual"].abs()
        <= config.max_log_dimension_residual + 1e-7
    )
    assert torch.all((output["candidate_iou"] >= 0) & (output["candidate_iou"] <= 1))
    assert torch.all(
        (output["improvement_probability"] >= 0)
        & (output["improvement_probability"] <= 1)
    )
    assert torch.all(output["uncertainty"] <= config.maximum_uncertainty)
    assert torch.all(output["uncertainty"] >= config.minimum_uncertainty)


def test_checkpoint_roundtrip_builder_and_strict_schema(tmp_path: Path):
    torch.manual_seed(12)
    model = SGCDetInspiredLocalSparseRefiner().eval()
    path = tmp_path / "sparse_refiner.pt"
    result_path = save_sgcdet_sparse_refiner_checkpoint(
        model, path, metadata={"split": "train-only", "epoch": 3}
    )
    assert result_path == path
    rebuilt = build_sgcdet_sparse_refiner(
        enabled=True, checkpoint_path=path
    )
    assert rebuilt is not None
    assert not rebuilt.training
    assert rebuilt.config.architecture_dict() == model.config.architecture_dict()
    for key, value in model.state_dict().items():
        assert torch.equal(value, rebuilt.state_dict()[key])
    assert build_sgcdet_sparse_refiner(enabled=False) is None

    payload = make_sgcdet_sparse_refiner_checkpoint(model)
    assert payload["reference"] == SGCDET_SPARSE_REFINER_REFERENCE
    payload["schema"] = "wrong"
    invalid = tmp_path / "invalid.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match="schema"):
        load_sgcdet_sparse_refiner_checkpoint(model, invalid)

    different = SGCDetInspiredLocalSparseRefiner(
        SGCDetLocalSparseRefinerConfig(head_hidden_dim=64)
    )
    with pytest.raises(ValueError, match="config"):
        load_sgcdet_sparse_refiner_checkpoint(different, path)


def test_forward_is_deterministic_and_occupancy_keeps_gradient():
    torch.manual_seed(21)
    model = SGCDetInspiredLocalSparseRefiner()
    inputs = _inputs(batch=1, views=2, points=30)
    first = model(*(value[:1] for value in inputs))
    second = model(*(value[:1] for value in inputs))
    for key in (
        "occupancy_logits",
        "selected_indices",
        "center_residual",
        "candidate_iou",
    ):
        assert torch.equal(first[key], second[key])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        first["occupancy_logits"], first["occupancy_targets"]
    )
    loss.backward()
    assert model.fine_occupancy_head[-1].weight.grad is not None
    assert torch.isfinite(model.fine_occupancy_head[-1].weight.grad).all()


def test_numpy_residual_application_clips_untrusted_values():
    config = SGCDetLocalSparseRefinerConfig()
    boxes = np.array([[0, 0, 0, 2, 4, 6, 77]], dtype=np.float32)
    result = apply_sgcdet_sparse_residual_numpy(
        boxes,
        np.array([[100, -100, 0]], dtype=np.float32),
        np.array([[100, -100, 0]], dtype=np.float32),
        config=config,
    )
    np.testing.assert_allclose(result[0, :3], [0.3, -0.6, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[0, 3:6], [2.5, 3.2, 6.0], atol=1e-6)
    assert result[0, 6] == 77
