from __future__ import annotations

import hashlib
import math

import pytest
import torch

from boxfusion.p1_geometry_loss import decode_p1g_residual_aabb
from boxfusion.p1_geometry_refiner import (
    P1G_ARCHITECTURE,
    P1G_BASE_DECODER,
    P1G_BASE_REGRESSION_ENCODING,
    P1G_CHECKPOINT_SCHEMA,
    P1G_REGRESSION_ENCODING,
    P1GeometryRegressionHead,
    load_p1g_checkpoint,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_geometry_head_shape_zero_initialization_and_config():
    model = P1GeometryRegressionHead(hidden_dim=8)
    assert tuple(name for name, _ in model.named_parameters()) == (
        "correction.weight",
        "correction.bias",
    )
    values = model(torch.randn((3, 8)))
    assert values.shape == (3, 6)
    torch.testing.assert_close(values, torch.zeros_like(values))
    config = model.model_config(
        max_center_offset=1.0,
        min_box_extent=0.08,
        max_box_extent=4.0,
    )
    assert config["architecture"] == P1G_ARCHITECTURE
    assert config["regression_encoding"] == P1G_REGRESSION_ENCODING
    assert (
        config["base_regression_encoding"]
        == P1G_BASE_REGRESSION_ENCODING
    )
    assert config["base_decoder"] == P1G_BASE_DECODER
    reconstructed = P1GeometryRegressionHead.from_model_config(config)
    assert reconstructed.hidden_dim == 8


def test_epoch_minus_one_head_preserves_frozen_p1s_clip_exp_decode():
    model = P1GeometryRegressionHead(hidden_dim=8)
    raw = torch.tensor(
        [
            [0.25, -0.75, 0.0, math.log(0.5), 0.0, math.log(2.0)],
            [10.0, -10.0, 0.0, -10.0, 10.0, math.log(1.0)],
        ],
        dtype=torch.float64,
    )
    anchors = torch.tensor(
        [[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]], dtype=torch.float64
    )
    correction = model(
        torch.randn((2, 8), dtype=torch.float32)
    ).to(dtype=torch.float64)
    observed = decode_p1g_residual_aabb(
        raw,
        correction,
        anchors,
        max_center_offset=1.0,
        min_box_extent=0.08,
        max_box_extent=4.0,
        adapter_epsilon=1e-6,
    )
    expected = torch.cat(
        (
            anchors + torch.clamp(raw[:, :3], -1.0, 1.0),
            torch.exp(
                torch.clamp(
                    raw[:, 3:],
                    math.log(0.08),
                    math.log(4.0),
                )
            ),
        ),
        dim=1,
    )
    torch.testing.assert_close(
        observed[0], expected[0], rtol=1e-12, atol=1e-12
    )
    center_error = torch.abs(observed[1, :3] - expected[1, :3])
    log_extent_error = torch.abs(
        torch.log(observed[1, 3:]) - torch.log(expected[1, 3:])
    ) / (math.log(4.0) - math.log(0.08))
    assert float(center_error.max()) <= 1.01e-6
    assert float(log_extent_error.max()) <= 1.01e-6


def test_geometry_head_rejects_invalid_features():
    model = P1GeometryRegressionHead(hidden_dim=8)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros((2, 7)))
    invalid = torch.zeros((2, 8))
    invalid[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(invalid)
    with pytest.raises(ValueError, match="exceed"):
        model.model_config(
            max_center_offset=1.0,
            min_box_extent=1.0,
            max_box_extent=0.5,
        )


def test_checkpoint_loader_binds_base_and_scene_splits(tmp_path):
    base_sha = _sha("p1s")
    model = P1GeometryRegressionHead(hidden_dim=8)
    config = model.model_config(
        max_center_offset=1.0,
        min_box_extent=0.08,
        max_box_extent=4.0,
    )
    payload = {
        "schema": P1G_CHECKPOINT_SCHEMA,
        "architecture": P1G_ARCHITECTURE,
        "regression_encoding": P1G_REGRESSION_ENCODING,
        "initialization": (
            "zero_residual_correction_function_preserving_v2"
        ),
        "observer_only": True,
        "uses_ground_truth": False,
        "class_agnostic": True,
        "semantic_features": False,
        "model_config": config,
        "decoder_config": {
            "encoding": P1G_REGRESSION_ENCODING,
            "adapter_epsilon": config["adapter_epsilon"],
            "max_center_offset": config["max_center_offset"],
            "min_box_extent": config["min_box_extent"],
            "max_box_extent": config["max_box_extent"],
        },
        "state_dict": model.state_dict(),
        "provenance": {
            "p1s_checkpoint_sha256": base_sha,
            "fit_scene_ids": ["scene0000_00"],
            "cal_scene_ids": ["scene0001_00"],
            "audit_scene_ids": ["scene0002_00"],
            "forbidden_overlap": [],
            "fit_scene_list_sha256": _sha("fit"),
            "cal_scene_list_sha256": _sha("cal"),
            "audit_scene_list_sha256": _sha("audit"),
            "forbidden_scene_list_sha256": _sha("forbidden"),
            "dataset_fingerprint_sha256": _sha("dataset"),
        },
    }
    path = tmp_path / "head.pt"
    torch.save(payload, path)
    loaded, metadata, checkpoint_sha = load_p1g_checkpoint(
        path, expected_p1s_checkpoint_sha256=base_sha
    )
    assert loaded.hidden_dim == 8
    assert metadata["schema"] == P1G_CHECKPOINT_SCHEMA
    assert len(checkpoint_sha) == 64
    with pytest.raises(ValueError, match="different P1S"):
        load_p1g_checkpoint(
            path, expected_p1s_checkpoint_sha256=_sha("other")
        )
    payload["provenance"]["audit_scene_ids"] = ["scene0000_00"]
    torch.save(payload, path)
    with pytest.raises(ValueError, match="overlaps"):
        load_p1g_checkpoint(
            path, expected_p1s_checkpoint_sha256=base_sha
        )
