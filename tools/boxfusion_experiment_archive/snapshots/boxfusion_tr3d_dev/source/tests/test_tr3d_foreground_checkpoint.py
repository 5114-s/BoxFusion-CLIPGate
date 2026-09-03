from __future__ import annotations

import hashlib
from collections import OrderedDict

import pytest


torch = pytest.importorskip("torch")

from boxfusion.tr3d_foreground_checkpoint import (  # noqa: E402
    BIAS_KEY,
    KERNEL_KEY,
    METHOD,
    collapse_sigmoid_classes_to_foreground,
    convert_checkpoint,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checkpoint(path):
    kernel = torch.arange(128 * 18, dtype=torch.float32).reshape(128, 18)
    kernel = (kernel - kernel.mean()) / 1000.0
    bias = torch.linspace(-5.0, -4.0, 18).reshape(1, 18)
    untouched = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state = OrderedDict(
        [
            ("backbone.example", untouched),
            ("head.conv_reg.kernel", torch.ones(128, 6)),
            (BIAS_KEY, bias),
            (KERNEL_KEY, kernel),
        ]
    )
    state._metadata = OrderedDict([("", {"version": 1})])
    payload = {
        "meta": {"epoch": 12, "name": "synthetic"},
        "state_dict": state,
        "optimizer": {
            "state": {0: {"exp_avg": torch.tensor([2.0])}},
            "param_groups": [{"params": [0]}],
        },
    }
    with path.open("wb") as handle:
        torch.save(payload, handle)
    return payload


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def test_union_collapse_matches_prior_and_local_gradient():
    generator = torch.Generator().manual_seed(7)
    kernel = torch.randn(128, 18, generator=generator)
    bias = torch.linspace(-5.0, -3.5, 18).reshape(1, 18)
    target_kernel, target_bias, details = (
        collapse_sigmoid_classes_to_foreground(kernel, bias)
    )

    probabilities = torch.sigmoid(bias.double().reshape(-1))
    union = 1.0 - torch.prod(1.0 - probabilities)
    assert torch.sigmoid(target_bias.double()).item() == pytest.approx(
        union.item(), abs=2e-8
    )
    assert details["source_class_count"] == 18
    assert target_kernel.shape == (128, 1)
    assert target_bias.shape == (1, 1)

    # Check the analytic first-order collapse against a central finite
    # difference of the exact nonlinear union logit.
    direction = torch.randn(128, generator=generator, dtype=torch.float64)
    eps = 1e-6

    def exact_union_logit(scale):
        logits = bias.double().reshape(-1) + (
            kernel.double().T @ (direction * scale)
        )
        probs = torch.sigmoid(logits)
        log_q = torch.log1p(-probs).sum()
        return torch.log(-torch.expm1(log_q)) - log_q

    finite_difference = (
        exact_union_logit(eps) - exact_union_logit(-eps)
    ) / (2.0 * eps)
    collapsed_gradient = (
        target_kernel.double().reshape(-1) @ direction
    )
    assert collapsed_gradient.item() == pytest.approx(
        finite_difference.item(), rel=2e-5, abs=2e-5
    )


def test_conversion_changes_only_classifier_and_is_deterministic(tmp_path):
    source = tmp_path / "source.pth"
    original = _checkpoint(source)
    output_a = tmp_path / "a.pth"
    output_b = tmp_path / "b.pth"
    json_a = tmp_path / "a.json"
    json_b = tmp_path / "b.json"

    record_a = convert_checkpoint(
        source,
        output_a,
        json_a,
        expected_source_sha256=_sha(source),
    )
    record_b = convert_checkpoint(
        source,
        output_b,
        json_b,
        expected_source_sha256=_sha(source),
    )
    converted = _load(output_a)

    assert _sha(output_a) == _sha(output_b)
    assert record_a["method"] == METHOD
    assert record_a["output"]["sha256"] == _sha(output_a)
    assert record_b["output"]["sha256"] == _sha(output_b)
    assert record_a["untouched_state_dict_exact"] is True
    assert record_a["modified_state_dict_keys"] == [KERNEL_KEY, BIAS_KEY]
    assert converted["meta"] == original["meta"]
    assert converted["optimizer"]["param_groups"] == (
        original["optimizer"]["param_groups"]
    )
    assert torch.equal(
        converted["optimizer"]["state"][0]["exp_avg"],
        original["optimizer"]["state"][0]["exp_avg"],
    )
    for key in original["state_dict"]:
        if key not in (KERNEL_KEY, BIAS_KEY):
            assert torch.equal(
                converted["state_dict"][key], original["state_dict"][key]
            )
    assert converted["state_dict"][KERNEL_KEY].shape == (128, 1)
    assert converted["state_dict"][BIAS_KEY].shape == (1, 1)
    assert converted["state_dict"]._metadata == original["state_dict"]._metadata


def test_conversion_refuses_overwrite_and_bad_hash(tmp_path):
    source = tmp_path / "source.pth"
    _checkpoint(source)
    output = tmp_path / "output.pth"
    provenance = tmp_path / "output.json"
    output.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="overwrite"):
        convert_checkpoint(source, output, provenance)
    assert output.read_bytes() == b"keep"
    output.unlink()

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        convert_checkpoint(
            source,
            output,
            provenance,
            expected_source_sha256="0" * 64,
        )
    assert not output.exists()
    assert not provenance.exists()


def test_wrong_official_classifier_shape_fails_closed(tmp_path):
    source = tmp_path / "bad.pth"
    payload = _checkpoint(source)
    payload["state_dict"][KERNEL_KEY] = torch.zeros(127, 18)
    with source.open("wb") as handle:
        torch.save(payload, handle)
    output = tmp_path / "output.pth"
    provenance = tmp_path / "output.json"
    with pytest.raises(ValueError, match="shape must be"):
        convert_checkpoint(source, output, provenance)
    assert not output.exists()
    assert not provenance.exists()
