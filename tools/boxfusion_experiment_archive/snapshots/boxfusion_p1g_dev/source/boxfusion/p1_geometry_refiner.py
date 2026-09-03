"""Strict P1G geometry-only residual-correction head and checkpoint loader.

P1G never proposes an anchor and never predicts a score.  It consumes the
frozen hidden row of a P1S anchor and predicts only a six-dimensional
correction to the frozen P1S raw geometry.  The correction layer is
zero-initialized, so the shared adapter/decoder reproduces the old P1S
clip/exp box before training.  Candidate selection, IDs, scores, raw boxes and
NMS therefore remain owned by the frozen P1S checkpoint.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from boxfusion.p1_geometry_loss import P1G_DEFAULT_ADAPTER_EPSILON

P1G_CHECKPOINT_SCHEMA = "boxfusion.p1g_geometry_refiner.v2"
P1G_ARCHITECTURE = "linear_residual_correction_v2"
P1G_REGRESSION_ENCODING = (
    "frozen_p1s_clip_exp_to_bounded_logits_plus_residual_correction_v2"
)
P1G_BASE_REGRESSION_ENCODING = "center_delta_m_log_size_m"
P1G_BASE_DECODER = "center_clip_then_log_extent_clip_exp"

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _positive_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a positive finite number") from error
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


class P1GeometryRegressionHead(nn.Module):
    """One zero-initialized correction row per frozen P1S hidden anchor."""

    architecture = P1G_ARCHITECTURE

    def __init__(self, hidden_dim: int = 64, regression_dim: int = 6) -> None:
        super().__init__()
        self.hidden_dim = _positive_integer(hidden_dim, "hidden_dim")
        if isinstance(regression_dim, bool) or int(regression_dim) != 6:
            raise ValueError("regression_dim must equal 6")
        self.regression_dim = 6
        self.correction = nn.Linear(self.hidden_dim, self.regression_dim)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)

    def forward(self, encoded_features: torch.Tensor) -> torch.Tensor:
        if not isinstance(encoded_features, torch.Tensor):
            raise TypeError("encoded_features must be a torch.Tensor")
        if (
            encoded_features.ndim != 2
            or encoded_features.shape[1] != self.hidden_dim
        ):
            raise ValueError(
                "encoded_features must have shape "
                f"[N,{self.hidden_dim}]"
            )
        if not encoded_features.is_floating_point():
            raise TypeError("encoded_features must use a floating dtype")
        if not bool(torch.isfinite(encoded_features).all()):
            raise ValueError("encoded_features must be finite")
        return self.correction(encoded_features)

    def model_config(
        self,
        *,
        max_center_offset: float,
        min_box_extent: float,
        max_box_extent: float,
        adapter_epsilon: float = P1G_DEFAULT_ADAPTER_EPSILON,
    ) -> dict[str, Any]:
        offset = _positive_float(max_center_offset, "max_center_offset")
        minimum = _positive_float(min_box_extent, "min_box_extent")
        maximum = _positive_float(max_box_extent, "max_box_extent")
        epsilon = _positive_float(adapter_epsilon, "adapter_epsilon")
        if maximum <= minimum:
            raise ValueError("max_box_extent must exceed min_box_extent")
        if epsilon >= 0.5:
            raise ValueError("adapter_epsilon must be smaller than 0.5")
        return {
            "architecture": P1G_ARCHITECTURE,
            "hidden_dim": self.hidden_dim,
            "regression_dim": self.regression_dim,
            "regression_encoding": P1G_REGRESSION_ENCODING,
            "base_regression_encoding": P1G_BASE_REGRESSION_ENCODING,
            "base_decoder": P1G_BASE_DECODER,
            "adapter_epsilon": epsilon,
            "max_center_offset": offset,
            "min_box_extent": minimum,
            "max_box_extent": maximum,
            "candidate_contract": (
                "frozen_p1s_ids_scores_order_raw_nms"
            ),
        }

    @classmethod
    def from_model_config(
        cls, config: Mapping[str, Any]
    ) -> "P1GeometryRegressionHead":
        if not isinstance(config, Mapping):
            raise TypeError("model_config must be a mapping")
        required = {
            "architecture",
            "hidden_dim",
            "regression_dim",
            "regression_encoding",
            "base_regression_encoding",
            "base_decoder",
            "adapter_epsilon",
            "max_center_offset",
            "min_box_extent",
            "max_box_extent",
            "candidate_contract",
        }
        missing = sorted(required - set(config))
        unknown = sorted(set(config) - required)
        if missing or unknown:
            raise ValueError(
                f"invalid P1G model_config missing={missing} "
                f"unknown={unknown}"
            )
        if config["architecture"] != P1G_ARCHITECTURE:
            raise ValueError("P1G model_config architecture mismatch")
        if config["regression_encoding"] != P1G_REGRESSION_ENCODING:
            raise ValueError("P1G regression encoding mismatch")
        if (
            config["base_regression_encoding"]
            != P1G_BASE_REGRESSION_ENCODING
        ):
            raise ValueError("P1G base regression encoding mismatch")
        if config["base_decoder"] != P1G_BASE_DECODER:
            raise ValueError("P1G base decoder mismatch")
        if config["candidate_contract"] != (
            "frozen_p1s_ids_scores_order_raw_nms"
        ):
            raise ValueError("P1G candidate contract mismatch")
        model = cls(
            hidden_dim=config["hidden_dim"],
            regression_dim=config["regression_dim"],
        )
        # Validate bounds even though they are applied by the shared decoder.
        model.model_config(
            max_center_offset=config["max_center_offset"],
            min_box_extent=config["min_box_extent"],
            max_box_extent=config["max_box_extent"],
            adapter_epsilon=config["adapter_epsilon"],
        )
        return model


def _scene_ids(
    value: Any, *, name: str, allow_empty: bool = False
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a scene-id sequence")
    result = tuple(value)
    if not result and not allow_empty:
        raise ValueError(f"{name} cannot be empty")
    if (
        any(
            not isinstance(scene, str)
            or _SCENE_PATTERN.fullmatch(scene) is None
            for scene in result
        )
        or len(set(result)) != len(result)
    ):
        raise ValueError(f"{name} contains invalid or duplicate scene IDs")
    return result


def load_p1g_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_p1s_checkpoint_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[
    P1GeometryRegressionHead,
    Mapping[str, Any],
    str,
]:
    """Load a fail-closed train-only P1G checkpoint."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_sha = str(expected_p1s_checkpoint_sha256).lower()
    if _SHA256_PATTERN.fullmatch(expected_sha) is None:
        raise ValueError("expected P1S SHA256 is invalid")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("P1G checkpoint must contain a mapping")
    if payload.get("schema") != P1G_CHECKPOINT_SCHEMA:
        raise ValueError("P1G checkpoint schema mismatch")
    if payload.get("architecture") != P1G_ARCHITECTURE:
        raise ValueError("P1G checkpoint architecture mismatch")
    if payload.get("regression_encoding") != P1G_REGRESSION_ENCODING:
        raise ValueError("P1G checkpoint regression encoding mismatch")
    if payload.get("initialization") != (
        "zero_residual_correction_function_preserving_v2"
    ):
        raise ValueError("P1G checkpoint initialization contract mismatch")
    if payload.get("observer_only") is not True:
        raise ValueError("P1G checkpoint is not observer-only")
    if payload.get("uses_ground_truth") is not False:
        raise ValueError("P1G runtime contract may not use ground truth")
    if payload.get("class_agnostic") is not True:
        raise ValueError("P1G checkpoint must be class agnostic")
    if payload.get("semantic_features") is not False:
        raise ValueError("P1G checkpoint may not use semantic features")
    model_config = payload.get("model_config")
    decoder_config = payload.get("decoder_config")
    state_dict = payload.get("state_dict")
    provenance = payload.get("provenance")
    if (
        not isinstance(model_config, Mapping)
        or not isinstance(decoder_config, Mapping)
        or not isinstance(state_dict, Mapping)
        or not isinstance(provenance, Mapping)
    ):
        raise ValueError("P1G checkpoint lacks strict model/provenance data")
    expected_decoder = {
        "encoding": P1G_REGRESSION_ENCODING,
        "adapter_epsilon": model_config.get("adapter_epsilon"),
        "max_center_offset": model_config.get("max_center_offset"),
        "min_box_extent": model_config.get("min_box_extent"),
        "max_box_extent": model_config.get("max_box_extent"),
    }
    if dict(decoder_config) != expected_decoder:
        raise ValueError("P1G decoder_config/model_config mismatch")
    observed_base_sha = str(
        provenance.get("p1s_checkpoint_sha256", "")
    ).lower()
    if (
        _SHA256_PATTERN.fullmatch(observed_base_sha) is None
        or observed_base_sha != expected_sha
    ):
        raise ValueError("P1G checkpoint binds a different P1S checkpoint")
    fit = set(_scene_ids(provenance.get("fit_scene_ids"), name="fit"))
    cal = set(_scene_ids(provenance.get("cal_scene_ids"), name="cal"))
    audit = set(
        _scene_ids(provenance.get("audit_scene_ids"), name="audit")
    )
    forbidden = _scene_ids(
        provenance.get("forbidden_overlap"),
        name="forbidden_overlap",
        allow_empty=True,
    )
    if fit & cal or fit & audit or cal & audit:
        raise ValueError("P1G fit/cal/audit provenance overlaps")
    if forbidden:
        raise ValueError("P1G checkpoint provenance overlaps validation")
    for name in (
        "fit_scene_list_sha256",
        "cal_scene_list_sha256",
        "audit_scene_list_sha256",
        "forbidden_scene_list_sha256",
        "dataset_fingerprint_sha256",
    ):
        if _SHA256_PATTERN.fullmatch(
            str(provenance.get(name, "")).lower()
        ) is None:
            raise ValueError(f"P1G provenance has invalid {name}")
    model = P1GeometryRegressionHead.from_model_config(model_config)
    model.load_state_dict(dict(state_dict), strict=True)
    model.to(device)
    model.eval()
    return model, payload, sha256_file(path)


__all__ = [
    "P1G_ARCHITECTURE",
    "P1G_BASE_DECODER",
    "P1G_BASE_REGRESSION_ENCODING",
    "P1G_CHECKPOINT_SCHEMA",
    "P1G_REGRESSION_ENCODING",
    "P1GeometryRegressionHead",
    "load_p1g_checkpoint",
    "sha256_file",
]
