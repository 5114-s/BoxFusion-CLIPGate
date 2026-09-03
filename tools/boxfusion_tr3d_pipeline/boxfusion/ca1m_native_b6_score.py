"""Isolated 14-D CA-1M-native B6 score hook.

This module intentionally does not import :mod:`boxfusion.quality_score`.
That runtime is frozen to the ScanNet 12-column feature contract, whereas
the CA-1M-native observer emits an exact 14-column depth/free-space schema.

The default hook is ``observer`` and is a strict no-op.  ``active`` mode is
fail-closed: it requires a train-only checkpoint and companion manifest whose
``activation_authorized`` fields are both true.  Applying the hook can change
only confidence scores; OBB corners, row count, and row order are immutable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .ca1m_native_b6_observer import FEATURE_NAMES, SCHEMA as OBSERVER_SCHEMA


CHECKPOINT_SCHEMA = "boxfusion.ca1m_native_b6_iou_mlp.v1"
CHECKPOINT_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_checkpoint_manifest.v1"
HOOK_DIAGNOSTIC_SCHEMA = "boxfusion.ca1m_native_b6_score_hook.v1"
OUTPUT_NAMES = ("predicted_iou", "prob_iou_015", "prob_iou_025", "prob_iou_050")
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _regular(path: str | os.PathLike[str], label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {resolved}")
    return resolved


def _scalar(values: Mapping[str, np.ndarray], name: str, expected: Any) -> None:
    value = np.asarray(values[name])
    if value.shape != () or value.item() != expected:
        raise ValueError(f"checkpoint scalar {name} disagrees with {expected!r}")


def _stable_sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    result = np.empty_like(value)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _write_npz_create_only(path: Path, payload: Mapping[str, np.ndarray]) -> None:
    target = Path(path)
    if target.is_symlink() or target.exists():
        raise FileExistsError(f"refusing existing native-B6 score diagnostic: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    np.savez_compressed(buffer, **payload)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target.parent, prefix=f".{target.name}.",
            suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(
            f"refusing existing native-B6 score diagnostic: {target}"
        ) from error
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


@dataclass(frozen=True)
class CA1MNativeB6ScoreConfig:
    mode: str = "observer"
    checkpoint: str = ""
    checkpoint_manifest: str = ""
    diagnostics_root: str = ""

    @property
    def active(self) -> bool:
        return self.mode == "active"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "CA1MNativeB6ScoreConfig":
        raw = dict(value or {})
        mode = str(raw.get("mode", "observer")).strip().lower()
        if mode not in {"observer", "active"}:
            raise ValueError("CA1M native-B6 score mode must be observer or active")
        diagnostics = dict(raw.get("diagnostics") or {})
        result = cls(
            mode=mode,
            checkpoint=str(raw.get("checkpoint", "")),
            checkpoint_manifest=str(raw.get("checkpoint_manifest", "")),
            diagnostics_root=str(diagnostics.get("root", "")),
        )
        if result.active:
            if not result.checkpoint or not result.checkpoint_manifest:
                raise ValueError("active native-B6 scoring requires checkpoint and manifest")
            if diagnostics.get("enabled") is not True or not result.diagnostics_root:
                raise ValueError("active native-B6 scoring requires diagnostics.enabled/root")
        return result


@dataclass(frozen=True)
class CA1MNativeB6ScorePrediction:
    components: np.ndarray
    quality_scores: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class CA1MNativeB6ScoreResult:
    corners: np.ndarray
    scores: np.ndarray
    source_indices: np.ndarray
    mode: str
    applied_count: int
    checkpoint_sha256: str
    diagnostic_path: str


class CA1MNativeB6Scorer:
    """Pickle-free scorer for the exact native 14-D checkpoint schema."""

    def __init__(
        self,
        *,
        weights: Sequence[np.ndarray],
        biases: Sequence[np.ndarray],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        ranking_weights: np.ndarray,
        detector_blend: float,
        activation_authorized: bool,
        checkpoint_path: Path,
        checkpoint_sha256: str,
        manifest_path: Path,
        manifest_sha256: str,
    ) -> None:
        self.weights = tuple(np.asarray(value, dtype=np.float64) for value in weights)
        self.biases = tuple(np.asarray(value, dtype=np.float64) for value in biases)
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.ranking_weights = np.asarray(ranking_weights, dtype=np.float64)
        self.detector_blend = float(detector_blend)
        self.activation_authorized = bool(activation_authorized)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.manifest_path = manifest_path
        self.manifest_sha256 = manifest_sha256
        for array in (
            *self.weights, *self.biases, self.feature_mean, self.feature_scale,
            self.ranking_weights,
        ):
            array.setflags(write=False)

    def predict(self, features: Any, detector_scores: Any) -> CA1MNativeB6ScorePrediction:
        inputs = np.asarray(features, dtype=np.float64)
        detector = np.asarray(detector_scores, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"native-B6 features must have shape [N,{len(FEATURE_NAMES)}]")
        if detector.shape != (len(inputs),):
            raise ValueError("detector scores must align with native-B6 features")
        if (
            not np.isfinite(inputs).all() or not np.isfinite(detector).all()
            or np.any(inputs < 0.0) or np.any(inputs > 1.0)
            or np.any(detector < 0.0) or np.any(detector > 1.0)
        ):
            raise ValueError("native-B6 features/scores must be finite in [0,1]")
        value = (inputs - self.feature_mean) / self.feature_scale
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight + bias
            if index + 1 < len(self.weights):
                value = np.maximum(value, 0.0)
        raw = _stable_sigmoid(value)
        components = raw.copy()
        components[:, 1:] = np.minimum.accumulate(components[:, 1:], axis=1)
        quality = components @ self.ranking_weights
        scores = self.detector_blend * detector + (1.0 - self.detector_blend) * quality
        scores = np.clip(scores, 0.0, 1.0)
        return CA1MNativeB6ScorePrediction(
            components=np.asarray(components, dtype=np.float64),
            quality_scores=np.asarray(quality, dtype=np.float64),
            scores=np.asarray(scores, dtype=np.float64),
        )


def load_ca1m_native_b6_scorer(
    checkpoint: str | os.PathLike[str],
    checkpoint_manifest: str | os.PathLike[str],
    *,
    require_activation_authorized: bool,
) -> CA1MNativeB6Scorer:
    checkpoint_path = _regular(checkpoint, "CA1M native-B6 checkpoint")
    manifest_path = _regular(checkpoint_manifest, "CA1M native-B6 checkpoint manifest")
    checkpoint_sha = sha256_file(checkpoint_path)
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    required_manifest = {
        "schema": CHECKPOINT_MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"native-B6 checkpoint manifest field {key} disagrees")
    record = manifest.get("checkpoint") or {}
    if Path(str(record.get("path", ""))).resolve() != checkpoint_path:
        raise ValueError("native-B6 checkpoint path differs from its manifest")
    if str(record.get("sha256")) != checkpoint_sha:
        raise ValueError("native-B6 checkpoint SHA256 differs from its manifest")
    manifest_authorized = manifest.get("activation_authorized") is True

    with np.load(checkpoint_path, allow_pickle=False) as archive:
        if "num_layers" not in archive.files:
            raise ValueError("native-B6 checkpoint is missing num_layers")
        layer_value = np.asarray(archive["num_layers"])
        if layer_value.shape != () or layer_value.dtype.kind not in "iu":
            raise ValueError("native-B6 num_layers must be a scalar integer")
        layer_count = int(layer_value.item())
        if layer_count < 1:
            raise ValueError("native-B6 num_layers must be positive")
        expected_keys = {
            "schema", "complete", "train_only", "validation_ground_truth_access",
            "activation_authorized", "feature_names", "output_names",
            "iou_thresholds", "ranking_weights", "detector_blend",
            "preserve_original_floor", "monotonic_probability_projection",
            "strict_iou_thresholds", "feature_mean", "feature_scale",
            "num_layers", "training_folds", "heldout_dev_fold",
        }
        expected_keys.update(f"weight_{index}" for index in range(layer_count))
        expected_keys.update(f"bias_{index}" for index in range(layer_count))
        if set(archive.files) != expected_keys:
            raise ValueError("native-B6 checkpoint keys do not exactly match its schema")
        values = {name: np.array(archive[name], copy=True) for name in archive.files}

    for name, expected in (
        ("schema", CHECKPOINT_SCHEMA), ("complete", True), ("train_only", True),
        ("validation_ground_truth_access", False),
        ("preserve_original_floor", False),
        ("monotonic_probability_projection", True),
        ("strict_iou_thresholds", True), ("heldout_dev_fold", 0),
    ):
        _scalar(values, name, expected)
    checkpoint_authorized = bool(np.asarray(values["activation_authorized"]).item())
    if checkpoint_authorized != manifest_authorized:
        raise ValueError("checkpoint and manifest activation authorization disagree")
    if require_activation_authorized and not checkpoint_authorized:
        raise PermissionError("native-B6 checkpoint is not activation_authorized")
    names = tuple(str(value) for value in np.asarray(values["feature_names"]).tolist())
    if names != FEATURE_NAMES:
        raise ValueError("native-B6 checkpoint is not the exact isolated 14-D schema")
    outputs = tuple(str(value) for value in np.asarray(values["output_names"]).tolist())
    if outputs != OUTPUT_NAMES:
        raise ValueError("native-B6 output schema disagrees")
    if not np.allclose(values["iou_thresholds"], IOU_THRESHOLDS, rtol=0, atol=1e-8):
        raise ValueError("native-B6 IoU thresholds disagree")
    if not np.array_equal(values["training_folds"], np.asarray((1, 2, 3, 4))):
        raise ValueError("native-B6 deploy checkpoint did not use folds 1--4")

    feature_mean = np.asarray(values["feature_mean"], dtype=np.float64)
    feature_scale = np.asarray(values["feature_scale"], dtype=np.float64)
    ranking_weights = np.asarray(values["ranking_weights"], dtype=np.float64)
    detector_blend = float(np.asarray(values["detector_blend"]).item())
    if feature_mean.shape != (len(FEATURE_NAMES),) or feature_scale.shape != feature_mean.shape:
        raise ValueError("native-B6 feature normalization has the wrong dimension")
    if not np.isfinite(feature_mean).all() or not np.isfinite(feature_scale).all() or np.any(feature_scale <= 0.0):
        raise ValueError("native-B6 feature normalization is invalid")
    if ranking_weights.shape != (len(OUTPUT_NAMES),) or np.any(ranking_weights < 0.0) or not np.isfinite(ranking_weights).all():
        raise ValueError("native-B6 ranking weights are invalid")
    if not np.isclose(ranking_weights.sum(), 1.0, rtol=0, atol=1e-6):
        raise ValueError("native-B6 ranking weights must sum to one")
    if not np.isfinite(detector_blend) or not 0.0 <= detector_blend <= 1.0:
        raise ValueError("native-B6 detector blend is invalid")
    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    input_dim = len(FEATURE_NAMES)
    for index in range(layer_count):
        weight = np.asarray(values[f"weight_{index}"], dtype=np.float64)
        bias = np.asarray(values[f"bias_{index}"], dtype=np.float64)
        if weight.ndim != 2 or weight.shape[0] != input_dim or bias.shape != (weight.shape[1],):
            raise ValueError(f"native-B6 MLP layer {index} shape is invalid")
        if not np.isfinite(weight).all() or not np.isfinite(bias).all():
            raise ValueError(f"native-B6 MLP layer {index} is non-finite")
        weights.append(weight)
        biases.append(bias)
        input_dim = weight.shape[1]
    if input_dim != len(OUTPUT_NAMES):
        raise ValueError("native-B6 MLP output dimension is invalid")
    return CA1MNativeB6Scorer(
        weights=weights, biases=biases, feature_mean=feature_mean,
        feature_scale=feature_scale, ranking_weights=ranking_weights,
        detector_blend=detector_blend,
        activation_authorized=checkpoint_authorized,
        checkpoint_path=checkpoint_path, checkpoint_sha256=checkpoint_sha,
        manifest_path=manifest_path, manifest_sha256=manifest_sha,
    )


def load_native_observer_diagnostic(
    path: str | os.PathLike[str],
    *,
    scene_id: str,
    corners: Any,
    scores: Any,
) -> dict[str, np.ndarray]:
    diagnostic_path = _regular(path, "CA1M native-B6 observer diagnostic")
    expected_corners = np.asarray(corners, dtype=np.float32)
    expected_scores = np.asarray(scores, dtype=np.float32)
    count = len(expected_corners)
    if expected_corners.shape != (count, 8, 3) or expected_scores.shape != (count,):
        raise ValueError("native-B6 input prediction shape is invalid")
    with np.load(diagnostic_path, allow_pickle=False) as archive:
        required = {
            "schema", "complete", "observer_only", "mutation_enabled",
            "applied_count", "ground_truth_access", "clip_access", "scene_id",
            "result_indices", "corners", "scores", "feature_names", "features",
            "valid_evidence",
        }
        if not required.issubset(set(archive.files)):
            raise ValueError("native-B6 observer diagnostic fields are missing")
        values = {name: np.array(archive[name], copy=True) for name in required}
    scalar_contract = {
        "schema": OBSERVER_SCHEMA, "complete": True, "observer_only": True,
        "mutation_enabled": False, "applied_count": 0,
        "ground_truth_access": False, "clip_access": False,
        "scene_id": str(scene_id),
    }
    for key, expected in scalar_contract.items():
        value = np.asarray(values[key])
        if value.shape != () or value.item() != expected:
            raise ValueError(f"native-B6 observer field {key} disagrees")
    if not np.array_equal(values["result_indices"], np.arange(count, dtype=np.int64)):
        raise ValueError("native-B6 result_indices must preserve full row order")
    if not np.array_equal(values["corners"], expected_corners):
        raise ValueError("native-B6 diagnostic OBBs differ from same-run prediction")
    if not np.array_equal(values["scores"], expected_scores):
        raise ValueError("native-B6 diagnostic scores differ from same-run prediction")
    names = tuple(str(value) for value in np.asarray(values["feature_names"]).tolist())
    features = np.asarray(values["features"], dtype=np.float64)
    if names != FEATURE_NAMES or features.shape != (count, len(FEATURE_NAMES)):
        raise ValueError("native-B6 diagnostic is not the exact 14-D feature schema")
    if not np.isfinite(features).all() or np.any(features < 0.0) or np.any(features > 1.0):
        raise ValueError("native-B6 diagnostic features must be finite in [0,1]")
    if not np.array_equal(features[:, 0].astype(np.float32), expected_scores):
        raise ValueError("native-B6 detector_score feature differs from prediction score")
    valid = np.asarray(values["valid_evidence"])
    if valid.shape != (count,) or valid.dtype != np.bool_:
        raise ValueError("native-B6 valid_evidence vector is invalid")
    return {
        "result_indices": np.asarray(values["result_indices"], dtype=np.int64),
        "features": features,
        "valid_evidence": valid,
        "diagnostic_sha256": np.asarray(sha256_file(diagnostic_path)),
    }


class CA1MNativeB6ScoreHook:
    def __init__(self, config: CA1MNativeB6ScoreConfig):
        self.config = config
        self.scorer = (
            load_ca1m_native_b6_scorer(
                config.checkpoint, config.checkpoint_manifest,
                require_activation_authorized=True,
            )
            if config.active else None
        )

    @property
    def active(self) -> bool:
        return self.config.active

    def apply(
        self,
        *,
        scene_id: str,
        corners: Any,
        scores: Any,
        observer_diagnostic: str | os.PathLike[str],
    ) -> CA1MNativeB6ScoreResult:
        input_corners = np.asarray(corners)
        input_scores = np.asarray(scores)
        if input_corners.ndim != 3 or input_corners.shape[1:] != (8, 3):
            raise ValueError("native-B6 hook corners must have shape [N,8,3]")
        if input_scores.shape != (len(input_corners),):
            raise ValueError("native-B6 hook scores must have shape [N]")
        if not np.isfinite(input_corners).all() or not np.isfinite(input_scores).all():
            raise ValueError("native-B6 hook inputs must be finite")
        frozen_corners = np.array(input_corners, dtype=np.float32, order="C", copy=True)
        frozen_scores = np.array(input_scores, dtype=np.float32, order="C", copy=True)
        source_indices = np.arange(len(frozen_scores), dtype=np.int64)
        if not self.active:
            return CA1MNativeB6ScoreResult(
                corners=frozen_corners, scores=frozen_scores,
                source_indices=source_indices, mode="observer", applied_count=0,
                checkpoint_sha256="", diagnostic_path="",
            )
        assert self.scorer is not None
        evidence = load_native_observer_diagnostic(
            observer_diagnostic, scene_id=str(scene_id),
            corners=frozen_corners, scores=frozen_scores,
        )
        prediction = self.scorer.predict(evidence["features"], frozen_scores)
        output_scores = np.asarray(prediction.scores, dtype=np.float32)
        if output_scores.shape != frozen_scores.shape or not np.isfinite(output_scores).all():
            raise RuntimeError("native-B6 scorer returned invalid scores")
        changed = int(np.count_nonzero(output_scores != frozen_scores))
        target = (
            Path(self.config.diagnostics_root)
            / f"{scene_id}_ca1m_native_b6_score.npz"
        )
        payload = {
            "schema": np.asarray(HOOK_DIAGNOSTIC_SCHEMA),
            "complete": np.asarray(True, dtype=np.bool_),
            "mode": np.asarray("active"),
            "score_only": np.asarray(True, dtype=np.bool_),
            "obb_mutation_enabled": np.asarray(False, dtype=np.bool_),
            "row_count_mutation_enabled": np.asarray(False, dtype=np.bool_),
            "row_order_mutation_enabled": np.asarray(False, dtype=np.bool_),
            "ground_truth_access": np.asarray(False, dtype=np.bool_),
            "scene_id": np.asarray(str(scene_id)),
            "source_indices": source_indices,
            "input_corners_sha256": np.asarray(_array_sha256(frozen_corners)),
            "output_corners_sha256": np.asarray(_array_sha256(frozen_corners)),
            "input_scores": frozen_scores,
            "output_scores": output_scores,
            "quality_scores": prediction.quality_scores.astype(np.float32),
            "components": prediction.components.astype(np.float32),
            "valid_evidence": evidence["valid_evidence"],
            "changed_scores": np.asarray(changed, dtype=np.int64),
            "checkpoint_sha256": np.asarray(self.scorer.checkpoint_sha256),
            "checkpoint_manifest_sha256": np.asarray(self.scorer.manifest_sha256),
            "observer_diagnostic_sha256": evidence["diagnostic_sha256"],
        }
        _write_npz_create_only(target, payload)
        if not np.array_equal(input_corners, np.asarray(corners)) or not np.array_equal(input_scores, np.asarray(scores)):
            raise RuntimeError("native-B6 score hook mutated caller inputs")
        return CA1MNativeB6ScoreResult(
            corners=frozen_corners, scores=output_scores,
            source_indices=source_indices, mode="active", applied_count=changed,
            checkpoint_sha256=self.scorer.checkpoint_sha256,
            diagnostic_path=str(target.resolve()),
        )

    @staticmethod
    def summary_text(result: CA1MNativeB6ScoreResult) -> str:
        return (
            "CA-1M native B6 score summary | "
            f"mode={result.mode}, rows={len(result.scores)}, "
            f"changed_scores={result.applied_count}, "
            "obb/count/order_changed=0/0/0"
        )


def build_ca1m_native_b6_score_hook(cfg: Mapping[str, Any]) -> CA1MNativeB6ScoreHook:
    return CA1MNativeB6ScoreHook(
        CA1MNativeB6ScoreConfig.from_mapping(cfg.get("ca1m_native_b6_score"))
    )


__all__ = [
    "CA1MNativeB6ScoreConfig", "CA1MNativeB6ScoreHook",
    "CA1MNativeB6ScorePrediction", "CA1MNativeB6ScoreResult",
    "CA1MNativeB6Scorer", "CHECKPOINT_MANIFEST_SCHEMA", "CHECKPOINT_SCHEMA",
    "HOOK_DIAGNOSTIC_SCHEMA", "IOU_THRESHOLDS", "OUTPUT_NAMES",
    "build_ca1m_native_b6_score_hook", "load_ca1m_native_b6_scorer",
    "load_native_observer_diagnostic", "sha256_file",
]
