"""Deterministic initialization of a one-class TR3D head.

The official ScanNet TR3D head is trained with 18 independent sigmoid focal
loss outputs.  A one-class foreground head cannot represent the exact
non-linear union of those logits with one affine layer.  This module therefore
uses the first-order Taylor approximation of the independent-Bernoulli union
logit at the zero input feature.  It exactly matches the source foreground
prior at that point and matches its local feature gradient.

This is an initialization conversion only.  It is not an 18-to-1 model
equivalence and it is not a replacement for foreground fine-tuning.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import Tensor


SCHEMA = "boxfusion.tr3d_foreground_checkpoint_init.v1"
METHOD = "independent_bernoulli_union_logit_first_order_at_zero_feature"
KERNEL_KEY = "head.conv_cls.kernel"
BIAS_KEY = "head.conv_cls.bias"
EXPECTED_KERNEL_SHAPE = (128, 18)
EXPECTED_BIAS_SHAPE = (1, 18)
TARGET_KERNEL_SHAPE = (128, 1)
TARGET_BIAS_SHAPE = (1, 1)


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of *path* without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: Tensor) -> str:
    """Hash the exact contiguous CPU tensor bytes."""
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def _state_fingerprint(
    state_dict: Mapping[str, Tensor], *, exclude: Tuple[str, ...] = ()
) -> str:
    """Hash names, shapes, dtypes and bytes in stable state-dict order."""
    excluded = set(exclude)
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        if key in excluded:
            continue
        tensor = state_dict[key]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"state_dict[{key!r}] is not a tensor")
        header = (
            f"{key}\0{tuple(tensor.shape)!r}\0{tensor.dtype}\0".encode(
                "utf-8"
            )
        )
        digest.update(header)
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def collapse_sigmoid_classes_to_foreground(
    kernel: Tensor, bias: Tensor
) -> Tuple[Tensor, Tensor, Dict[str, Any]]:
    """Collapse 18 one-vs-rest logits to a local foreground approximation.

    For source logits ``z_c(x) = b_c + w_c^T x``, let

    ``p_fg(x) = 1 - product_c(1 - sigmoid(z_c(x)))``.

    The returned affine logit is the first-order Taylor approximation of
    ``logit(p_fg(x))`` at ``x=0``.  Consequently:

    ``b_fg = logit(1 - product_c(1 - sigmoid(b_c)))``

    ``w_fg = sum_c [sigmoid(b_c) / p_fg(0)] * w_c``.
    """
    if tuple(kernel.shape) != EXPECTED_KERNEL_SHAPE:
        raise ValueError(
            f"{KERNEL_KEY} shape must be {EXPECTED_KERNEL_SHAPE}, "
            f"got {tuple(kernel.shape)}"
        )
    if tuple(bias.shape) != EXPECTED_BIAS_SHAPE:
        raise ValueError(
            f"{BIAS_KEY} shape must be {EXPECTED_BIAS_SHAPE}, "
            f"got {tuple(bias.shape)}"
        )
    if not kernel.is_floating_point() or not bias.is_floating_point():
        raise TypeError("TR3D classifier kernel and bias must be floating point")
    if kernel.dtype != bias.dtype:
        raise TypeError(
            "TR3D classifier kernel and bias must have the same dtype"
        )
    if not torch.isfinite(kernel).all() or not torch.isfinite(bias).all():
        raise ValueError("TR3D classifier kernel and bias must be finite")

    # Float64 makes this tiny reduction insensitive to source float32
    # accumulation order.  The final values are cast back to the source dtype.
    source_bias = bias.detach().cpu().to(torch.float64).reshape(-1)
    source_kernel = kernel.detach().cpu().to(torch.float64)
    probabilities = torch.sigmoid(source_bias)
    log_background_probability = torch.log1p(-probabilities).sum()
    foreground_probability = -torch.expm1(log_background_probability)
    if not 0.0 < float(foreground_probability) < 1.0:
        raise ValueError("collapsed foreground prior is outside (0, 1)")

    foreground_bias = (
        torch.log(foreground_probability) - log_background_probability
    ).reshape(1, 1)
    gradient_weights = probabilities / foreground_probability
    foreground_kernel = (
        source_kernel * gradient_weights.reshape(1, -1)
    ).sum(dim=1, keepdim=True)

    foreground_kernel = foreground_kernel.to(dtype=kernel.dtype)
    foreground_bias = foreground_bias.to(dtype=bias.dtype)
    if tuple(foreground_kernel.shape) != TARGET_KERNEL_SHAPE:
        raise AssertionError("unexpected collapsed kernel shape")
    if tuple(foreground_bias.shape) != TARGET_BIAS_SHAPE:
        raise AssertionError("unexpected collapsed bias shape")

    target_prior = torch.sigmoid(
        foreground_bias.to(torch.float64).reshape(())
    )
    details: Dict[str, Any] = {
        "source_class_count": EXPECTED_KERNEL_SHAPE[1],
        "target_class_count": 1,
        "expansion_point": "input_feature_x_equals_zero",
        "source_independent_union_probability_at_expansion": float(
            foreground_probability
        ),
        "target_sigmoid_probability_at_expansion": float(target_prior),
        "probability_rounding_error_after_source_dtype_cast": abs(
            float(target_prior) - float(foreground_probability)
        ),
        "gradient_weight_min": float(gradient_weights.min()),
        "gradient_weight_max": float(gradient_weights.max()),
        "gradient_weight_sum": float(gradient_weights.sum()),
    }
    return foreground_kernel, foreground_bias, details


def convert_payload(
    payload: MutableMapping[str, Any],
) -> Tuple[MutableMapping[str, Any], Dict[str, Any]]:
    """Convert a loaded official checkpoint payload in place.

    All top-level objects, optimizer state, metadata, and every other
    state-dict tensor are retained.  Exactly two state-dict tensors change.
    """
    if not isinstance(payload, MutableMapping):
        raise TypeError("checkpoint payload must be a mutable mapping")
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, MutableMapping):
        raise ValueError("checkpoint must contain a state_dict mapping")
    if KERNEL_KEY not in state_dict or BIAS_KEY not in state_dict:
        raise KeyError(
            f"checkpoint must contain {KERNEL_KEY!r} and {BIAS_KEY!r}"
        )

    kernel = state_dict[KERNEL_KEY]
    bias = state_dict[BIAS_KEY]
    if not isinstance(kernel, Tensor) or not isinstance(bias, Tensor):
        raise TypeError("classifier state entries must be torch tensors")
    untouched_before = _state_fingerprint(
        state_dict, exclude=(KERNEL_KEY, BIAS_KEY)
    )
    source_records = {
        KERNEL_KEY: {
            "shape": list(kernel.shape),
            "dtype": str(kernel.dtype),
            "sha256": tensor_sha256(kernel),
        },
        BIAS_KEY: {
            "shape": list(bias.shape),
            "dtype": str(bias.dtype),
            "sha256": tensor_sha256(bias),
        },
    }

    foreground_kernel, foreground_bias, math_details = (
        collapse_sigmoid_classes_to_foreground(kernel, bias)
    )
    state_dict[KERNEL_KEY] = foreground_kernel
    state_dict[BIAS_KEY] = foreground_bias

    untouched_after = _state_fingerprint(
        state_dict, exclude=(KERNEL_KEY, BIAS_KEY)
    )
    if untouched_after != untouched_before:
        raise AssertionError("an untouched state-dict tensor changed")
    target_records = {
        KERNEL_KEY: {
            "shape": list(foreground_kernel.shape),
            "dtype": str(foreground_kernel.dtype),
            "sha256": tensor_sha256(foreground_kernel),
        },
        BIAS_KEY: {
            "shape": list(foreground_bias.shape),
            "dtype": str(foreground_bias.dtype),
            "sha256": tensor_sha256(foreground_bias),
        },
    }
    conversion = {
        "schema": SCHEMA,
        "role": "initialization_only_not_trained",
        "method": METHOD,
        "modified_state_dict_keys": [KERNEL_KEY, BIAS_KEY],
        "source_classifier": source_records,
        "target_classifier": target_records,
        "untouched_state_dict_tensor_count": len(state_dict) - 2,
        "untouched_state_dict_sha256_before": untouched_before,
        "untouched_state_dict_sha256_after": untouched_after,
        "untouched_state_dict_exact": True,
        "top_level_payload_preservation": (
            "meta and optimizer are retained; use this artifact with "
            "load_from, never resume, because optimizer classifier slots "
            "still describe the source 18-class parameter"
        ),
        "math": math_details,
        "limitations": [
            (
                "one affine logit cannot exactly represent the nonlinear "
                "union of 18 affine sigmoid logits"
            ),
            "the converted classifier is an initialization and requires training",
            "no accuracy or runtime improvement is implied",
        ],
    }
    return payload, conversion


def _torch_load(path: Path) -> MutableMapping[str, Any]:
    # This converter only accepts the pinned, SHA-verified official artifact.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def _write_torch_temp(payload: Any, directory: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".tr3d-foreground-", suffix=".pth.tmp", dir=directory
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            # A file object gives PyTorch a stable "archive" zip root instead
            # of deriving it from the output filename.
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _write_json_temp(payload: Mapping[str, Any], directory: Path) -> Path:
    import json

    descriptor, raw_path = tempfile.mkstemp(
        prefix=".tr3d-foreground-", suffix=".json.tmp", dir=directory
    )
    temporary = Path(raw_path)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _publish_no_overwrite(temporary: Path, destination: Path) -> None:
    """Atomically publish by hard link, failing if destination exists."""
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"refusing to overwrite {destination}") from None


def convert_checkpoint(
    source: str | Path,
    output: str | Path,
    provenance: str | Path,
    *,
    expected_source_sha256: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert, atomically publish, and return the provenance record."""
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    provenance_path = Path(provenance).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if output_path == source_path or provenance_path == source_path:
        raise ValueError("source, output, and provenance paths must differ")
    if output_path == provenance_path:
        raise ValueError("output checkpoint and provenance paths must differ")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if provenance_path.exists():
        raise FileExistsError(f"refusing to overwrite {provenance_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)

    source_sha = sha256_file(source_path)
    if (
        expected_source_sha256 is not None
        and source_sha.lower() != expected_source_sha256.lower()
    ):
        raise ValueError(
            "source checkpoint SHA256 mismatch: "
            f"{source_sha} != {expected_source_sha256.lower()}"
        )

    payload = _torch_load(source_path)
    payload, conversion = convert_payload(payload)
    temporary_checkpoint = _write_torch_temp(payload, output_path.parent)
    temporary_provenance: Optional[Path] = None
    checkpoint_published = False
    try:
        output_sha = sha256_file(temporary_checkpoint)
        record = {
            **conversion,
            "source": {
                "filename": source_path.name,
                "bytes": source_path.stat().st_size,
                "sha256": source_sha,
            },
            "output": {
                "filename": output_path.name,
                "bytes": temporary_checkpoint.stat().st_size,
                "sha256": output_sha,
            },
        }
        temporary_provenance = _write_json_temp(
            record, provenance_path.parent
        )
        _publish_no_overwrite(temporary_checkpoint, output_path)
        checkpoint_published = True
        try:
            _publish_no_overwrite(temporary_provenance, provenance_path)
        except Exception:
            output_path.unlink()
            checkpoint_published = False
            raise
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
        if temporary_provenance is not None:
            temporary_provenance.unlink(missing_ok=True)

    if not checkpoint_published:
        raise AssertionError("checkpoint publication did not complete")
    if sha256_file(output_path) != record["output"]["sha256"]:
        raise AssertionError("published checkpoint SHA256 changed")
    return record
