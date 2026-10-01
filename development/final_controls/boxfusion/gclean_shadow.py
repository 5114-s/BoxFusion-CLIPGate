"""SMOV-clean Group3D-lite association observer (Gclean-shadow).

This module is a deliberately output-inert sidecar.  It adapts the direct,
signed 5 cm voxel keys produced by :mod:`boxfusion.smov_fragments` to the
already-audited causal memory and frozen Group3D-lite matcher used by
``GrawShadow``.  The adapter does not reconstruct voxel keys from floating
point centroids, and it never mutates BoxFusion rows, geometry, scores,
classes, native association state, or the supplied SMOV batch.

The query is begin-frame-past only: native association reserves its matched
past tracks, counterfactual matching runs against the remaining snapshot, and
current clean fragments are committed to observer memory only afterwards.
Malformed clean fragments abstain independently (fail open) while structural
trace mismatches are rejected so a caller cannot silently compare misaligned
proposals.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
import json
import os
import tempfile
import time
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np

from boxfusion.graw_fragments import (
    PreparedRawKeyframe,
    RawFragmentCoverage,
    RawProposalDiagnostic,
    RawViewFragment,
)
from boxfusion.graw_shadow import GrawFrameToken, GrawShadow
from boxfusion.group3d_lite import PairEvidenceResult
from boxfusion.observer_track_registry import IdentityResolution
from boxfusion.smov_fragments import PreparedKeyframe


SCHEMA = "boxfusion.gclean_shadow.v1"
FRAGMENT_SOURCE = "smov_clean"
MODE = "shadow"

# Executable limits are private literals.  Public module names elsewhere are
# audit metadata and rebinding them cannot enlarge this observer's envelope.
_F_MAX_INPUT_PROPOSALS = 4096
_F_MAX_PROPOSALS = 64
_F_MAX_TRACKS = 1024
_F_MAX_VIEWS_PER_TRACK = 5
_F_MAX_VOXELS_PER_VIEW = 512
_F_MAX_UNION_VOXELS = 1024
_F_MIN_VOXELS = 16
_F_COORDINATE_LIMIT = 1 << 52
_F_TIMING_WINDOW = 4096
_F_DIAGNOSTIC_BYTES = 32 * 1024 * 1024
_F_JSON_DEPTH = 16
_F_JSON_CONTAINER_ITEMS = 65_536
_F_MAX_DIAGNOSTIC_FRAMES = 16_384


@dataclass(frozen=True)
class GcleanFrameToken:
    """Opaque transaction token; only the creating sidecar may consume it."""

    serial: int
    frame_id: int
    _inner: GrawFrameToken


@dataclass(frozen=True)
class GcleanCounterfactualAssociation:
    proposal_id: int
    native_track_id: int
    past_track_id: int
    dice: float
    jaccard: float
    intersection: int
    centroid_distance: float


@dataclass(frozen=True)
class GcleanShadowResult:
    frame_id: int
    begin_track_ids: tuple[int, ...]
    reserved_past_track_ids: tuple[int, ...]
    eligible_past_track_ids: tuple[int, ...]
    candidate_proposal_ids: tuple[int, ...]
    candidate_native_track_ids: tuple[int, ...]
    associations: tuple[GcleanCounterfactualAssociation, ...]
    matcher_diagnostics: Mapping[str, object]
    adapter_diagnostics: Mapping[str, object]
    memory_track_ids: tuple[int, ...]
    smov_prepare_elapsed_ms: float
    voxel_adapter_elapsed_ms: float
    finish_elapsed_ms: float
    total_observer_elapsed_ms: float
    fragment_source: str = FRAGMENT_SOURCE
    mode: str = MODE
    # Populated only for a downstream probability observer.  The default
    # serializer intentionally omits it to preserve the v1 Gclean artifact.
    pair_evidence: Optional[PairEvidenceResult] = None


def _clean_voxel_keys(value: object) -> np.ndarray:
    """Validate, own, sort, and freeze one direct SMOV voxel-key set."""

    if not isinstance(value, np.ndarray):
        raise ValueError("voxel_keys_must_be_numpy")
    if value.ndim != 2 or value.shape[1:] != (3,):
        raise ValueError("invalid_voxel_shape")
    if not np.issubdtype(value.dtype, np.signedinteger):
        raise ValueError("non_signed_integer_voxels")
    if len(value) > _F_MAX_VOXELS_PER_VIEW:
        raise ValueError("voxel_cap")
    keys = np.asarray(value, dtype=np.int64)
    if keys.size and (
        np.any(keys > _F_COORDINATE_LIMIT)
        or np.any(keys < -_F_COORDINATE_LIMIT)
    ):
        raise ValueError("coordinate_overflow")
    keys = np.unique(keys, axis=0)
    if len(keys) < _F_MIN_VOXELS:
        raise ValueError("insufficient_voxels")
    result = np.ascontiguousarray(keys, dtype=np.int64).copy()
    result.setflags(write=False)
    return result


def _raw_coverage(clean: object, output_voxels: int = 0) -> RawFragmentCoverage:
    return RawFragmentCoverage(
        effective_stride=int(getattr(clean, "effective_stride", 0)),
        sampled_rays=int(getattr(clean, "sampled_rays", 0)),
        usable_rays=int(getattr(clean, "usable_rays", 0)),
        unique_voxels=int(getattr(clean, "unique_voxels", output_voxels)),
        output_voxels=int(output_voxels),
        valid_depth_ratio=float(getattr(clean, "valid_depth_ratio", 0.0)),
    )


def _convert_fragment(fragment: object, voxel_keys: np.ndarray) -> RawViewFragment:
    """Make the compatibility view consumed by the proven causal sidecar."""

    return RawViewFragment(
        proposal_id=int(getattr(fragment, "proposal_id")),
        frame_id=int(getattr(fragment, "frame_id")),
        score=float(getattr(fragment, "score")),
        crop_xyxy_depth=getattr(fragment, "crop_xyxy_depth"),
        depth_shape=tuple(getattr(fragment, "depth_shape")),
        proposal_to_depth_affine=getattr(fragment, "proposal_to_depth_affine"),
        intrinsics=getattr(fragment, "intrinsics"),
        camera_to_world=getattr(fragment, "camera_to_world"),
        voxel_keys=voxel_keys,
        coverage=_raw_coverage(getattr(fragment, "coverage"), len(voxel_keys)),
    )


def _adapt_clean_batch(
    batch: PreparedKeyframe,
) -> tuple[PreparedRawKeyframe, Mapping[str, object], float]:
    """Adapt per proposal; bad geometry abstains without poisoning the frame."""

    started = time.perf_counter_ns()
    proposal_ids = tuple(int(value) for value in batch.proposal_ids)
    if len(proposal_ids) > _F_MAX_INPUT_PROPOSALS:
        raise ValueError("SMOV batch exceeds the hard input proposal cap")
    if len(set(proposal_ids)) != len(proposal_ids) or any(
        value < 0 for value in proposal_ids
    ):
        raise ValueError("SMOV proposal IDs must be unique and nonnegative")
    if len(batch.diagnostics) != len(proposal_ids):
        raise ValueError("SMOV diagnostics must align with proposal IDs")
    diagnostic_ids = tuple(int(item.proposal_id) for item in batch.diagnostics)
    if diagnostic_ids != proposal_ids:
        raise ValueError("SMOV diagnostic order must align with proposal IDs")
    selected_count = sum(int(bool(item.selected)) for item in batch.diagnostics)
    if selected_count > _F_MAX_PROPOSALS:
        raise ValueError("SMOV selected proposals exceed the hard cap of 64")

    converted: list[RawProposalDiagnostic] = []
    failures: Counter[str] = Counter()
    accepted = 0
    for diagnostic in batch.diagnostics:
        fragment = None
        reason = diagnostic.reason
        output_voxels = 0
        if not bool(diagnostic.selected):
            reason = reason or "not_selected"
        elif diagnostic.fragment is None:
            reason = reason or "smov_fragment_abstained"
        else:
            try:
                keys = _clean_voxel_keys(
                    getattr(diagnostic.fragment, "voxel_keys", None)
                )
                fragment = _convert_fragment(diagnostic.fragment, keys)
                if (
                    fragment.proposal_id != int(diagnostic.proposal_id)
                    or fragment.frame_id != int(batch.frame_id)
                ):
                    raise ValueError("fragment_identity_mismatch")
                output_voxels = len(keys)
                reason = None
                accepted += 1
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                reason = str(error) or type(error).__name__
                fragment = None
        if fragment is None:
            failures[str(reason or "unknown")] += 1
        converted.append(
            RawProposalDiagnostic(
                proposal_id=int(diagnostic.proposal_id),
                selected=bool(diagnostic.selected),
                reason=reason,
                coverage=_raw_coverage(diagnostic.coverage, output_voxels),
                elapsed_ms=float(diagnostic.elapsed_ms),
                fragment=fragment,
            )
        )

    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    adapter = MappingProxyType(
        {
            "fragment_source": FRAGMENT_SOURCE,
            "input_proposals": len(proposal_ids),
            "selected_proposals": selected_count,
            "converted_fragments": accepted,
            "abstained_fragments": len(converted) - accepted,
            "failure_reasons": MappingProxyType(dict(sorted(failures.items()))),
            "elapsed_ms": elapsed_ms,
        }
    )
    raw = PreparedRawKeyframe(
        scene_id=str(batch.scene_id),
        frame_id=int(batch.frame_id),
        proposal_ids=proposal_ids,
        selected_proposal_ids=tuple(
            int(item.proposal_id) for item in batch.diagnostics if item.selected
        ),
        diagnostics=tuple(converted),
        elapsed_ms=elapsed_ms,
    )
    return raw, adapter, elapsed_ms


def _timing(values: deque[float]) -> Mapping[str, object]:
    samples = tuple(float(value) for value in values)
    array = np.asarray(samples, dtype=np.float64)
    return MappingProxyType(
        {
            "count": len(samples),
            "samples_ms": samples,
            "mean_ms": float(np.mean(array)) if len(array) else 0.0,
            "p50_ms": float(np.percentile(array, 50)) if len(array) else 0.0,
            "p95_ms": float(np.percentile(array, 95)) if len(array) else 0.0,
            "max_ms": float(np.max(array)) if len(array) else 0.0,
        }
    )


class GcleanShadow:
    """Output-inert SMOV-clean observer with causal Group3D-lite matching."""

    def __init__(self) -> None:
        self._inner = GrawShadow()
        self._pending: Optional[GcleanFrameToken] = None
        self._serial = 0
        self._stats: Counter[str] = Counter()
        self._failure_reasons: Counter[str] = Counter()
        self._adapter_timings: deque[float] = deque(maxlen=_F_TIMING_WINDOW)
        self._finish_timings: deque[float] = deque(maxlen=_F_TIMING_WINDOW)
        self._total_timings: deque[float] = deque(maxlen=_F_TIMING_WINDOW)

    @property
    def pending(self) -> bool:
        return self._pending is not None

    @property
    def memory_track_ids(self) -> tuple[int, ...]:
        if self._pending is not None:
            raise RuntimeError("memory is not externally stable during a keyframe")
        return self._inner.memory_track_ids

    def begin_keyframe(
        self,
        frame_id: int,
        *,
        active_track_ids: Optional[Sequence[int]] = None,
    ) -> GcleanFrameToken:
        if self._pending is not None:
            raise RuntimeError("a Gclean keyframe is already pending")
        inner = self._inner.begin_keyframe(
            frame_id, active_track_ids=active_track_ids
        )
        self._serial += 1
        token = GcleanFrameToken(self._serial, int(frame_id), inner)
        self._pending = token
        return token

    def abort_keyframe(self, token: GcleanFrameToken) -> None:
        if token is not self._pending:
            raise RuntimeError("abort must use the exact pending Gclean token")
        self._inner.abort_keyframe(token._inner)
        self._pending = None
        self._stats["aborted_keyframes"] += 1

    def finish_keyframe(
        self,
        token: GcleanFrameToken,
        *,
        batch: PreparedKeyframe,
        resolution: IdentityResolution,
        reserved_past_track_ids: Optional[Sequence[int]] = None,
        unmatched_retained_proposal_ids: Optional[Sequence[int]] = None,
        collect_pair_evidence: bool = False,
    ) -> GcleanShadowResult:
        if token is not self._pending:
            raise RuntimeError("finish must use the exact pending Gclean token")
        raw_batch, adapter, adapter_ms = _adapt_clean_batch(batch)
        try:
            raw_result = self._inner.finish_keyframe(
                token._inner,
                batch=raw_batch,
                resolution=resolution,
                reserved_past_track_ids=reserved_past_track_ids,
                unmatched_retained_proposal_ids=unmatched_retained_proposal_ids,
                collect_pair_evidence=collect_pair_evidence,
            )
        except Exception:
            # Preserve a still-live transaction so the integration wrapper can
            # explicitly abort it.  If the inner transaction already closed,
            # mirror that state and never pretend a token remains consumable.
            if not self._inner.pending:
                self._pending = None
            raise
        self._pending = None

        associations = tuple(
            GcleanCounterfactualAssociation(
                proposal_id=item.proposal_id,
                native_track_id=item.native_track_id,
                past_track_id=item.past_track_id,
                dice=item.dice,
                jaccard=item.jaccard,
                intersection=item.intersection,
                centroid_distance=item.centroid_distance,
            )
            for item in raw_result.associations
        )
        smov_ms = float(batch.elapsed_ms)
        finish_ms = float(raw_result.elapsed_ms)
        total_ms = smov_ms + adapter_ms + finish_ms
        self._adapter_timings.append(adapter_ms)
        self._finish_timings.append(finish_ms)
        self._total_timings.append(total_ms)
        self._stats["keyframes"] += 1
        self._stats["candidate_proposals"] += len(
            raw_result.candidate_proposal_ids
        )
        self._stats["counterfactual_associations"] += len(associations)
        self._stats["converted_fragments"] += int(
            adapter["converted_fragments"]
        )
        self._stats["fragment_abstentions"] += int(
            adapter["abstained_fragments"]
        )
        if bool(raw_result.matcher_diagnostics.get("fail_open", False)):
            self._stats["matcher_fail_open"] += 1
        for reason, count in dict(adapter["failure_reasons"]).items():
            self._failure_reasons[str(reason)] += int(count)

        return GcleanShadowResult(
            frame_id=raw_result.frame_id,
            begin_track_ids=raw_result.begin_track_ids,
            reserved_past_track_ids=raw_result.reserved_past_track_ids,
            eligible_past_track_ids=raw_result.eligible_past_track_ids,
            candidate_proposal_ids=raw_result.candidate_proposal_ids,
            candidate_native_track_ids=raw_result.candidate_native_track_ids,
            associations=associations,
            matcher_diagnostics=raw_result.matcher_diagnostics,
            adapter_diagnostics=adapter,
            memory_track_ids=raw_result.memory_track_ids,
            smov_prepare_elapsed_ms=smov_ms,
            voxel_adapter_elapsed_ms=adapter_ms,
            finish_elapsed_ms=finish_ms,
            total_observer_elapsed_ms=total_ms,
            pair_evidence=raw_result.pair_evidence,
        )

    def diagnostics(self) -> Mapping[str, object]:
        inner = self._inner.diagnostics()
        return MappingProxyType(
            {
                "schema": SCHEMA,
                "mode": MODE,
                "fragment_source": FRAGMENT_SOURCE,
                "pending": self._pending is not None,
                "last_frame": inner["last_frame"],
                "memory_track_ids": tuple(inner["memory_track_ids"]),
                "caps": MappingProxyType(
                    {
                        "max_proposals_per_keyframe": _F_MAX_PROPOSALS,
                        "max_tracks": _F_MAX_TRACKS,
                        "max_views_per_track": _F_MAX_VIEWS_PER_TRACK,
                        "max_voxels_per_view": _F_MAX_VOXELS_PER_VIEW,
                        "max_union_voxels_per_track": _F_MAX_UNION_VOXELS,
                        "min_voxels": _F_MIN_VOXELS,
                    }
                ),
                "stats": MappingProxyType(dict(self._stats)),
                "failure_reasons": MappingProxyType(
                    dict(sorted(self._failure_reasons.items()))
                ),
                "timing": MappingProxyType(
                    {
                        "voxel_adapter": _timing(self._adapter_timings),
                        "finish": _timing(self._finish_timings),
                        "total_observer": _timing(self._total_timings),
                    }
                ),
            }
        )


def gclean_result_to_dict(result: GcleanShadowResult) -> dict[str, object]:
    return {
        "mode": MODE,
        "fragment_source": FRAGMENT_SOURCE,
        "frame_id": result.frame_id,
        "begin_track_ids": list(result.begin_track_ids),
        "reserved_past_track_ids": list(result.reserved_past_track_ids),
        "eligible_past_track_ids": list(result.eligible_past_track_ids),
        "candidate_proposal_ids": list(result.candidate_proposal_ids),
        "candidate_native_track_ids": list(result.candidate_native_track_ids),
        "associations": [asdict(value) for value in result.associations],
        "matcher_diagnostics": dict(result.matcher_diagnostics),
        "adapter_diagnostics": {
            **dict(result.adapter_diagnostics),
            "failure_reasons": dict(
                result.adapter_diagnostics.get("failure_reasons", {})
            ),
        },
        "memory_track_ids": list(result.memory_track_ids),
        "smov_prepare_elapsed_ms": result.smov_prepare_elapsed_ms,
        "voxel_adapter_elapsed_ms": result.voxel_adapter_elapsed_ms,
        "finish_elapsed_ms": result.finish_elapsed_ms,
        "total_observer_elapsed_ms": result.total_observer_elapsed_ms,
    }


def _plain_json_value(value: object, *, _depth: int = 0) -> object:
    """Copy the bounded diagnostic vocabulary into JSON-native containers."""

    if _depth > _F_JSON_DEPTH:
        raise ValueError("Gclean diagnostic nesting exceeds the depth cap")
    if isinstance(value, Mapping):
        if len(value) > _F_JSON_CONTAINER_ITEMS:
            raise ValueError("Gclean diagnostic mapping exceeds the item cap")
        return {
            str(key): _plain_json_value(item, _depth=_depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        if len(value) > _F_JSON_CONTAINER_ITEMS:
            raise ValueError("Gclean diagnostic sequence exceeds the item cap")
        return [
            _plain_json_value(item, _depth=_depth + 1) for item in value
        ]
    if isinstance(value, np.generic):
        return _plain_json_value(value.item(), _depth=_depth + 1)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "Gclean diagnostics contain a non-JSON value: " + type(value).__name__
    )


def write_gclean_shadow_diagnostics(
    path: os.PathLike[str] | str,
    *,
    scene_id: str,
    results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    trace_valid: bool,
) -> str:
    """Atomically write one bounded, machine-readable scene diagnostic."""

    destination = os.path.abspath(os.fspath(path))
    if len(results) > _F_MAX_DIAGNOSTIC_FRAMES:
        raise ValueError("Gclean diagnostic frame count exceeds the hard cap")
    payload = {
        "schema": SCHEMA,
        "mode": MODE,
        "fragment_source": FRAGMENT_SOURCE,
        "scene_id": str(scene_id),
        "trace_valid": bool(trace_valid),
        "frame_count": len(results),
        "frames": _plain_json_value(results),
        "summary": _plain_json_value(summary),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > _F_DIAGNOSTIC_BYTES:
        raise ValueError("Gclean diagnostic exceeds the 32 MiB cap")
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(destination) + ".",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return destination


__all__ = [
    "FRAGMENT_SOURCE",
    "GcleanCounterfactualAssociation",
    "GcleanFrameToken",
    "GcleanShadow",
    "GcleanShadowResult",
    "MODE",
    "SCHEMA",
    "gclean_result_to_dict",
    "write_gclean_shadow_diagnostics",
]
