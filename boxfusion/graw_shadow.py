"""Causal, bounded Group3D-lite counterfactual association for Graw-shadow.

The sidecar consumes raw 5 cm voxel fragments and stable native identities.
It queries only a begin-frame-past memory, excludes every past track already
reserved by native association, and commits current observations only after
all counterfactual decisions are complete.  It never mutates BoxFusion rows,
scores, classes, geometry, or fusion history.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
import time
import tempfile
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np

from boxfusion.graw_fragments import PreparedRawKeyframe, RawViewFragment
from boxfusion.group3d_lite import (
    Association,
    EvidenceDiagnostics,
    MatchResult,
    PairEvidenceResult,
    PreparedTrackSnapshot,
    extract_prepared_pair_evidence,
    match_prepared,
    match_prepared_proposals,
    prepare_proposals,
    prepare_track_snapshot,
)
from boxfusion.observer_track_registry import IdentityResolution


SCHEMA = "boxfusion.graw_shadow.v1"

_F_MAX_TRACKS = 1024
_F_MAX_VIEWS = 5
_F_MAX_VOXELS_PER_VIEW = 512
_F_MAX_UNION_VOXELS = 1024


def _readonly_voxels(value: object) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 2
        or array.shape[1:] != (3,)
        or not np.issubdtype(array.dtype, np.signedinteger)
    ):
        raise ValueError("voxel keys must be a signed-integer Nx3 array")
    result = np.ascontiguousarray(array, dtype=np.int64).copy()
    if len(result) > 1:
        result = np.unique(result, axis=0)
    result.setflags(write=False)
    return result


def _bounded_rows(value: np.ndarray, maximum: int) -> np.ndarray:
    rows = _readonly_voxels(value)
    if len(rows) <= maximum:
        return rows
    indices = (np.arange(maximum, dtype=np.int64) * len(rows)) // maximum
    result = np.ascontiguousarray(rows[indices], dtype=np.int64)
    result.setflags(write=False)
    return result


def _view_rank(view: RawViewFragment) -> tuple[float, float, float, int]:
    return (
        -float(len(view.voxel_keys)),
        -float(view.coverage.valid_depth_ratio),
        -float(view.score),
        int(view.proposal_id),
    )


def _bounded_views(views: Sequence[RawViewFragment]) -> list[RawViewFragment]:
    per_frame: dict[int, RawViewFragment] = {}
    for view in views:
        current = per_frame.get(int(view.frame_id))
        if current is None or _view_rank(view) < _view_rank(current):
            per_frame[int(view.frame_id)] = view
    retained = [per_frame[frame_id] for frame_id in sorted(per_frame)]
    retained = retained[-_F_MAX_VIEWS:]
    if not retained:
        return []
    base, remainder = divmod(_F_MAX_UNION_VOXELS, len(retained))
    bounded: list[RawViewFragment] = []
    for index, view in enumerate(retained):
        allowance = min(
            _F_MAX_VOXELS_PER_VIEW,
            base + (1 if index < remainder else 0),
        )
        voxels = _bounded_rows(view.voxel_keys, allowance)
        coverage = replace(view.coverage, output_voxels=len(voxels))
        bounded.append(replace(view, voxel_keys=voxels, coverage=coverage))
    return bounded


@dataclass(frozen=True)
class GrawFrameToken:
    serial: int
    frame_id: int
    begin_track_ids: tuple[int, ...]
    snapshot_track_ids: tuple[int, ...]
    snapshot: Optional[PreparedTrackSnapshot]
    snapshot_error: Optional[str]


@dataclass(frozen=True)
class CounterfactualAssociation:
    proposal_id: int
    native_track_id: int
    past_track_id: int
    dice: float
    jaccard: float
    intersection: int
    centroid_distance: float


@dataclass(frozen=True)
class GrawShadowResult:
    frame_id: int
    begin_track_ids: tuple[int, ...]
    reserved_past_track_ids: tuple[int, ...]
    eligible_past_track_ids: tuple[int, ...]
    candidate_proposal_ids: tuple[int, ...]
    candidate_native_track_ids: tuple[int, ...]
    associations: tuple[CounterfactualAssociation, ...]
    matcher_diagnostics: Mapping[str, object]
    memory_track_ids: tuple[int, ...]
    elapsed_ms: float
    # Optional downstream observer evidence.  It is deliberately absent from
    # the legacy JSON serializer, so the default Graw/Gclean artifacts and
    # schemas remain byte-for-byte compatible.
    pair_evidence: Optional[PairEvidenceResult] = None


class GrawShadow:
    """Training-free Graw shadow with native-lineage voxel memory."""

    def __init__(self) -> None:
        self._memory: dict[int, list[RawViewFragment]] = {}
        self._pending: Optional[GrawFrameToken] = None
        self._serial = 0
        self._last_frame: Optional[int] = None
        self._stats = {
            "keyframes": 0,
            "candidate_proposals": 0,
            "counterfactual_associations": 0,
            "matcher_fail_open": 0,
            "fragment_commits": 0,
            "fragment_abstentions": 0,
        }

    @property
    def memory_track_ids(self) -> tuple[int, ...]:
        if self._pending is not None:
            raise RuntimeError("memory is not externally stable during a keyframe")
        return tuple(sorted(self._memory))

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def begin_keyframe(
        self,
        frame_id: int,
        *,
        active_track_ids: Optional[Sequence[int]] = None,
    ) -> GrawFrameToken:
        if self._pending is not None:
            raise RuntimeError("a Graw keyframe is already pending")
        if isinstance(frame_id, bool) or not isinstance(frame_id, (int, np.integer)):
            raise ValueError("frame_id must be an integer")
        frame_id = int(frame_id)
        if frame_id < 0 or (self._last_frame is not None and frame_id <= self._last_frame):
            raise ValueError("frame_id must be nonnegative and strictly increasing")
        if active_track_ids is None:
            begin_track_ids = tuple(sorted(self._memory))
        else:
            if isinstance(active_track_ids, (str, bytes)):
                raise ValueError("active_track_ids must be a bounded sequence")
            normalized: list[int] = []
            for value in active_track_ids:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, np.integer)
                ):
                    raise ValueError("active_track_ids must contain integers")
                normalized.append(int(value))
            begin_track_ids = tuple(normalized)
            if (
                len(begin_track_ids) > _F_MAX_TRACKS
                or len(set(begin_track_ids)) != len(begin_track_ids)
                or any(value < 0 for value in begin_track_ids)
            ):
                raise ValueError("active_track_ids are invalid or exceed the cap")
            begin_track_ids = tuple(sorted(begin_track_ids))
        if not set(self._memory).issubset(begin_track_ids):
            raise RuntimeError("Graw memory is not a subset of native active tracks")
        snapshot_track_ids = tuple(sorted(self._memory))
        tracks = tuple(
            {
                "id": track_id,
                "views": tuple(view.voxel_keys for view in self._memory[track_id]),
            }
            for track_id in snapshot_track_ids
        )
        prepared = prepare_track_snapshot(tracks)
        snapshot_error = (
            None if prepared.snapshot is not None else prepared.diagnostics.code
        )
        self._serial += 1
        token = GrawFrameToken(
            serial=self._serial,
            frame_id=frame_id,
            begin_track_ids=begin_track_ids,
            snapshot_track_ids=snapshot_track_ids,
            snapshot=prepared.snapshot,
            snapshot_error=snapshot_error,
        )
        self._pending = token
        return token

    def abort_keyframe(self, token: GrawFrameToken) -> None:
        if token is not self._pending:
            raise RuntimeError("abort must use the exact pending Graw token")
        self._pending = None

    @staticmethod
    def _matcher_diagnostics(result: MatchResult, snapshot_error: Optional[str]) -> Mapping[str, object]:
        values = asdict(result.diagnostics)
        if snapshot_error is not None:
            values["snapshot_error"] = snapshot_error
        return MappingProxyType(values)

    def _commit_native_memory(
        self,
        batch: PreparedRawKeyframe,
        resolution: IdentityResolution,
    ) -> None:
        active = set(int(value) for value in resolution.active_track_ids)
        staged = {track_id: list(views) for track_id, views in self._memory.items()}

        for raw_source, raw_target in sorted(resolution.track_aliases.items()):
            source, target = int(raw_source), int(raw_target)
            source_views = staged.pop(source, [])
            if source_views:
                staged[target] = _bounded_views(
                    [*staged.get(target, []), *source_views]
                )

        by_proposal = {row.proposal_id: row for row in batch.diagnostics}
        candidates: dict[int, list[RawViewFragment]] = {}
        for proposal_id, track_id in zip(
            resolution.proposal_ids, resolution.proposal_track_ids
        ):
            diagnostic = by_proposal.get(int(proposal_id))
            if (
                track_id is None
                or int(track_id) not in active
                or diagnostic is None
                or diagnostic.fragment is None
            ):
                self._stats["fragment_abstentions"] += 1
                continue
            candidates.setdefault(int(track_id), []).append(diagnostic.fragment)

        for track_id in sorted(candidates):
            view = min(candidates[track_id], key=_view_rank)
            if track_id not in staged and len(staged) >= _F_MAX_TRACKS:
                self._stats["fragment_abstentions"] += 1
                continue
            staged[track_id] = _bounded_views([*staged.get(track_id, []), view])
            self._stats["fragment_commits"] += 1

        self._memory = {
            track_id: staged[track_id]
            for track_id in sorted(staged)
            if track_id in active and staged[track_id]
        }
        if len(self._memory) > _F_MAX_TRACKS:
            raise RuntimeError("Graw memory exceeded the hard track cap")

    def finish_keyframe(
        self,
        token: GrawFrameToken,
        *,
        batch: PreparedRawKeyframe,
        resolution: IdentityResolution,
        reserved_past_track_ids: Optional[Sequence[int]] = None,
        unmatched_retained_proposal_ids: Optional[Sequence[int]] = None,
        collect_pair_evidence: bool = False,
    ) -> GrawShadowResult:
        started = time.perf_counter_ns()
        if token is not self._pending:
            raise RuntimeError("finish must use the exact pending Graw token")
        if not isinstance(collect_pair_evidence, (bool, np.bool_)):
            raise ValueError("collect_pair_evidence must be a boolean")
        collect_pair_evidence = bool(collect_pair_evidence)
        if batch.frame_id != token.frame_id or resolution.frame_id != token.frame_id:
            raise ValueError("Graw frame IDs are not aligned")
        if tuple(batch.proposal_ids) != tuple(resolution.proposal_ids):
            raise ValueError("Graw proposals are not aligned with native identities")

        past = set(token.begin_track_ids)
        reserved = {
            int(track_id)
            for track_id in resolution.proposal_track_ids
            if track_id is not None and int(track_id) in past
        }
        for source, target in resolution.track_aliases.items():
            if int(source) in past:
                reserved.add(int(source))
            if int(target) in past:
                reserved.add(int(target))
        if reserved_past_track_ids is not None:
            supplied_reserved = tuple(int(value) for value in reserved_past_track_ids)
            if any(value not in past for value in supplied_reserved):
                raise ValueError("reserved IDs must belong to begin-frame past")
            reserved.update(supplied_reserved)

        allowed_unmatched = None
        if unmatched_retained_proposal_ids is not None:
            allowed_unmatched = {int(value) for value in unmatched_retained_proposal_ids}
            if not allowed_unmatched.issubset(resolution.proposal_ids):
                raise ValueError("unmatched proposal IDs must belong to the keyframe")

        by_proposal = {row.proposal_id: row for row in batch.diagnostics}
        unmatched_by_track: dict[int, list[RawViewFragment]] = {}
        for proposal_id, track_id in zip(
            resolution.proposal_ids, resolution.proposal_track_ids
        ):
            diagnostic = by_proposal.get(int(proposal_id))
            if (
                track_id is None
                or int(track_id) in past
                or (
                    allowed_unmatched is not None
                    and int(proposal_id) not in allowed_unmatched
                )
                or diagnostic is None
                or diagnostic.fragment is None
            ):
                continue
            unmatched_by_track.setdefault(int(track_id), []).append(
                diagnostic.fragment
            )

        selected: list[tuple[int, RawViewFragment]] = []
        for native_track_id in sorted(unmatched_by_track):
            selected.append(
                (
                    native_track_id,
                    min(unmatched_by_track[native_track_id], key=_view_rank),
                )
            )
        candidate_to_native = {
            view.proposal_id: native_track_id for native_track_id, view in selected
        }
        proposals = tuple(
            {"id": view.proposal_id, "score": view.score, "voxels": view.voxel_keys}
            for _, view in selected
        )
        eligible_ids = tuple(
            track_id for track_id in token.snapshot_track_ids if track_id not in reserved
        )
        eligible_mask = np.asarray(
            [track_id not in reserved for track_id in token.snapshot_track_ids],
            dtype=np.bool_,
        )
        pair_evidence: Optional[PairEvidenceResult] = None
        if token.snapshot is None:
            from boxfusion.group3d_lite_oracle import Diagnostics

            diagnostics = Diagnostics(fail_open=True, code=token.snapshot_error or "snapshot_failure")
            match = MatchResult((), diagnostics)
            if collect_pair_evidence:
                pair_evidence = PairEvidenceResult(
                    (),
                    EvidenceDiagnostics(
                        fail_open=True,
                        code=token.snapshot_error or "snapshot_failure",
                    ),
                )
        elif collect_pair_evidence:
            # Prepare the exact current proposal set once.  Both native-frozen
            # Group3D matching and the PUF evidence observer read the same
            # sealed proposal batch and begin-frame-past track snapshot.
            prepared = prepare_proposals(proposals)
            if prepared.batch is None:
                match = MatchResult((), prepared.diagnostics)
                pair_evidence = PairEvidenceResult(
                    (),
                    EvidenceDiagnostics(
                        fail_open=True,
                        code=prepared.diagnostics.code,
                        selected_proposal_ids=prepared.diagnostics.selected_proposal_ids,
                        skipped_proposal_ids=prepared.diagnostics.skipped_proposal_ids,
                        fragment_errors=prepared.diagnostics.fragment_errors,
                    ),
                )
            else:
                match = match_prepared_proposals(
                    prepared.batch, token.snapshot, eligible_mask
                )
                pair_evidence = extract_prepared_pair_evidence(
                    prepared.batch, token.snapshot, eligible_mask
                )
        else:
            match = match_prepared(proposals, token.snapshot, eligible_mask)

        associations: list[CounterfactualAssociation] = []
        for item in match.associations:
            native_track_id = candidate_to_native.get(int(item.proposal_id))
            if native_track_id is None:
                raise RuntimeError("matcher returned an unknown Graw proposal ID")
            associations.append(
                CounterfactualAssociation(
                    proposal_id=int(item.proposal_id),
                    native_track_id=int(native_track_id),
                    past_track_id=int(item.track_id),
                    dice=float(item.dice),
                    jaccard=float(item.jaccard),
                    intersection=int(item.intersection),
                    centroid_distance=float(item.centroid_distance),
                )
            )

        self._commit_native_memory(batch, resolution)
        self._pending = None
        self._last_frame = token.frame_id
        self._stats["keyframes"] += 1
        self._stats["candidate_proposals"] += len(selected)
        self._stats["counterfactual_associations"] += len(associations)
        if match.diagnostics.fail_open:
            self._stats["matcher_fail_open"] += 1

        return GrawShadowResult(
            frame_id=token.frame_id,
            begin_track_ids=token.begin_track_ids,
            reserved_past_track_ids=tuple(sorted(reserved)),
            eligible_past_track_ids=eligible_ids,
            candidate_proposal_ids=tuple(view.proposal_id for _, view in selected),
            candidate_native_track_ids=tuple(track_id for track_id, _ in selected),
            associations=tuple(associations),
            matcher_diagnostics=self._matcher_diagnostics(match, token.snapshot_error),
            memory_track_ids=tuple(sorted(self._memory)),
            elapsed_ms=(time.perf_counter_ns() - started) / 1e6,
            pair_evidence=pair_evidence,
        )

    def diagnostics(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "schema": SCHEMA,
                "pending": self._pending is not None,
                "last_frame": self._last_frame,
                "memory_track_ids": tuple(sorted(self._memory)),
                "stats": MappingProxyType(dict(self._stats)),
            }
        )


def graw_result_to_dict(
    result: GrawShadowResult,
    *,
    raw_prepare_elapsed_ms: Optional[float] = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "frame_id": result.frame_id,
        "begin_track_ids": list(result.begin_track_ids),
        "reserved_past_track_ids": list(result.reserved_past_track_ids),
        "eligible_past_track_ids": list(result.eligible_past_track_ids),
        "candidate_proposal_ids": list(result.candidate_proposal_ids),
        "candidate_native_track_ids": list(result.candidate_native_track_ids),
        "associations": [asdict(value) for value in result.associations],
        "matcher_diagnostics": dict(result.matcher_diagnostics),
        "memory_track_ids": list(result.memory_track_ids),
        "finish_elapsed_ms": result.elapsed_ms,
    }
    if raw_prepare_elapsed_ms is not None:
        payload["raw_prepare_elapsed_ms"] = float(raw_prepare_elapsed_ms)
        payload["total_observer_elapsed_ms"] = float(
            raw_prepare_elapsed_ms + result.elapsed_ms
        )
    return payload


def write_graw_shadow_diagnostics(
    path: os.PathLike[str] | str,
    *,
    scene_id: str,
    results: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    trace_valid: bool,
) -> str:
    destination = os.path.abspath(os.fspath(path))
    payload = {
        "schema": SCHEMA,
        "scene_id": str(scene_id),
        "trace_valid": bool(trace_valid),
        "frame_count": len(results),
        "frames": [dict(value) for value in results],
        "summary": dict(summary),
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(encoded) > 32 * 1024 * 1024:
        raise ValueError("Graw diagnostic exceeds the 32 MiB cap")
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
    "CounterfactualAssociation",
    "GrawFrameToken",
    "GrawShadow",
    "GrawShadowResult",
    "SCHEMA",
    "graw_result_to_dict",
    "write_graw_shadow_diagnostics",
]
