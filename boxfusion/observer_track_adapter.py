"""Fail-open adapter for observer-only native-track identity diagnostics.

This module is deliberately independent from :mod:`demo`.  It wraps
``ObserverTrackRegistry`` with the transaction boundary needed by an online
observer such as Group3D-lite:

* snapshot the begin-frame past;
* attach only after ``BoxManager.init_new_predictions``;
* mirror native association and row reindexing through the existing
  ``BoxManager`` hooks;
* finalize before ``BoxFuser.boxfusion``;
* expose native-matched, native-unmatched, and reserved stable identities;
* map terminal output rows back to stable track IDs.

The adapter is observer-only.  Runtime failures invalidate its trace and
disable it for the remainder of the scene, but never become a native pipeline
decision.  All borrowed index containers are copied to bounded immutable
tuples before inspection.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import operator
import os
from pathlib import Path
import random
import re
import tempfile
from types import MappingProxyType
from typing import Iterator, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from boxfusion.observer_track_registry import (
    IdentityResolution,
    KeyframeToken,
    ObserverTrackRegistry,
)


# These public names document the protocol.  Enforcement below also uses
# literals, so rebinding a module constant cannot raise a live resource cap.
HARD_MAX_TRACE_FRAMES = 4096
HARD_MAX_TRACE_PROPOSALS = 131072
HARD_MAX_ERRORS = 64
HARD_MAX_ERROR_CHARS = 512
HARD_MAX_DIGEST_FIELDS = 64
HARD_MAX_DIGEST_NODES = 65536
HARD_MAX_DIGEST_BYTES = 128 * 1024 * 1024
HARD_MAX_DIAGNOSTIC_BYTES = 32 * 1024 * 1024

_SCENE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


class ObserverAdapterError(RuntimeError):
    """Base class for adapter configuration and audit errors."""


class ObserverBoundaryError(ObserverAdapterError):
    """Raised by the explicit native/RNG digest assertion helper."""


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    result = int(result)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _bounded_integer_tuple(
    name: str,
    values: object,
    *,
    maximum_length: int,
    unique: bool,
) -> tuple[int, ...]:
    """Copy a bounded random-access container without consuming iterators."""

    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a bounded indexable container")
    try:
        length = len(values)  # type: ignore[arg-type]
        get_item = values.__getitem__  # type: ignore[attr-defined]
    except (AttributeError, TypeError) as error:
        raise ValueError(
            f"{name} must be a bounded indexable container"
        ) from error
    if length > maximum_length:
        raise ValueError(f"{name} exceeds the cap of {maximum_length}")
    result = tuple(
        _integer(f"{name} item", get_item(index))
        for index in range(length)
    )
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


@dataclass(frozen=True)
class ObserverAdapterConfig:
    """Frozen observer limits.  ``disabled`` is the safe default."""

    mode: str = "disabled"
    diagnostics_root: Optional[str] = None
    max_active_rows: int = 1024
    max_new_proposals: int = 4096
    max_trace_frames: int = 4096
    max_trace_proposals: int = 131072

    def __post_init__(self) -> None:
        if self.mode not in ("disabled", "shadow"):
            raise ValueError(
                "observer mode must be 'disabled' or 'shadow'; active "
                "association is not implemented by this adapter"
            )
        active = _integer("max_active_rows", self.max_active_rows, 1)
        proposals = _integer("max_new_proposals", self.max_new_proposals, 1)
        frames = _integer("max_trace_frames", self.max_trace_frames, 1)
        total = _integer("max_trace_proposals", self.max_trace_proposals, 1)
        if active > 1024:
            raise ValueError("max_active_rows cannot exceed 1024")
        if proposals > 4096:
            raise ValueError("max_new_proposals cannot exceed 4096")
        if frames > 4096:
            raise ValueError("max_trace_frames cannot exceed 4096")
        if total > 131072:
            raise ValueError("max_trace_proposals cannot exceed 131072")
        if self.diagnostics_root is not None and not isinstance(
            self.diagnostics_root, (str, os.PathLike)
        ):
            raise ValueError("diagnostics_root must be a path or null")

    @classmethod
    def from_mapping(cls, config: Optional[Mapping[str, object]]) -> "ObserverAdapterConfig":
        values = config or {}
        if not hasattr(values, "get"):
            raise ValueError("observer configuration must be a mapping")
        return cls(
            mode=str(values.get("mode", "disabled")),
            diagnostics_root=(
                None
                if values.get("diagnostics_root") is None
                else os.fspath(values.get("diagnostics_root"))
            ),
            max_active_rows=values.get("max_active_rows", 1024),  # type: ignore[arg-type]
            max_new_proposals=values.get("max_new_proposals", 4096),  # type: ignore[arg-type]
            max_trace_frames=values.get("max_trace_frames", 4096),  # type: ignore[arg-type]
            max_trace_proposals=values.get("max_trace_proposals", 131072),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class AdapterFrameToken:
    """Opaque token for one adapter transaction."""

    serial: int
    frame_id: int


@dataclass(frozen=True)
class FrameIdentityTrace:
    """Stable identity trace for one successfully observed keyframe."""

    frame_id: int
    proposal_ids: tuple[int, ...]
    begin_past_track_ids: tuple[int, ...]
    proposal_track_ids: tuple[Optional[int], ...]
    native_target_track_ids: tuple[tuple[int, ...], ...]
    native_status: tuple[str, ...]
    reserved_past_track_ids: tuple[int, ...]
    native_matched_proposal_ids: tuple[int, ...]
    native_unmatched_retained_proposal_ids: tuple[int, ...]
    native_unmatched_dropped_proposal_ids: tuple[int, ...]
    active_track_ids: tuple[int, ...]
    track_aliases: Mapping[int, int]


@dataclass(frozen=True)
class TerminalIdentityMapping:
    """Stable identities aligned with terminal serialized output rows."""

    snapshot_frame_id: Optional[int]
    native_row_count: int
    kept_native_indices: tuple[int, ...]
    output_track_ids: tuple[int, ...]


@dataclass
class _PendingFrame:
    adapter_token: AdapterFrameToken
    registry_token: Optional[KeyframeToken]
    proposal_ids: tuple[int, ...]
    begin_past_track_ids: tuple[int, ...]
    initial_rows: tuple[int, ...]
    parent: dict[int, int]
    reserved_past_track_ids: set[int]
    attached: bool = False


class NativeRNGDigest(NamedTuple):
    native_sha256: str
    rng_sha256: str


class _DigestBudget:
    __slots__ = ("nodes", "bytes")

    def __init__(self) -> None:
        self.nodes = 0
        self.bytes = 0

    def add(self, *, nodes: int = 1, byte_count: int = 0) -> None:
        self.nodes += nodes
        self.bytes += byte_count
        if self.nodes > 65536:
            raise ValueError("native digest exceeds 65536 nodes")
        if self.bytes > 134217728:
            raise ValueError("native digest exceeds 128 MiB")


def _hash_piece(hasher: "hashlib._Hash", value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, "little", signed=False))
    hasher.update(value)


def _hash_value(
    hasher: "hashlib._Hash", value: object, budget: _DigestBudget, depth: int = 0
) -> None:
    if depth > 16:
        raise ValueError("native digest exceeds nesting depth 16")
    budget.add()
    if value is None:
        _hash_piece(hasher, b"none")
        return
    if isinstance(value, (bool, np.bool_)):
        _hash_piece(hasher, b"bool:1" if bool(value) else b"bool:0")
        return
    if isinstance(value, (int, np.integer)):
        _hash_piece(hasher, b"int:" + str(int(value)).encode("ascii"))
        return
    if isinstance(value, (float, np.floating)):
        _hash_piece(hasher, b"float:" + float(value).hex().encode("ascii"))
        return
    if isinstance(value, str):
        raw = value.encode("utf-8")
        budget.add(nodes=0, byte_count=len(raw))
        _hash_piece(hasher, b"str:" + raw)
        return
    if isinstance(value, bytes):
        budget.add(nodes=0, byte_count=len(value))
        _hash_piece(hasher, b"bytes:" + value)
        return
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("object arrays are unsupported by native digest")
        array = np.ascontiguousarray(value)
        raw = memoryview(array).cast("B").tobytes()
        budget.add(nodes=0, byte_count=len(raw))
        _hash_piece(hasher, b"ndarray")
        _hash_piece(hasher, array.dtype.str.encode("ascii"))
        _hash_piece(hasher, json.dumps(array.shape).encode("ascii"))
        _hash_piece(hasher, raw)
        return

    # Torch is optional for unit-level use of this module.  Import lazily and
    # hash a detached CPU copy so the borrowed tensor is never changed.
    try:
        import torch
    except ImportError:  # pragma: no cover - production environment has torch
        torch = None  # type: ignore[assignment]
    if torch is not None and torch.is_tensor(value):
        tensor = value.detach()
        if tensor.layout != torch.strided:
            raise ValueError("non-strided tensors are unsupported by native digest")
        cpu = tensor.contiguous().cpu()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        budget.add(nodes=0, byte_count=len(raw))
        _hash_piece(hasher, b"tensor")
        _hash_piece(hasher, str(cpu.dtype).encode("ascii"))
        _hash_piece(hasher, json.dumps(tuple(cpu.shape)).encode("ascii"))
        _hash_piece(hasher, raw)
        return
    if isinstance(value, Mapping):
        if len(value) > 65536:
            raise ValueError("native digest mapping exceeds 65536 items")
        if any(not isinstance(key, str) for key in value):
            raise ValueError("native digest mapping keys must be strings")
        _hash_piece(hasher, b"mapping")
        for key in sorted(value):
            _hash_value(hasher, key, budget, depth + 1)
            _hash_value(hasher, value[key], budget, depth + 1)
        return
    if isinstance(value, (tuple, list)):
        if len(value) > 65536:
            raise ValueError("native digest sequence exceeds 65536 items")
        _hash_piece(hasher, b"tuple" if isinstance(value, tuple) else b"list")
        for item in value:
            _hash_value(hasher, item, budget, depth + 1)
        return
    raise ValueError(
        f"unsupported native digest value type: {type(value).__name__}"
    )


def _mapping_digest(fields: Mapping[str, object]) -> str:
    if not isinstance(fields, Mapping):
        raise ValueError("native_fields must be a mapping")
    if len(fields) > 64:
        raise ValueError("native_fields exceeds the cap of 64 fields")
    digest = hashlib.sha256()
    _hash_value(digest, fields, _DigestBudget())
    return digest.hexdigest()


def _capture_rng_state() -> tuple[object, object, object, Optional[tuple[object, ...]]]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        import torch
    except ImportError:  # pragma: no cover
        return python_state, numpy_state, None, None
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states: Optional[tuple[object, ...]] = None
    if torch.cuda.is_initialized():
        cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    return python_state, numpy_state, cpu_state, cuda_states


def _rng_state_digest(state: tuple[object, object, object, object]) -> str:
    digest = hashlib.sha256()
    _hash_value(digest, state, _DigestBudget())
    return digest.hexdigest()


def _restore_rng_state(state: tuple[object, object, object, object]) -> None:
    python_state, numpy_state, cpu_state, cuda_states = state
    random.setstate(python_state)  # type: ignore[arg-type]
    np.random.set_state(numpy_state)  # type: ignore[arg-type]
    if cpu_state is None:
        return
    import torch

    torch.random.set_rng_state(cpu_state)  # type: ignore[arg-type]
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(list(cuda_states))  # type: ignore[arg-type]


def capture_native_rng_digest(native_fields: Mapping[str, object]) -> NativeRNGDigest:
    """Digest selected native fields and all already-initialized RNG streams."""

    rng_state = _capture_rng_state()
    return NativeRNGDigest(
        native_sha256=_mapping_digest(native_fields),
        rng_sha256=_rng_state_digest(rng_state),
    )


def assert_native_rng_digest_unchanged(
    before: NativeRNGDigest, after: NativeRNGDigest
) -> None:
    """Explicit assertion used by smoke tests around pure observer calls."""

    changed: list[str] = []
    if before.native_sha256 != after.native_sha256:
        changed.append("native fields")
    if before.rng_sha256 != after.rng_sha256:
        changed.append("RNG state")
    if changed:
        raise ObserverBoundaryError("observer boundary changed " + " and ".join(changed))


class ObserverTrackAdapter:
    """Bounded, fail-open transaction adapter for a single scene."""

    def __init__(
        self,
        config: Optional[ObserverAdapterConfig] = None,
        *,
        scene_id: str,
    ) -> None:
        self.config = config or ObserverAdapterConfig()
        if not isinstance(self.config, ObserverAdapterConfig):
            raise ValueError("config must be ObserverAdapterConfig")
        if not isinstance(scene_id, str) or _SCENE_ID.fullmatch(scene_id) is None:
            raise ValueError("scene_id must contain only safe filename characters")
        self.scene_id = scene_id
        self._registry = ObserverTrackRegistry(
            max_active_rows=self.config.max_active_rows,
            max_new_proposals=self.config.max_new_proposals,
        )
        self._serial = 0
        self._pending: Optional[_PendingFrame] = None
        self._traces: list[FrameIdentityTrace] = []
        self._trace_proposal_count = 0
        self._valid = True
        self._errors: list[str] = []
        self._terminal: Optional[TerminalIdentityMapping] = None
        self._boundary_checks = 0
        self._boundary_violations = 0

    @property
    def enabled(self) -> bool:
        return self.config.mode == "shadow" and self._valid

    @property
    def trace_valid(self) -> bool:
        return self._valid

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(self._errors)

    @property
    def frame_traces(self) -> tuple[FrameIdentityTrace, ...]:
        return tuple(self._traces)

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        if not self.enabled:
            return ()
        return self._registry.active_track_ids

    def _bounded_error(self, where: str, error: object) -> str:
        try:
            message = f"{where}: {type(error).__name__}: {error}"
        except Exception:
            message = f"{where}: unprintable observer error"
        return message[:512]

    def _invalidate(self, where: str, error: object) -> None:
        if len(self._errors) < 64:
            self._errors.append(self._bounded_error(where, error))
        pending = self._pending
        if pending is not None and pending.registry_token is not None:
            try:
                if self._registry.pending:
                    self._registry.abort_keyframe(pending.registry_token)
            except Exception:
                pass
        self._pending = None
        self._valid = False

    def begin_keyframe(
        self, frame_id: int, proposal_ids: Sequence[int]
    ) -> Optional[AdapterFrameToken]:
        """Begin an actual inference keyframe; never raises into native code."""

        if self.config.mode == "disabled" or not self._valid:
            return None
        try:
            if self._pending is not None:
                raise ObserverAdapterError("a frame transaction is already pending")
            frame = _integer("frame_id", frame_id)
            proposals = _bounded_integer_tuple(
                "proposal_ids",
                proposal_ids,
                maximum_length=min(self.config.max_new_proposals, 4096),
                unique=True,
            )
            if len(self._traces) >= min(self.config.max_trace_frames, 4096):
                raise ObserverAdapterError("frame trace cap exceeded")
            if (
                self._trace_proposal_count + len(proposals)
                > min(self.config.max_trace_proposals, 131072)
            ):
                raise ObserverAdapterError("proposal trace cap exceeded")
            begin_past = self._registry.active_track_ids
            registry_token = self._registry.begin_keyframe(frame, proposals)
            self._serial += 1
            adapter_token = AdapterFrameToken(self._serial, frame)
            initial = begin_past + proposals
            self._pending = _PendingFrame(
                adapter_token=adapter_token,
                registry_token=registry_token,
                proposal_ids=proposals,
                begin_past_track_ids=begin_past,
                initial_rows=initial,
                parent={identity: identity for identity in initial},
                reserved_past_track_ids=set(),
            )
            return adapter_token
        except Exception as error:
            self._invalidate("begin_keyframe", error)
            return None

    def mark_non_keyframe(self, frame_id: int) -> bool:
        if self.config.mode == "disabled" or not self._valid:
            return False
        try:
            self._registry.mark_non_keyframe(frame_id)
            return True
        except Exception as error:
            self._invalidate("mark_non_keyframe", error)
            return False

    def _require(self, token: AdapterFrameToken) -> _PendingFrame:
        if self._pending is None or token is not self._pending.adapter_token:
            raise ObserverAdapterError("operation must use the exact pending token")
        return self._pending

    @staticmethod
    def _find(parent: dict[int, int], identity: int) -> int:
        root = identity
        while parent[root] != root:
            root = parent[root]
        while identity != root:
            previous = parent[identity]
            parent[identity] = root
            identity = previous
        return root

    def attach(self, box_manager: object, token: Optional[AdapterFrameToken]) -> bool:
        """Attach after native proposal rows have been appended."""

        if token is None or not self.enabled:
            return False
        try:
            pending = self._require(token)
            if pending.registry_token is None:
                # Empty keyframes have no appended rows and require no hooks.
                return True
            if pending.attached:
                raise ObserverAdapterError("frame is already attached")
            attach = getattr(box_manager, "attach_observer_track_registry")
            attached = bool(attach(self, token))
            if not attached:
                raise ObserverAdapterError("BoxManager rejected observer attachment")
            pending.attached = True
            return True
        except Exception as error:
            self._invalidate("attach", error)
            return False

    # The following four methods implement the protocol expected by the
    # BoxManager observer hooks.  They intentionally raise on an observer
    # contract error; BoxManager catches that error, aborts this adapter, and
    # continues its native operation with the original borrowed arguments.
    def assert_native_row_count(
        self, token: AdapterFrameToken, native_row_count: int
    ) -> None:
        pending = self._require(token)
        if pending.registry_token is None:
            expected = len(pending.begin_past_track_ids)
            if _integer("native_row_count", native_row_count) != expected:
                raise ObserverAdapterError("empty-keyframe native row mismatch")
            return
        self._registry.assert_native_row_count(
            pending.registry_token, native_row_count
        )

    def record_association(
        self,
        token: AdapterFrameToken,
        winner_row: int,
        loser_rows: Sequence[int],
        *,
        stage: str,
    ) -> int:
        pending = self._require(token)
        if pending.registry_token is None:
            raise ObserverAdapterError("empty keyframe cannot associate rows")
        winner = _integer("winner_row", winner_row)
        losers = _bounded_integer_tuple(
            "loser_rows",
            loser_rows,
            maximum_length=5119,
            unique=True,
        )
        indices = (winner, *losers)
        if winner in losers:
            raise ValueError("winner_row cannot also be a loser row")
        if any(index >= len(pending.initial_rows) for index in indices):
            raise ObserverAdapterError("association row index is out of range")

        canonical = self._registry.record_association(
            pending.registry_token, winner, losers, stage=stage
        )
        identities = tuple(pending.initial_rows[index] for index in indices)
        past = set(pending.begin_past_track_ids)
        pending.reserved_past_track_ids.update(
            identity for identity in identities if identity in past
        )
        roots = {self._find(pending.parent, identity) for identity in identities}
        component_root = min(roots)
        for root in roots:
            pending.parent[root] = component_root
        pending.parent[component_root] = component_root
        return canonical

    def apply_keep(
        self, token: AdapterFrameToken, keep_indices: Sequence[int]
    ) -> None:
        pending = self._require(token)
        if pending.registry_token is None:
            raise ObserverAdapterError("empty keyframe cannot reindex rows")
        keep = _bounded_integer_tuple(
            "keep_indices", keep_indices, maximum_length=5120, unique=True
        )
        self._registry.apply_keep(pending.registry_token, keep)

    def abort_keyframe(self, token: AdapterFrameToken) -> None:
        """Fail open for native BoxManager and permanently invalidate trace."""

        try:
            pending = self._require(token)
            if pending.registry_token is not None and self._registry.pending:
                self._registry.abort_keyframe(pending.registry_token)
        except Exception:
            pass
        self._pending = None
        if len(self._errors) < 64:
            self._errors.append("box_manager_hook: observer transaction aborted")
        self._valid = False

    def _native_targets(
        self, pending: _PendingFrame
    ) -> tuple[tuple[int, ...], ...]:
        past = set(pending.begin_past_track_ids)
        components: dict[int, set[int]] = {}
        for identity in pending.initial_rows:
            root = self._find(pending.parent, identity)
            components.setdefault(root, set()).add(identity)
        return tuple(
            tuple(sorted(components[self._find(pending.parent, proposal)] & past))
            for proposal in pending.proposal_ids
        )

    def _trace_from_resolution(
        self, pending: _PendingFrame, resolution: IdentityResolution
    ) -> FrameIdentityTrace:
        targets = self._native_targets(pending)
        statuses: list[str] = []
        matched: list[int] = []
        unmatched_dropped: list[int] = []
        unmatched_retained_by_track: dict[int, int] = {}
        for proposal, committed, native_targets in zip(
            pending.proposal_ids, resolution.proposal_track_ids, targets
        ):
            if native_targets:
                matched.append(proposal)
                statuses.append(
                    "matched_past_retained"
                    if committed is not None
                    else "matched_past_dropped"
                )
            elif committed is None:
                unmatched_dropped.append(proposal)
                statuses.append("unmatched_dropped")
            else:
                statuses.append("unmatched_retained")
                previous = unmatched_retained_by_track.get(committed)
                if previous is None or proposal < previous:
                    unmatched_retained_by_track[committed] = proposal
        unmatched_retained = tuple(
            proposal
            for _, proposal in sorted(unmatched_retained_by_track.items())
        )
        return FrameIdentityTrace(
            frame_id=resolution.frame_id,
            proposal_ids=tuple(resolution.proposal_ids),
            begin_past_track_ids=tuple(pending.begin_past_track_ids),
            proposal_track_ids=tuple(resolution.proposal_track_ids),
            native_target_track_ids=targets,
            native_status=tuple(statuses),
            reserved_past_track_ids=tuple(
                sorted(pending.reserved_past_track_ids)
            ),
            native_matched_proposal_ids=tuple(matched),
            native_unmatched_retained_proposal_ids=unmatched_retained,
            native_unmatched_dropped_proposal_ids=tuple(unmatched_dropped),
            active_track_ids=tuple(resolution.active_track_ids),
            track_aliases=MappingProxyType(dict(resolution.track_aliases)),
        )

    def finalize(
        self, box_manager: object, token: Optional[AdapterFrameToken]
    ) -> Optional[FrameIdentityTrace]:
        """Detach and commit just before native ``BoxFuser.boxfusion``."""

        if token is None or not self.enabled:
            return None
        pending: Optional[_PendingFrame] = None
        try:
            pending = self._require(token)
            if pending.registry_token is None:
                trace = FrameIdentityTrace(
                    frame_id=token.frame_id,
                    proposal_ids=(),
                    begin_past_track_ids=pending.begin_past_track_ids,
                    proposal_track_ids=(),
                    native_target_track_ids=(),
                    native_status=(),
                    reserved_past_track_ids=(),
                    native_matched_proposal_ids=(),
                    native_unmatched_retained_proposal_ids=(),
                    native_unmatched_dropped_proposal_ids=(),
                    active_track_ids=pending.begin_past_track_ids,
                    track_aliases=MappingProxyType({}),
                )
            else:
                detach = getattr(box_manager, "detach_observer_track_registry")
                attached_registry, attached_token, hook_error = detach()
                pending.attached = False
                if hook_error is not None:
                    raise ObserverAdapterError(
                        f"BoxManager observer hook failed: {hook_error}"
                    )
                if attached_registry is not self or attached_token is not token:
                    raise ObserverAdapterError("BoxManager observer attachment drifted")
                native_rows = len(getattr(box_manager, "fusion_list"))
                self._registry.assert_native_row_count(
                    pending.registry_token, native_rows
                )
                resolution = self._registry.finalize_keyframe(
                    pending.registry_token
                )
                trace = self._trace_from_resolution(pending, resolution)
            self._pending = None
            self._traces.append(trace)
            self._trace_proposal_count += len(trace.proposal_ids)
            return trace
        except Exception as error:
            # Detach best-effort even when a final handshake fails.  Native
            # state is neither read for decisions nor changed here.
            if pending is not None and pending.attached:
                try:
                    getattr(box_manager, "detach_observer_track_registry")()
                except Exception:
                    pass
            self._invalidate("finalize", error)
            return None

    def terminal_mapping(
        self,
        kept_native_indices: Sequence[int],
        *,
        native_row_count: int,
        current_frame_id: Optional[int] = None,
        close: bool = True,
    ) -> Optional[TerminalIdentityMapping]:
        """Map a copied terminal keep-index list to output stable IDs."""

        if not self.enabled:
            return None
        try:
            if self._pending is not None:
                raise ObserverAdapterError("terminal mapping cannot bypass a keyframe")
            native_count = _integer("native_row_count", native_row_count)
            active = self._registry.active_track_ids
            if native_count != len(active):
                raise ObserverAdapterError(
                    "terminal native/observer row-count mismatch"
                )
            keep = _bounded_integer_tuple(
                "kept_native_indices",
                kept_native_indices,
                maximum_length=min(self.config.max_active_rows, 1024),
                unique=True,
            )
            if any(index >= native_count for index in keep):
                raise ObserverAdapterError("terminal keep index is out of range")
            snapshot = self._registry.terminal_snapshot_frame(
                current_frame_id=current_frame_id, close=close
            )
            mapping = TerminalIdentityMapping(
                snapshot_frame_id=snapshot,
                native_row_count=native_count,
                kept_native_indices=keep,
                output_track_ids=tuple(active[index] for index in keep),
            )
            self._terminal = mapping
            return mapping
        except Exception as error:
            self._invalidate("terminal_mapping", error)
            return None

    @contextmanager
    def observer_boundary(
        self,
        native_fields: Mapping[str, object],
        *,
        label: str,
    ) -> Iterator[None]:
        """Fail-open native/RNG identity audit around pure observer work.

        Observer exceptions and digest violations are suppressed so native
        inference can continue.  Any changed RNG stream is restored before
        control returns.  A native-field violation cannot be safely restored,
        so it invalidates the trace and is made visible in diagnostics.
        """

        if self.config.mode == "disabled" or not self._valid:
            yield
            return
        before_rng = _capture_rng_state()
        try:
            before = NativeRNGDigest(
                _mapping_digest(native_fields), _rng_state_digest(before_rng)
            )
        except Exception as error:
            self._invalidate(f"boundary[{label}].capture", error)
            try:
                yield
            except Exception:
                pass
            return
        observer_error: Optional[Exception] = None
        try:
            yield
        except Exception as error:  # observer-only errors never escape
            observer_error = error
        after_rng = _capture_rng_state()
        try:
            after = NativeRNGDigest(
                _mapping_digest(native_fields), _rng_state_digest(after_rng)
            )
            self._boundary_checks += 1
            assert_native_rng_digest_unchanged(before, after)
        except Exception as error:
            self._boundary_violations += 1
            try:
                _restore_rng_state(before_rng)
            except Exception as restore_error:
                self._invalidate(
                    f"boundary[{label}].rng_restore", restore_error
                )
            self._invalidate(f"boundary[{label}]", error)
        else:
            # Restore first, then invalidate for a raised observer exception.
            # The states normally compare equal; restoration is unnecessary.
            if observer_error is not None:
                self._invalidate(f"boundary[{label}].observer", observer_error)

    @staticmethod
    def _trace_json(trace: FrameIdentityTrace) -> dict[str, object]:
        return {
            "frame_id": trace.frame_id,
            "proposal_ids": list(trace.proposal_ids),
            "begin_past_track_ids": list(trace.begin_past_track_ids),
            "proposal_track_ids": list(trace.proposal_track_ids),
            "native_target_track_ids": [
                list(values) for values in trace.native_target_track_ids
            ],
            "native_status": list(trace.native_status),
            "reserved_past_track_ids": list(trace.reserved_past_track_ids),
            "native_matched_proposal_ids": list(
                trace.native_matched_proposal_ids
            ),
            "native_unmatched_retained_proposal_ids": list(
                trace.native_unmatched_retained_proposal_ids
            ),
            "native_unmatched_dropped_proposal_ids": list(
                trace.native_unmatched_dropped_proposal_ids
            ),
            "active_track_ids": list(trace.active_track_ids),
            "track_aliases": [
                [source, target]
                for source, target in sorted(trace.track_aliases.items())
            ],
        }

    def diagnostics(self) -> Mapping[str, object]:
        terminal = None
        if self._terminal is not None:
            terminal = {
                "snapshot_frame_id": self._terminal.snapshot_frame_id,
                "native_row_count": self._terminal.native_row_count,
                "kept_native_indices": list(
                    self._terminal.kept_native_indices
                ),
                "output_track_ids": list(self._terminal.output_track_ids),
            }
        payload = {
            "schema_version": 1,
            "scene_id": self.scene_id,
            "mode": self.config.mode,
            "trace_valid": self._valid,
            "errors": list(self._errors),
            "limits": {
                "max_active_rows": min(self.config.max_active_rows, 1024),
                "max_new_proposals": min(self.config.max_new_proposals, 4096),
                "max_trace_frames": min(self.config.max_trace_frames, 4096),
                "max_trace_proposals": min(
                    self.config.max_trace_proposals, 131072
                ),
            },
            "frame_count": len(self._traces),
            "proposal_trace_count": self._trace_proposal_count,
            "boundary_checks": self._boundary_checks,
            "boundary_violations": self._boundary_violations,
            "frames": [self._trace_json(trace) for trace in self._traces],
            "terminal": terminal,
        }
        return MappingProxyType(payload)

    def write_diagnostics(
        self, path: Optional[os.PathLike[str] | str] = None
    ) -> Optional[Path]:
        """Atomically write one bounded scene JSON diagnostic."""

        if path is None:
            if self.config.diagnostics_root is None:
                return None
            destination = (
                Path(self.config.diagnostics_root)
                / f"{self.scene_id}.observer_tracks.json"
            )
        else:
            destination = Path(path)
        payload = json.dumps(
            dict(self.diagnostics()), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(payload) > 33554432:
            self._invalidate(
                "write_diagnostics",
                ObserverAdapterError("diagnostic exceeds 32 MiB"),
            )
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: Optional[str] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, destination)
            temporary_name = None
            return destination
        except Exception as error:
            self._invalidate("write_diagnostics", error)
            return None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass


def build_observer_track_adapter(
    cfg: Optional[Mapping[str, object]], *, scene_id: str
) -> ObserverTrackAdapter:
    """Build from the top-level BoxFusion config without modifying it.

    The accepted location is ``association.group3d_lite``.  Absence of any
    section is exactly equivalent to ``mode: disabled``.
    """

    top = cfg or {}
    association = top.get("association", {})  # type: ignore[union-attr]
    if association is None:
        association = {}
    if not hasattr(association, "get"):
        raise ValueError("association configuration must be a mapping")
    section = association.get("group3d_lite", {})
    if section is None:
        section = {}
    return ObserverTrackAdapter(
        ObserverAdapterConfig.from_mapping(section), scene_id=scene_id
    )


__all__ = [
    "AdapterFrameToken",
    "FrameIdentityTrace",
    "HARD_MAX_DIAGNOSTIC_BYTES",
    "HARD_MAX_DIGEST_BYTES",
    "HARD_MAX_DIGEST_FIELDS",
    "HARD_MAX_DIGEST_NODES",
    "HARD_MAX_ERRORS",
    "HARD_MAX_ERROR_CHARS",
    "HARD_MAX_TRACE_FRAMES",
    "HARD_MAX_TRACE_PROPOSALS",
    "NativeRNGDigest",
    "ObserverAdapterConfig",
    "ObserverAdapterError",
    "ObserverBoundaryError",
    "ObserverTrackAdapter",
    "TerminalIdentityMapping",
    "assert_native_rng_digest_unchanged",
    "build_observer_track_adapter",
    "capture_native_rng_digest",
]
