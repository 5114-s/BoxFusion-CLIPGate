"""Stable native-track identity for an observer-only BoxFusion sidecar.

The registry mirrors *row lifecycle* but never reads or changes BoxFusion
geometry, confidence, semantics, or fusion history.  Native association code
only reports the winner and suppressed row indices.  The registry unions the
corresponding observer identities, mirrors native keep-index operations, and
returns proposal-to-track identities for a separately prepared shadow batch.

It deliberately does not use ``fusion_list``.  The outer index of that list is
transient, while an inner list contains selected observation-history indices
and is not a complete association lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
import operator
from types import MappingProxyType
from typing import Mapping, NamedTuple, Optional, Sequence


# Absolute ceilings are deliberately not constructor defaults alone.  A caller
# may request a smaller per-registry budget, but can never raise either budget
# above these observer-memory bounds.
HARD_MAX_ACTIVE_ROWS = 1024
HARD_MAX_NEW_PROPOSALS = 4096
HARD_MAX_TOTAL_ROWS = 5120


class RegistryError(RuntimeError):
    """Base class for observer-registry contract failures."""


class RegistryStateError(RegistryError):
    """Raised when calls violate the begin/associate/keep/finalize protocol."""


class RegistryIntegrityError(RegistryError):
    """Raised when row identity is ambiguous or inconsistent."""


def _integer(name: str, value: object, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _integer_tuple(
    name: str,
    values: object,
    *,
    unique: bool,
    maximum_length: int,
) -> tuple[int, ...]:
    # Hook arguments are borrowed from native code.  Read only replayable,
    # random-access containers so an observer can never advance a one-shot
    # iterator before native BoxManager consumes it.  This covers list/tuple,
    # NumPy arrays, and Torch tensors without importing either dependency.
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
class KeyframeToken:
    """Opaque transaction token returned for a non-empty model keyframe."""

    serial: int
    frame_id: int


@dataclass(frozen=True)
class IdentityResolution:
    """Identity result aligned with the proposals supplied at begin time."""

    frame_id: int
    proposal_ids: tuple[int, ...]
    proposal_track_ids: tuple[Optional[int], ...]
    active_track_ids: tuple[int, ...]
    track_aliases: Mapping[int, int]


@dataclass
class _Pending:
    token: KeyframeToken
    proposal_ids: tuple[int, ...]
    first_new_row: int
    prior_rows: tuple[int, ...]
    prior_max_proposal_id: Optional[int]
    captured_proposal_tracks: Optional[tuple[int, ...]] = None
    association_closed: bool = False


class _RegistryLimits(NamedTuple):
    """Construction-time budgets; frozen so live code cannot raise a cap."""

    max_active_rows: int
    max_new_proposals: int


class ObserverTrackRegistry:
    """Bounded, deterministic observer identity aligned to native active rows.

    Integration protocol for a non-empty model keyframe::

        token = registry.begin_keyframe(frame_id, proposal_ids)
        registry.record_association(token, winner, losers, stage="spatial")
        registry.record_association(token, winner, losers,
                                    stage="correspondence")
        registry.apply_keep(token, native_keep_indices)       # BoxManager.update
        registry.apply_keep(token, native_valid_indices)      # check_valid_num
        resolution = registry.finalize_keyframe(token)

    ``apply_keep`` captures the current proposal identities before the first
    native reindexing.  Further keep operations may therefore delete or reorder
    active rows without losing proposal alignment.
    """

    __slots__ = (
        "_limits",
        "_row_track_ids",
        "_parent",
        "_pending",
        "_serial",
        "_max_proposal_id",
        "_last_seen_frame",
        "_last_committed_frame",
        "_closed",
        "_stats",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if name == "_limits":
            try:
                object.__getattribute__(self, "_limits")
            except AttributeError:
                pass
            else:
                raise AttributeError("observer registry limits are immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        *,
        max_active_rows: int = HARD_MAX_ACTIVE_ROWS,
        max_new_proposals: int = HARD_MAX_NEW_PROPOSALS,
    ) -> None:
        active_limit = _integer("max_active_rows", max_active_rows, 1)
        proposal_limit = _integer(
            "max_new_proposals", max_new_proposals, 1
        )
        # Keep the enforcement literals independent of rebindable module
        # names; exported constants are documentation/API, not a bypass knob.
        if active_limit > 1024:
            raise ValueError(
                f"max_active_rows cannot exceed hard cap "
                f"{1024}"
            )
        if proposal_limit > 4096:
            raise ValueError(
                f"max_new_proposals cannot exceed hard cap "
                f"{4096}"
            )
        self._limits = _RegistryLimits(active_limit, proposal_limit)
        self._row_track_ids: list[int] = []
        self._parent: dict[int, int] = {}
        self._pending: Optional[_Pending] = None
        self._serial = 0
        self._max_proposal_id: Optional[int] = None
        self._last_seen_frame: Optional[int] = None
        self._last_committed_frame: Optional[int] = None
        self._closed = False
        self._stats = {
            "model_keyframes": 0,
            "empty_keyframes": 0,
            "non_keyframes": 0,
            "spatial_edges": 0,
            "correspondence_edges": 0,
            "keep_operations": 0,
            "native_row_count_checks": 0,
            "aborted_keyframes": 0,
        }

    @property
    def max_active_rows(self) -> int:
        return self._active_row_cap()

    @property
    def max_new_proposals(self) -> int:
        return self._new_proposal_cap()

    def _active_row_cap(self) -> int:
        configured = _integer(
            "stored max_active_rows", self._limits.max_active_rows, 1
        )
        # The literal is an independent final clamp even if trusted-looking
        # module names and the private limits slot are both forcibly rebound.
        return min(configured, HARD_MAX_ACTIVE_ROWS, 1024)

    def _new_proposal_cap(self) -> int:
        configured = _integer(
            "stored max_new_proposals", self._limits.max_new_proposals, 1
        )
        return min(configured, HARD_MAX_NEW_PROPOSALS, 4096)

    def _total_row_cap(self) -> int:
        configured_total = self._active_row_cap() + self._new_proposal_cap()
        return min(configured_total, HARD_MAX_TOTAL_ROWS, 5120)

    def _assert_stored_rows_bounded(
        self, *, allow_temporary_rows: bool, context: str
    ) -> None:
        row_count = len(self._row_track_ids)
        cap = (
            self._total_row_cap()
            if allow_temporary_rows
            else self._active_row_cap()
        )
        if row_count > cap:
            cap_name = "temporary total" if allow_temporary_rows else "active"
            raise RegistryIntegrityError(
                f"{context} exceeds the {cap_name} row cap of {cap}"
            )

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        if self._pending is not None:
            raise RegistryStateError(
                "active identities are not externally stable during a keyframe"
            )
        self._assert_stored_rows_bounded(
            allow_temporary_rows=False,
            context="externally visible active rows",
        )
        return tuple(self._row_track_ids)

    @property
    def last_committed_frame(self) -> Optional[int]:
        return self._last_committed_frame

    @property
    def pending(self) -> bool:
        return self._pending is not None

    def _check_open_idle(self) -> None:
        if self._closed:
            raise RegistryStateError("observer registry is closed")
        if self._pending is not None:
            raise RegistryStateError("a keyframe transaction is already pending")

    def _next_frame(self, frame_id: object) -> int:
        result = _integer("frame_id", frame_id)
        if self._last_seen_frame is not None and result <= self._last_seen_frame:
            raise ValueError("frame_id must be strictly increasing")
        return result

    def mark_non_keyframe(self, frame_id: int) -> None:
        """Record a frame at which no model inference ran; state is unchanged."""

        self._check_open_idle()
        frame_id = self._next_frame(frame_id)
        self._last_seen_frame = frame_id
        self._stats["non_keyframes"] += 1

    def begin_keyframe(
        self, frame_id: int, proposal_ids: Sequence[int]
    ) -> Optional[KeyframeToken]:
        """Append proposal rows for one *actual model-inference* keyframe.

        An empty proposal list is an immediate no-op and returns ``None``.  It
        does not retire active tracks and does not change the terminal snapshot
        frame, matching the observer adapter's skip-empty behavior.
        """

        self._check_open_idle()
        self._assert_stored_rows_bounded(
            allow_temporary_rows=False,
            context="stored pre-keyframe rows",
        )
        frame_id = self._next_frame(frame_id)
        proposal_cap = self._new_proposal_cap()
        proposals = _integer_tuple(
            "proposal_ids",
            proposal_ids,
            unique=True,
            maximum_length=proposal_cap,
        )
        if (
            proposals
            and self._max_proposal_id is not None
            and min(proposals) <= self._max_proposal_id
        ):
            raise ValueError(
                "proposal IDs must be globally increasing across model keyframes"
            )
        total_cap = self._total_row_cap()
        if len(self._row_track_ids) + len(proposals) > total_cap:
            raise RegistryIntegrityError(
                f"pre-association rows exceed the temporary total row cap "
                f"of {total_cap}"
            )

        self._last_seen_frame = frame_id
        self._stats["model_keyframes"] += 1
        if not proposals:
            self._stats["empty_keyframes"] += 1
            return None

        self._serial += 1
        token = KeyframeToken(self._serial, frame_id)
        prior_rows = tuple(self._row_track_ids)
        first_new = len(self._row_track_ids)
        self._row_track_ids.extend(proposals)
        identities = set(self._row_track_ids)
        self._parent = {identity: identity for identity in identities}
        self._pending = _Pending(
            token=token,
            proposal_ids=proposals,
            first_new_row=first_new,
            prior_rows=prior_rows,
            prior_max_proposal_id=self._max_proposal_id,
        )
        self._max_proposal_id = max(proposals)
        return token

    def _require(self, token: KeyframeToken) -> _Pending:
        if self._pending is None or token is not self._pending.token:
            raise RegistryStateError("operation must use the exact pending token")
        return self._pending

    def assert_native_row_count(
        self, token: KeyframeToken, native_row_count: int
    ) -> None:
        """Handshake observer rows against native ``fusion_list`` rows.

        Integration calls this when attaching and immediately before and after
        every native outer-row reindex.  It detects a missed hook or incorrect
        attach point before observer identities can silently drift.
        """

        self._require(token)
        native_count = _integer("native_row_count", native_row_count)
        self._stats["native_row_count_checks"] += 1
        observer_count = len(self._row_track_ids)
        total_cap = self._total_row_cap()
        if native_count > total_cap:
            raise RegistryIntegrityError(
                f"native rows exceed the temporary total row cap of {total_cap}"
            )
        if observer_count > total_cap:
            raise RegistryIntegrityError(
                f"observer rows exceed the temporary total row cap of {total_cap}"
            )
        if native_count != observer_count:
            raise RegistryIntegrityError(
                "native/observer row-count mismatch: "
                f"native={native_count}, observer={observer_count}"
            )

    def _find(self, identity: int) -> int:
        parent = self._parent.get(identity)
        if parent is None:
            raise RegistryIntegrityError(f"unknown observer identity {identity}")
        path: list[int] = []
        while parent != identity:
            path.append(identity)
            identity = parent
            parent = self._parent[identity]
        for item in path:
            self._parent[item] = identity
        return identity

    def _capture_current(self, pending: _Pending) -> tuple[int, ...]:
        self._assert_stored_rows_bounded(
            allow_temporary_rows=True,
            context="proposal capture rows",
        )
        if pending.captured_proposal_tracks is None:
            stop = pending.first_new_row + len(pending.proposal_ids)
            if stop > len(self._row_track_ids):
                raise RegistryIntegrityError(
                    "native rows were reindexed before proposal identities were captured"
                )
            pending.captured_proposal_tracks = tuple(
                self._find(identity)
                for identity in self._row_track_ids[pending.first_new_row:stop]
            )
        return pending.captured_proposal_tracks

    def record_association(
        self,
        token: KeyframeToken,
        winner_row: int,
        loser_rows: Sequence[int],
        *,
        stage: str,
    ) -> int:
        """Union native rows for one accepted spatial/correspondence event."""

        pending = self._require(token)
        if pending.association_closed:
            raise RegistryStateError("association cannot occur after native reindexing")
        self._assert_stored_rows_bounded(
            allow_temporary_rows=True,
            context="association rows",
        )
        if stage not in ("spatial", "correspondence"):
            raise ValueError("stage must be 'spatial' or 'correspondence'")
        winner = _integer("winner_row", winner_row)
        total_cap = self._total_row_cap()
        losers = _integer_tuple(
            "loser_rows",
            loser_rows,
            unique=True,
            maximum_length=total_cap - 1,
        )
        rows = (winner, *losers)
        if winner in losers:
            raise ValueError("winner_row cannot also be a loser row")
        if any(row >= len(self._row_track_ids) for row in rows):
            raise RegistryIntegrityError("association row index is out of range")
        roots = {self._find(self._row_track_ids[row]) for row in rows}
        canonical = min(roots)
        for root in roots:
            self._parent[root] = canonical
        # Resolve canonical itself in case assignments above included it.
        self._parent[canonical] = canonical
        self._stats[stage + "_edges"] += len(losers)
        return canonical

    def apply_keep(
        self, token: KeyframeToken, keep_indices: Sequence[int]
    ) -> None:
        """Mirror one native row keep/reorder operation.

        The first call corresponds to ``BoxManager.update``.  A second call may
        mirror ``check_valid_num``.  Proposal identities are captured exactly
        once, before the first reindexing.
        """

        pending = self._require(token)
        self._assert_stored_rows_bounded(
            allow_temporary_rows=True,
            context="pre-keep rows",
        )
        keep = _integer_tuple(
            "keep_indices",
            keep_indices,
            unique=True,
            maximum_length=self._total_row_cap(),
        )
        if any(index >= len(self._row_track_ids) for index in keep):
            raise RegistryIntegrityError("keep index is out of range")
        self._capture_current(pending)
        pending.association_closed = True
        self._row_track_ids = [
            self._find(self._row_track_ids[index]) for index in keep
        ]
        self._stats["keep_operations"] += 1

    def finalize_keyframe(self, token: KeyframeToken) -> IdentityResolution:
        """Finalize active rows and return proposal-aligned stable identities."""

        pending = self._require(token)
        self._assert_stored_rows_bounded(
            allow_temporary_rows=True,
            context="pre-finalize rows",
        )
        active_cap = self._active_row_cap()
        if len(self._row_track_ids) > active_cap:
            raise RegistryIntegrityError(
                f"active observer-track cap of {active_cap} exceeded"
            )
        proposal_tracks = self._capture_current(pending)
        active = tuple(self._find(identity) for identity in self._row_track_ids)
        if len(set(active)) != len(active):
            raise RegistryIntegrityError(
                "multiple native active rows resolve to the same observer track"
            )
        active_set = set(active)
        aligned: tuple[Optional[int], ...] = tuple(
            identity if identity in active_set else None
            for identity in proposal_tracks
        )

        # Only emit aliases whose final canonical target is still an active
        # native row.  Chained unions are compressed to that final target.
        # Fully retired components must not be resurrected by a shadow commit.
        aliases = {
            identity: self._find(identity)
            for identity in tuple(self._parent)
            if (
                self._find(identity) != identity
                and self._find(identity) in active_set
            )
        }
        result = IdentityResolution(
            frame_id=pending.token.frame_id,
            proposal_ids=pending.proposal_ids,
            proposal_track_ids=aligned,
            active_track_ids=active,
            track_aliases=MappingProxyType(dict(sorted(aliases.items()))),
        )
        self._row_track_ids = list(active)
        self._parent = {}
        self._pending = None
        self._last_committed_frame = pending.token.frame_id
        return result

    def abort_keyframe(self, token: KeyframeToken) -> None:
        """Restore the pre-keyframe observer state after an adapter failure."""

        pending = self._require(token)
        self._row_track_ids = list(pending.prior_rows)
        self._max_proposal_id = pending.prior_max_proposal_id
        self._parent = {}
        self._pending = None
        self._stats["aborted_keyframes"] += 1

    def terminal_snapshot_frame(
        self, *, current_frame_id: Optional[int] = None, close: bool = False
    ) -> Optional[int]:
        """Return the last committed model keyframe, never a stale terminal frame."""

        if self._pending is not None:
            raise RegistryStateError("terminal observation cannot bypass a keyframe")
        if current_frame_id is not None:
            current = _integer("current_frame_id", current_frame_id)
            if (
                self._last_seen_frame is not None
                and current < self._last_seen_frame
            ):
                raise ValueError("terminal frame cannot precede the last seen frame")
        if not isinstance(close, bool):
            raise ValueError("close must be a boolean")
        if close:
            self._closed = True
        return self._last_committed_frame

    def diagnostics(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "active_rows": len(self._row_track_ids),
                "last_seen_frame": self._last_seen_frame,
                "last_committed_frame": self._last_committed_frame,
                "pending": self._pending is not None,
                "closed": self._closed,
                "stats": MappingProxyType(dict(self._stats)),
            }
        )


__all__ = [
    "HARD_MAX_ACTIVE_ROWS",
    "HARD_MAX_NEW_PROPOSALS",
    "HARD_MAX_TOTAL_ROWS",
    "IdentityResolution",
    "KeyframeToken",
    "ObserverTrackRegistry",
    "RegistryError",
    "RegistryIntegrityError",
    "RegistryStateError",
]
