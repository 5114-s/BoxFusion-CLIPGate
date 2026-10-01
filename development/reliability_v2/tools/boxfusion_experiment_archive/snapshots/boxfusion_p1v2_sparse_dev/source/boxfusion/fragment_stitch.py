"""Deterministic observer-only stitching of Mask Graph fragments.

This module turns lifecycle snapshots from
``OnlineRefinementController._live_mask_graph_snapshots`` into immutable
*candidate* clusters.  It deliberately does not mutate snapshots, tracks, or
detections, and it has no ground-truth, model, or file-system dependency.

Two fragments can be linked only when all of the following hold:

* both have the same normalized, non-empty label;
* their event frames are sufficiently separated;
* either their AABB IoU is high, or their intersection-over-smaller-volume is
  high while their centers remain close.

Fragments are partitioned with a deterministic anchor-clique procedure.  An
anchor is selected from its current direct-neighbor evidence; its neighbors
are then considered in decreasing anchor-pair IoU order and admitted only
when they are directly compatible with *every* member already admitted.
Unlike union-find, this cannot close an A--B--C chain whose endpoints are
incompatible.  Cluster-level score, lifecycle, evidence-count, and
existing-confirmation gates are then applied.  The safe default is disabled,
so importing or calling this module cannot affect released output unless an
experiment opts in explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral, Real
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_FRAGMENT_STITCH_CONFIG: Dict[str, object] = {
    # Observer safety default.
    "enabled": False,
    # Pair compatibility is:
    # IoU >= minimum_pair_iou OR
    # (containment >= minimum_pair_containment AND
    #  center distance <= maximum_center_distance).
    "minimum_pair_iou": 0.40,
    "minimum_pair_containment": 0.60,
    "maximum_center_distance": 0.25,
    # A cluster must contain at least one strong fragment, while the mean gate
    # suppresses clusters formed by one strong and many weak fragments.
    "minimum_max_detector_score": 0.85,
    "minimum_mean_detector_score": 0.70,
    # Repeated proposals from nearby frames are not independent identity
    # evidence.
    "minimum_event_frame_separation": 5,
    # "Live" means an active or archived member.  Absorbed/discarded-only
    # clusters remain diagnostic and are excluded by default.
    "require_live_member": True,
}


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bounded_float(name: str, value: object) -> float:
    result = _finite_float(name, value)
    if result < 0.0 or result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _strict_int(name: str, value: object, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a boolean")
    return bool(value)


def _normalize_label(label: str) -> str:
    return " ".join(label.casefold().replace("_", " ").replace("-", " ").split())


def resolve_fragment_stitch_config(
    config: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    """Return a detached and strictly validated stitch configuration."""

    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise ValueError("fragment_stitch config must be a mapping")

    unknown = sorted(set(config) - set(DEFAULT_FRAGMENT_STITCH_CONFIG))
    if unknown:
        raise ValueError(
            "Unknown fragment_stitch config key(s): " + ", ".join(unknown)
        )

    resolved = dict(DEFAULT_FRAGMENT_STITCH_CONFIG)
    resolved.update(config)

    for key in ("enabled", "require_live_member"):
        resolved[key] = _strict_bool(
            f"fragment_stitch.{key}", resolved[key]
        )

    for key in (
        "minimum_pair_iou",
        "minimum_pair_containment",
        "minimum_max_detector_score",
        "minimum_mean_detector_score",
    ):
        resolved[key] = _bounded_float(
            f"fragment_stitch.{key}", resolved[key]
        )

    resolved["maximum_center_distance"] = _finite_float(
        "fragment_stitch.maximum_center_distance",
        resolved["maximum_center_distance"],
    )
    if float(resolved["maximum_center_distance"]) < 0.0:
        raise ValueError(
            "fragment_stitch.maximum_center_distance must be non-negative"
        )

    resolved["minimum_event_frame_separation"] = _strict_int(
        "fragment_stitch.minimum_event_frame_separation",
        resolved["minimum_event_frame_separation"],
        1,
    )

    return resolved


def _readonly_box(value: object, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except Exception as error:
        raise ValueError(f"{name} cannot be converted to an array") from error
    if (
        array.shape != (6,)
        or not np.issubdtype(array.dtype, np.number)
    ):
        raise ValueError(f"{name} must have numeric shape [6]")
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array[3:6] <= 0.0):
        raise ValueError(f"{name} dimensions must be positive")
    result = np.array(array, dtype=np.float32, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class _FragmentSnapshot:
    track_id: int
    lifecycle_state: str
    event_frame: int
    box: np.ndarray
    view_count: int
    node_count: int
    edge_count: int
    memory_geometry_points: int
    detector_score: float
    label: str
    graph_confirmed: bool


@dataclass(frozen=True)
class FragmentStitchPair:
    """One immutable compatible edge between two fragment snapshots."""

    track_ids: Tuple[int, int]
    event_frames: Tuple[int, int]
    iou: float
    containment: float
    center_distance: float
    compatibility_branch: str

    def __post_init__(self) -> None:
        if (
            len(self.track_ids) != 2
            or self.track_ids[0] >= self.track_ids[1]
        ):
            raise ValueError(
                "fragment stitch pair track_ids must be strictly increasing"
            )
        if len(self.event_frames) != 2:
            raise ValueError(
                "fragment stitch pair event_frames must contain two values"
            )
        for name, value in (
            ("pair iou", self.iou),
            ("pair containment", self.containment),
        ):
            _bounded_float(name, value)
        distance = _finite_float(
            "pair center_distance", self.center_distance
        )
        if distance < 0.0:
            raise ValueError("pair center_distance must be non-negative")
        if self.compatibility_branch not in {
            "iou",
            "containment",
            "iou+containment",
        }:
            raise ValueError("invalid fragment stitch compatibility branch")


@dataclass(frozen=True)
class FragmentStitchCandidate:
    """One immutable observer candidate produced by fragment stitching.

    ``states`` and ``event_frames`` are aligned with the sorted ``track_ids``.
    ``pair_metrics`` contains every accepted edge internal to the clique, also
    sorted by track IDs.  The representative is the deterministic partition
    anchor.  Its box is a detached read-only copy and is never written back to
    any online track.
    """

    track_ids: Tuple[int, ...]
    representative_track_id: int
    box: np.ndarray
    label: str
    states: Tuple[str, ...]
    total_views: int
    event_frames: Tuple[int, ...]
    edge_count: int
    pair_metrics: Tuple[FragmentStitchPair, ...]
    min_pair_iou: float
    min_pair_containment: float
    max_pair_center_distance: float
    max_detector_score: float
    mean_detector_score: float

    def __post_init__(self) -> None:
        if len(self.track_ids) < 2:
            raise ValueError(
                "fragment stitch candidate must contain at least two tracks"
            )
        if tuple(sorted(set(self.track_ids))) != self.track_ids:
            raise ValueError(
                "fragment stitch candidate track_ids must be unique and sorted"
            )
        if self.representative_track_id not in self.track_ids:
            raise ValueError(
                "representative_track_id must belong to the candidate"
            )
        if not isinstance(self.label, str) or not self.label:
            raise ValueError(
                "fragment stitch candidate label must be non-empty"
            )
        if (
            len(self.states) != len(self.track_ids)
            or len(self.event_frames) != len(self.track_ids)
        ):
            raise ValueError(
                "candidate states/event_frames must align with track_ids"
            )
        _strict_int("candidate total_views", self.total_views, 2)
        _strict_int("candidate edge_count", self.edge_count, 1)
        if self.edge_count != len(self.pair_metrics):
            raise ValueError(
                "candidate edge_count must equal the pair metric count"
            )
        if len(set(self.event_frames)) < 2:
            raise ValueError(
                "fragment stitch candidate needs at least two event frames"
            )
        for name, value in (
            ("candidate min_pair_iou", self.min_pair_iou),
            (
                "candidate min_pair_containment",
                self.min_pair_containment,
            ),
            ("candidate max_detector_score", self.max_detector_score),
            ("candidate mean_detector_score", self.mean_detector_score),
        ):
            _bounded_float(name, value)
        distance = _finite_float(
            "candidate max_pair_center_distance",
            self.max_pair_center_distance,
        )
        if distance < 0.0:
            raise ValueError(
                "candidate max_pair_center_distance must be non-negative"
            )
        object.__setattr__(
            self,
            "box",
            _readonly_box(self.box, name="candidate box"),
        )


def _snapshot_from_mapping(
    row: Mapping[str, object],
    *,
    index: int,
) -> _FragmentSnapshot:
    required = (
        "track_id",
        "lifecycle_state",
        "event_frame",
        "box",
        "view_count",
        "node_count",
        "edge_count",
        "memory_geometry_points",
        "mean_detector_score",
        "label",
        "graph_confirmed",
    )
    missing = [name for name in required if name not in row]
    if missing:
        raise ValueError(
            f"fragment snapshot {index} is missing key(s): "
            + ", ".join(missing)
        )

    track_id = _strict_int(
        f"fragment snapshot {index} track_id", row["track_id"], 0
    )
    event_frame = _strict_int(
        f"fragment snapshot {index} event_frame",
        row["event_frame"],
        0,
    )
    if not isinstance(row["lifecycle_state"], str):
        raise ValueError(
            f"fragment snapshot {index} lifecycle_state must be a string"
        )
    lifecycle_state = row["lifecycle_state"].strip().casefold()
    if not lifecycle_state:
        raise ValueError(
            f"fragment snapshot {index} lifecycle_state must be non-empty"
        )
    if not isinstance(row["label"], str):
        raise ValueError(
            f"fragment snapshot {index} label must be a string"
        )
    label = _normalize_label(row["label"])

    counts = {}
    for name in (
        "view_count",
        "node_count",
        "edge_count",
        "memory_geometry_points",
    ):
        counts[name] = _strict_int(
            f"fragment snapshot {index} {name}", row[name], 0
        )

    return _FragmentSnapshot(
        track_id=track_id,
        lifecycle_state=lifecycle_state,
        event_frame=event_frame,
        box=_readonly_box(
            row["box"], name=f"fragment snapshot {index} box"
        ),
        view_count=counts["view_count"],
        node_count=counts["node_count"],
        edge_count=counts["edge_count"],
        memory_geometry_points=counts["memory_geometry_points"],
        detector_score=_bounded_float(
            f"fragment snapshot {index} mean_detector_score",
            row["mean_detector_score"],
        ),
        label=label,
        graph_confirmed=_strict_bool(
            f"fragment snapshot {index} graph_confirmed",
            row["graph_confirmed"],
        ),
    )


def _aabb_pair_metrics(
    first: np.ndarray,
    second: np.ndarray,
) -> Tuple[float, float, float]:
    first_min = np.asarray(first[:3], dtype=np.float64) - 0.5 * np.asarray(
        first[3:6], dtype=np.float64
    )
    first_max = np.asarray(first[:3], dtype=np.float64) + 0.5 * np.asarray(
        first[3:6], dtype=np.float64
    )
    second_min = np.asarray(second[:3], dtype=np.float64) - 0.5 * np.asarray(
        second[3:6], dtype=np.float64
    )
    second_max = np.asarray(second[:3], dtype=np.float64) + 0.5 * np.asarray(
        second[3:6], dtype=np.float64
    )
    overlap = np.maximum(
        np.minimum(first_max, second_max) - np.maximum(first_min, second_min),
        0.0,
    )
    intersection = float(np.prod(overlap))
    first_volume = float(np.prod(np.asarray(first[3:6], dtype=np.float64)))
    second_volume = float(np.prod(np.asarray(second[3:6], dtype=np.float64)))
    union = first_volume + second_volume - intersection
    iou = 0.0 if union <= 0.0 else intersection / union
    smaller = min(first_volume, second_volume)
    containment = 0.0 if smaller <= 0.0 else intersection / smaller
    center_distance = float(
        np.linalg.norm(
            np.asarray(first[:3], dtype=np.float64)
            - np.asarray(second[:3], dtype=np.float64)
        )
    )
    return (
        float(np.clip(iou, 0.0, 1.0)),
        float(np.clip(containment, 0.0, 1.0)),
        center_distance,
    )


def _compatible_pair(
    first: _FragmentSnapshot,
    second: _FragmentSnapshot,
    config: Mapping[str, object],
) -> Optional[FragmentStitchPair]:
    if not first.label or first.label != second.label:
        return None
    if (
        abs(first.event_frame - second.event_frame)
        < int(config["minimum_event_frame_separation"])
    ):
        return None

    iou, containment, center_distance = _aabb_pair_metrics(
        first.box, second.box
    )
    iou_compatible = iou >= float(config["minimum_pair_iou"])
    containment_compatible = (
        containment >= float(config["minimum_pair_containment"])
        and center_distance <= float(config["maximum_center_distance"])
    )
    if not (iou_compatible or containment_compatible):
        return None
    if iou_compatible and containment_compatible:
        branch = "iou+containment"
    elif iou_compatible:
        branch = "iou"
    else:
        branch = "containment"

    if first.track_id < second.track_id:
        ordered = (first, second)
    else:
        ordered = (second, first)
    return FragmentStitchPair(
        track_ids=(ordered[0].track_id, ordered[1].track_id),
        event_frames=(ordered[0].event_frame, ordered[1].event_frame),
        iou=iou,
        containment=containment,
        center_distance=center_distance,
        compatibility_branch=branch,
    )


def build_fragment_stitch_candidates(
    snapshots: Sequence[Mapping[str, object]],
    config: Optional[Mapping[str, object]] = None,
) -> Tuple[FragmentStitchCandidate, ...]:
    """Build deterministic observer candidates without mutating any input.

    Input rows are validated even when the feature is disabled.  This makes a
    malformed diagnostics contract fail at its source rather than silently
    changing an ablation.  A disabled, valid invocation returns an empty
    tuple.
    """

    resolved = resolve_fragment_stitch_config(config)
    if isinstance(snapshots, (str, bytes)) or not isinstance(
        snapshots, Sequence
    ):
        raise ValueError("fragment snapshots must be a sequence of mappings")

    rows: List[_FragmentSnapshot] = []
    seen_track_ids = set()
    for index, snapshot in enumerate(snapshots):
        if not isinstance(snapshot, Mapping):
            raise ValueError(
                f"fragment snapshot {index} must be a mapping"
            )
        row = _snapshot_from_mapping(snapshot, index=index)
        if row.track_id in seen_track_ids:
            raise ValueError(
                f"duplicate fragment snapshot track_id: {row.track_id}"
            )
        seen_track_ids.add(row.track_id)
        rows.append(row)

    rows.sort(key=lambda row: (row.label, row.track_id))
    if not bool(resolved["enabled"]) or len(rows) < 2:
        return ()

    accepted_pairs: Dict[Tuple[int, int], FragmentStitchPair] = {}
    for first_index, first in enumerate(rows):
        for second_index in range(first_index + 1, len(rows)):
            second = rows[second_index]
            if second.label != first.label:
                # Rows are label-sorted, so no later row can match.
                break
            pair = _compatible_pair(first, second, resolved)
            if pair is None:
                continue
            accepted_pairs[(first_index, second_index)] = pair

    def pair_key(first_index: int, second_index: int) -> Tuple[int, int]:
        if first_index < second_index:
            return first_index, second_index
        return second_index, first_index

    def compatible_neighbors(
        anchor_index: int,
        available: Sequence[int],
    ) -> List[int]:
        return [
            index
            for index in available
            if index != anchor_index
            and pair_key(anchor_index, index) in accepted_pairs
        ]

    # Partition each normalized-label group independently.  Keeping the
    # partition itself separate from cluster gates is intentional: rejected
    # clusters cannot donate members to a second, less-conservative grouping.
    label_indices: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        if row.label:
            label_indices.setdefault(row.label, []).append(index)

    partitions: List[Tuple[int, Tuple[int, ...]]] = []
    for label in sorted(label_indices):
        unassigned = set(label_indices[label])
        while unassigned:
            available = tuple(
                sorted(unassigned, key=lambda index: rows[index].track_id)
            )
            anchor_candidates = []
            for index in available:
                neighbors = compatible_neighbors(index, available)
                if not neighbors:
                    continue
                iou_sum = math.fsum(
                    accepted_pairs[pair_key(index, neighbor)].iou
                    for neighbor in sorted(
                        neighbors, key=lambda item: rows[item].track_id
                    )
                )
                distinct_neighbor_frames = len(
                    {rows[neighbor].event_frame for neighbor in neighbors}
                )
                anchor_candidates.append(
                    (
                        (
                            -iou_sum,
                            -distinct_neighbor_frames,
                            -rows[index].view_count,
                            -rows[index].memory_geometry_points,
                            -rows[index].detector_score,
                            rows[index].track_id,
                        ),
                        index,
                        neighbors,
                    )
                )
            if not anchor_candidates:
                break

            _, anchor_index, neighbors = min(
                anchor_candidates, key=lambda item: item[0]
            )
            neighbors = sorted(
                neighbors,
                key=lambda index: (
                    -accepted_pairs[
                        pair_key(anchor_index, index)
                    ].iou,
                    -accepted_pairs[
                        pair_key(anchor_index, index)
                    ].containment,
                    accepted_pairs[
                        pair_key(anchor_index, index)
                    ].center_distance,
                    -rows[index].view_count,
                    -rows[index].memory_geometry_points,
                    -rows[index].detector_score,
                    rows[index].track_id,
                ),
            )

            clique = [anchor_index]
            for neighbor in neighbors:
                if all(
                    pair_key(neighbor, member) in accepted_pairs
                    for member in clique
                ):
                    clique.append(neighbor)
            if len(clique) < 2:
                # An anchor candidate always has a direct neighbor, so this is
                # defensive and prevents an accidental infinite loop if the
                # admission rule changes later.
                unassigned.remove(anchor_index)
                continue
            partitions.append((anchor_index, tuple(clique)))
            unassigned.difference_update(clique)

    candidates: List[FragmentStitchCandidate] = []
    for anchor_index, indices in partitions:
        if len(indices) < 2:
            continue
        members = [rows[index] for index in indices]
        if len({row.track_id for row in members}) < 2:
            continue
        if len({row.event_frame for row in members}) < 2:
            continue
        total_views = sum(row.view_count for row in members)
        if total_views < 2:
            continue
        if any(row.graph_confirmed for row in members):
            continue
        if bool(resolved["require_live_member"]) and not any(
            row.lifecycle_state in {"active", "archived"}
            for row in members
        ):
            continue

        detector_scores = [row.detector_score for row in members]
        max_detector_score = float(max(detector_scores))
        mean_detector_score = float(np.mean(detector_scores))
        if max_detector_score < float(
            resolved["minimum_max_detector_score"]
        ):
            continue
        if mean_detector_score < float(
            resolved["minimum_mean_detector_score"]
        ):
            continue

        index_set = set(indices)
        pair_metrics = tuple(
            sorted(
                (
                    pair
                    for (first_index, second_index), pair
                    in accepted_pairs.items()
                    if (
                        first_index in index_set
                        and second_index in index_set
                    )
                ),
                key=lambda pair: pair.track_ids,
            )
        )
        if not pair_metrics:
            continue
        anchor = rows[anchor_index]
        ordered_members = sorted(members, key=lambda row: row.track_id)
        candidates.append(
            FragmentStitchCandidate(
                track_ids=tuple(row.track_id for row in ordered_members),
                representative_track_id=anchor.track_id,
                box=anchor.box,
                label=anchor.label,
                states=tuple(
                    row.lifecycle_state for row in ordered_members
                ),
                total_views=total_views,
                event_frames=tuple(
                    row.event_frame for row in ordered_members
                ),
                edge_count=len(pair_metrics),
                pair_metrics=pair_metrics,
                min_pair_iou=min(pair.iou for pair in pair_metrics),
                min_pair_containment=min(
                    pair.containment for pair in pair_metrics
                ),
                max_pair_center_distance=max(
                    pair.center_distance for pair in pair_metrics
                ),
                max_detector_score=max_detector_score,
                mean_detector_score=mean_detector_score,
            )
        )

    candidates.sort(
        key=lambda candidate: (
            candidate.label,
            candidate.representative_track_id,
            candidate.track_ids,
        )
    )
    return tuple(candidates)


__all__ = [
    "DEFAULT_FRAGMENT_STITCH_CONFIG",
    "FragmentStitchCandidate",
    "FragmentStitchPair",
    "build_fragment_stitch_candidates",
    "resolve_fragment_stitch_config",
]
