#!/usr/bin/env python3
"""Materialize a terminal, create-only Group3D counterfactual prediction root.

Graw-shadow, Gclean-shadow, or PUF-Gclean-shadow records associations that *would* have merged a
newly-created native track into a past native track.  This tool follows both
identities
through every later native ``track_aliases`` event and asks what that merge
would change at the terminal output boundary.  It never changes box geometry,
scores, classes, or the order of retained rows.  For an association whose two
lineages both survive as distinct terminal rows, only the candidate row is
deleted.  Repeated requests to delete the same row are deduplicated.

The materializer is deliberately fail-closed.  Invalid shadow traces, scene
set drift, ambiguous terminal row identities, malformed aliases, and alias
cycles abort the entire materialization before the requested output root is
published.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import pickle
import re
import shutil
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


OBSERVER_SUFFIX = ".observer_tracks.json"
GRAW_SUFFIX = ".graw_shadow.json"
GCLEAN_SUFFIX = ".gclean_shadow.json"
PUF_SUFFIX = ".puf_gclean_shadow.json"
PREDICTION_SUFFIX = "_boxes.pkl"
AUDIT_FILENAME = "graw_counterfactual_audit.json"
GCLEAN_AUDIT_FILENAME = "gclean_counterfactual_audit.json"
PUF_AUDIT_FILENAME = "puf_gclean_counterfactual_audit.json"
AUDIT_SCHEMA = "boxfusion.graw_counterfactual.v1"
GCLEAN_AUDIT_SCHEMA = "boxfusion.gclean_counterfactual.v1"
PUF_AUDIT_SCHEMA = "boxfusion.puf_gclean_counterfactual.v1"
GRAW_SCHEMA = "boxfusion.graw_shadow.v1"
GCLEAN_SCHEMA = "boxfusion.gclean_shadow.v1"
PUF_SCHEMA = "boxfusion.puf_gclean_shadow.v1"
GCLEAN_FRAGMENT_SOURCE = "smov_clean"
PUF_CANDIDATE_SOURCE = "gclean_positive_overlap_top8"
SCENE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")

MAX_SCENES = 4096
MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_FRAMES = 4096
MAX_ASSOCIATIONS = 131072
MAX_PREDICTION_ROWS = 4096


class GrawCounterfactualError(ValueError):
    """Raised when a create-only counterfactual cannot be proven safe."""


@dataclass(frozen=True)
class ShadowSpec:
    kind: str
    suffix: str
    schema: str
    fragment_source: str
    candidate_source: Optional[str]
    audit_filename: str
    audit_schema: str
    label: str


_SHADOW_SPECS: Mapping[str, ShadowSpec] = {
    "graw": ShadowSpec(
        kind="graw",
        suffix=GRAW_SUFFIX,
        schema=GRAW_SCHEMA,
        fragment_source="raw_depth",
        candidate_source=None,
        audit_filename=AUDIT_FILENAME,
        audit_schema=AUDIT_SCHEMA,
        label="Graw-shadow",
    ),
    "gclean": ShadowSpec(
        kind="gclean",
        suffix=GCLEAN_SUFFIX,
        schema=GCLEAN_SCHEMA,
        fragment_source=GCLEAN_FRAGMENT_SOURCE,
        candidate_source=None,
        audit_filename=GCLEAN_AUDIT_FILENAME,
        audit_schema=GCLEAN_AUDIT_SCHEMA,
        label="Gclean-shadow",
    ),
    "puf": ShadowSpec(
        kind="puf",
        suffix=PUF_SUFFIX,
        schema=PUF_SCHEMA,
        fragment_source=GCLEAN_FRAGMENT_SOURCE,
        candidate_source=PUF_CANDIDATE_SOURCE,
        audit_filename=PUF_AUDIT_FILENAME,
        audit_schema=PUF_AUDIT_SCHEMA,
        label="PUF-Gclean-shadow",
    ),
}


@dataclass(frozen=True)
class ObserverFrame:
    frame_id: int
    proposal_ids: tuple[int, ...]
    proposal_track_ids: tuple[Optional[int], ...]
    native_status: tuple[str, ...]
    begin_past_track_ids: tuple[int, ...]
    active_track_ids: tuple[int, ...]
    aliases: Mapping[int, int]


@dataclass(frozen=True)
class ObserverScene:
    frames: tuple[ObserverFrame, ...]
    output_track_ids: tuple[int, ...]
    kept_native_indices: tuple[int, ...]
    native_row_count: int
    snapshot_frame_id: Optional[int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Integral, np.integer)
    ):
        raise GrawCounterfactualError(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise GrawCounterfactualError(f"{label} must be at least {minimum}")
    return result


def _as_finite_float(
    value: object,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    strict_minimum: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (numbers.Real, np.floating)
    ):
        raise GrawCounterfactualError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GrawCounterfactualError(f"{label} must be a finite number")
    if minimum is not None and (
        result <= minimum if strict_minimum else result < minimum
    ):
        comparison = "greater than" if strict_minimum else "at least"
        raise GrawCounterfactualError(
            f"{label} must be {comparison} {minimum}"
        )
    if maximum is not None and result > maximum:
        raise GrawCounterfactualError(f"{label} must be at most {maximum}")
    return result


def _int_tuple(
    value: object,
    label: str,
    *,
    maximum: int = MAX_ASSOCIATIONS,
    unique: bool = False,
) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise GrawCounterfactualError(f"{label} must be a JSON list")
    if len(value) > maximum:
        raise GrawCounterfactualError(f"{label} exceeds the cap of {maximum}")
    result = tuple(_as_int(item, f"{label}[{index}]") for index, item in enumerate(value))
    if unique and len(set(result)) != len(result):
        raise GrawCounterfactualError(f"{label} contains duplicate identities")
    return result


def _optional_int_tuple(
    value: object, label: str, *, maximum: int = MAX_ASSOCIATIONS
) -> tuple[Optional[int], ...]:
    if not isinstance(value, list):
        raise GrawCounterfactualError(f"{label} must be a JSON list")
    if len(value) > maximum:
        raise GrawCounterfactualError(f"{label} exceeds the cap of {maximum}")
    result: list[Optional[int]] = []
    for index, item in enumerate(value):
        result.append(None if item is None else _as_int(item, f"{label}[{index}]"))
    return tuple(result)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise GrawCounterfactualError(f"{label} must be a JSON object")
    return value


def _shadow_spec(kind: object) -> ShadowSpec:
    if not isinstance(kind, str) or kind not in _SHADOW_SPECS:
        raise GrawCounterfactualError(
            "shadow_kind must be exactly one of: " + ", ".join(_SHADOW_SPECS)
        )
    return _SHADOW_SPECS[kind]


def read_scene_list(path: str | Path) -> tuple[str, ...]:
    source = Path(path).resolve()
    if not source.is_file():
        raise GrawCounterfactualError(f"scene list is not a regular file: {source}")
    rows = source.read_text(encoding="utf-8").splitlines()
    if not rows or len(rows) > MAX_SCENES:
        raise GrawCounterfactualError(
            f"scene list must contain between 1 and {MAX_SCENES} scenes"
        )
    scenes: list[str] = []
    for line_number, raw in enumerate(rows, start=1):
        if raw != raw.strip() or not raw:
            raise GrawCounterfactualError(
                f"invalid whitespace or empty scene at line {line_number}"
            )
        if SCENE_ID_RE.fullmatch(raw) is None or raw in {".", ".."}:
            raise GrawCounterfactualError(
                f"unsafe scene ID at line {line_number}: {raw!r}"
            )
        scenes.append(raw)
    if len(set(scenes)) != len(scenes):
        raise GrawCounterfactualError("scene list contains duplicate scene IDs")
    return tuple(scenes)


def _discover_scene_set(root: Path, suffix: str) -> set[str]:
    if not root.is_dir():
        raise GrawCounterfactualError(f"input root is not a directory: {root}")
    scenes: set[str] = set()
    for path in root.glob(f"*{suffix}"):
        if not path.is_file():
            continue
        scene = path.name[: -len(suffix)]
        if SCENE_ID_RE.fullmatch(scene) is None or scene in {".", ".."}:
            raise GrawCounterfactualError(f"unsafe scene filename: {path.name!r}")
        if scene in scenes:
            raise GrawCounterfactualError(f"duplicate scene input for {scene}")
        scenes.add(scene)
    return scenes


def _require_exact_scene_set(
    root: Path, suffix: str, expected_scenes: Sequence[str], label: str
) -> None:
    discovered = _discover_scene_set(root, suffix)
    expected = set(expected_scenes)
    missing = sorted(expected - discovered)
    extra = sorted(discovered - expected)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise GrawCounterfactualError(
            f"{label} scene set mismatch: " + "; ".join(details)
        )


def _load_json(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise GrawCounterfactualError(f"missing JSON input: {path}")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise GrawCounterfactualError(f"JSON input exceeds 32 MiB: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GrawCounterfactualError(f"could not read JSON {path}: {error}") from error
    return _mapping(payload, str(path))


def _parse_aliases(value: object, label: str) -> dict[int, int]:
    if not isinstance(value, list):
        raise GrawCounterfactualError(f"{label} must be a JSON list")
    if len(value) > MAX_ASSOCIATIONS:
        raise GrawCounterfactualError(f"{label} exceeds the alias cap")
    result: dict[int, int] = {}
    for index, pair in enumerate(value):
        if not isinstance(pair, list) or len(pair) != 2:
            raise GrawCounterfactualError(f"{label}[{index}] must be [source,target]")
        source = _as_int(pair[0], f"{label}[{index}].source")
        target = _as_int(pair[1], f"{label}[{index}].target")
        if source == target:
            raise GrawCounterfactualError(f"{label}[{index}] is a self-alias")
        if source in result:
            raise GrawCounterfactualError(f"{label} has duplicate source {source}")
        result[source] = target
    return result


def _assert_alias_graph_acyclic(edges: Mapping[int, int], scene: str) -> None:
    state: dict[int, int] = {}

    def visit(identity: int) -> None:
        marker = state.get(identity, 0)
        if marker == 1:
            raise GrawCounterfactualError(
                f"{scene}: alias graph contains a cycle through track {identity}"
            )
        if marker == 2:
            return
        state[identity] = 1
        target = edges.get(identity)
        if target is not None:
            visit(target)
        state[identity] = 2

    for source in edges:
        visit(source)


def _parse_observer(path: Path, scene: str) -> ObserverScene:
    payload = _load_json(path)
    if payload.get("schema_version") != 1:
        raise GrawCounterfactualError(f"{scene}: unsupported observer schema")
    if payload.get("scene_id") != scene:
        raise GrawCounterfactualError(f"{scene}: observer scene_id mismatch")
    if payload.get("mode") != "shadow":
        raise GrawCounterfactualError(f"{scene}: observer mode is not shadow")
    if payload.get("trace_valid") is not True:
        raise GrawCounterfactualError(f"{scene}: observer trace_valid is not true")
    if payload.get("errors") != []:
        raise GrawCounterfactualError(f"{scene}: observer contains recorded errors")
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) > MAX_FRAMES:
        raise GrawCounterfactualError(f"{scene}: observer frames are invalid or unbounded")
    if _as_int(payload.get("frame_count"), f"{scene}.observer.frame_count") != len(raw_frames):
        raise GrawCounterfactualError(f"{scene}: observer frame_count mismatch")

    frames: list[ObserverFrame] = []
    previous_frame = -1
    all_proposals: set[int] = set()
    all_aliases: dict[int, int] = {}
    for frame_index, raw_frame in enumerate(raw_frames):
        frame = _mapping(raw_frame, f"{scene}.observer.frames[{frame_index}]")
        frame_id = _as_int(frame.get("frame_id"), f"{scene}.observer.frame_id")
        if frame_id <= previous_frame:
            raise GrawCounterfactualError(
                f"{scene}: observer frame IDs are not strictly increasing"
            )
        previous_frame = frame_id
        proposals = _int_tuple(
            frame.get("proposal_ids"),
            f"{scene}.frame[{frame_id}].proposal_ids",
            unique=True,
        )
        duplicate_proposals = all_proposals.intersection(proposals)
        if duplicate_proposals:
            raise GrawCounterfactualError(
                f"{scene}: proposal identities are reused across frames"
            )
        all_proposals.update(proposals)
        proposal_tracks = _optional_int_tuple(
            frame.get("proposal_track_ids"),
            f"{scene}.frame[{frame_id}].proposal_track_ids",
        )
        status_value = frame.get("native_status")
        if not isinstance(status_value, list) or any(
            not isinstance(item, str) for item in status_value
        ):
            raise GrawCounterfactualError(
                f"{scene}.frame[{frame_id}].native_status must be strings"
            )
        statuses = tuple(status_value)
        if len(proposals) != len(proposal_tracks) or len(proposals) != len(statuses):
            raise GrawCounterfactualError(
                f"{scene}.frame[{frame_id}] proposal-aligned fields differ in length"
            )
        begin_past = _int_tuple(
            frame.get("begin_past_track_ids"),
            f"{scene}.frame[{frame_id}].begin_past_track_ids",
            maximum=1024,
            unique=True,
        )
        active = _int_tuple(
            frame.get("active_track_ids"),
            f"{scene}.frame[{frame_id}].active_track_ids",
            maximum=1024,
            unique=True,
        )
        aliases = _parse_aliases(
            frame.get("track_aliases"), f"{scene}.frame[{frame_id}].track_aliases"
        )
        if any(target not in set(active) for target in aliases.values()):
            raise GrawCounterfactualError(
                f"{scene}.frame[{frame_id}] alias target is not an active track"
            )
        for source, target in aliases.items():
            if source in all_aliases:
                raise GrawCounterfactualError(
                    f"{scene}: alias source {source} is assigned in multiple frames"
                )
            all_aliases[source] = target
        frames.append(
            ObserverFrame(
                frame_id=frame_id,
                proposal_ids=proposals,
                proposal_track_ids=proposal_tracks,
                native_status=statuses,
                begin_past_track_ids=begin_past,
                active_track_ids=active,
                aliases=aliases,
            )
        )
    _assert_alias_graph_acyclic(all_aliases, scene)

    terminal = _mapping(payload.get("terminal"), f"{scene}.observer.terminal")
    native_row_count = _as_int(
        terminal.get("native_row_count"), f"{scene}.terminal.native_row_count"
    )
    if native_row_count > 1024:
        raise GrawCounterfactualError(f"{scene}: terminal native row cap exceeded")
    kept = _int_tuple(
        terminal.get("kept_native_indices"),
        f"{scene}.terminal.kept_native_indices",
        maximum=1024,
        unique=True,
    )
    output_ids = _int_tuple(
        terminal.get("output_track_ids"),
        f"{scene}.terminal.output_track_ids",
        maximum=1024,
        unique=True,
    )
    if len(kept) != len(output_ids):
        raise GrawCounterfactualError(f"{scene}: terminal row mapping length mismatch")
    if any(index >= native_row_count for index in kept):
        raise GrawCounterfactualError(f"{scene}: terminal kept index is out of range")
    snapshot_raw = terminal.get("snapshot_frame_id")
    snapshot = (
        None
        if snapshot_raw is None
        else _as_int(snapshot_raw, f"{scene}.terminal.snapshot_frame_id")
    )
    if frames:
        frame_ids = {frame.frame_id for frame in frames}
        if snapshot is not None and snapshot not in frame_ids:
            raise GrawCounterfactualError(
                f"{scene}: terminal snapshot frame is absent from observer frames"
            )
        final_active = frames[-1].active_track_ids
        if native_row_count != len(final_active):
            raise GrawCounterfactualError(
                f"{scene}: terminal native_row_count disagrees with final active rows"
            )
        expected_output_ids = tuple(final_active[index] for index in kept)
        if output_ids != expected_output_ids:
            raise GrawCounterfactualError(
                f"{scene}: terminal output_track_ids disagree with kept native rows"
            )
    elif output_ids or native_row_count or snapshot is not None:
        raise GrawCounterfactualError(
            f"{scene}: nonempty terminal mapping has no observer frames"
        )
    return ObserverScene(
        frames=tuple(frames),
        output_track_ids=output_ids,
        kept_native_indices=kept,
        native_row_count=native_row_count,
        snapshot_frame_id=snapshot,
    )


def _load_prediction(path: Path) -> tuple[object, object, tuple[object, ...]]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise GrawCounterfactualError(f"could not load prediction {path}: {error}") from error
    if type(payload) not in (list, tuple) or len(payload) != 1:
        raise GrawCounterfactualError(
            f"prediction {path} must be a one-element built-in list or tuple"
        )
    rows = payload[0]
    if type(rows) not in (list, tuple) or len(rows) > MAX_PREDICTION_ROWS:
        raise GrawCounterfactualError(
            f"prediction {path} rows must be a bounded built-in list or tuple"
        )
    for row_index, row in enumerate(rows):
        if type(row) not in (list, tuple) or len(row) != 3:
            raise GrawCounterfactualError(
                f"prediction {path} row {row_index} must be (class,corners,score)"
            )
        class_id, corners, score = row
        if isinstance(class_id, (bool, np.bool_)) or not isinstance(
            class_id, numbers.Integral
        ):
            raise GrawCounterfactualError(
                f"prediction {path} row {row_index} class is not integral"
            )
        if isinstance(score, (bool, np.bool_)) or not isinstance(score, numbers.Real):
            raise GrawCounterfactualError(
                f"prediction {path} row {row_index} score is not numeric"
            )
        if not math.isfinite(float(score)):
            raise GrawCounterfactualError(
                f"prediction {path} row {row_index} score is non-finite"
            )
        array = np.asarray(corners)
        if array.shape != (8, 3) or not np.issubdtype(array.dtype, np.number):
            raise GrawCounterfactualError(
                f"prediction {path} row {row_index} corners must be numeric (8,3)"
            )
        if not np.isfinite(array).all():
            raise GrawCounterfactualError(
                f"prediction {path} row {row_index} corners are non-finite"
            )
    return payload, rows, tuple(rows)


def _rebuild_prediction(payload: object, rows: object, retained: Sequence[object]) -> object:
    retained_rows = list(retained) if type(rows) is list else tuple(retained)
    return [retained_rows] if type(payload) is list else (retained_rows,)


def _row_pickle_sha256(rows: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        encoded = pickle.dumps(row, protocol=pickle.HIGHEST_PROTOCOL)
        digest.update(len(encoded).to_bytes(8, "little", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _resolve_later_aliases(
    identity: int,
    *,
    after_frame_id: int,
    observer_frames: Sequence[ObserverFrame],
    scene: str,
) -> tuple[int, list[dict[str, int]]]:
    current = identity
    seen = {current}
    path: list[dict[str, int]] = []
    for frame in observer_frames:
        if frame.frame_id <= after_frame_id:
            continue
        # Aliases are expected to be compressed, but closure makes malformed
        # chained maps fail deterministically instead of silently truncating.
        while current in frame.aliases:
            target = frame.aliases[current]
            if target in seen:
                raise GrawCounterfactualError(
                    f"{scene}: alias cycle while resolving terminal track {identity}"
                )
            path.append(
                {"frame_id": frame.frame_id, "source": current, "target": target}
            )
            seen.add(target)
            current = target
    return current, path


def _parse_graw_associations(
    path: Path,
    scene: str,
    observer: ObserverScene,
    *,
    shadow_spec: ShadowSpec,
) -> list[dict[str, object]]:
    payload = _load_json(path)
    if payload.get("schema") != shadow_spec.schema:
        raise GrawCounterfactualError(
            f"{scene}: unsupported {shadow_spec.label} schema"
        )
    if payload.get("scene_id") != scene:
        raise GrawCounterfactualError(
            f"{scene}: {shadow_spec.label} scene_id mismatch"
        )
    if payload.get("trace_valid") is not True:
        raise GrawCounterfactualError(
            f"{scene}: {shadow_spec.label} trace_valid is not true"
        )
    if shadow_spec.kind in {"gclean", "puf"}:
        if payload.get("mode") != "shadow":
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} mode is not shadow"
            )
        if payload.get("fragment_source") != shadow_spec.fragment_source:
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} fragment_source is not "
                f"{shadow_spec.fragment_source}"
            )
        if shadow_spec.kind == "puf":
            if payload.get("candidate_source") != shadow_spec.candidate_source:
                raise GrawCounterfactualError(
                    f"{scene}: PUF-Gclean candidate_source mismatch"
                )
            if payload.get("birth_enabled") is not False:
                raise GrawCounterfactualError(
                    f"{scene}: PUF-Gclean birth_enabled must be false"
                )
        summary = _mapping(
            payload.get("summary"), f"{scene}.{shadow_spec.kind}.summary"
        )
        if summary.get("schema") != shadow_spec.schema:
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} summary schema mismatch"
            )
        if summary.get("mode") != "shadow":
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} summary mode is not shadow"
            )
        if summary.get("fragment_source") != shadow_spec.fragment_source:
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} summary fragment_source mismatch"
            )
        if shadow_spec.kind == "puf":
            if summary.get("candidate_source") != shadow_spec.candidate_source:
                raise GrawCounterfactualError(
                    f"{scene}: PUF-Gclean summary candidate_source mismatch"
                )
            if summary.get("birth_enabled") is not False:
                raise GrawCounterfactualError(
                    f"{scene}: PUF-Gclean summary birth_enabled must be false"
                )
            if summary.get("fail_open") is not False:
                raise GrawCounterfactualError(
                    f"{scene}: PUF-Gclean summary fail_open must be false"
                )
        if summary.get("pending") is not False:
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} summary is not terminal "
                "(pending must be false)"
            )
    raw_frames = payload.get("frames")
    if not isinstance(raw_frames, list) or len(raw_frames) > MAX_FRAMES:
        raise GrawCounterfactualError(
            f"{scene}: {shadow_spec.label} frames are invalid or unbounded"
        )
    trace_label = shadow_spec.kind
    if _as_int(
        payload.get("frame_count"), f"{scene}.{trace_label}.frame_count"
    ) != len(raw_frames):
        raise GrawCounterfactualError(
            f"{scene}: {shadow_spec.label} frame_count mismatch"
        )
    observer_by_frame = {frame.frame_id: frame for frame in observer.frames}
    previous_frame = -1
    association_count = 0
    seen_associations: set[tuple[int, int]] = set()
    records: list[dict[str, object]] = []
    for frame_index, raw_frame in enumerate(raw_frames):
        frame = _mapping(raw_frame, f"{scene}.{trace_label}.frames[{frame_index}]")
        frame_id = _as_int(
            frame.get("frame_id"), f"{scene}.{trace_label}.frame_id"
        )
        if frame_id <= previous_frame:
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} frame IDs are not strictly increasing"
            )
        previous_frame = frame_id
        if frame_id not in observer_by_frame:
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} frame {frame_id} has no observer "
                "identity frame"
            )
        if shadow_spec.kind in {"gclean", "puf"}:
            if frame.get("mode") != "shadow":
                raise GrawCounterfactualError(
                    f"{scene}: {shadow_spec.label} frame {frame_id} mode is not shadow"
                )
            if frame.get("fragment_source") != shadow_spec.fragment_source:
                raise GrawCounterfactualError(
                    f"{scene}: {shadow_spec.label} frame {frame_id} "
                    "fragment_source mismatch"
                )
            if shadow_spec.kind == "puf":
                if frame.get("schema") != shadow_spec.schema:
                    raise GrawCounterfactualError(
                        f"{scene}: PUF-Gclean frame {frame_id} schema mismatch"
                    )
                if frame.get("candidate_source") != shadow_spec.candidate_source:
                    raise GrawCounterfactualError(
                        f"{scene}: PUF-Gclean frame {frame_id} "
                        "candidate_source mismatch"
                    )
                if frame.get("birth_enabled") is not False:
                    raise GrawCounterfactualError(
                        f"{scene}: PUF-Gclean frame {frame_id} "
                        "birth_enabled must be false"
                    )
                if frame.get("fail_open") is not False:
                    raise GrawCounterfactualError(
                        f"{scene}: PUF-Gclean frame {frame_id} "
                        "fail_open must be false"
                    )
        candidate_proposals = _int_tuple(
            frame.get("candidate_proposal_ids"),
            f"{scene}.{trace_label}.frame[{frame_id}].candidate_proposal_ids",
            unique=True,
        )
        candidate_tracks = _int_tuple(
            frame.get("candidate_native_track_ids"),
            f"{scene}.{trace_label}.frame[{frame_id}].candidate_native_track_ids",
            unique=True,
        )
        if len(candidate_proposals) != len(candidate_tracks):
            raise GrawCounterfactualError(
                f"{scene}.{trace_label}.frame[{frame_id}] candidate fields differ "
                "in length"
            )
        candidate_map = dict(zip(candidate_proposals, candidate_tracks))
        raw_associations = frame.get("associations")
        if not isinstance(raw_associations, list):
            raise GrawCounterfactualError(
                f"{scene}.{trace_label}.frame[{frame_id}].associations must be a list"
            )
        association_count += len(raw_associations)
        if association_count > MAX_ASSOCIATIONS:
            raise GrawCounterfactualError(
                f"{scene}: {shadow_spec.label} association cap exceeded"
            )
        observed = observer_by_frame[frame_id]
        proposal_index = {
            proposal_id: index for index, proposal_id in enumerate(observed.proposal_ids)
        }
        puf_target_ids: set[int] = set()
        for association_index, raw_association in enumerate(raw_associations):
            association = _mapping(
                raw_association,
                f"{scene}.{trace_label}.frame[{frame_id}].associations"
                f"[{association_index}]",
            )
            proposal_id = _as_int(association.get("proposal_id"), "proposal_id")
            candidate_id = _as_int(
                association.get("native_track_id"), "native_track_id"
            )
            target_id = _as_int(association.get("past_track_id"), "past_track_id")
            probability_fields: dict[str, object] = {}
            if shadow_spec.kind == "puf":
                beta_track = _as_finite_float(
                    association.get("beta_track"),
                    "beta_track",
                    minimum=0.0,
                    maximum=1.0,
                )
                beta_null = _as_finite_float(
                    association.get("beta_null"),
                    "beta_null",
                    minimum=0.0,
                    maximum=1.0,
                )
                margin = _as_finite_float(
                    association.get("margin"),
                    "margin",
                    minimum=0.0,
                    strict_minimum=True,
                )
                if beta_track <= beta_null:
                    raise GrawCounterfactualError(
                        f"{scene}: PUF-Gclean beta_track must exceed beta_null"
                    )
                if association.get("birth_enabled") is not False:
                    raise GrawCounterfactualError(
                        f"{scene}: PUF-Gclean association birth_enabled must be false"
                    )
                if target_id in puf_target_ids:
                    raise GrawCounterfactualError(
                        f"{scene}: PUF-Gclean active-safe associations contain "
                        f"a duplicate past_track_id {target_id}"
                    )
                puf_target_ids.add(target_id)
                probability_fields = {
                    "beta_track": beta_track,
                    "beta_null": beta_null,
                    "margin": margin,
                    "birth_enabled": False,
                }
            key = (frame_id, proposal_id)
            if key in seen_associations:
                raise GrawCounterfactualError(
                    f"{scene}: duplicate shadow association for frame/proposal {key}"
                )
            seen_associations.add(key)
            if candidate_id == target_id:
                raise GrawCounterfactualError(
                    f"{scene}: shadow association already has identical lineages"
                )
            if candidate_map.get(proposal_id) != candidate_id:
                raise GrawCounterfactualError(
                    f"{scene}: association is absent from the "
                    f"{shadow_spec.label} candidate mapping"
                )
            index = proposal_index.get(proposal_id)
            if index is None:
                raise GrawCounterfactualError(
                    f"{scene}: associated proposal is absent from observer trace"
                )
            if observed.proposal_track_ids[index] != candidate_id:
                raise GrawCounterfactualError(
                    f"{scene}: candidate native identity disagrees with observer trace"
                )
            if observed.native_status[index] != "unmatched_retained":
                raise GrawCounterfactualError(
                    f"{scene}: {shadow_spec.label} candidate is not "
                    "native-unmatched-retained"
                )
            if target_id not in set(observed.begin_past_track_ids):
                raise GrawCounterfactualError(
                    f"{scene}: {shadow_spec.label} target is not a "
                    "begin-frame-past identity"
                )
            if candidate_id not in set(observed.active_track_ids) or target_id not in set(
                observed.active_track_ids
            ):
                raise GrawCounterfactualError(
                    f"{scene}: association endpoint is not active after native association"
                )
            records.append(
                {
                    "frame_id": frame_id,
                    "proposal_id": proposal_id,
                    "candidate_native_track_id": candidate_id,
                    "past_native_track_id": target_id,
                    **probability_fields,
                }
            )
    return records


def _analyze_scene(
    *,
    scene: str,
    observer_path: Path,
    graw_path: Path,
    prediction_path: Path,
    shadow_spec: ShadowSpec,
) -> tuple[dict[str, object], object]:
    observer = _parse_observer(observer_path, scene)
    raw_associations = _parse_graw_associations(
        graw_path, scene, observer, shadow_spec=shadow_spec
    )
    payload, rows_container, rows = _load_prediction(prediction_path)
    if len(rows) != len(observer.output_track_ids):
        raise GrawCounterfactualError(
            f"{scene}: prediction rows ({len(rows)}) do not match terminal "
            f"output_track_ids ({len(observer.output_track_ids)})"
        )
    output_row_by_track = {
        track_id: index for index, track_id in enumerate(observer.output_track_ids)
    }
    final_active = set(observer.frames[-1].active_track_ids) if observer.frames else set()
    classified: list[dict[str, object]] = []
    deletion_rows: set[int] = set()
    for association in raw_associations:
        frame_id = int(association["frame_id"])
        candidate_id = int(association["candidate_native_track_id"])
        target_id = int(association["past_native_track_id"])
        terminal_candidate, candidate_path = _resolve_later_aliases(
            candidate_id,
            after_frame_id=frame_id,
            observer_frames=observer.frames,
            scene=scene,
        )
        terminal_target, target_path = _resolve_later_aliases(
            target_id,
            after_frame_id=frame_id,
            observer_frames=observer.frames,
            scene=scene,
        )
        candidate_row = output_row_by_track.get(terminal_candidate)
        target_row = output_row_by_track.get(terminal_target)
        if terminal_candidate == terminal_target:
            classification = "later-native-same"
        elif candidate_row is None:
            classification = "candidate-dropped"
        elif target_row is None:
            classification = "target-dropped"
        else:
            classification = "both-survive-distinct"
            deletion_rows.add(candidate_row)
        classified.append(
            {
                **association,
                "terminal_candidate_track_id": terminal_candidate,
                "terminal_target_track_id": terminal_target,
                "candidate_alias_path": candidate_path,
                "target_alias_path": target_path,
                "candidate_output_row": candidate_row,
                "target_output_row": target_row,
                "candidate_terminal_state": (
                    "output"
                    if candidate_row is not None
                    else "postprocess-dropped"
                    if terminal_candidate in final_active
                    else "native-dropped"
                ),
                "target_terminal_state": (
                    "output"
                    if target_row is not None
                    else "postprocess-dropped"
                    if terminal_target in final_active
                    else "native-dropped"
                ),
                "classification": classification,
                "deletes_candidate_row": classification == "both-survive-distinct",
            }
        )

    deleted = tuple(sorted(deletion_rows))
    retained_indices = tuple(index for index in range(len(rows)) if index not in deletion_rows)
    retained_rows = tuple(rows[index] for index in retained_indices)
    output_payload = _rebuild_prediction(payload, rows_container, retained_rows)
    counts = Counter(str(item["classification"]) for item in classified)
    scene_audit: dict[str, object] = {
        "scene_id": scene,
        "shadow_kind": shadow_spec.kind,
        "fragment_source": shadow_spec.fragment_source,
        "input_prediction_sha256": _sha256(prediction_path),
        "observer_sha256": _sha256(observer_path),
        "shadow_diagnostic_sha256": _sha256(graw_path),
        "terminal_snapshot_frame_id": observer.snapshot_frame_id,
        "association_count": len(classified),
        "classification_counts": {
            label: counts.get(label, 0)
            for label in (
                "later-native-same",
                "candidate-dropped",
                "target-dropped",
                "both-survive-distinct",
            )
        },
        "input_row_count": len(rows),
        "output_row_count": len(retained_rows),
        "deleted_row_count": len(deleted),
        "deleted_native_rows": list(deleted),
        "deleted_terminal_track_ids": [observer.output_track_ids[index] for index in deleted],
        "retained_native_rows": list(retained_indices),
        "retained_row_payload_sha256": _row_pickle_sha256(retained_rows),
        "associations": classified,
    }
    if shadow_spec.candidate_source is not None:
        scene_audit["candidate_source"] = shadow_spec.candidate_source
        scene_audit["birth_enabled"] = False
    if shadow_spec.kind == "graw":
        # Compatibility for consumers of the original Graw audit.
        scene_audit["graw_shadow_sha256"] = scene_audit[
            "shadow_diagnostic_sha256"
        ]
    return scene_audit, output_payload


def _write_pickle(path: Path, payload: object) -> None:
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())


def materialize_graw_counterfactual(
    *,
    observer_root: str | Path,
    graw_root: str | Path,
    native_prediction_root: str | Path,
    output_prediction_root: str | Path,
    scene_list: str | Path,
    shadow_kind: str = "graw",
) -> dict[str, object]:
    """Validate all inputs and atomically publish a create-only root."""

    shadow_spec = _shadow_spec(shadow_kind)
    scene_path = Path(scene_list).resolve()
    scenes = read_scene_list(scene_path)
    observer_dir = Path(observer_root).resolve()
    graw_dir = Path(graw_root).resolve()
    native_dir = Path(native_prediction_root).resolve()
    output_dir = Path(output_prediction_root).resolve()
    if output_dir.exists():
        raise GrawCounterfactualError(
            f"output prediction root already exists; refusing to overwrite: {output_dir}"
        )
    _require_exact_scene_set(observer_dir, OBSERVER_SUFFIX, scenes, "observer")
    _require_exact_scene_set(
        graw_dir, shadow_spec.suffix, scenes, shadow_spec.label
    )
    _require_exact_scene_set(native_dir, PREDICTION_SUFFIX, scenes, "native prediction")

    analyzed: list[tuple[dict[str, object], object]] = []
    for scene in scenes:
        analyzed.append(
            _analyze_scene(
                scene=scene,
                observer_path=observer_dir / f"{scene}{OBSERVER_SUFFIX}",
                graw_path=graw_dir / f"{scene}{shadow_spec.suffix}",
                prediction_path=native_dir / f"{scene}{PREDICTION_SUFFIX}",
                shadow_spec=shadow_spec,
            )
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    published = False
    try:
        scene_audits: list[dict[str, object]] = []
        for scene, (scene_audit, output_payload) in zip(scenes, analyzed):
            output_path = staging / f"{scene}{PREDICTION_SUFFIX}"
            _write_pickle(output_path, output_payload)
            # Reloading validates that serialization preserved every retained
            # row payload exactly, not merely its numeric projection.
            _, _, reloaded_rows = _load_prediction(output_path)
            retained_digest = _row_pickle_sha256(reloaded_rows)
            if retained_digest != scene_audit["retained_row_payload_sha256"]:
                raise GrawCounterfactualError(
                    f"{scene}: retained row payload changed during serialization"
                )
            scene_audit["output_prediction_sha256"] = _sha256(output_path)
            scene_audits.append(scene_audit)

        category_totals = Counter()
        for scene_audit in scene_audits:
            category_totals.update(scene_audit["classification_counts"])
        audit: dict[str, object] = {
            "schema": shadow_spec.audit_schema,
            "shadow_kind": shadow_spec.kind,
            "fragment_source": shadow_spec.fragment_source,
            "contract": {
                "mode": "create-only-terminal-counterfactual",
                "association_resolution": "all strictly later native track_aliases",
                "only_mutation": "delete candidate row for both-survive-distinct",
                "retained_row_order": "unchanged",
                "retained_class_score_geometry": "exact pickle payload identity",
                "duplicate_candidate_deletions": "deduplicated",
                "failure_policy": "fail-closed before output-root publication",
            },
            "scene_list": {
                "path": str(scene_path),
                "sha256": _sha256(scene_path),
                "count": len(scenes),
                "scenes": list(scenes),
            },
            "inputs": {
                "observer_root": str(observer_dir),
                "shadow_root": str(graw_dir),
                "shadow_kind": shadow_spec.kind,
                "fragment_source": shadow_spec.fragment_source,
                "native_prediction_root": str(native_dir),
            },
            "output_prediction_root": str(output_dir),
            "totals": {
                "scene_count": len(scenes),
                "association_count": sum(
                    int(item["association_count"]) for item in scene_audits
                ),
                "classification_counts": {
                    label: category_totals.get(label, 0)
                    for label in (
                        "later-native-same",
                        "candidate-dropped",
                        "target-dropped",
                        "both-survive-distinct",
                    )
                },
                "input_row_count": sum(
                    int(item["input_row_count"]) for item in scene_audits
                ),
                "output_row_count": sum(
                    int(item["output_row_count"]) for item in scene_audits
                ),
                "deleted_row_count": sum(
                    int(item["deleted_row_count"]) for item in scene_audits
                ),
            },
            "scenes": scene_audits,
            "validation": {
                "passed": True,
                "trace_valid_required": True,
                "scene_sets_exact": True,
                "terminal_row_mapping_unambiguous": True,
                "alias_graph_acyclic": True,
            },
        }
        if shadow_spec.candidate_source is not None:
            audit["candidate_source"] = shadow_spec.candidate_source
            audit["birth_enabled"] = False
            audit["inputs"]["candidate_source"] = shadow_spec.candidate_source
            audit["inputs"]["birth_enabled"] = False
        if shadow_spec.kind == "graw":
            # Preserve the legacy path key for existing Graw audit consumers.
            audit["inputs"]["graw_root"] = str(graw_dir)
        encoded = json.dumps(
            audit, indent=2, sort_keys=True, allow_nan=False
        ).encode("utf-8")
        if len(encoded) > MAX_JSON_BYTES:
            raise GrawCounterfactualError("counterfactual audit exceeds 32 MiB")
        audit_path = staging / shadow_spec.audit_filename
        with audit_path.open("wb") as handle:
            handle.write(encoded)
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, output_dir)
        published = True
        return audit
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observer-root", required=True)
    parser.add_argument(
        "--graw-root",
        "--shadow-root",
        dest="graw_root",
        required=True,
        help="Graw/Gclean/PUF-Gclean shadow diagnostic root",
    )
    parser.add_argument(
        "--shadow-kind",
        choices=tuple(_SHADOW_SPECS),
        default="graw",
        help="strict diagnostic format to materialize (default: graw)",
    )
    parser.add_argument("--native-prediction-root", required=True)
    parser.add_argument("--output-prediction-root", required=True)
    parser.add_argument("--scene-list", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    audit = materialize_graw_counterfactual(
        observer_root=args.observer_root,
        graw_root=args.graw_root,
        native_prediction_root=args.native_prediction_root,
        output_prediction_root=args.output_prediction_root,
        scene_list=args.scene_list,
        shadow_kind=args.shadow_kind,
    )
    shadow_spec = _shadow_spec(args.shadow_kind)
    print(
        json.dumps(
            {
                "audit": str(
                    Path(args.output_prediction_root).resolve()
                    / shadow_spec.audit_filename
                ),
                "totals": audit["totals"],
                "validation": audit["validation"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
