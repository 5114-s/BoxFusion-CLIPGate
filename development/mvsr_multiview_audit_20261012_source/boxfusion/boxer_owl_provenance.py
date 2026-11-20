"""Strict no-GT provenance from Boxer 3D rows to OWL 2D rows.

The frozen Boxer producer creates one 3D prediction for every OWL detection,
then applies only the 3D-confidence mask before writing ``boxer_3dbbs.csv``.
That mask preserves order.  Consequently, within one ``time_ns`` group, the
Boxer ``(normalized name, sem_id)`` sequence must be a one-to-one subsequence
of the corresponding OWL sequence.

This module deliberately refuses to guess when repeated semantic keys admit
more than one order-preserving embedding.  It reads only the two CSV files
named by the production contract; it has no GT/oracle input or fallback.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA = "boxfusion.boxer_owl_provenance.v1"

BOXER_COLUMNS = (
    "time_ns",
    "tx_world_object",
    "ty_world_object",
    "tz_world_object",
    "qw_world_object",
    "qx_world_object",
    "qy_world_object",
    "qz_world_object",
    "scale_x",
    "scale_y",
    "scale_z",
    "name",
    "instance",
    "sem_id",
    "prob",
)

OWL_COLUMNS = (
    "time_ns",
    "frame_id",
    "sensor",
    "device",
    "img_width",
    "img_height",
    "x1",
    "y1",
    "x2",
    "y2",
    "name",
    "instance",
    "sem_id",
    "prob",
)

_NONNEGATIVE_INTEGER = re.compile(r"^(?:0|[1-9][0-9]*)$")


class ProvenanceError(RuntimeError):
    """Raised when exact order-preserving provenance cannot be established."""


@dataclass(frozen=True)
class _SemanticRow:
    global_row_index: int
    csv_line_number: int
    time_ns: int
    normalized_name: str
    sem_id: int

    @property
    def key(self) -> tuple[str, int]:
        return self.normalized_name, self.sem_id


@dataclass(frozen=True)
class _ParsedCsv:
    path: Path
    sha256: str
    byte_count: int
    rows: tuple[_SemanticRow, ...]
    groups: Mapping[int, tuple[_SemanticRow, ...]]


def normalize_name(value: str) -> str:
    """Apply only the producer-compatible name normalization.

    Boxer lowercases and strips decoded text before writing it.  We mirror
    those two operations and intentionally do not merge underscores, hyphens,
    punctuation, or internal whitespace.
    """

    if not isinstance(value, str) or "\x00" in value:
        raise ProvenanceError("semantic name must be NUL-free text")
    normalized = value.strip().lower()
    if not normalized:
        raise ProvenanceError("semantic name is empty after normalization")
    return normalized


def _parse_nonnegative_integer(
    value: str, *, field: str, path: Path, line_number: int
) -> int:
    if not isinstance(value, str) or _NONNEGATIVE_INTEGER.fullmatch(value) is None:
        raise ProvenanceError(
            f"invalid non-negative integer {field} at {path}:{line_number}: {value!r}"
        )
    return int(value)


def _resolve_contract_path(path: str | Path, expected_name: str) -> Path:
    candidate = Path(path).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise ProvenanceError(f"missing provenance input: {candidate}") from error
    if not resolved.is_file():
        raise ProvenanceError(f"provenance input is not a file: {resolved}")
    if resolved.name != expected_name:
        raise ProvenanceError(
            f"expected production input named {expected_name}, received {resolved.name}"
        )
    return resolved


def _group_rows(
    rows: Iterable[_SemanticRow], *, path: Path
) -> dict[int, tuple[_SemanticRow, ...]]:
    mutable: dict[int, list[_SemanticRow]] = {}
    closed: set[int] = set()
    previous: int | None = None
    for row in rows:
        if previous is None:
            mutable[row.time_ns] = []
        elif row.time_ns != previous:
            closed.add(previous)
            if row.time_ns in closed:
                raise ProvenanceError(
                    f"non-contiguous time_ns group {row.time_ns} in {path}"
                )
            if row.time_ns <= previous:
                raise ProvenanceError(
                    f"time_ns groups are not strictly increasing in {path}: "
                    f"{previous} then {row.time_ns}"
                )
            mutable[row.time_ns] = []
        mutable[row.time_ns].append(row)
        previous = row.time_ns
    return {time_ns: tuple(group) for time_ns, group in mutable.items()}


def _read_semantic_csv(
    path: str | Path,
    *,
    expected_name: str,
    expected_columns: tuple[str, ...],
) -> _ParsedCsv:
    resolved = _resolve_contract_path(path, expected_name)
    try:
        payload = resolved.read_bytes()
        text = payload.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ProvenanceError(f"could not read UTF-8 CSV: {resolved}") from error
    digest = hashlib.sha256(payload).hexdigest()
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        header = next(reader)
    except StopIteration as error:
        raise ProvenanceError(f"empty CSV: {resolved}") from error
    if tuple(header) != expected_columns:
        raise ProvenanceError(
            f"unexpected {expected_name} header: {tuple(header)!r}"
        )

    rows: list[_SemanticRow] = []
    for global_row_index, values in enumerate(reader):
        line_number = reader.line_num
        if len(values) != len(expected_columns):
            raise ProvenanceError(
                f"CSV column count mismatch at {resolved}:{line_number}: "
                f"{len(values)} != {len(expected_columns)}"
            )
        row = dict(zip(expected_columns, values))
        rows.append(
            _SemanticRow(
                global_row_index=global_row_index,
                csv_line_number=line_number,
                time_ns=_parse_nonnegative_integer(
                    row["time_ns"],
                    field="time_ns",
                    path=resolved,
                    line_number=line_number,
                ),
                normalized_name=normalize_name(row["name"]),
                sem_id=_parse_nonnegative_integer(
                    row["sem_id"],
                    field="sem_id",
                    path=resolved,
                    line_number=line_number,
                ),
            )
        )
    groups = _group_rows(rows, path=resolved)
    return _ParsedCsv(
        path=resolved,
        sha256=digest,
        byte_count=len(payload),
        rows=tuple(rows),
        groups=groups,
    )


def _unique_subsequence_indices(
    boxer_rows: tuple[_SemanticRow, ...],
    owl_rows: tuple[_SemanticRow, ...],
    *,
    time_ns: int,
) -> tuple[int, ...]:
    """Return the unique per-frame OWL indices, or fail closed.

    Greedy left-to-right matching gives the component-wise earliest embedding;
    greedy right-to-left matching gives the latest.  An embedding is unique iff
    both vectors are equal.
    """

    earliest: list[int] = []
    cursor = 0
    for boxer_row in boxer_rows:
        while cursor < len(owl_rows) and owl_rows[cursor].key != boxer_row.key:
            cursor += 1
        if cursor == len(owl_rows):
            raise ProvenanceError(
                f"Boxer rows are not an OWL semantic subsequence at time_ns={time_ns}; "
                f"no ordered match for Boxer global row {boxer_row.global_row_index} "
                f"key={boxer_row.key!r}"
            )
        earliest.append(cursor)
        cursor += 1

    latest_reversed: list[int] = []
    cursor = len(owl_rows) - 1
    for boxer_row in reversed(boxer_rows):
        while cursor >= 0 and owl_rows[cursor].key != boxer_row.key:
            cursor -= 1
        if cursor < 0:
            raise ProvenanceError(
                f"internal subsequence inconsistency at time_ns={time_ns}"
            )
        latest_reversed.append(cursor)
        cursor -= 1
    latest = list(reversed(latest_reversed))
    if earliest != latest:
        raise ProvenanceError(
            f"ambiguous Boxer-to-OWL ordered provenance at time_ns={time_ns}: "
            f"earliest={earliest}, latest={latest}"
        )
    return tuple(earliest)


def build_boxer_owl_provenance(
    boxer_csv: str | Path,
    owl_csv: str | Path,
) -> dict[str, Any]:
    """Build deterministic, JSON-serializable no-GT row provenance.

    Row indices are global, zero-based data-row indices; CSV headers are not
    counted.  Any missing, reordered, duplicated-ambiguously, or schema-invalid
    relation raises :class:`ProvenanceError`.
    """

    boxer = _read_semantic_csv(
        boxer_csv,
        expected_name="boxer_3dbbs.csv",
        expected_columns=BOXER_COLUMNS,
    )
    owl = _read_semantic_csv(
        owl_csv,
        expected_name="owl_2dbbs.csv",
        expected_columns=OWL_COLUMNS,
    )
    if boxer.path.parent != owl.path.parent:
        raise ProvenanceError(
            "Boxer and OWL CSVs must resolve to the same scene directory"
        )

    missing_times = [time_ns for time_ns in boxer.groups if time_ns not in owl.groups]
    if missing_times:
        raise ProvenanceError(
            f"Boxer time_ns groups missing from OWL CSV: {missing_times}"
        )

    mappings: list[dict[str, Any]] = []
    matched_owl_indices: set[int] = set()
    per_time_matched: dict[int, int] = {}
    for time_ns, boxer_rows in boxer.groups.items():
        owl_rows = owl.groups[time_ns]
        owl_frame_indices = _unique_subsequence_indices(
            boxer_rows, owl_rows, time_ns=time_ns
        )
        per_time_matched[time_ns] = len(owl_frame_indices)
        for boxer_frame_index, (boxer_row, owl_frame_index) in enumerate(
            zip(boxer_rows, owl_frame_indices)
        ):
            owl_row = owl_rows[owl_frame_index]
            if owl_row.global_row_index in matched_owl_indices:
                raise ProvenanceError(
                    f"OWL global row mapped more than once: {owl_row.global_row_index}"
                )
            matched_owl_indices.add(owl_row.global_row_index)
            mappings.append(
                {
                    "time_ns": time_ns,
                    "boxer_row_index": boxer_row.global_row_index,
                    "owl_row_index": owl_row.global_row_index,
                    "boxer_frame_row_index": boxer_frame_index,
                    "owl_frame_row_index": owl_frame_index,
                    "normalized_name": boxer_row.normalized_name,
                    "sem_id": boxer_row.sem_id,
                }
            )

    if len(mappings) != len(boxer.rows):
        raise ProvenanceError("not every Boxer data row received provenance")
    if [row["boxer_row_index"] for row in mappings] != list(range(len(boxer.rows))):
        raise ProvenanceError("Boxer global row mapping is not complete and ordered")

    frame_statistics = []
    for time_ns, owl_rows in owl.groups.items():
        boxer_count = len(boxer.groups.get(time_ns, ()))
        frame_statistics.append(
            {
                "time_ns": time_ns,
                "owl_row_count": len(owl_rows),
                "boxer_row_count": boxer_count,
                "mapped_row_count": per_time_matched.get(time_ns, 0),
                "unmapped_owl_row_count": len(owl_rows) - boxer_count,
            }
        )

    return {
        "schema": SCHEMA,
        "mode": "no_gt_unique_order_preserving_subsequence",
        "gt_access": False,
        "oracle_access": False,
        "scene_id": boxer.path.parent.name,
        "contract": {
            "group_field": "time_ns",
            "match_fields": ["normalized_name", "sem_id"],
            "name_normalization": "strip_then_lower_only",
            "relation": "strict_unique_one_to_one_subsequence_per_time_ns",
            "row_index_basis": "global_0_based_data_row_excluding_header",
        },
        "inputs": {
            "boxer_3dbbs_csv": {
                "path": str(boxer.path),
                "sha256": boxer.sha256,
                "byte_count": boxer.byte_count,
                "row_count": len(boxer.rows),
            },
            "owl_2dbbs_csv": {
                "path": str(owl.path),
                "sha256": owl.sha256,
                "byte_count": owl.byte_count,
                "row_count": len(owl.rows),
            },
        },
        "mappings": mappings,
        "frame_statistics": frame_statistics,
        "statistics": {
            "owl_frame_count": len(owl.groups),
            "boxer_frame_count": len(boxer.groups),
            "owl_only_frame_count": len(set(owl.groups) - set(boxer.groups)),
            "owl_row_count": len(owl.rows),
            "boxer_row_count": len(boxer.rows),
            "mapped_row_count": len(mappings),
            "unmapped_owl_row_count": len(owl.rows) - len(mappings),
        },
    }


def map_boxer_rows_to_owl(
    boxer_csv: str | Path,
    owl_csv: str | Path,
) -> dict[str, Any]:
    """Compatibility alias with an action-oriented name for S2 callers."""

    return build_boxer_owl_provenance(boxer_csv, owl_csv)


__all__ = [
    "BOXER_COLUMNS",
    "OWL_COLUMNS",
    "ProvenanceError",
    "SCHEMA",
    "build_boxer_owl_provenance",
    "map_boxer_rows_to_owl",
    "normalize_name",
]
