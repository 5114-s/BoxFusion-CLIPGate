from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest

from boxfusion.boxer_owl_provenance import (
    BOXER_COLUMNS,
    OWL_COLUMNS,
    ProvenanceError,
    build_boxer_owl_provenance,
    map_boxer_rows_to_owl,
)


SCENE = "scene0001_00"


def _owl_row(time_ns: int, name: str, sem_id: int) -> list[str]:
    return [
        str(time_ns),
        str(time_ns // 10),
        "scannet",
        "ScanNet",
        "960",
        "960",
        "1.0",
        "2.0",
        "3.0",
        "4.0",
        name,
        "-1",
        str(sem_id),
        "0.75",
    ]


def _boxer_row(time_ns: int, name: str, sem_id: int) -> list[str]:
    return [
        str(time_ns),
        "1.0",
        "2.0",
        "3.0",
        "1.0",
        "0.0",
        "0.0",
        "0.0",
        "0.5",
        "0.6",
        "0.7",
        name,
        "0",
        str(sem_id),
        "0.8",
    ]


def _write_csv(path: Path, header: tuple[str, ...], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def _fixture(
    tmp_path: Path,
    *,
    owl_rows: list[list[str]],
    boxer_rows: list[list[str]],
) -> tuple[Path, Path]:
    scene = tmp_path / SCENE
    owl = scene / "owl_2dbbs.csv"
    boxer = scene / "boxer_3dbbs.csv"
    _write_csv(owl, OWL_COLUMNS, owl_rows)
    _write_csv(boxer, BOXER_COLUMNS, boxer_rows)
    return boxer, owl


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builds_unique_global_zero_based_provenance_and_statistics(
    tmp_path: Path,
) -> None:
    boxer, owl = _fixture(
        tmp_path,
        owl_rows=[
            _owl_row(0, " Chair ", 1),
            _owl_row(0, "table", 2),
            _owl_row(0, "lamp", 3),
            _owl_row(10, "door", 4),  # valid OWL-only frame
            _owl_row(20, "cabinet", 5),
            _owl_row(20, "window", 6),
        ],
        boxer_rows=[
            _boxer_row(0, "chair", 1),
            _boxer_row(0, "LAMP", 3),
            _boxer_row(20, "window", 6),
        ],
    )

    first = build_boxer_owl_provenance(boxer, owl)
    second = map_boxer_rows_to_owl(boxer, owl)
    assert first == second
    assert first["gt_access"] is False
    assert first["oracle_access"] is False
    assert first["scene_id"] == SCENE
    assert first["inputs"]["boxer_3dbbs_csv"]["sha256"] == _sha256(boxer)
    assert first["inputs"]["owl_2dbbs_csv"]["sha256"] == _sha256(owl)
    assert [
        (row["boxer_row_index"], row["owl_row_index"])
        for row in first["mappings"]
    ] == [(0, 0), (1, 2), (2, 5)]
    assert [row["boxer_frame_row_index"] for row in first["mappings"]] == [0, 1, 0]
    assert [row["owl_frame_row_index"] for row in first["mappings"]] == [0, 2, 1]
    assert [row["normalized_name"] for row in first["mappings"]] == [
        "chair",
        "lamp",
        "window",
    ]
    assert first["statistics"] == {
        "owl_frame_count": 3,
        "boxer_frame_count": 2,
        "owl_only_frame_count": 1,
        "owl_row_count": 6,
        "boxer_row_count": 3,
        "mapped_row_count": 3,
        "unmapped_owl_row_count": 3,
    }
    assert first["frame_statistics"][1] == {
        "time_ns": 10,
        "owl_row_count": 1,
        "boxer_row_count": 0,
        "mapped_row_count": 0,
        "unmapped_owl_row_count": 1,
    }


def test_rejects_ambiguous_repeated_semantic_key(tmp_path: Path) -> None:
    boxer, owl = _fixture(
        tmp_path,
        owl_rows=[_owl_row(0, "chair", 1), _owl_row(0, "chair", 1)],
        boxer_rows=[_boxer_row(0, "chair", 1)],
    )
    with pytest.raises(ProvenanceError, match="ambiguous.*time_ns=0"):
        build_boxer_owl_provenance(boxer, owl)


@pytest.mark.parametrize(
    ("owl_rows", "boxer_rows"),
    [
        (
            [_owl_row(0, "chair", 1), _owl_row(0, "table", 2)],
            [_boxer_row(0, "table", 2), _boxer_row(0, "chair", 1)],
        ),
        (
            [_owl_row(0, "chair", 1)],
            [_boxer_row(0, "chair", 2)],
        ),
        (
            [_owl_row(0, "chair", 1)],
            [_boxer_row(0, "chair", 1), _boxer_row(0, "chair", 1)],
        ),
    ],
)
def test_rejects_order_semantic_or_one_to_one_mismatch(
    tmp_path: Path,
    owl_rows: list[list[str]],
    boxer_rows: list[list[str]],
) -> None:
    boxer, owl = _fixture(tmp_path, owl_rows=owl_rows, boxer_rows=boxer_rows)
    with pytest.raises(ProvenanceError, match="not an OWL semantic subsequence"):
        build_boxer_owl_provenance(boxer, owl)


def test_rejects_boxer_timestamp_absent_from_owl(tmp_path: Path) -> None:
    boxer, owl = _fixture(
        tmp_path,
        owl_rows=[_owl_row(0, "chair", 1)],
        boxer_rows=[_boxer_row(10, "chair", 1)],
    )
    with pytest.raises(ProvenanceError, match="missing from OWL CSV"):
        build_boxer_owl_provenance(boxer, owl)


def test_rejects_noncontiguous_or_nonmonotonic_timestamp_groups(
    tmp_path: Path,
) -> None:
    boxer, owl = _fixture(
        tmp_path,
        owl_rows=[
            _owl_row(0, "chair", 1),
            _owl_row(10, "table", 2),
            _owl_row(0, "lamp", 3),
        ],
        boxer_rows=[],
    )
    with pytest.raises(ProvenanceError, match="non-contiguous time_ns group"):
        build_boxer_owl_provenance(boxer, owl)


def test_rejects_header_drift_and_malformed_integer(tmp_path: Path) -> None:
    boxer, owl = _fixture(
        tmp_path,
        owl_rows=[_owl_row(0, "chair", 1)],
        boxer_rows=[_boxer_row(0, "chair", 1)],
    )
    wrong_header = list(OWL_COLUMNS)
    wrong_header[-1] = "score"
    _write_csv(owl, tuple(wrong_header), [_owl_row(0, "chair", 1)])
    with pytest.raises(ProvenanceError, match="unexpected owl_2dbbs.csv header"):
        build_boxer_owl_provenance(boxer, owl)

    _write_csv(owl, OWL_COLUMNS, [_owl_row(0, "chair", 1)])
    bad_boxer = _boxer_row(0, "chair", 1)
    bad_boxer[BOXER_COLUMNS.index("sem_id")] = "1.0"
    _write_csv(boxer, BOXER_COLUMNS, [bad_boxer])
    with pytest.raises(ProvenanceError, match="invalid non-negative integer sem_id"):
        build_boxer_owl_provenance(boxer, owl)


def test_requires_exact_production_basenames_and_same_scene(tmp_path: Path) -> None:
    boxer, owl = _fixture(
        tmp_path,
        owl_rows=[_owl_row(0, "chair", 1)],
        boxer_rows=[_boxer_row(0, "chair", 1)],
    )
    renamed = boxer.with_name("candidate_3dbbs.csv")
    renamed.write_bytes(boxer.read_bytes())
    with pytest.raises(ProvenanceError, match="expected production input named"):
        build_boxer_owl_provenance(renamed, owl)

    other_owl = tmp_path / "scene0002_00" / "owl_2dbbs.csv"
    _write_csv(other_owl, OWL_COLUMNS, [_owl_row(0, "chair", 1)])
    with pytest.raises(ProvenanceError, match="same scene directory"):
        build_boxer_owl_provenance(boxer, other_owl)


def test_header_only_boxer_maps_deterministically_to_nothing(tmp_path: Path) -> None:
    boxer, owl = _fixture(
        tmp_path,
        owl_rows=[_owl_row(0, "chair", 1)],
        boxer_rows=[],
    )
    result = build_boxer_owl_provenance(boxer, owl)
    assert result["mappings"] == []
    assert result["statistics"]["mapped_row_count"] == 0
    assert result["statistics"]["unmapped_owl_row_count"] == 1
