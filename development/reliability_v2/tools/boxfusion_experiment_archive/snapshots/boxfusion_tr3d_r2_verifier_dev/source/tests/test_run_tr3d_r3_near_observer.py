from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from boxfusion.tr3d_r2_provenance import canonical_json_sha256
from boxfusion.tr3d_r3_observer import TR3D_R3_NEAR_ANCHOR_IOU
from tools.run_tr3d_r3_near_observer import (
    CONFIG_SCHEMA,
    _config,
    _optional_parent_contract,
    _write_create_only,
)


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        r2a_cache_root=None,
        r2a_export_report=None,
        frames_root=None,
        r2b_cache_root=None,
        r2b_export_report=None,
        parent_cache_root=tmp_path / "parent",
        prefix_manifest=tmp_path / "prefix.jsonl",
        scene_list=tmp_path / "scenes.txt",
        prefix_id="p100",
    )


def test_optional_parent_contract_has_explicit_absence_sentinels(
    tmp_path: Path,
) -> None:
    hashes, reports = _optional_parent_contract(
        _args(tmp_path), ["scene0001_00"]
    )
    assert reports == {"r2a_enabled": False, "r2b_enabled": False}
    assert all(value == "" for value in hashes.values())
    config = _config(False, False)
    assert config["schema"] == CONFIG_SCHEMA
    assert config["observer_only"] is True
    assert config["mutation_enabled"] is False
    assert config["applied_count"] == 0
    assert config["ground_truth_access"] is False
    assert config["axis_alignment_is_ground_truth"] is False
    assert config["near_anchor_iou"] == TR3D_R3_NEAR_ANCHOR_IOU
    assert len(canonical_json_sha256(config)) == 64


def test_r2b_without_r2a_is_refused(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.r2b_cache_root = tmp_path / "r2b"
    args.r2b_export_report = tmp_path / "r2b.json"
    with pytest.raises(ValueError, match="requires the exact R2a"):
        _optional_parent_contract(args, ["scene0001_00"])


def test_report_writer_is_create_only_and_readonly(tmp_path: Path) -> None:
    target = tmp_path / "report.json"
    _write_create_only(target, "{}\n")
    assert target.read_text() == "{}\n"
    assert target.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="immutable R3"):
        _write_create_only(target, "{}\n")
