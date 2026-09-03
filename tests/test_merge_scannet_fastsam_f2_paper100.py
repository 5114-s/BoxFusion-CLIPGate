from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import merge_scannet_fastsam_f2_paper100 as merger
from test_run_scannet_fastsam_f2_paper100 import _prepare_f2


def test_merge_validates_scene_npz_identity_and_publishes_oracle_contract(
    tmp_path: Path,
) -> None:
    manifest, _f0_receipt, _calls = _prepare_f2(tmp_path)
    shard = tmp_path / "f2/shards/shard-000-of-001.json"
    assert shard.is_file()
    scene_list = Path(manifest["scene_list"]["path"])
    receipt = merger.merge_f2(
        shard_paths=(shard,),
        scene_list_path=scene_list,
        output_dir=tmp_path / "f2/final",
        _expected_scene_count=1,
    )
    assert receipt["complete"]
    assert receipt["overall_pass"]
    assert receipt["gates"]["overall_pass"] == receipt["overall_pass"]
    expected_runtime_gates = [
        "provider_runtime_p95_ms",
        "complete_runtime_p95_ms",
        "complete_runtime_max_ms",
        "amortized_complete_ms_per_source_frame",
        "amortized_f2_core_ms_per_source_frame",
        "gpu_peak_memory_bytes",
    ]
    assert receipt["gates"]["runtime"]["gate_names"] == expected_runtime_gates
    for name in expected_runtime_gates:
        assert set(receipt["gates"][name]) == {
            "actual",
            "comparator",
            "threshold",
            "passed",
        }
    assert receipt["totals"]["scene_count"] == 1
    assert receipt["totals"]["keyframe_count"] == 2
    assert receipt["totals"]["successful_frame_count"] == 2
    assert receipt["totals"]["source_count"] == 2
    assert receipt["totals"]["identity_verified_source_count"] == 2
    assert receipt["coverage"]["identity_ratio"] == 1.0
    assert receipt["runtime"]["f2_candidate_diagnostics"][
        "f2_candidate_local_ms"
    ]["count"] == 2
    scene = receipt["scenes"][0]
    assert set(scene) >= {"scene_id", "scene_index", "sidecar", "evidence_npz"}


def test_merge_refuses_changed_evidence_hash(tmp_path: Path) -> None:
    manifest, _f0_receipt, _calls = _prepare_f2(tmp_path)
    shard_path = tmp_path / "f2/shards/shard-000-of-001.json"
    shard = json.loads(shard_path.read_text())
    shard["scenes"][0]["evidence_npz_sha256"] = "0" * 64
    changed = tmp_path / "changed-shard.json"
    changed.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(merger.F2MergeError, match="evidence reference differs"):
        merger.merge_f2(
            shard_paths=(changed,),
            scene_list_path=Path(manifest["scene_list"]["path"]),
            output_dir=tmp_path / "final",
            _expected_scene_count=1,
        )


def test_merge_is_create_only(tmp_path: Path) -> None:
    manifest, _f0_receipt, _calls = _prepare_f2(tmp_path)
    kwargs = {
        "shard_paths": (tmp_path / "f2/shards/shard-000-of-001.json",),
        "scene_list_path": Path(manifest["scene_list"]["path"]),
        "output_dir": tmp_path / "f2/final",
        "_expected_scene_count": 1,
    }
    merger.merge_f2(**kwargs)
    with pytest.raises(merger.F2MergeError, match="refusing to overwrite"):
        merger.merge_f2(**kwargs)
