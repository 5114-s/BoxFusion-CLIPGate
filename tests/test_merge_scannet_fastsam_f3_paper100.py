from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import merge_scannet_fastsam_f3_paper100 as merger
import audit_scannet_fastsam_f3_paper100_oracle as oracle
from test_run_scannet_fastsam_f3_paper100 import _prepare_f3


def test_merge_validates_partition_causality_runtime_and_oracle_schema(
    tmp_path: Path,
) -> None:
    manifest, _inputs = _prepare_f3(tmp_path)
    shard = tmp_path / "f3/shards/shard-000-of-001.json"
    receipt = merger.merge_f3(
        shard_paths=(shard,),
        output_dir=tmp_path / "f3/final",
        _expected_scene_count=1,
    )
    assert receipt["complete"]
    assert receipt["integrity"]["overall_pass"]
    assert receipt["causality"]["overall_pass"]
    assert receipt["overall_pass"] == receipt["runtime"]["overall_pass"]
    assert receipt["coverage"]["scene_order"] == ["scene0000_00"]
    assert receipt["coverage"]["keyframe_count"] == 3
    assert receipt["coverage"]["source_count"] == 3
    assert receipt["totals"]["track_count"] == manifest["totals"]["track_count"]
    assert set(receipt["runtime"]["gates"]) == {
        "f3_incremental_mean_ms",
        "f3_incremental_p95_ms",
        "amortized_f3_ms_per_source_frame",
        "composed_complete_p95_ms",
        "composed_complete_max_ms",
        "amortized_composed_complete_ms_per_source_frame",
        "new_gpu_allocation_bytes",
        "total_gpu_peak_memory_bytes",
    }
    for gate in receipt["runtime"]["gates"].values():
        assert set(gate) == {"actual", "threshold", "comparator", "passed"}

    rows, signature, _runtime_pass, _runtime = oracle._validate_f3_receipt(
        receipt,
        ["scene0000_00"],
        expected_scene_count=1,
        expected_keyframe_count=3,
        expected_successful_frame_count=3,
        expected_source_count=3,
    )
    scene_row = rows["scene0000_00"]
    sidecar_path = Path(scene_row["sidecar"]["path"])
    sidecar = json.loads(sidecar_path.read_text())
    tracks, counts, diagnostics = oracle._load_f3_tracks(
        path=sidecar_path,
        f0_path=Path(sidecar["inputs"]["f0_sidecar"]["path"]),
        scene="scene0000_00",
        scene_index=0,
        alignment=np.eye(4, dtype=np.float64),
        receipt_sidecar_sha256=scene_row["sidecar"]["sha256"],
        run_signature_sha256=signature,
    )
    assert counts["source_count"] == 3
    assert tracks
    assert diagnostics["causality"]["overall_pass"]


def test_merge_rejects_assignment_order_not_matching_f0_rank(tmp_path: Path) -> None:
    manifest, _inputs = _prepare_f3(tmp_path)
    scene_path = Path(manifest["scenes"][0]["sidecar_path"])
    scene = json.loads(scene_path.read_text())
    # Synthetic fixture has one source/frame; make the source identity itself
    # differ so the same strict zip-order validator is exercised.
    scene["frames"][0]["assignments"][0]["source_id"] = "tampered/source"
    changed_scene = tmp_path / "changed-scene.json"
    changed_scene.write_text(json.dumps(scene), encoding="utf-8")
    shard_path = tmp_path / "f3/shards/shard-000-of-001.json"
    shard = json.loads(shard_path.read_text())
    shard["scenes"][0]["sidecar_path"] = str(changed_scene)
    import hashlib

    shard["scenes"][0]["sidecar_sha256"] = hashlib.sha256(
        changed_scene.read_bytes()
    ).hexdigest()
    changed_shard = tmp_path / "changed-shard.json"
    changed_shard.write_text(json.dumps(shard), encoding="utf-8")
    with pytest.raises(merger.F3MergeError, match="assignment order differs"):
        merger.merge_f3(
            shard_paths=(changed_shard,),
            output_dir=tmp_path / "changed-final",
            _expected_scene_count=1,
        )


def test_merge_is_create_only(tmp_path: Path) -> None:
    _manifest, _inputs = _prepare_f3(tmp_path)
    kwargs = {
        "shard_paths": (tmp_path / "f3/shards/shard-000-of-001.json",),
        "output_dir": tmp_path / "f3/final",
        "_expected_scene_count": 1,
    }
    merger.merge_f3(**kwargs)
    with pytest.raises(merger.F3MergeError, match="refusing to overwrite"):
        merger.merge_f3(**kwargs)
