from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT / "tools", ROOT / "tests"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import merge_scannet_fastsam_f4_boxer_paper100 as merger
import run_scannet_fastsam_f4_boxer_paper100 as runner
from test_run_scannet_fastsam_f4_boxer_paper100 import _prepare_inputs, _run_shard


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
    ).hexdigest()


def _run_both(tmp_path: Path) -> tuple[dict, tuple[Path, Path]]:
    inputs = _prepare_inputs(tmp_path)
    _run_shard(inputs, 0, [])
    _run_shard(inputs, 1, [])
    shards = (
        inputs["output"] / "shards/shard-000-of-002.json",
        inputs["output"] / "shards/shard-001-of-002.json",
    )
    return inputs, shards


def test_merge_proves_partition_lineage_causality_and_runtime(tmp_path: Path) -> None:
    _inputs, shards = _run_both(tmp_path)
    receipt = merger.merge_f4(
        shard_paths=shards,
        output_dir=tmp_path / "final",
        expected_scene_count=2,
        expected_keyframes=2,
        expected_successful_frames=2,
        expected_sources=2,
    )
    assert receipt["complete"]
    assert receipt["overall_pass"]
    assert receipt["coverage"]["scene_order"] == ["scene0000_00", "scene0001_00"]
    assert receipt["coverage"]["exact_source_partition"]
    assert receipt["coverage"]["exact_source_order"]
    assert receipt["totals"]["source_count"] == 2
    assert receipt["totals"]["valid_hb_count"] == 2
    assert receipt["causality"]["maximum_lookahead_frames"] == 0
    assert all(gate["pass"] for gate in receipt["gates"].values())
    assert receipt["oracle_authorization"]["active_birth_authorized"] is False


def test_merge_rejects_even_resealed_h0_mutation(tmp_path: Path) -> None:
    _inputs, shards = _run_both(tmp_path)
    shard = json.loads(shards[0].read_text())
    scene_path = Path(shard["scenes"][0]["sidecar"]["path"])
    scene = json.loads(scene_path.read_text())
    scene["frames"][0]["sources"][0]["hypotheses"]["H0"]["center"][0] += 0.25
    scene["content_sha256"] = _canonical({key: value for key, value in scene.items() if key != "content_sha256"})
    changed_scene = tmp_path / "changed-scene.json"
    changed_scene.write_text(json.dumps(scene, sort_keys=True), encoding="utf-8")
    shard["scenes"][0]["sidecar"] = {"path": str(changed_scene.resolve()), "sha256": _hash(changed_scene)}
    shard["content_sha256"] = _canonical({key: value for key, value in shard.items() if key not in ("content_sha256", "manifest_path")})
    changed_shard = tmp_path / "changed-shard.json"
    changed_shard.write_text(json.dumps(shard, sort_keys=True), encoding="utf-8")
    with pytest.raises(merger.F4MergeError, match="H0/HL/HLG"):
        merger.merge_f4(
            shard_paths=(changed_shard, shards[1]),
            output_dir=tmp_path / "bad-final",
            expected_scene_count=2,
            expected_keyframes=2,
            expected_successful_frames=2,
            expected_sources=2,
        )


def test_merge_is_create_only_and_cli_has_no_forbidden_inputs(tmp_path: Path) -> None:
    _inputs, shards = _run_both(tmp_path)
    kwargs = dict(
        shard_paths=shards,
        output_dir=tmp_path / "final",
        expected_scene_count=2,
        expected_keyframes=2,
        expected_successful_frames=2,
        expected_sources=2,
    )
    merger.merge_f4(**kwargs)
    with pytest.raises(merger.F4MergeError, match="refusing to overwrite"):
        merger.merge_f4(**kwargs)
    options = {option for action in merger._parser()._actions for option in action.option_strings}
    assert not any(token in option for option in options for token in ("gt", "oracle", "prediction", "native", "evaluator"))


def test_cold_warmup_deadline_misses_remain_visible_but_do_not_fail_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _prepare_inputs(tmp_path)
    timestamps = iter((0, 900_000_000, 1_000_000_000, 1_900_000_000))
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: next(timestamps))
    shard0 = _run_shard(inputs, 0, [])
    shard1 = _run_shard(inputs, 1, [])
    for shard in (shard0, shard1):
        frame = json.loads(Path(shard["scenes"][0]["sidecar"]["path"]).read_text())["frames"][0]
        assert frame["runtime"]["f4_warmup_excluded"] is True
        assert frame["runtime"]["gap25_deadline_missed"] is True
        assert frame["runtime"]["gap25_deadline_missed_warm"] is False
        assert shard["runtime"]["gap25_all_deadline_miss_count"] == 1
        assert shard["runtime"]["gap25_warm_deadline_miss_count"] == 0

    receipt = merger.merge_f4(
        shard_paths=(
            inputs["output"] / "shards/shard-000-of-002.json",
            inputs["output"] / "shards/shard-001-of-002.json",
        ),
        output_dir=tmp_path / "cold-final",
        expected_scene_count=2,
        expected_keyframes=2,
        expected_successful_frames=2,
        expected_sources=2,
    )
    assert receipt["runtime"]["gap25_all_deadline_miss_count"] == 2
    assert receipt["runtime"]["gap25_warm_deadline_miss_count"] == 0
    assert receipt["gates"]["gap25_warm_deadline_miss_count"]["pass"] is True
    assert receipt["overall_pass"] is True
