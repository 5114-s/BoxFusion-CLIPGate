from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import summarize_stream3dv3_live_official100 as summary_tool


FINGERPRINT_A = "a" * 64
FINGERPRINT_B = "b" * 64


def _diagnostic(
    scene: str,
    *,
    raw_frames: int = 100,
    pipeline_seconds: float = 4.0,
    fingerprint: str = FINGERPRINT_A,
    keyframes: int = 4,
    births: int = 1,
    deadline_misses: int = 0,
    keyframe_max_ms: float = 200.0,
) -> dict[str, object]:
    native = 10
    return {
        "schema": summary_tool.DIAGNOSTIC_SCHEMA,
        "complete": True,
        "scene_id": scene,
        "raw_frame_count": raw_frames,
        "pipeline_seconds": pipeline_seconds,
        "run_fingerprint": fingerprint,
        "training_free": True,
        "past_only": True,
        "query_before_commit": True,
        "selection_and_acceptance_held_out": True,
        "future_access_count": 0,
        "ground_truth_access": False,
        "annotation_access": False,
        "evaluator_access": False,
        "proposal_cache_access": False,
        "teacher_cache_access": False,
        "terminal_cache_access": False,
        "native_scores_preserved": True,
        "target_end_to_end_fps": 20.0,
        "addon_deadline_ms": 285.0,
        "counts": {
            "keyframes": keyframes,
            "addon_deadline_misses": deadline_misses,
            "native": native,
            "births": births,
            "overlays": 0,
            "output": native + births,
            "accepted_track_pool": births,
        },
        "trigger": {},
        "gate_rejections": {},
        "f4_per_track": {},
        "timing_ms": {
            "keyframe_total": {
                "count": keyframes,
                "mean": min(100.0, keyframe_max_ms),
                "p50": min(90.0, keyframe_max_ms),
                "p95": min(180.0, keyframe_max_ms),
                "max": keyframe_max_ms,
            }
        },
        "bounded": {
            "max_track_views": 6,
            "max_f4_views_per_track": 2,
            "max_f4_sources_per_batch": 6,
            "prelift_top_k": 6,
            "max_births_per_scene": 2,
        },
        "f3": {
            "keyframes": keyframes,
            "audit_complete": True,
            "max_logical_accessed_ordinal": keyframes - 1,
        },
        "sam3": {"max_queue_depth": 0, "drop_count": 0},
        "peak_cuda_allocated_bytes": 0,
        "peak_cuda_reserved_bytes": 0,
    }


def _write_scene(root: Path, scene: str, payload: dict[str, object]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{scene}.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )


def _scene_list(root: Path, scenes: list[str]) -> Path:
    path = root / "scenes.txt"
    path.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    return path


def test_exact_fps_uses_total_raw_frames_over_total_pipeline_seconds(
    tmp_path: Path,
) -> None:
    scenes = ["scene0000_00", "scene0001_00"]
    diagnostics = tmp_path / "diagnostics"
    _write_scene(
        diagnostics,
        scenes[0],
        _diagnostic(scenes[0], raw_frames=100, pipeline_seconds=4.0),
    )
    _write_scene(
        diagnostics,
        scenes[1],
        _diagnostic(scenes[1], raw_frames=100, pipeline_seconds=6.0),
    )

    result = summary_tool.summarize(
        scene_list=_scene_list(tmp_path, scenes),
        diagnostics_root=diagnostics,
        minimum_fps=20.0,
    )

    assert result["status"]["artifacts_complete"] is True
    assert result["status"]["official100_complete"] is False
    assert result["runtime"]["raw_frame_count"] == 200
    assert result["runtime"]["pipeline_seconds"] == pytest.approx(10.0)
    assert result["runtime"]["aggregate_fps"] == pytest.approx(20.0)
    assert result["runtime"]["exact"] is True
    assert result["status"]["throughput_contract_pass"] is True
    assert result["status"]["diagnostic_contract_pass"] is True
    assert result["run_fingerprint"] == FINGERPRINT_A


def test_minimum_fps_is_not_rounded_and_can_fail_an_otherwise_valid_set(
    tmp_path: Path,
) -> None:
    scene = "scene0000_00"
    diagnostics = tmp_path / "diagnostics"
    _write_scene(
        diagnostics,
        scene,
        _diagnostic(scene, raw_frames=101, pipeline_seconds=5.0),
    )
    result = summary_tool.summarize(
        scene_list=_scene_list(tmp_path, [scene]),
        diagnostics_root=diagnostics,
        minimum_fps=20.2001,
    )

    assert result["runtime"]["aggregate_fps"] == pytest.approx(20.2)
    assert result["status"]["throughput_contract_pass"] is False


def test_contract_audit_reports_causal_heldout_cache_native_birth_deadline_and_fingerprint(
    tmp_path: Path,
) -> None:
    scenes = ["scene0000_00", "scene0001_00"]
    diagnostics = tmp_path / "diagnostics"
    _write_scene(diagnostics, scenes[0], _diagnostic(scenes[0]))
    broken = _diagnostic(
        scenes[1],
        fingerprint=FINGERPRINT_B,
        births=3,
        deadline_misses=1,
        keyframe_max_ms=300.0,
    )
    broken["past_only"] = False
    broken["future_access_count"] = 1
    broken["selection_and_acceptance_held_out"] = False
    broken["proposal_cache_access"] = True
    broken["native_scores_preserved"] = False
    _write_scene(diagnostics, scenes[1], broken)

    result = summary_tool.summarize(
        scene_list=_scene_list(tmp_path, scenes),
        diagnostics_root=diagnostics,
        minimum_fps=20.0,
    )
    status = result["status"]
    assert status["causal_contract_pass"] is False
    assert status["held_out_contract_pass"] is False
    assert status["cache_contract_pass"] is False
    assert status["native_score_contract_pass"] is False
    assert status["fingerprint_contract_pass"] is False
    assert status["birth_contract_pass"] is False
    assert status["deadline_contract_pass"] is False
    assert status["diagnostic_contract_pass"] is False
    kinds = {row["kind"] for row in result["issues"]}
    assert "causal_contract_mismatch" in kinds
    assert "held_out_contract_mismatch" in kinds
    assert "cache_contract_mismatch" in kinds
    assert "native_score_contract_mismatch" in kinds
    assert "birth_cap_exceeded" in kinds
    assert "cross_scene_fingerprint_mismatch" in kinds


def test_missing_exact_runtime_or_malformed_fingerprint_invalidates_scene(
    tmp_path: Path,
) -> None:
    scenes = ["scene0000_00", "scene0001_00"]
    diagnostics = tmp_path / "diagnostics"
    missing_runtime = _diagnostic(scenes[0])
    del missing_runtime["pipeline_seconds"]
    _write_scene(diagnostics, scenes[0], missing_runtime)
    malformed_fingerprint = _diagnostic(scenes[1], fingerprint="A" * 64)
    _write_scene(diagnostics, scenes[1], malformed_fingerprint)

    result = summary_tool.summarize(
        scene_list=_scene_list(tmp_path, scenes),
        diagnostics_root=diagnostics,
    )

    assert result["coverage"]["valid_scene_count"] == 0
    assert result["coverage"]["invalid_scenes"] == scenes
    assert result["runtime"]["aggregate_fps"] is None
    kinds = {row["kind"] for row in result["issues"]}
    assert "invalid_positive_number" in kinds
    assert "invalid_run_fingerprint" in kinds


def test_official100_pass_and_cli_require_flags(tmp_path: Path) -> None:
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    diagnostics = tmp_path / "diagnostics"
    for scene in scenes:
        _write_scene(
            diagnostics,
            scene,
            _diagnostic(scene, raw_frames=100, pipeline_seconds=4.0),
        )
    scene_list = _scene_list(tmp_path, scenes)
    output = tmp_path / "summary.json"

    exit_code = summary_tool.main(
        [
            "--scene-list",
            str(scene_list),
            "--diagnostics-root",
            str(diagnostics),
            "--output",
            str(output),
            "--minimum-fps",
            "20",
            "--require-complete",
            "--require-realtime-pass",
        ]
    )

    assert exit_code == 0
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["status"]["official100_complete"] is True
    assert saved["status"]["strict_realtime_online_pass"] is True
    assert saved["runtime"]["aggregate_fps"] == pytest.approx(25.0)


def test_require_flags_fail_for_partial_or_slow_official_set(tmp_path: Path) -> None:
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    diagnostics = tmp_path / "diagnostics"
    for scene in scenes[:-1]:
        _write_scene(
            diagnostics,
            scene,
            _diagnostic(scene, raw_frames=100, pipeline_seconds=5.0),
        )
    scene_list = _scene_list(tmp_path, scenes)
    partial_output = tmp_path / "partial.json"
    assert (
        summary_tool.main(
            [
                "--scene-list",
                str(scene_list),
                "--diagnostics-root",
                str(diagnostics),
                "--output",
                str(partial_output),
                "--require-complete",
            ]
        )
        == 1
    )

    _write_scene(
        diagnostics,
        scenes[-1],
        _diagnostic(scenes[-1], raw_frames=100, pipeline_seconds=5.0),
    )
    slow_output = tmp_path / "slow.json"
    assert (
        summary_tool.main(
            [
                "--scene-list",
                str(scene_list),
                "--diagnostics-root",
                str(diagnostics),
                "--output",
                str(slow_output),
                "--minimum-fps",
                "20.1",
                "--require-realtime-pass",
            ]
        )
        == 1
    )
    saved = json.loads(slow_output.read_text(encoding="utf-8"))
    assert saved["status"]["official100_complete"] is True
    assert saved["status"]["throughput_contract_pass"] is False
    assert saved["status"]["strict_realtime_online_pass"] is False


def test_duplicate_scene_list_and_invalid_minimum_are_rejected(tmp_path: Path) -> None:
    duplicate = _scene_list(tmp_path, ["scene0000_00", "scene0000_00"])
    with pytest.raises(summary_tool.SummaryInputError, match="duplicate"):
        summary_tool.read_scene_list(duplicate)
    with pytest.raises(summary_tool.SummaryInputError, match="minimum_fps"):
        summary_tool.summarize(
            scene_list=_scene_list(tmp_path, ["scene0000_00"]),
            diagnostics_root=tmp_path / "diagnostics",
            minimum_fps=0.0,
        )

