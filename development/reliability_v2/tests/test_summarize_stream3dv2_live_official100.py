from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import summarize_stream3dv2_live_official100 as summary_tool


def _diagnostic(
    scene: str,
    *,
    keyframes: int = 4,
    mean_ms: float = 100.0,
    p50_ms: float = 90.0,
    p95_ms: float = 140.0,
    max_ms: float = 160.0,
    births: int = 2,
    overlays: int = 1,
    future: int = 0,
    late: int = 0,
    dropped_late: int | None = None,
    queue_max: int = 1,
    deadline_misses: int = 0,
    peak_allocated: int = 2 * 1024**3,
) -> dict[str, object]:
    if dropped_late is None:
        dropped_late = late
    native = 10
    return {
        "schema": summary_tool.DIAGNOSTIC_SCHEMA,
        "complete": True,
        "scene_id": scene,
        "training_free": True,
        "gt_access": False,
        "annotation_access": False,
        "evaluator_access": False,
        "proposal_cache_access": False,
        "teacher_cache_access": False,
        "terminal_cache_access": False,
        "past_only": True,
        "query_before_commit": True,
        "terminal_output_decision": True,
        "native_scores_preserved": True,
        "bounded": {
            "sam3_queue_capacity": 1,
            "state_max_tracks": 1024,
            "state_max_views_per_track": 5,
            "semantic_views": 8,
            "fastsam_candidates_per_keyframe": 16,
        },
        "counts": {
            "keyframes": keyframes,
            "deadline_misses": deadline_misses,
            "native": native,
            "candidates": 3,
            "births": births,
            "overlays": overlays,
            "output": native + births,
            "sam3_submitted": 2,
            "sam3_results_accepted": 1,
            "sam3_result_drops": late,
            "sam3_submit_drops": 1,
        },
        "timing_ms": {
            "keyframe_total": {
                "count": keyframes,
                "mean": mean_ms,
                "p50": p50_ms,
                "p95": p95_ms,
                "max": max_ms,
            },
            "fastsam": {
                "count": keyframes,
                "mean": mean_ms / 2,
                "p50": p50_ms / 2,
                "p95": p95_ms / 2,
                "max": max_ms / 2,
            },
        },
        "deadline_ms": 833.333,
        "state": {
            "committed_frame_count": keyframes,
            "committed_view_count": 2,
            "retired_track_count": 1,
            "terminal_finalized_track_count": 1,
            "peak_live_track_count": 2,
            "peak_live_view_count": 3,
            "live_track_count": 0,
            "live_view_count": 0,
            "last_committed_frame_ordinal": keyframes - 1,
            "future_access_count": future,
            "query_before_commit": True,
        },
        "sam3": {
            "queue_depth": 0,
            "max_queue_depth": queue_max,
            "submitted": 2,
            "completed": 1,
            "delivered": 1 - int(late > 0),
            "drop_count": late,
            "dropped_late": dropped_late,
            "late_count": late,
            "worker_error_count": 0,
        },
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_allocated + 1024,
        "future_access_count": future,
        "late_result_count": late,
    }


def _write_scene(
    root: Path,
    scene: str,
    diagnostic: dict[str, object],
    *,
    cost: float,
    fps: float,
) -> None:
    diag_root = root / "diagnostics"
    log_root = root / "logs"
    diag_root.mkdir(exist_ok=True)
    log_root.mkdir(exist_ok=True)
    (diag_root / f"{scene}.json").write_text(json.dumps(diagnostic), encoding="utf-8")
    (log_root / f"{scene}.log").write_text(
        f"Cost: {cost:.2f} s Average FPS: {fps:.2f}\n"
        "Strict live summary | past_only=True future_access=0 queue_capacity=1\n",
        encoding="utf-8",
    )


def _scene_list(root: Path, scenes: list[str]) -> Path:
    path = root / "scenes.txt"
    path.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    return path


def test_complete_summary_uses_duration_weighted_fps_and_declares_quantile_approximation(
    tmp_path: Path,
) -> None:
    scenes = ["scene0000_00", "scene0001_00"]
    _write_scene(
        tmp_path,
        scenes[0],
        _diagnostic(scenes[0], keyframes=2, mean_ms=100, p50_ms=90, p95_ms=120, max_ms=130),
        cost=10,
        fps=20,
    )
    _write_scene(
        tmp_path,
        scenes[1],
        _diagnostic(
            scenes[1],
            keyframes=6,
            mean_ms=200,
            p50_ms=180,
            p95_ms=240,
            max_ms=260,
            births=1,
            overlays=0,
            peak_allocated=3 * 1024**3,
        ),
        cost=30,
        fps=10,
    )
    result = summary_tool.summarize(
        scene_list=_scene_list(tmp_path, scenes),
        diagnostics_root=tmp_path / "diagnostics",
        scene_log_root=tmp_path / "logs",
    )

    assert result["status"]["coverage"] == "complete"
    assert result["status"]["official100_complete"] is False
    assert result["internal_runtime"]["aggregate_internal_fps_approx"] == pytest.approx(12.5)
    stage = result["stage_latency_ms"]["stages"]["keyframe_total"]
    assert stage["sample_count"] == 8
    assert stage["mean"] == pytest.approx(175.0)
    assert stage["p50_weighted_scene_quantile_approx"] == pytest.approx(157.5)
    assert stage["p95_weighted_scene_quantile_approx"] == pytest.approx(210.0)
    assert stage["max"] == 260
    assert stage["raw_samples_available"] is False
    assert "approximate" in result["stage_latency_ms"]["method"]["p50_p95"]
    assert result["counts"]["births"] == 3
    assert result["counts"]["overlays"] == 1
    assert result["cuda_main_process_peak"]["allocated_gib_max"] == 3.0
    assert result["cuda_main_process_peak"]["includes_sam3_subprocess"] is False


def test_missing_scene_is_marked_partial_and_require_complete_exits_one(tmp_path: Path) -> None:
    scenes = ["scene0000_00", "scene0001_00"]
    _write_scene(
        tmp_path,
        scenes[0],
        _diagnostic(scenes[0]),
        cost=10,
        fps=20,
    )
    scene_list = _scene_list(tmp_path, scenes)
    result = summary_tool.summarize(
        scene_list=scene_list,
        diagnostics_root=tmp_path / "diagnostics",
        scene_log_root=tmp_path / "logs",
    )
    assert result["status"]["partial"] is True
    assert result["coverage"]["valid_scene_count"] == 1
    assert result["coverage"]["missing_diagnostics"] == [scenes[1]]
    assert result["coverage"]["missing_scene_logs"] == [scenes[1]]

    output = tmp_path / "summary.json"
    exit_code = summary_tool.main(
        [
            "--scene-list",
            str(scene_list),
            "--diagnostics-root",
            str(tmp_path / "diagnostics"),
            "--scene-log-root",
            str(tmp_path / "logs"),
            "--output",
            str(output),
            "--require-complete",
        ]
    )
    assert exit_code == 1
    assert json.loads(output.read_text(encoding="utf-8"))["status"]["partial"] is True


def test_future_queue_and_undropped_late_result_fail_closed_contract(tmp_path: Path) -> None:
    scene = "scene0000_00"
    _write_scene(
        tmp_path,
        scene,
        _diagnostic(scene, future=1, late=1, dropped_late=0, queue_max=2),
        cost=10,
        fps=20,
    )
    result = summary_tool.summarize(
        scene_list=_scene_list(tmp_path, [scene]),
        diagnostics_root=tmp_path / "diagnostics",
        scene_log_root=tmp_path / "logs",
    )
    assert result["coverage"]["valid_scene_count"] == 1
    assert result["coverage"]["invalid_scenes"] == []
    assert result["causality_and_queue"]["future_access_count"] == 1
    assert result["causality_and_queue"]["late_result_count"] == 1
    assert result["causality_and_queue"]["queue_max_depth"] == 2
    assert result["status"]["causal_contract_pass"] is False
    kinds = {row["kind"] for row in result["issues"]}
    assert "queue_capacity_exceeded" in kinds
    assert "late_result_not_fail_closed" in kinds


def test_exact_100_valid_scenes_sets_official100_complete(tmp_path: Path) -> None:
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    for scene in scenes:
        _write_scene(tmp_path, scene, _diagnostic(scene), cost=1, fps=12)
    result = summary_tool.summarize(
        scene_list=_scene_list(tmp_path, scenes),
        diagnostics_root=tmp_path / "diagnostics",
        scene_log_root=tmp_path / "logs",
    )
    assert result["status"]["official100_complete"] is True
    assert result["status"]["strict_realtime_online_pass"] is True
    assert result["coverage"]["valid_scene_count"] == 100
    assert result["internal_runtime"]["aggregate_internal_fps_approx"] == 12.0


def test_duplicate_scene_list_is_rejected(tmp_path: Path) -> None:
    path = _scene_list(tmp_path, ["scene0000_00", "scene0000_00"])
    with pytest.raises(summary_tool.SummaryInputError, match="duplicate"):
        summary_tool.read_scene_list(path)
