from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from boxfusion.yidu_ablation import YIDU_STAGE_TO_PROFILE
from tools.build_yidu_run_manifest import build


def _file(path: Path, value: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return str(path)


def _arguments(
    tmp_path: Path,
    *,
    stage: str,
    teacher: bool,
) -> argparse.Namespace:
    cache = tmp_path / "teacher"
    cache.mkdir()
    _file(cache / "manifest.json", "{}")
    return argparse.Namespace(
        output=str(tmp_path / "logs" / "run" / "run_manifest.json"),
        stage=stage,
        profile=YIDU_STAGE_TO_PROFILE[stage],
        config=_file(tmp_path / "config.yaml", "config"),
        scene_list=_file(tmp_path / "scenes.txt", "scene0000_00\n"),
        b6_checkpoint=_file(tmp_path / "b6.npz", "b6"),
        yoloe_checkpoint=_file(tmp_path / "yoloe.pt", "yoloe"),
        teacher_cache=str(cache) if teacher else None,
        teacher_namespace="teacher-v1" if teacher else None,
        cache_missing_policy="error",
        live_root=str(tmp_path / "live"),
        frames_root=str(tmp_path / "frames"),
        prediction_root=str(tmp_path / "predictions"),
        log_root=str(tmp_path / "logs" / "run"),
        diagnostics_root=str(tmp_path / "diagnostics"),
        evaluation_root=str(tmp_path / "evaluation"),
        minimum_extent=0.4,
        post_minimum_extent="disabled",
        gate_checkpoint=None,
        gate_training_archive=None,
        gate_train_scene_list=None,
        gate_forbidden_scene_list=None,
        inference_seed=0,
        evaluation_seed=0,
        python_executable="/usr/bin/python3",
        python_version="3",
        torch_version="test",
    )


def test_b0_manifest_excludes_inactive_teacher(tmp_path: Path) -> None:
    payload = build(_arguments(tmp_path, stage="B0", teacher=True))

    assert payload["schema"] == "boxfusion.yidu.run_manifest.v3"
    assert payload["teacher_cache"] is None
    assert payload["teacher_namespace"] is None
    assert payload["teacher_metadata_sha256"] is None
    assert payload["cache_missing_policy"] is None


def test_observer_manifest_requires_teacher_pair(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires teacher"):
        build(_arguments(tmp_path, stage="A1", teacher=False))


def test_observer_manifest_records_active_teacher(tmp_path: Path) -> None:
    payload = build(_arguments(tmp_path, stage="A2", teacher=True))

    assert payload["teacher_cache"] == str(
        (tmp_path / "teacher").resolve()
    )
    assert payload["teacher_namespace"] == "teacher-v1"
    assert payload["teacher_metadata_sha256"] is not None
