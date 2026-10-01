from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import run_boxfusion_sequences as runner


def _config(path: Path, diagnostics_root: Path, *, enabled: bool = True) -> Path:
    payload = {
        "data": {
            "datadir": "./data/scannet_val/scene0011_01/frames",
            "output_dir": "./unused-native-output",
        },
        "openbox_smov_r2": {
            "enabled": enabled,
            "diagnostics": {"root": str(diagnostics_root)},
        },
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _scene_list(path: Path, scenes: list[str]) -> Path:
    path.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    return path


def _argv(
    *,
    config: Path,
    scene_list: Path,
    native_root: Path,
    diagnostics_root: Path,
    log_root: Path,
    num_shards: int = 1,
    shard_index: int = 0,
) -> list[str]:
    return [
        "run_boxfusion_sequences.py",
        "--dataset",
        "scannet",
        "--seq-list",
        str(scene_list),
        "--config",
        str(config),
        "--output-dir",
        str(native_root),
        "--openbox-smov-r2-diagnostics-root",
        str(diagnostics_root),
        "--log-dir",
        str(log_root),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
    ]


def _successful_demo(calls: list[list[str]]):
    def run(cmd, **_kwargs):
        calls.append(list(cmd))
        scene = cmd[cmd.index("--seq") + 1]
        native_root = Path(cmd[cmd.index("--output-dir") + 1])
        diagnostics_root = Path(
            cmd[cmd.index("--openbox-smov-r2-diagnostics-root") + 1]
        )
        native_root.mkdir(parents=True, exist_ok=True)
        diagnostics_root.mkdir(parents=True, exist_ok=True)
        (native_root / f"{scene}_boxes.pkl").touch()
        runner.np.savez_compressed(
            diagnostics_root / f"{scene}{runner.R2_SIDECAR_SUFFIX}",
            schema=runner.np.asarray(runner.R2_EXPECTED_SCHEMA),
        )
        return subprocess.CompletedProcess(cmd, 0)

    return run


def test_r2_paired_resume_forwards_explicit_diagnostics_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    diagnostics = tmp_path / "diagnostics"
    native = tmp_path / "native"
    logs = tmp_path / "logs"
    config = _config(tmp_path / "config.yaml", tmp_path / "config-diag")
    scenes = _scene_list(
        tmp_path / "scenes.txt", ["scene0000_00", "scene0001_00"]
    )
    argv = _argv(
        config=config,
        scene_list=scenes,
        native_root=native,
        diagnostics_root=diagnostics,
        log_root=logs,
    )
    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _successful_demo(calls))
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == 0
    assert len(calls) == 2
    assert all(
        Path(call[call.index("--openbox-smov-r2-diagnostics-root") + 1])
        == diagnostics.resolve()
        for call in calls
    )

    calls.clear()
    monkeypatch.setattr(sys, "argv", argv)
    assert runner.main() == 0
    assert calls == []


@pytest.mark.parametrize("present", ["prediction", "sidecar"])
def test_r2_refuses_half_complete_create_only_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    present: str,
):
    diagnostics = tmp_path / "diagnostics"
    native = tmp_path / "native"
    diagnostics.mkdir()
    native.mkdir()
    scene = "scene0000_00"
    if present == "prediction":
        (native / f"{scene}_boxes.pkl").touch()
    else:
        (diagnostics / f"{scene}{runner.R2_SIDECAR_SUFFIX}").touch()
    config = _config(tmp_path / "config.yaml", diagnostics)
    scenes = _scene_list(tmp_path / "scenes.txt", [scene])
    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _successful_demo(calls))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            config=config,
            scene_list=scenes,
            native_root=native,
            diagnostics_root=diagnostics,
            log_root=tmp_path / "logs",
        ),
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert calls == []


def test_r2_rejects_duplicate_scene_ids_and_equal_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    diagnostics = tmp_path / "diagnostics"
    config = _config(tmp_path / "config.yaml", diagnostics)
    duplicate_list = _scene_list(
        tmp_path / "duplicates.txt", ["scene0000_00", "scene0000_00"]
    )
    base = _argv(
        config=config,
        scene_list=duplicate_list,
        native_root=tmp_path / "native",
        diagnostics_root=diagnostics,
        log_root=tmp_path / "logs",
    )
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit) as duplicate_error:
        runner.main()
    assert duplicate_error.value.code == 2

    unique_list = _scene_list(tmp_path / "unique.txt", ["scene0000_00"])
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            config=config,
            scene_list=unique_list,
            native_root=diagnostics,
            diagnostics_root=diagnostics,
            log_root=tmp_path / "logs",
        ),
    )
    with pytest.raises(SystemExit) as roots_error:
        runner.main()
    assert roots_error.value.code == 2


def test_r2_shards_are_disjoint_with_shared_artifact_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    diagnostics = tmp_path / "diagnostics"
    native = tmp_path / "native"
    config = _config(tmp_path / "config.yaml", diagnostics)
    scene_ids = [f"scene{index:04d}_00" for index in range(6)]
    scenes = _scene_list(tmp_path / "scenes.txt", scene_ids)
    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _successful_demo(calls))

    observed: list[set[str]] = []
    for shard_index in (0, 1):
        monkeypatch.setattr(
            sys,
            "argv",
            _argv(
                config=config,
                scene_list=scenes,
                native_root=native,
                diagnostics_root=diagnostics,
                log_root=tmp_path / "logs",
                num_shards=2,
                shard_index=shard_index,
            ),
        )
        start = len(calls)
        assert runner.main() == 0
        observed.append(
            {call[call.index("--seq") + 1] for call in calls[start:]}
        )

    assert observed[0].isdisjoint(observed[1])
    assert observed[0] | observed[1] == set(scene_ids)


def test_r2_demo_zero_exit_without_both_artifacts_is_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    diagnostics = tmp_path / "diagnostics"
    native = tmp_path / "native"
    config = _config(tmp_path / "config.yaml", diagnostics)
    scenes = _scene_list(tmp_path / "scenes.txt", ["scene0000_00"])
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 0),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            config=config,
            scene_list=scenes,
            native_root=native,
            diagnostics_root=diagnostics,
            log_root=tmp_path / "logs",
        ),
    )
    assert runner.main() == 1


def test_r2_refuses_v1_sidecar_resume_in_visibility_v2_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    diagnostics = tmp_path / "diagnostics"
    native = tmp_path / "native"
    diagnostics.mkdir()
    native.mkdir()
    scene = "scene0000_00"
    (native / f"{scene}_boxes.pkl").touch()
    runner.np.savez_compressed(
        diagnostics / f"{scene}{runner.R2_SIDECAR_SUFFIX}",
        schema=runner.np.asarray("boxfusion.openbox_smov_r2_shadow.v1"),
    )
    config = _config(tmp_path / "config.yaml", diagnostics)
    scenes = _scene_list(tmp_path / "scenes.txt", [scene])
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            config=config,
            scene_list=scenes,
            native_root=native,
            diagnostics_root=diagnostics,
            log_root=tmp_path / "logs",
        ),
    )
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
