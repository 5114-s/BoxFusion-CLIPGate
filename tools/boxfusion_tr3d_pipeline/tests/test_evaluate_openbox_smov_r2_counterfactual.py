from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import evaluate_openbox_smov_r2_counterfactual as paired  # noqa: E402


def _case(
    tmp_path: Path, scene_count: int = 2
) -> tuple[argparse.Namespace, tuple[str, ...]]:
    scenes = tuple(f"scene{index:04d}_00" for index in range(scene_count))
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    native_root = tmp_path / "native"
    counterfactual_root = tmp_path / "counterfactual"
    scans_root = tmp_path / "scans"
    gt_root = tmp_path / "gt"
    for root in (native_root, counterfactual_root, scans_root, gt_root):
        root.mkdir()
    for scene in scenes:
        (native_root / f"{scene}_boxes.pkl").write_bytes(
            f"native-{scene}".encode("utf-8")
        )
        (counterfactual_root / f"{scene}_boxes.pkl").write_bytes(
            f"counterfactual-{scene}".encode("utf-8")
        )
        scene_root = scans_root / scene
        scene_root.mkdir()
        (scene_root / f"{scene}.txt").write_text(
            "axisAlignment = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n",
            encoding="utf-8",
        )
        for suffix in paired._GT_SUFFIXES:
            (gt_root / f"{scene}{suffix}").write_bytes(b"gt")
    return (
        argparse.Namespace(
            scene_list=scene_list,
            native_root=native_root,
            counterfactual_root=counterfactual_root,
            scans_root=scans_root,
            gt_root=gt_root,
            log_root=tmp_path / "logs",
            report=tmp_path / "report.json",
            python=Path(sys.executable),
            seed=7,
            gpu=2,
            expected_scene_count=scene_count,
        ),
        scenes,
    )


def _output(
    scenes: tuple[str, ...],
    metrics: tuple[tuple[float, float, float], ...],
) -> str:
    lines: list[str] = []
    for index, scene in enumerate(scenes):
        lines.extend((f"Eval batch: {index}", f"scan_idx {scene}"))
    for (_, raw_threshold), values in zip(paired.THRESHOLDS, metrics):
        lines.append(f"---------- iou_thresh: {raw_threshold} ----------")
        lines.extend(
            f"eval {name}: {value:.6f}"
            for name, value in zip(paired.METRIC_NAMES, values)
        )
    return "\n".join(lines) + "\n"


def test_runs_fixed_paired_commands_and_writes_delta_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, scenes = _case(tmp_path)
    native_values = ((0.40, 0.50, 0.60), (0.30, 0.40, 0.50), (0.20, 0.30, 0.40))
    counterfactual_values = (
        (0.42, 0.51, 0.62),
        (0.33, 0.43, 0.53),
        (0.25, 0.35, 0.45),
    )
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        prediction_root = Path(argv[argv.index("--pred_root") + 1])
        values = (
            native_values
            if prediction_root == args.native_root.resolve()
            else counterfactual_values
        )
        return subprocess.CompletedProcess(argv, 0, stdout=_output(scenes, values))

    monkeypatch.setattr(paired.subprocess, "run", fake_run)
    report = paired.evaluate_pair(args)

    assert len(calls) == 2
    native_argv, native_kwargs = calls[0]
    counterfactual_argv, counterfactual_kwargs = calls[1]
    assert isinstance(native_argv, list) and isinstance(counterfactual_argv, list)
    paired.validate_paired_argv(native_argv, counterfactual_argv)
    for argv in (native_argv, counterfactual_argv):
        assert argv[0] == str(Path(sys.executable).resolve())
        assert argv[1] == str(paired.EVALUATOR.resolve())
        assert argv[argv.index("--seed") + 1] == "7"
        assert argv[argv.index("--gpu") + 1] == "2"
        assert argv[argv.index("--batch_size") + 1] == "1"
        assert argv[argv.index("--num_point") + 1] == "40000"
        assert argv[argv.index("--ap_iou_thresholds") + 1] == "0.15,0.25,0.5"
    for kwargs in (native_kwargs, counterfactual_kwargs):
        assert kwargs["shell"] is False
        assert kwargs["check"] is False
        assert kwargs["cwd"] == paired.EVALUATOR.parent.resolve()
        assert "PYTHONPATH" not in kwargs["env"]

    assert report["schema"] == paired.SCHEMA
    assert report["complete"] is True
    assert report["scene_count"] == 2
    assert report["materialization_invoked"] is False
    assert report["native"]["AP15"] == {
        "mAP": 0.4,
        "APrec": 0.5,
        "ARecall": 0.6,
    }
    assert report["counterfactual"]["AP50"]["mAP"] == 0.25
    assert report["delta"]["AP25"]["mAP"] == pytest.approx(0.03)
    assert report["delta"]["AP25"]["APrec"] == pytest.approx(0.03)
    assert report["delta"]["AP25"]["ARecall"] == pytest.approx(0.03)
    assert report["delta_percentage_points"]["AP50"]["mAP"] == pytest.approx(5.0)
    assert json.loads(args.report.read_text(encoding="utf-8")) == report
    for path in (args.report, args.log_root / "native.log", args.log_root / "counterfactual.log"):
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0


@pytest.mark.parametrize("root_name", ("native_root", "counterfactual_root"))
def test_requires_exact_prediction_file_sets_before_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_name: str,
) -> None:
    args, _ = _case(tmp_path)
    root = Path(getattr(args, root_name))
    (root / "extra_boxes.pkl").write_bytes(b"extra")
    called = False

    def forbidden(*_args: object, **_kwargs: object) -> None:
        nonlocal called
        called = True
        raise AssertionError("subprocess must not run")

    monkeypatch.setattr(paired.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match="file set differs"):
        paired.evaluate_pair(args)
    assert called is False
    assert not args.log_root.exists()
    assert not args.report.exists()


def test_expected_scene_count_is_strict_and_test_override_is_explicit(
    tmp_path: Path,
) -> None:
    args, _ = _case(tmp_path)
    args.expected_scene_count = 100
    with pytest.raises(ValueError, match="exactly 100 scenes"):
        paired.evaluate_pair(args)
    parsed = paired.parser().parse_args(
        [
            "--scene-list",
            str(args.scene_list),
            "--native-root",
            str(args.native_root),
            "--counterfactual-root",
            str(args.counterfactual_root),
            "--scans-root",
            str(args.scans_root),
            "--gt-root",
            str(args.gt_root),
            "--log-root",
            str(args.log_root),
            "--report",
            str(args.report),
        ]
    )
    assert parsed.expected_scene_count == 100


@pytest.mark.parametrize("collision", ("report", "log_root"))
def test_existing_report_or_log_root_fails_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    args, _ = _case(tmp_path)
    path = Path(getattr(args, collision))
    if collision == "report":
        path.write_text("sealed", encoding="utf-8")
    else:
        path.mkdir()
    monkeypatch.setattr(
        paired.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )
    with pytest.raises(FileExistsError, match="refusing existing"):
        paired.evaluate_pair(args)


@pytest.mark.parametrize("collision", ("report", "log_root"))
def test_broken_output_symlink_fails_before_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    args, _ = _case(tmp_path)
    path = Path(getattr(args, collision))
    path.symlink_to(tmp_path / f"missing-{collision}")
    monkeypatch.setattr(
        paired.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess must not run"),
    )
    with pytest.raises(FileExistsError, match="refusing existing"):
        paired.evaluate_pair(args)


def test_evaluator_failure_writes_only_create_only_failure_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = _case(tmp_path)
    calls = 0

    def fail(argv: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        return subprocess.CompletedProcess(argv, 9, stdout="synthetic evaluator failure\n")

    monkeypatch.setattr(paired.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match=r"evaluator failed \(9\)"):
        paired.evaluate_pair(args)
    assert calls == 1
    assert (args.log_root / "native.log").read_text() == "synthetic evaluator failure\n"
    assert stat.S_IMODE((args.log_root / "native.log").stat().st_mode) & 0o222 == 0
    assert not (args.log_root / "counterfactual.log").exists()
    assert not args.report.exists()


def test_success_with_malformed_metrics_keeps_log_but_no_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, scenes = _case(tmp_path)
    malformed = "\n".join(
        [
            *(line for index, scene in enumerate(scenes) for line in (
                f"Eval batch: {index}", f"scan_idx {scene}"
            )),
            "---------- iou_thresh: 0.150000 ----------",
            "eval mAP: 0.400000",
            "eval APrec: 0.500000",
            "eval ARecall: 0.600000",
        ]
    )
    monkeypatch.setattr(
        paired.subprocess,
        "run",
        lambda argv, **_kwargs: subprocess.CompletedProcess(argv, 0, stdout=malformed),
    )
    with pytest.raises(ValueError, match="IoU threshold sequence differs"):
        paired.evaluate_pair(args)
    assert (args.log_root / "native.log").is_file()
    assert not args.report.exists()


def test_paired_argv_validator_rejects_any_third_difference(tmp_path: Path) -> None:
    args, _ = _case(tmp_path)
    common = dict(
        python=Path(sys.executable).resolve(),
        evaluator=paired.EVALUATOR.resolve(),
        scans_root=args.scans_root.resolve(),
        gt_root=args.gt_root.resolve(),
        scene_list=args.scene_list.resolve(),
        seed=0,
        gpu=0,
    )
    native = paired.evaluator_argv(
        **common,
        prediction_root=args.native_root.resolve(),
        dump_root=tmp_path / "native_dump",
    )
    counterfactual = paired.evaluator_argv(
        **common,
        prediction_root=args.counterfactual_root.resolve(),
        dump_root=tmp_path / "counterfactual_dump",
    )
    counterfactual[counterfactual.index("--seed") + 1] = "1"
    with pytest.raises(AssertionError, match="differ outside"):
        paired.validate_paired_argv(native, counterfactual)
