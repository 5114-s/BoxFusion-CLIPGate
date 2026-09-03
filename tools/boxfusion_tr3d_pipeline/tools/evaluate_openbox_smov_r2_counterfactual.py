#!/usr/bin/env python3
"""Run a fail-closed paired ScanNet AP evaluation for OpenBox-SMOV R2.

This driver consumes two already-materialized prediction trees.  It never
materializes, edits, or otherwise interprets predictions.  Both trees are
evaluated by the repository's unchanged ``evaluation/eval_scannet.py`` with
the same fixed command, scene list, ground truth, and seed.  The only command
differences are the prediction root and the isolated evaluator dump root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "evaluation/eval_scannet.py"
SCHEMA = "boxfusion.openbox_smov_r2_counterfactual_paired_eval.v1"
THRESHOLDS = (
    ("AP15", "0.150000"),
    ("AP25", "0.250000"),
    ("AP50", "0.500000"),
)
METRIC_NAMES = ("mAP", "APrec", "ARecall")

_SCENE_RE = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_BATCH_RE = re.compile(r"^Eval batch: ([0-9]+)$", re.MULTILINE)
_SCAN_RE = re.compile(r"^scan_idx (scene[0-9]{4}_[0-9]{2})$", re.MULTILINE)
_IOU_RE = re.compile(
    r"^-+ iou_thresh: ([0-9]+(?:[.][0-9]+)?) -+$", re.MULTILINE
)
_METRIC_RE = re.compile(
    r"^eval (mAP|APrec|ARecall): ([0-9]+(?:[.][0-9]+)?)$",
    re.MULTILINE,
)
_GT_SUFFIXES = ("_vert.npy", "_ins_label.npy", "_sem_label.npy", "_bbox.npy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.is_symlink() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {resolved}")
    return resolved


def _directory(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _executable(path: Path, label: str) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError(f"{label} must resolve to an executable file: {path}")
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def read_scenes(path: Path, expected_scene_count: int = 100) -> tuple[Path, tuple[str, ...]]:
    if isinstance(expected_scene_count, bool) or expected_scene_count < 1:
        raise ValueError("expected scene count must be a positive integer")
    scene_list = _regular_file(path, "scene list")
    scenes = tuple(
        line.strip()
        for line in scene_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(scenes) != expected_scene_count:
        raise ValueError(
            f"scene list must contain exactly {expected_scene_count} scenes; "
            f"found {len(scenes)}"
        )
    if len(set(scenes)) != len(scenes):
        raise ValueError("scene list contains duplicate scene IDs")
    invalid = [scene for scene in scenes if _SCENE_RE.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"scene list contains invalid ScanNet IDs: {invalid}")
    return scene_list, scenes


def exact_prediction_files(
    root: Path, scenes: Sequence[str], label: str
) -> tuple[Path, dict[str, Path]]:
    resolved = _directory(root, label)
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    actual = {entry.name for entry in resolved.iterdir()}
    if actual != expected:
        raise ValueError(
            f"{label} file set differs: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    files = {
        scene: _regular_file(
            resolved / f"{scene}_boxes.pkl", f"{label} prediction for {scene}"
        )
        for scene in scenes
    }
    return resolved, files


def prediction_inventory(files: Mapping[str, Path]) -> dict[str, Any]:
    digest = hashlib.sha256()
    total_bytes = 0
    for scene, path in files.items():
        size = path.stat().st_size
        file_hash = sha256_file(path)
        total_bytes += size
        digest.update(f"{scene}\t{size}\t{file_hash}\n".encode("utf-8"))
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def validate_prediction_inventory(
    root: Path,
    scenes: Sequence[str],
    label: str,
    expected: Mapping[str, Any],
) -> None:
    _, files = exact_prediction_files(root, scenes, label)
    if prediction_inventory(files) != dict(expected):
        raise RuntimeError(f"{label} changed during paired evaluation")


def validate_dataset_inputs(
    scans_root: Path, gt_root: Path, scenes: Sequence[str]
) -> tuple[Path, Path]:
    scans = _directory(scans_root, "ScanNet scans root")
    ground_truth = _directory(gt_root, "ScanNet prepared GT root")
    for scene in scenes:
        _regular_file(scans / scene / f"{scene}.txt", f"{scene} ScanNet metadata")
        for suffix in _GT_SUFFIXES:
            _regular_file(
                ground_truth / f"{scene}{suffix}",
                f"{scene} prepared GT {suffix}",
            )
    return scans, ground_truth


def parse_evaluator_output(
    text: str, scenes: Sequence[str]
) -> dict[str, dict[str, float]]:
    batches = [int(value) for value in _BATCH_RE.findall(text)]
    if batches != list(range(len(scenes))):
        raise ValueError(
            "official evaluator batch sequence differs: "
            f"observed={batches}, expected={list(range(len(scenes)))}"
        )
    observed_scenes = _SCAN_RE.findall(text)
    if observed_scenes != list(scenes):
        raise ValueError(
            "official evaluator scene order differs: "
            f"observed={observed_scenes}, expected={list(scenes)}"
        )
    observed_thresholds = _IOU_RE.findall(text)
    expected_thresholds = [raw for _, raw in THRESHOLDS]
    if observed_thresholds != expected_thresholds:
        raise ValueError(
            "official evaluator IoU threshold sequence differs: "
            f"observed={observed_thresholds}, expected={expected_thresholds}"
        )
    rows = _METRIC_RE.findall(text)
    expected_names = [name for _ in THRESHOLDS for name in METRIC_NAMES]
    if [name for name, _ in rows] != expected_names:
        raise ValueError("expected exactly three ordered official metric triplets")
    result: dict[str, dict[str, float]] = {}
    for index, (threshold, _) in enumerate(THRESHOLDS):
        chunk = rows[index * len(METRIC_NAMES) : (index + 1) * len(METRIC_NAMES)]
        values = {name: float(raw) for name, raw in chunk}
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
            raise ValueError(f"official evaluator emitted invalid {threshold} metrics")
        result[threshold] = values
    return result


def metric_delta(
    native: Mapping[str, Mapping[str, float]],
    counterfactual: Mapping[str, Mapping[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    delta = {
        threshold: {
            name: counterfactual[threshold][name] - native[threshold][name]
            for name in METRIC_NAMES
        }
        for threshold, _ in THRESHOLDS
    }
    percentage_points = {
        threshold: {name: value * 100.0 for name, value in row.items()}
        for threshold, row in delta.items()
    }
    return delta, percentage_points


def _write_create_only(path: Path, data: bytes) -> Path:
    raw = Path(path)
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"refusing existing output: {raw}")
    target = raw.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # Publish an already read-only inode so a post-link chmod failure
        # cannot leave a visible, writable create-only artifact.
        os.chmod(temporary_name, 0o444)
        try:
            os.link(temporary_name, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing existing output: {target}") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return target


def evaluator_argv(
    *,
    python: Path,
    evaluator: Path,
    scans_root: Path,
    gt_root: Path,
    scene_list: Path,
    prediction_root: Path,
    dump_root: Path,
    seed: int,
    gpu: int,
) -> list[str]:
    return [
        str(python),
        str(evaluator),
        "--dataset",
        "scannet",
        "--data_path",
        str(scans_root),
        "--gt_root",
        str(gt_root),
        "--dump_dir",
        str(dump_root),
        "--num_point",
        "40000",
        "--batch_size",
        "1",
        "--cluster_sampling",
        "seed_fps",
        "--ap_iou_thresholds",
        "0.15,0.25,0.5",
        "--use_3d_nms",
        "--use_cls_nms",
        "--per_class_proposal",
        "--num_workers",
        "0",
        "--gpu",
        str(gpu),
        "--seed",
        str(seed),
        "--scene_list",
        str(scene_list),
        "--pred_root",
        str(prediction_root),
    ]


def validate_paired_argv(native: Sequence[str], counterfactual: Sequence[str]) -> None:
    left, right = list(native), list(counterfactual)
    if len(left) != len(right):
        raise AssertionError("paired evaluator commands have different lengths")
    for flag in ("--dump_dir", "--pred_root"):
        if left.count(flag) != 1 or right.count(flag) != 1:
            raise AssertionError(f"paired evaluator command does not contain one {flag}")
        left[left.index(flag) + 1] = f"<{flag}>"
        right[right.index(flag) + 1] = f"<{flag}>"
    if left != right:
        raise AssertionError(
            "paired evaluator commands differ outside --dump_dir/--pred_root"
        )


def run_evaluator(
    *,
    argv: Sequence[str],
    evaluator_root: Path,
    runtime_root: Path,
    log_path: Path,
    scenes: Sequence[str],
) -> tuple[dict[str, dict[str, float]], str]:
    runtime_root.mkdir(parents=False, exist_ok=False)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONUNBUFFERED": "1",
            "TMPDIR": str(runtime_root),
            "TMP": str(runtime_root),
            "TEMP": str(runtime_root),
            "MPLCONFIGDIR": str(runtime_root / "mplconfig"),
        }
    )
    process = subprocess.run(
        list(argv),
        cwd=evaluator_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
        shell=False,
    )
    stdout = process.stdout if isinstance(process.stdout, str) else ""
    saved_log = _write_create_only(log_path, stdout.encode("utf-8"))
    if process.returncode != 0:
        raise RuntimeError(
            f"official ScanNet evaluator failed ({process.returncode}); "
            f"inspect {saved_log}"
        )
    return parse_evaluator_output(stdout, scenes), sha256_file(saved_log)


def evaluate_pair(args: argparse.Namespace) -> dict[str, Any]:
    expected_count = int(args.expected_scene_count)
    if expected_count < 1:
        raise ValueError("expected scene count must be positive")
    if args.seed < 0 or args.seed > 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32-1]")
    if args.gpu < 0:
        raise ValueError("gpu must be non-negative")

    raw_report = Path(args.report)
    raw_log_root = Path(args.log_root)
    if raw_report.exists() or raw_report.is_symlink():
        raise FileExistsError(f"refusing existing report: {raw_report}")
    if raw_log_root.exists() or raw_log_root.is_symlink():
        raise FileExistsError(f"refusing existing log root: {raw_log_root}")
    report_path = raw_report.resolve()
    log_root = raw_log_root.resolve()
    if report_path == log_root:
        raise ValueError("report path and log root must differ")

    scene_list, scenes = read_scenes(args.scene_list, expected_count)
    native_root, native_files = exact_prediction_files(
        args.native_root, scenes, "native prediction root"
    )
    counterfactual_root, counterfactual_files = exact_prediction_files(
        args.counterfactual_root, scenes, "counterfactual prediction root"
    )
    if native_root == counterfactual_root:
        raise ValueError("native and counterfactual prediction roots must differ")
    for output, label in ((log_root, "log root"), (report_path, "report")):
        for prediction_root, prediction_label in (
            (native_root, "native prediction root"),
            (counterfactual_root, "counterfactual prediction root"),
        ):
            if _is_within(output, prediction_root):
                raise ValueError(f"{label} must not be inside {prediction_label}")

    scans_root, gt_root = validate_dataset_inputs(
        args.scans_root, args.gt_root, scenes
    )
    evaluator = _regular_file(EVALUATOR, "fixed ScanNet evaluator")
    python = _executable(args.python, "evaluation Python")
    evaluator_hash = sha256_file(evaluator)
    scene_list_hash = sha256_file(scene_list)
    native_inventory = prediction_inventory(native_files)
    counterfactual_inventory = prediction_inventory(counterfactual_files)

    log_root.parent.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(exist_ok=False)
    native_log = log_root / "native.log"
    counterfactual_log = log_root / "counterfactual.log"
    native_argv = evaluator_argv(
        python=python,
        evaluator=evaluator,
        scans_root=scans_root,
        gt_root=gt_root,
        scene_list=scene_list,
        prediction_root=native_root,
        dump_root=log_root / "native_eval_dump",
        seed=args.seed,
        gpu=args.gpu,
    )
    counterfactual_argv = evaluator_argv(
        python=python,
        evaluator=evaluator,
        scans_root=scans_root,
        gt_root=gt_root,
        scene_list=scene_list,
        prediction_root=counterfactual_root,
        dump_root=log_root / "counterfactual_eval_dump",
        seed=args.seed,
        gpu=args.gpu,
    )
    validate_paired_argv(native_argv, counterfactual_argv)

    native_metrics, native_log_hash = run_evaluator(
        argv=native_argv,
        evaluator_root=evaluator.parent,
        runtime_root=log_root / "native_runtime",
        log_path=native_log,
        scenes=scenes,
    )
    if sha256_file(evaluator) != evaluator_hash:
        raise RuntimeError("fixed ScanNet evaluator changed during native evaluation")
    validate_prediction_inventory(
        native_root, scenes, "native prediction root", native_inventory
    )
    validate_prediction_inventory(
        counterfactual_root,
        scenes,
        "counterfactual prediction root",
        counterfactual_inventory,
    )

    counterfactual_metrics, counterfactual_log_hash = run_evaluator(
        argv=counterfactual_argv,
        evaluator_root=evaluator.parent,
        runtime_root=log_root / "counterfactual_runtime",
        log_path=counterfactual_log,
        scenes=scenes,
    )
    if sha256_file(evaluator) != evaluator_hash:
        raise RuntimeError("fixed ScanNet evaluator changed during counterfactual evaluation")
    if sha256_file(scene_list) != scene_list_hash:
        raise RuntimeError("scene list changed during paired evaluation")
    validate_prediction_inventory(
        native_root, scenes, "native prediction root", native_inventory
    )
    validate_prediction_inventory(
        counterfactual_root,
        scenes,
        "counterfactual prediction root",
        counterfactual_inventory,
    )

    delta, delta_points = metric_delta(native_metrics, counterfactual_metrics)
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": True,
        "dataset": "scannet",
        "dataset_split": (
            "official_validation_full100"
            if expected_count == 100
            else f"test_subset_{expected_count}"
        ),
        "scene_count": len(scenes),
        "paired_official_evaluation": True,
        "ground_truth_access": True,
        "training_invoked": False,
        "materialization_invoked": False,
        "activation_authorized": False,
        "scene_list": {
            "path": str(scene_list),
            "sha256": scene_list_hash,
        },
        "prediction_roots": {
            "native": {"path": str(native_root), **native_inventory},
            "counterfactual": {
                "path": str(counterfactual_root),
                **counterfactual_inventory,
            },
        },
        "evaluation_contract": {
            "evaluator": {
                "path": str(evaluator),
                "sha256": evaluator_hash,
                "unchanged_during_pair": True,
            },
            "python": str(python),
            "scans_root": str(scans_root),
            "gt_root": str(gt_root),
            "seed": int(args.seed),
            "gpu": int(args.gpu),
            "thresholds": [0.15, 0.25, 0.50],
            "native_argv": native_argv,
            "counterfactual_argv": counterfactual_argv,
            "only_argv_differences": ["--dump_dir", "--pred_root"],
        },
        "logs": {
            "native": {
                "path": str(native_log.resolve()),
                "sha256": native_log_hash,
            },
            "counterfactual": {
                "path": str(counterfactual_log.resolve()),
                "sha256": counterfactual_log_hash,
            },
        },
        "native": native_metrics,
        "counterfactual": counterfactual_metrics,
        "delta": delta,
        "delta_percentage_points": delta_points,
    }
    _write_create_only(
        report_path,
        (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    return report


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--native-root", type=Path, required=True)
    value.add_argument("--counterfactual-root", type=Path, required=True)
    value.add_argument("--scans-root", type=Path, required=True)
    value.add_argument("--gt-root", type=Path, required=True)
    value.add_argument("--log-root", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    value.add_argument("--python", type=Path, default=Path(sys.executable))
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument(
        "--expected-scene-count",
        type=_positive_int,
        default=100,
        help="Expected scene count; keep 100 for formal runs, override only in tests.",
    )
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = evaluate_pair(args)
    print(
        json.dumps(
            {
                "complete": True,
                "scene_count": report["scene_count"],
                "AP15_native_counterfactual_delta": [
                    report["native"]["AP15"]["mAP"],
                    report["counterfactual"]["AP15"]["mAP"],
                    report["delta"]["AP15"]["mAP"],
                ],
                "AP25_native_counterfactual_delta": [
                    report["native"]["AP25"]["mAP"],
                    report["counterfactual"]["AP25"]["mAP"],
                    report["delta"]["AP25"]["mAP"],
                ],
                "AP50_native_counterfactual_delta": [
                    report["native"]["AP50"]["mAP"],
                    report["counterfactual"]["AP50"]["mAP"],
                    report["delta"]["AP50"]["mAP"],
                ],
                "report": str(Path(args.report).resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
