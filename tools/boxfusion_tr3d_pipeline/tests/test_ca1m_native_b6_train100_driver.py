from __future__ import annotations

import fcntl
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
from argparse import Namespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tools" / "build_ca1m_native_b6_train100.py"
HOST = "https://ml-site.cdn-apple.com/datasets/ca1m"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract(root: Path, *, scenes: int = 100) -> tuple[Path, Path, Path, list[str]]:
    ids = [f"{42_000_000 + index:08d}" for index in range(scenes)]
    scene_ids = root / "scene_ids.txt"
    scene_ids.write_text("".join(f"{value}\n" for value in ids))
    val = root / "val.txt"
    val.write_text(f"{HOST}/val/ca1m-val-51000000.tar\n")
    manifest = root / "subset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "boxfusion.ca1m_native_b6_train_subset.v1",
                "source": {"train_val_overlap": [], "val_url_list_sha256": _sha(val)},
                "safety_contract": {
                    "train_only": True,
                    "validation_ground_truth_access": False,
                    "validation_scene_overlap_count": 0,
                },
                "entries": [
                    {
                        "rank": rank,
                        "scene_id": value,
                        "selection_key_sha256": f"{rank:064x}",
                        "tar_name": f"ca1m-train-{value}.tar",
                        "url": f"{HOST}/train/ca1m-train-{value}.tar",
                    }
                    for rank, value in enumerate(ids)
                ],
            }
        )
    )
    return manifest, scene_ids, val, ids


def _run(
    root: Path,
    manifest: Path,
    scene_ids: Path,
    val: Path,
    *,
    mode: str = "preflight",
    lock: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    harmless = root / "harmless.py"
    harmless.write_text("print('{}')\n")
    return subprocess.run(
        [
            sys.executable,
            str(DRIVER),
            "--mode",
            mode,
            "--subset-manifest",
            str(manifest),
            "--scene-ids",
            str(scene_ids),
            "--val-url-list",
            str(val),
            "--tar-root",
            str(root / "tars"),
            "--output-root",
            str(root / "output"),
            "--report-root",
            str(root / "reports"),
            "--lock",
            str(lock or root / "driver.lock"),
            "--python",
            sys.executable,
            "--builder",
            str(harmless),
            "--auditor",
            str(harmless),
        ],
        text=True,
        capture_output=True,
    )


def test_preflight_requires_exact100_and_never_builds(tmp_path: Path) -> None:
    manifest, scene_ids, val, _ = _contract(tmp_path)
    (tmp_path / "tars").mkdir()
    result = _run(tmp_path, manifest, scene_ids, val)
    assert result.returncode == 3, result.stderr
    report = json.loads(result.stdout)
    assert report["build_started"] is False
    assert report["readiness"]["counts"] == {
        "absent": 100,
        "complete": 0,
        "expected": 100,
        "hidden_tar_artifacts_ignored": 0,
        "invalid": 0,
        "partial_artifacts_ignored": 0,
    }
    assert not (tmp_path / "output").exists()


def test_wrong_scene_count_fails_closed(tmp_path: Path) -> None:
    manifest, scene_ids, val, _ = _contract(tmp_path, scenes=99)
    (tmp_path / "tars").mkdir()
    result = _run(tmp_path, manifest, scene_ids, val)
    assert result.returncode != 0
    assert "exactly 100" in result.stderr


def test_single_instance_lock_fails_closed(tmp_path: Path) -> None:
    manifest, scene_ids, val, _ = _contract(tmp_path)
    (tmp_path / "tars").mkdir()
    lock_path = tmp_path / "driver.lock"
    with lock_path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(tmp_path, manifest, scene_ids, val, lock=lock_path)
    assert result.returncode != 0
    assert "holds the lock" in result.stderr


def test_hidden_and_partial_tar_artifacts_are_reported_not_counted(tmp_path: Path) -> None:
    manifest, scene_ids, val, ids = _contract(tmp_path)
    tar_root = tmp_path / "tars"
    tar_root.mkdir()
    (tar_root / f"ca1m-train-{ids[0]}.tar.part").write_bytes(b"partial")
    (tar_root / ".download.lock").write_text("hidden")
    result = _run(tmp_path, manifest, scene_ids, val)
    assert result.returncode == 3, result.stderr
    readiness = json.loads(result.stdout)["readiness"]
    assert readiness["counts"]["partial_artifacts_ignored"] == 1
    assert readiness["counts"]["hidden_tar_artifacts_ignored"] == 1
    assert readiness["counts"]["complete"] == 0


def _load_driver():
    spec = importlib.util.spec_from_file_location("ca1m_train100_driver_test", DRIVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _all_tars(root: Path, ids: list[str]) -> Path:
    tar_root = root / "tars"
    tar_root.mkdir()
    payload = root / "payload.txt"
    payload.write_text("train-only\n")
    for scene_id in ids:
        with tarfile.open(tar_root / f"ca1m-train-{scene_id}.tar", "w") as archive:
            archive.add(payload, arcname=f"{scene_id}/first.txt")
    return tar_root


def _args(
    root: Path, manifest: Path, scene_ids: Path, val: Path, tar_root: Path
) -> Namespace:
    return Namespace(
        mode="run",
        subset_manifest=manifest,
        scene_ids=scene_ids,
        val_url_list=val,
        tar_root=tar_root,
        output_root=root / "output",
        report_root=root / "reports",
        lock=root / "driver.lock",
        python=Path(sys.executable),
        builder=root / "unused_builder.py",
        auditor=root / "unused_auditor.py",
        pixel_check="none",
    )


def test_run_is_resume_safe_exact100_and_frozen_completion(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _load_driver()
    manifest, scene_ids, val, ids = _contract(tmp_path)
    tar_root = _all_tars(tmp_path, ids)
    args = _args(tmp_path, manifest, scene_ids, val, tar_root)
    args.output_root.mkdir()
    hidden = args.output_root / ".42000001.failed.keep"
    hidden.mkdir()
    existing = args.output_root / ids[0]
    existing.mkdir()
    (existing / "derived_train_gt_manifest.json").write_text(
        json.dumps({"schema": driver.SCENE_SCHEMA, "scene_id": ids[0]})
    )
    built: list[str] = []

    def fake_build(_args, scene_id: str) -> None:
        built.append(scene_id)
        scene = _args.output_root / scene_id
        scene.mkdir()
        (scene / "derived_train_gt_manifest.json").write_text(
            json.dumps({"schema": driver.SCENE_SCHEMA, "scene_id": scene_id})
        )

    def fake_audit(_args, scene_id: str) -> dict:
        return {
            "ok": True,
            "scene_id": scene_id,
            "train_only": True,
            "validation_scene_overlap": False,
            "validation_ground_truth_access": False,
            "source_tar_sha256": hashlib.sha256(f"tar:{scene_id}".encode()).hexdigest(),
            "derived_train_gt_sha256": hashlib.sha256(f"gt:{scene_id}".encode()).hexdigest(),
            "counts": {"frames": 1, "derived_train_gt_boxes": 1},
        }

    monkeypatch.setattr(driver, "build_scene", fake_build)
    monkeypatch.setattr(driver, "audit_scene", fake_audit)
    first, status = driver.run(args)
    assert status == 0 and first["ok"] is True
    assert first["counts"] == {
        "built_this_run": 99,
        "exact_scenes": 100,
        "existing_full_audited_skip": 1,
        "hidden_output_artifacts_ignored": 1,
    }
    assert built == ids[1:]
    assert hidden.is_dir()
    completion = (args.report_root / "exact100_completion.json").read_bytes()

    built.clear()
    second, status = driver.run(args)
    assert status == 0 and second["counts"]["built_this_run"] == 0
    assert second["counts"]["existing_full_audited_skip"] == 100
    assert built == []
    assert (args.report_root / "exact100_completion.json").read_bytes() == completion


def test_tampered_frozen_completion_is_rejected_exactly(
    tmp_path: Path, monkeypatch
) -> None:
    driver = _load_driver()
    manifest, scene_ids, val, ids = _contract(tmp_path)
    args = _args(tmp_path, manifest, scene_ids, val, _all_tars(tmp_path, ids))

    def fake_build(_args, scene_id: str) -> None:
        scene = _args.output_root / scene_id
        scene.mkdir(parents=True)
        (scene / "derived_train_gt_manifest.json").write_text(
            json.dumps({"schema": driver.SCENE_SCHEMA, "scene_id": scene_id})
        )

    def fake_audit(_args, scene_id: str) -> dict:
        return {
            "ok": True,
            "scene_id": scene_id,
            "train_only": True,
            "validation_scene_overlap": False,
            "validation_ground_truth_access": False,
            "source_tar_sha256": hashlib.sha256(f"tar:{scene_id}".encode()).hexdigest(),
            "derived_train_gt_sha256": hashlib.sha256(f"gt:{scene_id}".encode()).hexdigest(),
            "counts": {"frames": 1, "derived_train_gt_boxes": 1},
        }

    monkeypatch.setattr(driver, "build_scene", fake_build)
    monkeypatch.setattr(driver, "audit_scene", fake_audit)
    first, status = driver.run(args)
    assert status == 0 and first["ok"] is True
    completion = args.report_root / "exact100_completion.json"
    latest_before = (args.report_root / "latest_run.json").read_bytes()
    completion.chmod(0o644)
    completion.write_bytes(completion.read_bytes() + b"\n")
    completion.chmod(0o444)

    try:
        driver.run(args)
    except ValueError as error:
        assert "refusing to change frozen completion report" in str(error)
    else:
        raise AssertionError("tampered frozen completion was accepted")
    # Freeze fails before latest_run is replaced, preserving the last valid run
    # report for diagnosis.
    assert (args.report_root / "latest_run.json").read_bytes() == latest_before


def test_unexpected_numeric_output_fails_before_build(tmp_path: Path, monkeypatch) -> None:
    driver = _load_driver()
    manifest, scene_ids, val, ids = _contract(tmp_path)
    args = _args(tmp_path, manifest, scene_ids, val, _all_tars(tmp_path, ids))
    args.output_root.mkdir()
    (args.output_root / "99999999").mkdir()
    called = False

    def forbidden(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("builder must not run")

    monkeypatch.setattr(driver, "build_scene", forbidden)
    try:
        driver.run(args)
    except ValueError as error:
        assert "outside frozen100" in str(error)
    else:
        raise AssertionError("unexpected numeric output was accepted")
    assert called is False
