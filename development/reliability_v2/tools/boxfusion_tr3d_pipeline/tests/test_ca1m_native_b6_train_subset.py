from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "prepare_ca1m_native_b6_train_subset.py"
HOST = "https://ml-site.cdn-apple.com/datasets/ca1m"


def _write_lists(root: Path, *, overlap: bool = False) -> tuple[Path, Path]:
    train = root / "train.txt"
    val = root / "val.txt"
    train_ids = [f"{42_000_000 + index:08d}" for index in range(12)]
    val_ids = [train_ids[0] if overlap else "51000000", "51000001", "51000002"]
    train.write_text(
        "".join(f"{HOST}/train/ca1m-train-{scene}.tar\n" for scene in train_ids)
    )
    val.write_text(
        "".join(f"{HOST}/val/ca1m-val-{scene}.tar\n" for scene in val_ids)
    )
    return train, val


def _run(
    tmp_path: Path,
    train: Path,
    val: Path,
    *extra: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--train-url-list",
            str(train),
            "--val-url-list",
            str(val),
            "--subset-size",
            "5",
            "--namespace",
            "unit-test-v1",
            "--output-dir",
            str(tmp_path / "manifest"),
            "--download-root",
            str(tmp_path / "tars"),
            *extra,
        ],
        text=True,
        capture_output=True,
    )


def test_deterministic_scene_selection_and_zero_val_overlap(tmp_path: Path) -> None:
    train, val = _write_lists(tmp_path)
    first = _run(tmp_path, train, val)
    assert first.returncode == 0, first.stderr
    output = tmp_path / "manifest"
    payload = json.loads((output / "subset_manifest.json").read_text())
    expected = sorted(
        [f"{42_000_000 + index:08d}" for index in range(12)],
        key=lambda scene: (hashlib.sha256(f"unit-test-v1\0{scene}".encode()).hexdigest(), scene),
    )[:5]
    assert [row["scene_id"] for row in payload["entries"]] == expected
    assert payload["safety_contract"] == {
        "automatic_download": False,
        "train_only": True,
        "training_started": False,
        "validation_ground_truth_access": False,
        "validation_scene_overlap_count": 0,
    }
    readiness = json.loads((output / "readiness.json").read_text())
    assert readiness["counts"] == {"absent": 5, "complete": 0, "expected": 5, "partial": 0}
    assert not readiness["ready"]
    before = (output / "subset_manifest.json").read_bytes()
    second = _run(tmp_path, train, val)
    assert second.returncode == 0, second.stderr
    assert (output / "subset_manifest.json").read_bytes() == before
    for name in ("subset_manifest.json", "scene_ids.txt", "urls.txt", "subset_manifest.tsv"):
        digest, recorded_name = (output / f"{name}.sha256").read_text().split()
        assert recorded_name == name
        assert digest == hashlib.sha256((output / name).read_bytes()).hexdigest()


def test_source_train_val_overlap_fails_closed(tmp_path: Path) -> None:
    train, val = _write_lists(tmp_path, overlap=True)
    result = _run(tmp_path, train, val)
    assert result.returncode != 0
    assert "overlap" in result.stderr
    assert not (tmp_path / "manifest" / "subset_manifest.json").exists()


def test_frozen_manifest_refuses_selection_change(tmp_path: Path) -> None:
    train, val = _write_lists(tmp_path)
    assert _run(tmp_path, train, val).returncode == 0
    result = _run(tmp_path, train, val, "--namespace", "changed-v2")
    assert result.returncode != 0
    assert "refusing to change frozen subset artifact" in result.stderr


def test_local_tar_readiness_and_optional_file_hash(tmp_path: Path) -> None:
    train, val = _write_lists(tmp_path)
    assert _run(tmp_path, train, val).returncode == 0
    output = tmp_path / "manifest"
    scene = (output / "scene_ids.txt").read_text().splitlines()[0]
    tar_root = tmp_path / "tars"
    tar_root.mkdir()
    source = tmp_path / "payload.txt"
    source.write_text("train-only\n")
    tar_path = tar_root / f"ca1m-train-{scene}.tar"
    with tarfile.open(tar_path, "w") as archive:
        archive.add(source, arcname=f"{scene}/payload.txt")
    result = _run(tmp_path, train, val, "--hash-existing")
    assert result.returncode == 0, result.stderr
    readiness = json.loads((output / "readiness.json").read_text())
    row = next(item for item in readiness["entries"] if item["scene_id"] == scene)
    assert row["ready"]
    assert row["file_sha256"] == hashlib.sha256(tar_path.read_bytes()).hexdigest()
    assert readiness["counts"] == {"absent": 4, "complete": 1, "expected": 5, "partial": 0}
    hashes = (output / "downloaded_sha256.tsv").read_text()
    assert scene in hashes and row["file_sha256"] in hashes


def test_require_complete_is_nonzero_without_downloading(tmp_path: Path) -> None:
    train, val = _write_lists(tmp_path)
    result = _run(tmp_path, train, val, "--require-complete")
    assert result.returncode == 3
    summary = json.loads(result.stdout)
    assert not summary["ready"]
    assert not summary["tool_download_started"]
