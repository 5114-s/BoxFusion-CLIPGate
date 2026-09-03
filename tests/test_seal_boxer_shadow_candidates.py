from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools import seal_boxer_shadow_candidates as seal


CSV_HEADER = ",".join(seal.CSV_COLUMNS) + "\n"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv_row(
    *,
    frame: int,
    center: tuple[float, float, float],
    probability: float,
    instance: int,
) -> str:
    return (
        f"{frame},{center[0]},{center[1]},{center[2]},"
        f"1,0,0,0,1,2,3,discarded_label,{instance},17,{probability}\n"
    )


def _make_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    boxer_root = tmp_path / "boxer"
    (boxer_root / "ckpts").mkdir(parents=True)
    (boxer_root / "owl").mkdir()
    (boxer_root / "boxernet").mkdir()
    run_boxer = boxer_root / "run_boxer.py"
    owl_checkpoint = boxer_root / "ckpts" / seal.EXPECTED_OWL_CHECKPOINT
    owl_text_cache = boxer_root / "ckpts" / seal.EXPECTED_OWL_TEXT_CACHE
    boxer_checkpoint = boxer_root / "ckpts" / seal.EXPECTED_BOXER_CHECKPOINT
    dinov3_checkpoint = boxer_root / "ckpts" / seal.EXPECTED_DINOV3_CHECKPOINT
    taxonomy = boxer_root / "owl" / f"{seal.EXPECTED_TAXONOMY}_classes.csv"
    owl_wrapper = boxer_root / "owl" / "owl_wrapper.py"
    boxernet_source = boxer_root / "boxernet" / "boxernet.py"
    run_boxer.write_text("# frozen runner\n", encoding="utf-8")
    owl_checkpoint.write_bytes(b"frozen owl")
    owl_text_cache.write_bytes(b"frozen text cache")
    boxer_checkpoint.write_bytes(b"frozen boxer")
    dinov3_checkpoint.write_bytes(b"frozen dino")
    taxonomy.write_text("chair\ntable\n", encoding="utf-8")
    owl_wrapper.write_text(
        "class OwlWrapper:\n"
        "    def __init__(self, device='cuda', nms_iou_threshold=0.5):\n"
        "        pass\n",
        encoding="utf-8",
    )
    boxernet_source.write_text("class BoxerNet:\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(seal, "EXPECTED_OWL_SHA256", _sha(owl_checkpoint))
    monkeypatch.setattr(
        seal, "EXPECTED_OWL_TEXT_CACHE_SHA256", _sha(owl_text_cache)
    )
    monkeypatch.setattr(seal, "EXPECTED_BOXER_SHA256", _sha(boxer_checkpoint))
    monkeypatch.setattr(seal, "EXPECTED_DINOV3_SHA256", _sha(dinov3_checkpoint))
    monkeypatch.setattr(seal, "EXPECTED_TAXONOMY_SHA256", _sha(taxonomy))
    monkeypatch.setattr(seal, "EXPECTED_RUN_BOXER_SHA256", _sha(run_boxer))
    monkeypatch.setattr(seal, "EXPECTED_OWL_WRAPPER_SHA256", _sha(owl_wrapper))
    monkeypatch.setattr(
        seal, "EXPECTED_BOXERNET_SOURCE_SHA256", _sha(boxernet_source)
    )
    monkeypatch.setattr(seal, "EXPECTED_TAXONOMY_COUNT", 2)

    scene_id = "scene0001_00"
    scene_root = tmp_path / "scenes"
    color_dir = scene_root / scene_id / "frames" / "color"
    pose_dir = scene_root / scene_id / "frames" / "pose"
    color_dir.mkdir(parents=True)
    pose_dir.mkdir()
    (color_dir / "0.jpg").write_bytes(b"not decoded by sealer")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [10.0, -2.0, 3.0]
    np.savetxt(pose_dir / "0.txt", pose)

    run_root = tmp_path / "run"
    raw_dir = run_root / "boxer_raw" / scene_id
    log_dir = run_root / "scenes"
    raw_dir.mkdir(parents=True)
    log_dir.mkdir()
    raw_csv = raw_dir / "boxer_3dbbs.csv"
    raw_csv.write_text(
        CSV_HEADER
        + _csv_row(frame=0, center=(1.0, 0.0, 0.0), probability=0.2, instance=4)
        + _csv_row(frame=0, center=(2.0, 0.0, 0.0), probability=0.9, instance=5)
        + _csv_row(frame=0, center=(3.0, 0.0, 0.0), probability=0.8, instance=6),
        encoding="utf-8",
    )
    tracked_csv = raw_dir / "boxer_3dbbs_tracked.csv"
    tracked_csv.write_text(
        CSV_HEADER
        + _csv_row(frame=0, center=(-1.0, 1.0, 2.0), probability=0.7, instance=99),
        encoding="utf-8",
    )
    (raw_dir / "owl_2dbbs.csv").write_text("sealed owl rows\n", encoding="utf-8")

    expected_ckpt = boxer_checkpoint.resolve()
    namespace = (
        "Namespace("
        f"input={str((scene_root / scene_id).resolve())!r}, "
        "skip_n=25, start_n=1, max_n=1, pinhole=False, camera='rgb', "
        "detector='owl', thresh2d=0.25, thresh3d=0.5, labels=['lvisplus'], "
        "detector_hw=960, write_name='boxer', skip_viz=True, cache2d=False, "
        "cache3d=False, no_sdp=False, no_csv=False, force_cpu=False, gt2d=False, "
        "fuse=False, track=True, "
        f"ckpt={str(expected_ckpt)!r}, force_precision='bfloat16', "
        f"output_dir={str((run_root / 'boxer_raw').resolve())!r}, "
        "viz_headless=False)\n"
    )
    log_text = (
        seal.GT_ACCESS_GUARD
        + "\n"
        + namespace
        + f"ScanNetLoader: {scene_id}, 1 frames, 0 3D boxes\n"
        + "Loaded OWLv2 on cuda with 2 text prompts, precision=bfloat16\n"
        + f'Loading checkpoint from "{expected_ckpt}"\n'
        + "==> Saved 3D BBs to path\n"
        + "==> Saved 2D BBs to path\n"
        + "==> Saved 1 tracked OBBs to path\n"
        + "1/1\n"
    )
    (log_dir / f"{scene_id}.log").write_text(log_text, encoding="utf-8")

    ledger = run_root / "frozen_inputs_sha256.txt"
    schedule = tmp_path / "cache" / scene_id / "manifest.json"
    schedule.parent.mkdir(parents=True)
    schedule.write_text(
        json.dumps({"record_count": 1, "recorded_frame_ids": [0]}),
        encoding="utf-8",
    )
    ledger.write_text(
        f"{_sha(run_boxer)}  {run_boxer.resolve()}\n"
        f"{_sha(owl_wrapper)}  {owl_wrapper.resolve()}\n"
        f"{_sha(boxernet_source)}  {boxernet_source.resolve()}\n"
        f"{_sha(owl_checkpoint)}  {owl_checkpoint.resolve()}\n"
        f"{_sha(owl_text_cache)}  {owl_text_cache.resolve()}\n"
        f"{_sha(boxer_checkpoint)}  {boxer_checkpoint.resolve()}\n",
        encoding="utf-8",
    )
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(f"{_sha(dinov3_checkpoint)}  {dinov3_checkpoint.resolve()}\n")
        handle.write(f"{_sha(schedule)}  {schedule.resolve()}\n")
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene_id + "\n", encoding="utf-8")
    native_line = (
        "0" * 64
        + "  "
        + str((tmp_path / f"{scene_id}_boxes.pkl").resolve())
        + "\n"
    )
    (run_root / "native_before_sha256.txt").write_text(native_line, encoding="utf-8")
    (run_root / "native_after_sha256.txt").write_text(native_line, encoding="utf-8")
    return {
        "boxer_root": boxer_root,
        "scene_root": scene_root,
        "run_root": run_root,
        "scene_list": scene_list,
        "raw_csv": raw_csv,
        "log": log_dir / f"{scene_id}.log",
    }


def _seal(paths: dict[str, Path], output_dir: Path) -> tuple[dict, Path, Path]:
    output_json = output_dir / "sealed.json"
    output_npz = output_dir / "sealed.npz"
    manifest = seal.seal_candidates(
        run_root=paths["run_root"],
        scene_root=paths["scene_root"],
        scene_list=paths["scene_list"],
        boxer_root=paths["boxer_root"],
        output_json=output_json,
        output_npz=output_npz,
    )
    return manifest, output_json, output_npz


def test_seals_geometry_only_and_restores_scannet_world_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(seal, "MAX_PER_FRAME_CANDIDATES", 2)
    manifest, output_json, output_npz = _seal(paths, tmp_path / "sealed")

    assert manifest["output_inert"] is True
    assert manifest["birth"] is False
    assert manifest["gt_access"] is False
    assert manifest["semantic_source_exported"] is False
    assert manifest["per_view_candidate_count"] == 2
    assert manifest["tracked_candidate_count"] == 1
    assert manifest["scenes"][0]["per_view_cap_dropped_rows"] == 1
    assert manifest["scenes"][0]["world_offset_xyz"] == [10.0, -2.0, 3.0]
    assert manifest["npz_sha256"] == _sha(output_npz)
    assert json.loads(output_json.read_text(encoding="utf-8")) == manifest

    with np.load(output_npz, allow_pickle=False) as arrays:
        assert "semantic_id" not in " ".join(arrays.files)
        assert "label" not in " ".join(arrays.files)
        np.testing.assert_array_equal(arrays["per_view_source_instance_id"], [5, 6])
        np.testing.assert_allclose(
            arrays["per_view_center_world"],
            [[12.0, -2.0, 3.0], [13.0, -2.0, 3.0]],
        )
        np.testing.assert_allclose(
            arrays["tracked_center_world"], [[9.0, -1.0, 5.0]]
        )


def test_output_pair_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    _, json_a, npz_a = _seal(paths, tmp_path / "a")
    _, json_b, npz_b = _seal(paths, tmp_path / "b")
    assert json_a.read_bytes() == json_b.read_bytes()
    assert npz_a.read_bytes() == npz_b.read_bytes()


def test_accepts_missing_terminal_csv_only_when_zero_active_is_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    tracked_csv = (
        paths["run_root"]
        / "boxer_raw"
        / "scene0001_00"
        / "boxer_3dbbs_tracked.csv"
    )
    tracked_csv.unlink()
    paths["log"].write_text(
        paths["log"].read_text(encoding="utf-8").replace(
            "==> Saved 1 tracked OBBs to path",
            "==> 0 active tracks from inline tracker",
        ),
        encoding="utf-8",
    )

    manifest, _, output_npz = _seal(paths, tmp_path / "zero_active")
    scene = manifest["scenes"][0]
    assert manifest["tracked_candidate_count"] == 0
    assert scene["tracked_input_rows"] == 0
    assert scene["tracked_csv_present"] is False
    assert scene["tracked_zero_active_verified"] is True
    assert scene["inputs"]["boxer_3dbbs_tracked_csv_sha256"] is None
    with np.load(output_npz, allow_pickle=False) as arrays:
        assert arrays["tracked_scene_index"].size == 0


def test_rejects_missing_terminal_csv_without_zero_active_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    (
        paths["run_root"]
        / "boxer_raw"
        / "scene0001_00"
        / "boxer_3dbbs_tracked.csv"
    ).unlink()
    with pytest.raises(seal.SealError, match="required file is absent"):
        _seal(paths, tmp_path / "missing_terminal")


def test_rejects_ambiguous_or_repeated_terminal_status_markers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    paths["log"].write_text(
        paths["log"].read_text(encoding="utf-8")
        + "==> 0 active tracks from inline tracker\n",
        encoding="utf-8",
    )
    with pytest.raises(seal.SealError, match="ambiguous terminal-track status"):
        _seal(paths, tmp_path / "dual_status")

    paths = _make_fixture(tmp_path / "repeat", monkeypatch)
    paths["log"].write_text(
        paths["log"].read_text(encoding="utf-8")
        + "==> Saved 1 tracked OBBs to duplicate path\n",
        encoding="utf-8",
    )
    with pytest.raises(seal.SealError, match="ambiguous terminal-track status"):
        _seal(paths, tmp_path / "repeat_status")


def test_rejects_zero_active_marker_with_nonempty_terminal_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    paths["log"].write_text(
        paths["log"].read_text(encoding="utf-8").replace(
            "==> Saved 1 tracked OBBs to path",
            "==> 0 active tracks from inline tracker",
        ),
        encoding="utf-8",
    )
    with pytest.raises(seal.SealError, match="reported zero active tracks"):
        _seal(paths, tmp_path / "false_zero")


def test_rejects_threshold_drift_without_writing_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    paths["log"].write_text(
        paths["log"].read_text(encoding="utf-8").replace(
            "thresh2d=0.25", "thresh2d=0.24"
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "rejected"
    with pytest.raises(seal.SealError, match="thresh2d"):
        _seal(paths, output_dir)
    assert not (output_dir / "sealed.json").exists()
    assert not (output_dir / "sealed.npz").exists()


def test_rejects_nonfinite_geometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    paths["raw_csv"].write_text(
        paths["raw_csv"].read_text(encoding="utf-8").replace("1.0,0.0,0.0", "nan,0.0,0.0", 1),
        encoding="utf-8",
    )
    with pytest.raises(seal.SealError, match="non-finite"):
        _seal(paths, tmp_path / "bad")


def test_refuses_to_overwrite_a_sealed_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    _seal(paths, tmp_path / "sealed")
    with pytest.raises(seal.SealError, match="refusing to overwrite"):
        _seal(paths, tmp_path / "sealed")


def test_excludes_loader_tail_frames_outside_the_sealed_t05_schedule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    scene_dir = paths["scene_root"] / "scene0001_00" / "frames"
    (scene_dir / "color" / "25.jpg").write_bytes(b"frame 25")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = [11.0, -2.0, 3.0]
    np.savetxt(scene_dir / "pose" / "25.txt", pose)
    with paths["raw_csv"].open("a", encoding="utf-8") as handle:
        handle.write(
            _csv_row(
                frame=25,
                center=(99.0, 0.0, 0.0),
                probability=1.0,
                instance=777,
            )
        )

    manifest, _, output_npz = _seal(paths, tmp_path / "sealed_extra")
    scene = manifest["scenes"][0]
    assert scene["per_view_extra_schedule_rows_excluded"] == 1
    assert scene["per_view_extra_schedule_frame_ids_excluded"] == [25]
    assert scene["tracked_schedule_clean"] is False
    assert scene["tracked_schedule_contaminated_rows_excluded"] == 1
    with np.load(output_npz, allow_pickle=False) as arrays:
        assert 777 not in arrays["per_view_source_instance_id"]
        assert arrays["tracked_instance_id"].size == 0


def test_cli_exposes_no_annotation_argument() -> None:
    destinations = {action.dest for action in seal._parser()._actions}
    assert destinations == {
        "help",
        "run_root",
        "scene_root",
        "scene_list",
        "boxer_root",
        "output_json",
        "output_npz",
    }


def test_rejects_missing_strict_no_gt_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    lines = paths["log"].read_text(encoding="utf-8").splitlines()
    assert lines.pop(0) == seal.GT_ACCESS_GUARD
    paths["log"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(seal.SealError, match="no-GT access guard"):
        _seal(paths, tmp_path / "unguarded")


def test_rejects_unsealed_owl_text_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _make_fixture(tmp_path, monkeypatch)
    ledger = paths["run_root"] / "frozen_inputs_sha256.txt"
    rows = [
        line
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if seal.EXPECTED_OWL_TEXT_CACHE not in line
    ]
    ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(seal.SealError, match="OWLv2 text cache"):
        _seal(paths, tmp_path / "unsealed_text")
