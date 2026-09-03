from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from tools import seal_scannet_stream_inputs as seal


SCENE = "scene0000_00"
ROOT = Path(__file__).resolve().parents[1]
PRODUCER_SOURCE = ROOT / "boxfusion" / "proposal_cache.py"


def _array_digest(value: object) -> str:
    if isinstance(value, torch.Tensor):
        canonical = value.detach().cpu().contiguous().clone()
        dtype = str(canonical.dtype)
        shape = tuple(canonical.shape)
        payload = canonical.view(torch.uint8).numpy().tobytes(order="C")
    else:
        array = np.ascontiguousarray(np.asarray(value))
        dtype = str(array.dtype)
        shape = tuple(array.shape)
        payload = array.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(dtype.encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(payload)
    return digest.hexdigest()


def _signature(inputs: object) -> dict[str, str]:
    assert isinstance(inputs, dict)
    return {field: _array_digest(inputs[field]) for field in seal.INPUT_FIELDS}


def _write_pose(path: Path, pose: np.ndarray) -> None:
    np.savetxt(path, pose, fmt="%.10g")


def _make_fixture(
    tmp_path: Path,
    *,
    schema: str = "boxfusion.cutr_postfilter_cache.v3",
    start: int = 0,
) -> dict[str, Path | int]:
    scene_root = tmp_path / "scenes"
    frames = scene_root / SCENE / "frames"
    color_dir = frames / "color"
    depth_dir = frames / "depth"
    pose_dir = frames / "pose"
    intrinsic_dir = frames / "intrinsic"
    for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
        directory.mkdir(parents=True, exist_ok=True)

    width, height = 6, 4
    for frame_id in range(5):
        # Deliberately use a different source RGB size: exact producer resize
        # and JPEG-decoder behavior are part of the signature contract.
        yy, xx = np.mgrid[:8, :10]
        color = np.stack(
            (
                (xx * 17 + frame_id * 11) % 256,
                (yy * 23 + frame_id * 7) % 256,
                ((xx + yy) * 13 + frame_id * 19) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        assert cv2.imwrite(str(color_dir / f"{frame_id}.jpg"), color)
        depth = (
            np.arange(width * height, dtype=np.uint16).reshape(height, width)
            + 1000
            + frame_id * 20
        )
        assert cv2.imwrite(str(depth_dir / f"{frame_id}.png"), depth)

    poses: list[np.ndarray] = []
    for frame_id in range(5):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = frame_id * 0.125
        poses.append(pose)
    poses[1] = np.eye(4, dtype=np.float64)
    poses[1][0, 0] = np.inf
    for frame_id, pose in enumerate(poses):
        _write_pose(pose_dir / f"{frame_id}.txt", pose)

    intrinsic = np.asarray(
        [
            [5.1, 0.0, 2.7, 0.0],
            [0.0, 5.2, 1.6, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    for index, name in enumerate(seal.INTRINSIC_FILES):
        value = intrinsic.copy()
        if name != "intrinsic_depth.txt":
            value[0, 3] = float(index + 1)
        np.savetxt(intrinsic_dir / name, value, fmt="%.10g")

    capture_source = tmp_path / "capture_stream.py"
    capture_source.write_text("# pinned ScannetDataset source fixture\n", encoding="utf-8")
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(SCENE + "\n", encoding="utf-8")
    schedule_root = tmp_path / "schedule"
    schedule_dir = schedule_root / SCENE
    schedule_dir.mkdir(parents=True)

    # The producer substitutes pose 1 with pose 0 before applying start.
    effective_poses = [poses[0], poses[0], poses[2], poses[3], poses[4]][start:]
    retained_count = 5 - start
    gap = 1
    final_native_frame = max(0, retained_count - gap - 1)
    frame_ids = list(range(0, final_native_frame + 1, gap))
    K_f32 = intrinsic[:3, :3].astype(np.float32)
    records = []
    for frame_id in frame_ids:
        source_id = frame_id + start
        inputs = seal._processed_inputs(
            color_path=color_dir / f"{source_id}.jpg",
            depth_path=depth_dir / f"{source_id}.png",
            effective_pose=effective_poses[frame_id],
            K_f32=K_f32,
            width=width,
            height=height,
            depth_scale=1000.0,
        )
        records.append(
            {
                "frame_id": frame_id,
                "input_signature": _signature(dict(inputs)),
            }
        )
    manifest = {
        "schema": schema,
        "scene_id": SCENE,
        "namespace": "unit-test",
        "producer_fingerprint": "f" * 64,
        "record_count": len(records),
        "recorded_frame_ids": frame_ids,
        "schedule": {
            "dataset_length": retained_count,
            "gap": gap,
            "terminal_policy": "upstream_boxfusion_early_exit_v1",
        },
        "records": records,
    }
    manifest_path = schedule_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "scene_root": scene_root,
        "scene_list": scene_list,
        "schedule_root": schedule_root,
        "manifest": manifest_path,
        "capture_source": capture_source,
        "producer_source": PRODUCER_SOURCE,
        "width": width,
        "height": height,
        "start": start,
    }


def _build(paths: dict[str, Path | int]) -> dict:
    return seal.build_seal(
        schedule_root=Path(paths["schedule_root"]),
        scene_root=Path(paths["scene_root"]),
        scene_list=Path(paths["scene_list"]),
        producer_source=Path(paths["producer_source"]),
        capture_source=Path(paths["capture_source"]),
        width=int(paths["width"]),
        height=int(paths["height"]),
        start=int(paths["start"]),
    )


def _verify(paths: dict[str, Path | int], sealed: Path) -> dict:
    return seal.verify_seal(
        seal_path=sealed,
        schedule_root=Path(paths["schedule_root"]),
        scene_root=Path(paths["scene_root"]),
        scene_list=Path(paths["scene_list"]),
        producer_source=Path(paths["producer_source"]),
        capture_source=Path(paths["capture_source"]),
        width=int(paths["width"]),
        height=int(paths["height"]),
        start=int(paths["start"]),
    )


@pytest.mark.parametrize(
    "schema",
    [
        "boxfusion.cutr_postfilter_cache.v2",
        "boxfusion.cutr_postfilter_cache.v3",
    ],
)
def test_exact_producer_signatures_and_deterministic_exclusive_seal(
    tmp_path: Path, schema: str
) -> None:
    paths = _make_fixture(tmp_path, schema=schema)
    first = _build(paths)
    second = _build(paths)
    assert first == second
    assert first["gt_access"] is False
    assert first["oracle_access"] is False
    assert first["signature_mode"] == "exact_producer_array_signature"
    assert first["raw_file_ledger_equivalent_to_producer_signature"] is False
    assert len(first["schedule_manifest_ledger_sha256"]) == 64
    assert len(first["tool"]["sha256"]) == 64
    scene = first["scenes"][0]
    assert scene["pose_ledger"]["all_invalid_source_frame_ids"] == [1]
    assert scene["pose_ledger"]["scheduled_invalid_source_frame_ids"] == [1]
    assert scene["records"][1]["pose_substituted_from_source_frame_id"] == 0

    seal_a = tmp_path / "seal-a.json"
    seal_b = tmp_path / "seal-b.json"
    seal.write_seal_exclusive(seal_a, first)
    seal.write_seal_exclusive(seal_b, second)
    assert seal_a.read_bytes() == seal_b.read_bytes()
    with pytest.raises(seal.StreamInputError, match="refusing to overwrite"):
        seal.write_seal_exclusive(seal_a, first)
    assert _verify(paths, seal_a)["verified"] is True


def test_start_keeps_pre_start_pose_substitution_state(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path, start=1)
    result = _build(paths)
    scene = result["scenes"][0]
    assert scene["source_frame_id_first"] == 1
    assert scene["records"][0]["pose_status"] == "substituted_previous_valid"
    assert scene["records"][0]["pose_substituted_from_source_frame_id"] == 0


@pytest.mark.parametrize("kind", ["depth", "pose", "intrinsic_depth"])
def test_processed_input_drift_fails_closed(tmp_path: Path, kind: str) -> None:
    paths = _make_fixture(tmp_path)
    sealed = tmp_path / "sealed.json"
    seal.write_seal_exclusive(sealed, _build(paths))
    frames = Path(paths["scene_root"]) / SCENE / "frames"
    if kind == "depth":
        path = frames / "depth" / "2.png"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert depth is not None
        depth[0, 0] += 17
        assert cv2.imwrite(str(path), depth)
    elif kind == "pose":
        path = frames / "pose" / "2.txt"
        pose = np.loadtxt(path)
        pose[1, 3] += 0.25
        _write_pose(path, pose)
    else:
        path = frames / "intrinsic" / "intrinsic_depth.txt"
        intrinsic = np.loadtxt(path)
        intrinsic[0, 0] += 0.125
        np.savetxt(path, intrinsic, fmt="%.10g")
    with pytest.raises(seal.StreamInputError, match="input signature mismatch"):
        _verify(paths, sealed)


def test_raw_byte_or_unused_calibration_drift_fails_ledger(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    sealed = tmp_path / "sealed.json"
    seal.write_seal_exclusive(sealed, _build(paths))

    # Appending after JPEG EOI preserves the decoded producer array, so this
    # specifically exercises the supplemental (non-equivalent) raw ledger.
    color = Path(paths["scene_root"]) / SCENE / "frames" / "color" / "0.jpg"
    with color.open("ab") as handle:
        handle.write(b"supplemental-ledger-drift")
    with pytest.raises(seal.StreamInputError, match="differs from the sealed ledger"):
        _verify(paths, sealed)


def test_manifest_signature_drift_fails_before_sealing(tmp_path: Path) -> None:
    paths = _make_fixture(tmp_path)
    manifest_path = Path(paths["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"][0]["input_signature"]["image"] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(seal.StreamInputError, match="input signature mismatch"):
        _build(paths)


def test_cli_has_no_gt_or_oracle_path_argument() -> None:
    parser = seal._build_parser()
    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    subparsers = next(
        action
        for action in parser._actions
        if hasattr(action, "choices") and action.choices
    )
    for command in subparsers.choices.values():
        option_strings.update(
            option for action in command._actions for option in action.option_strings
        )
    assert not any("gt" in option.lower() for option in option_strings)
    assert not any("oracle" in option.lower() for option in option_strings)
    with pytest.raises(SystemExit):
        parser.parse_args(["seal", "--gt-root", "/tmp/forbidden"])


def test_signature_ast_reuse_rejects_io_calls(tmp_path: Path) -> None:
    source = tmp_path / "mutated_producer.py"
    source.write_text(
        "def _canonical_tensor(value):\n"
        "    return value\n"
        "def _tensor_sha256(value):\n"
        "    return '0' * 64\n"
        "def _array_sha256(value):\n"
        "    return _tensor_sha256(value)\n"
        "def _input_signature(inputs):\n"
        "    open('/tmp/unsafe', 'w')\n"
        "    return {name: _array_sha256(value) for name, value in inputs.items()}\n",
        encoding="utf-8",
    )
    with pytest.raises(seal.StreamInputError, match="unsafe call.*open"):
        seal._producer_signature_function(source)
