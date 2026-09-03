from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from boxfusion.capture_stream import CA1MDataset


def _scene(root: Path, *, frames: int = 3) -> dict:
    (root / "rgb").mkdir(parents=True)
    (root / "depth").mkdir()
    for index in range(frames):
        (root / "rgb" / f"{index}.png").touch()
        (root / "depth" / f"{index}.png").touch()
    np.save(root / "all_poses.npy", np.repeat(np.eye(4)[None], frames, axis=0))
    intrinsic = np.asarray(
        [[500.0, 0.0, 255.5], [0.0, 501.0, 191.5], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    np.savetxt(root / "K_depth.txt", intrinsic)
    return {
        "data": {"start": 0, "datadir": str(root)},
        "cam": {"H": 384, "W": 512, "png_depth_scale": 1000.0},
    }


def test_ca1m_static_intrinsics_remain_backward_compatible(tmp_path: Path) -> None:
    cfg = _scene(tmp_path / "42444499")
    dataset = CA1MDataset(cfg)
    assert dataset.intrinsics_mode == "static_scene_intrinsics_v1"
    assert dataset.depth_intrinsics.shape == (3, 3, 3)
    assert np.array_equal(dataset.depth_intrinsics[0], dataset.K)
    assert np.array_equal(dataset.depth_intrinsics[0], dataset.depth_intrinsics[-1])


def test_ca1m_per_frame_intrinsics_and_start_slice(tmp_path: Path) -> None:
    root = tmp_path / "42444501"
    cfg = _scene(root)
    values = np.repeat(np.eye(3, dtype=np.float64)[None], 3, axis=0)
    values[:, 0, 0] = [500.0, 501.0, 502.0]
    values[:, 1, 1] = [510.0, 511.0, 512.0]
    values[:, 0, 2] = 255.5
    values[:, 1, 2] = 191.5
    np.save(root / "K_depth_per_frame.npy", values)
    cfg["data"]["start"] = 1
    dataset = CA1MDataset(cfg)
    assert dataset.intrinsics_mode == "per_frame_depth_intrinsics_v1"
    assert np.array_equal(dataset.depth_intrinsics, values[1:])


def test_ca1m_per_frame_intrinsics_fail_closed_on_bad_cardinality(
    tmp_path: Path,
) -> None:
    root = tmp_path / "42444503"
    cfg = _scene(root)
    np.save(root / "K_depth_per_frame.npy", np.repeat(np.eye(3)[None], 2, axis=0))
    with pytest.raises(ValueError, match="must match the sliced or full frame count"):
        CA1MDataset(cfg)
