from __future__ import annotations

import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "tools" / "audit_ca1m_native_b6_train_inputs.py"
MATERIALIZE = ROOT / "tools" / "materialize_ca1m_native_b6_train_config.py"


def _frozen_contract(root: Path, scenes: list[str]) -> tuple[Path, Path]:
    digest = __import__("hashlib").sha256(("\n".join(scenes) + "\n").encode()).hexdigest()
    manifest = root / "subset.json"
    manifest.write_text(json.dumps({
        "schema": "boxfusion.ca1m_native_b6_train_subset.v1",
        "selection": {"scene_ids_sha256": digest},
        "source": {"val_scene_count": 2},
        "safety_contract": {
            "train_only": True, "validation_ground_truth_access": False,
            "validation_scene_overlap_count": 0, "training_started": False,
        },
        "entries": [{"scene_id": scene} for scene in scenes],
    }))
    val = root / "val.txt"
    val.write_text(
        "https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-51000000.tar\n"
        "https://ml-site.cdn-apple.com/datasets/ca1m/val/ca1m-val-51000001.tar\n"
    )
    return manifest, val


def _scene(root: Path, scene: str, *, varying_k: bool = False) -> None:
    target = root / scene
    (target / "rgb").mkdir(parents=True)
    (target / "depth").mkdir()
    rgb = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.full((8, 10), 1000, dtype=np.uint16)
    for index in range(2):
        assert cv2.imwrite(str(target / "rgb" / f"{index}.png"), rgb)
        assert cv2.imwrite(str(target / "depth" / f"{index}.png"), depth)
    k = np.asarray([[8, 0, 4.5], [0, 8, 3.5], [0, 0, 1]], dtype=np.float64)
    np.savetxt(target / "K_depth.txt", k)
    np.savetxt(target / "K_rgb.txt", k)
    np.save(target / "all_poses.npy", np.repeat(np.eye(4)[None], 2, axis=0))
    np.save(target / "T_gravity.npy", np.repeat(np.eye(3)[None], 2, axis=0))
    if varying_k:
        values = np.repeat(k[None], 2, axis=0)
        values[1, 0, 0] += 0.1
        np.save(target / "K_depth_per_frame.npy", values)


def test_gt_free_train_input_audit_accepts_static_k(tmp_path: Path) -> None:
    scenes = ["42000000"]
    manifest, val = _frozen_contract(tmp_path, scenes)
    data = tmp_path / "data"
    _scene(data, scenes[0])
    # This deliberately invalid file proves that the input audit never loads GT.
    (data / scenes[0] / "after_filter_boxes.npy").write_bytes(b"not-numpy")
    output = tmp_path / "audit.json"
    result = subprocess.run([
        sys.executable, str(AUDIT), "--manifest", str(manifest),
        "--val-url-list", str(val), "--data-root", str(data), "--output", str(output),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["ground_truth_files_read"] == []
    assert not payload["validation_ground_truth_access"]
    assert payload["scenes"][0]["intrinsics_contract"]["mode"] == "static_scene_intrinsics_v1"


def test_train_input_audit_accepts_explicit_varying_per_frame_k(tmp_path: Path) -> None:
    scenes = ["42000000"]
    manifest, val = _frozen_contract(tmp_path, scenes)
    data = tmp_path / "data"
    _scene(data, scenes[0], varying_k=True)
    output = tmp_path / "varying_k_audit.json"
    result = subprocess.run([
        sys.executable, str(AUDIT), "--manifest", str(manifest),
        "--val-url-list", str(val), "--data-root", str(data), "--output", str(output),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    contract = json.loads(output.read_text())["scenes"][0]["intrinsics_contract"]
    assert contract["mode"] == "per_frame_intrinsics_v1"
    assert contract["variation_detected"] is True
    assert contract["loader_behavior"] == "K_depth_per_frame_npy"
    assert contract["rgb_intrinsics_behavior"] == (
        "K_rgb.txt_static; RGB resized to depth grid"
    )


def test_input_preflight_does_not_require_future_data(tmp_path: Path) -> None:
    manifest, val = _frozen_contract(tmp_path, ["42000000"])
    output = tmp_path / "preflight.json"
    result = subprocess.run([
        sys.executable, str(AUDIT), "--manifest", str(manifest),
        "--val-url-list", str(val), "--data-root", str(tmp_path / "absent"),
        "--preflight", "--output", str(output),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["scene_directories_missing"] == 1
    assert not payload["training_started"]


def test_materialized_observer_keeps_g0_replay_and_score04(tmp_path: Path) -> None:
    output = tmp_path / "observer.yaml"
    result = subprocess.run([
        sys.executable, str(MATERIALIZE),
        "--template", str(ROOT / "config" / "ca1m_native_b6_train100_g0_observer.yaml"),
        "--phase", "observer", "--data-root", str(tmp_path / "data"),
        "--output-root", str(tmp_path / "pred"), "--cache-root", str(tmp_path / "cache"),
        "--baseline-root", str(tmp_path / "baseline"),
        "--native-diagnostics-root", str(tmp_path / "native"),
        "--boxer-diagnostics-root", str(tmp_path / "boxer"), "--output", str(output),
    ], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    cfg = yaml.safe_load(output.read_text())
    assert cfg["detection"]["score_thresh"] == 0.4
    assert cfg["lifting"]["proposal_cache"]["mode"] == "replay"
    assert cfg["lifting"]["boxer"]["selective_gate"] == {
        "enabled": True, "max_center_shift_m": 0.1,
        "min_volume_ratio": 0.5, "max_volume_ratio": 2.0,
    }
    assert cfg["ca1m_native_b6_observer"]["observer_only"]
    with pytest.raises(Exception):
        subprocess.run([
            sys.executable, str(MATERIALIZE), "--template", str(output),
            "--phase", "observer", "--data-root", str(tmp_path / "data"),
            "--output-root", str(tmp_path / "pred"), "--cache-root", str(tmp_path / "cache"),
            "--baseline-root", str(tmp_path / "baseline"),
            "--native-diagnostics-root", str(tmp_path / "native"),
            "--boxer-diagnostics-root", str(tmp_path / "boxer"), "--output", str(output),
        ], check=True, capture_output=True)
