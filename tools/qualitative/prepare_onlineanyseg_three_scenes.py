#!/usr/bin/env python3
"""Prepare three read-only RGB-D sequences for official OnlineAnySeg.

The script creates links inside the workspace and writes CA-1M pose text files.
It never changes the original ScanNet or CA-1M datasets.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "external/OnlineAnySeg_runs_20261102"
SCANNET = Path("/extra/ZhaoX/scannet_data/scans")
CA1M = Path("/extra/ZhaoX/boxfusion_ca1m")
SCENES = ("scene0426_03", "scene0606_01")
CA_SCENE = "42897692"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        assert target.resolve() == source, (target, target.resolve(), source)
        return
    if target.exists():
        raise FileExistsError(target)
    target.symlink_to(source, target_is_directory=source.is_dir())


def prepare_scannet(manifest: dict) -> None:
    for scene in SCENES:
        source = SCANNET / scene
        destination = OUT / "data/scannet" / scene
        frames = destination / "frames"
        for folder in ("color", "depth", "intrinsic", "pose"):
            link(source / folder, frames / folder)
        mesh = source / f"{scene}_vh_clean_2.ply"
        link(mesh, destination / mesh.name)
        color_count = len(list((source / "color").glob("*.jpg")))
        depth_count = len(list((source / "depth").glob("*.png")))
        pose_count = len(list((source / "pose").glob("*.txt")))
        assert color_count == depth_count == pose_count > 0
        with Image.open(source / "depth/0.png") as image:
            depth_size = list(image.size)
        manifest["scenes"].append({"dataset": "ScanNet", "scene": scene,
            "source": str(source), "prepared": str(destination), "frames": color_count,
            "depth_size_wh": depth_size, "mesh": str(mesh), "mesh_sha256": sha256(mesh),
            "transformation": "No coordinate change during input preparation"})


def prepare_ca1m(manifest: dict) -> None:
    source = CA1M / CA_SCENE
    destination = OUT / "data/ca1m" / CA_SCENE
    link(source / "rgb", destination / "color")
    link(source / "depth", destination / "depth")
    intrinsic_source = source / "K_depth.txt"
    link(intrinsic_source, destination / "intrinsic_depth.txt")
    poses = np.load(source / "all_poses.npy")
    rgb_count = len(list((source / "rgb").glob("*.png")))
    depth_count = len(list((source / "depth").glob("*.png")))
    assert poses.shape == (rgb_count, 4, 4) and rgb_count == depth_count
    pose_dir = destination / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)
    for index, pose in enumerate(poses):
        path = pose_dir / f"{index}.txt"
        text = "\n".join(" ".join(f"{value:.10e}" for value in row) for row in pose) + "\n"
        if path.exists():
            assert path.read_text() == text
        else:
            path.write_text(text)
    with Image.open(source / "depth/0.png") as image:
        depth_size = list(image.size)
    manifest["scenes"].append({"dataset": "CA-1M", "scene": CA_SCENE,
        "source": str(source), "prepared": str(destination), "frames": rgb_count,
        "depth_size_wh": depth_size, "intrinsic": str(intrinsic_source),
        "intrinsic_sha256": sha256(intrinsic_source), "poses": str(source / "all_poses.npy"),
        "poses_sha256": sha256(source / "all_poses.npy"),
        "transformation": "No coordinate change; all_poses.npy exported verbatim as camera-to-world text matrices"})


def write_ca_config() -> Path:
    source = ROOT / "external/OnlineAnySeg_source_20261102/config/mydataset_cropformer.yaml"
    text = source.read_text()
    # PIL reports (width, height)=(384, 512); K_depth has (cx,cy)≈(191.6,254.9).
    text = text.replace('img_h: 480', 'img_h: 512', 1).replace('img_w: 640', 'img_w: 384', 1)
    target = OUT / "config/ca1m_cropformer_h512_w384.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    return target


def main() -> None:
    manifest = {"schema": "recar.onlineanyseg_three_scene_inputs.v1",
                "purpose": "Input adapter for the official OnlineAnySeg implementation",
                "original_data_modified": False, "scenes": []}
    prepare_scannet(manifest)
    prepare_ca1m(manifest)
    config = write_ca_config()
    manifest["ca1m_config"] = str(config)
    manifest["ca1m_config_sha256"] = sha256(config)
    manifest["official_protocol"] = {"ScanNet_mask_interval": 20,
        "ScanNet_mapping_keyframe_frequency": 10, "CA1M_mask_interval": 20,
        "CA1M_mapping_keyframe_frequency": 20,
        "note": "CA-1M uses OnlineAnySeg's documented custom-sequence interface with measured width=384, height=512 depth intrinsics"}
    path = OUT / "input_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
