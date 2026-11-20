#!/usr/bin/env python3
"""Benchmark one ScanNet scene for frozen original BoxFusion or final ReCaR-3D."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml


class CountedStream:
    def __init__(self, dataset):
        self.dataset = dataset
        self.consumed = 0

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def __iter__(self):
        for sample in self.dataset:
            self.consumed += 1
            yield sample


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prewarm_scene(root: Path) -> tuple[int, int, float]:
    """Read scene files into the OS page cache outside the timed interval."""
    started = time.perf_counter()
    files = 0
    size = 0
    for directory, _, names in os.walk(root, followlinks=True):
        for name in sorted(names):
            path = Path(directory) / name
            if not path.is_file():
                continue
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    size += len(chunk)
            files += 1
    if files == 0 or size == 0:
        raise RuntimeError(f"scene prewarm read no data: {root}")
    return files, size, time.perf_counter() - started


def load_models(demo, repository_root: Path):
    import torch

    checkpoint_path = repository_root / "models/cutr_rgbd.pth"
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )["model"]
    dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
    model = demo.make_cubify_transformer(
        dimension=dimension, depth_model=True
    ).eval().cuda()
    model.load_state_dict(checkpoint)
    del checkpoint
    clip_model, preprocess = demo.load_clip(
        str(repository_root / "models/open_clip_pytorch_model.bin")
    )
    categories = np.genfromtxt(
        repository_root / "data/panoptic_categories_nomerge.txt",
        delimiter="\n",
        dtype=str,
    )
    features = torch.load(
        repository_root / "data/class_features.pt", weights_only=False
    ).cuda()
    return model, clip_model, preprocess, categories, features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("original", "recar"), required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    source = args.source_root.resolve()
    repository = args.repository_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "runtime.json"
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite runtime receipt: {receipt_path}")

    sys.path.insert(0, str(source))
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device_index)
    import demo

    if args.arm == "original":
        config_path = source / "config/scannet.yaml"
    else:
        if args.config is None:
            raise ValueError("--config is required for the ReCaR arm")
        config_path = args.config.resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    frames = (
        repository
        / "upstream_clean/scannet_readme_frames"
        / args.scene
        / "frames"
    ).resolve()
    if not frames.is_dir():
        raise FileNotFoundError(frames)
    cfg["data"]["datadir"] = str(frames)
    cfg["data"]["output_dir"] = str(output / "native")
    (output / "native").mkdir(parents=True, exist_ok=True)
    cfg["vis"]["rerun"] = False
    cfg["eval"] = True
    cfg.setdefault("association", {}).setdefault("appearance_gate", {})[
        "enabled"
    ] = False

    if args.arm == "recar":
        # Run only the locked final route: PLR-v1 + CALR-v1 + MVSR-v2 Max.
        cfg["association"].setdefault("pvq_ar", {})["enabled"] = False
        cfg["lifting"]["proposal_cache"]["mode"] = "disabled"
        cfg["lifting"]["boxer"]["diagnostics_dir"] = str(
            output / "native_boxer_diagnostics"
        )
        online = cfg["online_candidate_map"]
        online["enabled"] = True
        online["output_root"] = str(output / "recar")
        online["diagnostics_root"] = str(output / "online_diagnostics")
        online["audit_output_root"] = None
        online["write_every_keyframe"] = False
        online["asynchronous"] = True
        online["provider"]["device"] = "cuda:0"
        online["provider"]["diagnostics_root"] = str(
            output / "provider_diagnostics"
        )
        online["provider"]["evidence_cache_root"] = None
        state = online["state"]
        state["audit_controls"] = False
        state["audit_plr_controls"] = False
        state["audit_direct_plr"] = False
        state["audit_raw_iou_max"] = False
        # Lock the exact submission route.  The source config was also used by
        # the completed standalone reproduction, but its factor-audit fields
        # must be disabled so shadow variants do not enter the timed path.
        state["audit_mvsr_factors"] = False
        state["mvsr_factor_variants"] = []
        state["selected_calr_v1"] = True
        state["m1p"]["use_children"] = False
        state["m2"]["support_mode"] = "max"
        state["m2"]["strength_variant"] = "no_visibility"
        cfg["runtime_benchmark"] = {
            "synchronize_cuda": True,
            "prepare_all_raw_frames": True,
            "receipt_path": str(receipt_path),
        }

    effective_path = output / "effective_config.yaml"
    effective_path.write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    cold_start = time.perf_counter()
    model, clip_model, preprocess, categories, features = load_models(
        demo, repository
    )
    raw_dataset = demo.get_dataset(cfg)
    raw_dataset.load_arkit_depth = True
    dataset = CountedStream(raw_dataset)
    augmentor = demo.Augmentor(("wide/image", "wide/depth"))
    preprocessor = demo.Preprocessor()
    warm_files, warm_bytes, warm_seconds = prewarm_scene(frames)
    torch.cuda.synchronize()
    loaded = time.perf_counter()

    if args.arm == "original":
        torch.cuda.reset_peak_memory_stats()
    try:
        demo.run(
            cfg,
            model,
            dataset,
            clip_model,
            preprocess,
            categories,
            features,
            augmentor,
            preprocessor,
            score_thresh=cfg["detection"]["score_thresh"],
            gap=cfg["data"]["gap"],
            re_vis=False,
        )
    except SystemExit as error:
        if error.code not in (None, 0):
            raise

    if args.arm == "original":
        torch.cuda.synchronize()
        finished = time.perf_counter()
        run_seconds = finished - loaded
        receipt = {
            "schema": "boxfusion.original_vs_recar.original_scene.v1",
            "raw_frames": int(dataset.consumed),
            "cost_seconds": float(run_seconds),
            "fps": float(dataset.consumed / run_seconds),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_synchronized": True,
            "prepare_all_raw_frames": True,
            "model_initialization_excluded": True,
            "visualization": False,
            "proposal_replay": False,
            "final_output_serialization_included": True,
            "device": torch.cuda.get_device_name(args.device_index),
            "torch": torch.__version__,
        }
    else:
        if not receipt_path.is_file():
            raise RuntimeError("ReCaR demo did not write its runtime receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    if int(receipt["raw_frames"]) != dataset.consumed:
        raise RuntimeError(
            f"frame mismatch: receipt={receipt['raw_frames']} "
            f"stream={dataset.consumed}"
        )
    receipt.update(
        {
            "arm": args.arm,
            "scene": args.scene,
            "gpu_count": 1,
            "gap": int(cfg["data"]["gap"]),
            "score_threshold": float(cfg["detection"]["score_thresh"]),
            "scene_cache_prewarmed": True,
            "prewarm_files": int(warm_files),
            "prewarm_bytes": int(warm_bytes),
            "prewarm_seconds_excluded": float(warm_seconds),
            "model_load_seconds_excluded": float(loaded - cold_start),
            "source_root": str(source),
            "source_demo_sha256": sha256(source / "demo.py"),
            "source_config_sha256": sha256(config_path),
            "effective_config_sha256": sha256(effective_path),
            "memory_measurement": "torch.cuda.max_memory_allocated",
        }
    )
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
