#!/usr/bin/env python3
"""Use BoxFusion's native wall-clock FPS boundary from verified cold files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import types
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


class NativeTimerProxy:
    """Record only ``demo.py`` wall-clock calls while forwarding all others."""

    def __init__(self, module: types.ModuleType):
        self._module = module
        self.wall_calls = []

    def time(self):
        value = self._module.time()
        self.wall_calls.append(value)
        return value

    def __getattr__(self, name):
        return getattr(self._module, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evict_and_verify_scene_files(root: Path) -> tuple[int, int, int]:
    """Evict scene file data and prove zero resident bytes before timing."""
    paths = []
    for directory, _, names in os.walk(root, followlinks=True):
        for name in names:
            path = Path(directory) / name
            if path.is_file():
                paths.append(path)
    paths.sort()
    if not paths:
        raise RuntimeError(f"no scene files found under {root}")
    advised_bytes = 0
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(descriptor).st_size
            os.posix_fadvise(
                descriptor, 0, 0, os.POSIX_FADV_DONTNEED
            )
            advised_bytes += size
        finally:
            os.close(descriptor)
    check = subprocess.run(
        [
            "fincore", "--bytes", "--noheadings", "--output", "RES",
            *(str(path) for path in paths),
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    resident_values = [
        int(line.strip()) for line in check.stdout.splitlines() if line.strip()
    ]
    if len(resident_values) != len(paths):
        raise RuntimeError(
            "fincore row count mismatch: "
            f"expected {len(paths)}, observed {len(resident_values)}"
        )
    resident_bytes = sum(resident_values)
    if resident_bytes != 0:
        raise RuntimeError(
            f"scene page-cache eviction incomplete: {resident_bytes} bytes resident"
        )
    return len(paths), advised_bytes, resident_bytes


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
        # Lock the submission configuration rather than timing the older
        # four-factor MVSR or any diagnostic shadow arms.  The primary online
        # reranker uses proposal score, 2D overlap and 3D consistency with
        # historical-maximum aggregation; visibility is deliberately absent.
        state["audit_mvsr_factors"] = False
        state["mvsr_factor_variants"] = []
        state["selected_calr_v1"] = True
        state["m1p"]["use_children"] = False
        state["m2"]["support_mode"] = "max"
        state["m2"]["strength_variant"] = "no_visibility"
        # Preserve the public demo's native timing path: time.time(), no CUDA
        # synchronization, no eager frame preparation, and no runtime receipt
        # hook that would extend the interval through output serialization.
        cfg["runtime_benchmark"] = {
            "synchronize_cuda": False,
            "prepare_all_raw_frames": False,
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
    evicted_files, evicted_bytes, residual_resident_bytes = (
        evict_and_verify_scene_files(frames)
    )
    loaded = time.perf_counter()

    torch.cuda.reset_peak_memory_stats()
    timer_proxy = NativeTimerProxy(demo.time)
    demo.time = timer_proxy
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
    if len(timer_proxy.wall_calls) != 2:
        raise RuntimeError(
            "native demo timing boundary changed: expected exactly two "
            f"time.time calls, observed {len(timer_proxy.wall_calls)}"
        )
    run_seconds = timer_proxy.wall_calls[1] - timer_proxy.wall_calls[0]
    if run_seconds <= 0:
        raise RuntimeError(f"invalid native demo duration: {run_seconds}")
    receipt = {
        "schema": "boxfusion.paper_native_wall_clock_scene.v1",
        "raw_frames": int(dataset.consumed),
        "cost_seconds": float(run_seconds),
        "fps": float(dataset.consumed / run_seconds),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "timing_protocol": "public_demo_native_time_time",
        "cold_page_cache_verified": True,
        "cache_eviction_method": "POSIX_FADV_DONTNEED_then_fincore",
        "cache_evicted_files": int(evicted_files),
        "cache_evicted_bytes": int(evicted_bytes),
        "resident_bytes_at_timing_boundary": int(residual_resident_bytes),
        "native_timer_call_count": len(timer_proxy.wall_calls),
        "cuda_synchronized": False,
        "prepare_all_raw_frames": False,
        "model_initialization_excluded": True,
        "visualization": False,
        "proposal_replay": False,
        "final_output_serialization_included": False,
        "device": torch.cuda.get_device_name(args.device_index),
        "torch": torch.__version__,
    }

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
            "scene_cache_prewarmed": False,
            "os_page_cache_control": "verified_cold_scene_files",
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
