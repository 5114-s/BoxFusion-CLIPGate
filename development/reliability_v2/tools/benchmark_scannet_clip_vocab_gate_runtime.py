#!/usr/bin/env python3
"""Benchmark the frozen CLIP-vocabulary gate on real three-view receipts.

This is a no-GT runtime-only benchmark.  It replays the exact RGB crops stored
in the CLIP shadow sidecar, loads the native ViT-H-14 checkpoint once, and
reports CPU preparation and warm GPU encode+473-way-score latency separately.
The GPU timing deliberately excludes checkpoint/model loading and host-to-device
transfer: an online deployment must retain both the model and the three input
tensors between calls to obtain this warm-gate number.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from run_scannet_raw_boxer_clip_vocab_shadow_full100 import (
    _crop_rgb,
    _find_color_path,
    _load_clip_runtime,
    _load_resized_rgb,
    _load_text_features,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _percentiles(values_ms: Sequence[float]) -> dict[str, float]:
    values = np.asarray(values_ms, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise RuntimeError("latency samples must be finite and non-empty")
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(values.max()),
    }


def _load_tracks(sidecar_path: Path) -> list[tuple[str, list[dict[str, Any]]]]:
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("gt_access") is not False or sidecar.get("evaluator_access") is not False:
        raise RuntimeError("sidecar is not a no-GT/no-evaluator artifact")
    tracks: list[tuple[str, list[dict[str, Any]]]] = []
    for scene in sorted(sidecar["scenes"]):
        for track_id in sorted(sidecar["scenes"][scene]["tracks"], key=int):
            evidence = sidecar["scenes"][scene]["tracks"][track_id]["evidence"]
            if len(evidence) != 3:
                raise RuntimeError(f"expected exactly three crops: {scene}/{track_id}")
            tracks.append((scene, evidence))
    if not tracks:
        raise RuntimeError("sidecar contains no receipt tracks")
    return tracks


def _load_and_preprocess_track(
    scene_root: Path,
    scene: str,
    evidence: Sequence[dict[str, Any]],
    preprocess: Any,
) -> torch.Tensor:
    tensors = []
    for row in evidence:
        path = _find_color_path(scene_root, scene, int(row["frame_id"]))
        rgb = _load_resized_rgb(
            path,
            int(row["owl_image_width"]),
            int(row["owl_image_height"]),
        )
        crop = _crop_rgb(rgb, row["owl_bbox_xyxy"])
        tensors.append(preprocess(crop))
    return torch.stack(tensors)


def _preload_resized_frames(
    scene_root: Path,
    tracks: Sequence[tuple[str, list[dict[str, Any]]]],
) -> dict[tuple[str, int, int, int], np.ndarray]:
    frames: dict[tuple[str, int, int, int], np.ndarray] = {}
    for scene, evidence in tracks:
        for row in evidence:
            key = (
                scene,
                int(row["frame_id"]),
                int(row["owl_image_width"]),
                int(row["owl_image_height"]),
            )
            if key not in frames:
                path = _find_color_path(scene_root, scene, key[1])
                frames[key] = _load_resized_rgb(path, key[2], key[3])
    return frames


def _preprocess_from_resized(
    scene: str,
    evidence: Sequence[dict[str, Any]],
    frames: dict[tuple[str, int, int, int], np.ndarray],
    preprocess: Any,
) -> torch.Tensor:
    tensors = []
    for row in evidence:
        key = (
            scene,
            int(row["frame_id"]),
            int(row["owl_image_width"]),
            int(row["owl_image_height"]),
        )
        tensors.append(preprocess(_crop_rgb(frames[key], row["owl_bbox_xyxy"])))
    return torch.stack(tensors)


def _timed_cpu_batches(
    tracks: Sequence[tuple[str, list[dict[str, Any]]]],
    function: Any,
) -> tuple[list[torch.Tensor], list[float]]:
    batches: list[torch.Tensor] = []
    timings: list[float] = []
    for scene, evidence in tracks:
        started = time.perf_counter_ns()
        batch = function(scene, evidence)
        timings.append((time.perf_counter_ns() - started) / 1e6)
        batches.append(batch)
    return batches, timings


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.device == "cpu" or not args.device.startswith("cuda"):
        raise RuntimeError("this benchmark requires an explicit CUDA device")
    if args.warmup < 1 or args.repeats < 1:
        raise RuntimeError("warmup and repeats must be positive")

    tracks = _load_tracks(args.sidecar)
    if args.max_tracks:
        tracks = tracks[: args.max_tracks]

    torch.cuda.set_device(torch.device(args.device))
    model_load_started = time.perf_counter_ns()
    model, preprocess = _load_clip_runtime(args.checkpoint, args.device)
    text_features = _load_text_features(args.class_features, args.device)
    torch.cuda.synchronize(args.device)
    model_load_ms = (time.perf_counter_ns() - model_load_started) / 1e6

    # Prime the OS page cache, PIL/torchvision transforms, and resize kernels.
    _load_and_preprocess_track(
        args.scene_root, tracks[0][0], tracks[0][1], preprocess
    )
    disk_batches, disk_ms = _timed_cpu_batches(
        tracks,
        lambda scene, evidence: _load_and_preprocess_track(
            args.scene_root, scene, evidence, preprocess
        ),
    )

    resized_frames = _preload_resized_frames(args.scene_root, tracks)
    _, crop_preprocess_ms = _timed_cpu_batches(
        tracks,
        lambda scene, evidence: _preprocess_from_resized(
            scene, evidence, resized_frames, preprocess
        ),
    )

    # Hold real batch-of-three tensors on the selected GPU.  The warm timing
    # below is therefore exactly encode_image + normalization + 473-way dot.
    device_batches = [batch.to(args.device) for batch in disk_batches]
    stream = itertools.cycle(device_batches)

    def score(batch: torch.Tensor) -> torch.Tensor:
        features = model.encode_image(batch).float()
        features = features / features.norm(dim=1, keepdim=True)
        return features @ text_features.T

    with torch.inference_mode():
        for _ in range(args.warmup):
            score(next(stream))
        torch.cuda.synchronize(args.device)

        gpu_ms: list[float] = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            batch = next(stream)
            start_event.record()
            similarities = score(batch)
            end_event.record()
            end_event.synchronize()
            if tuple(similarities.shape) != (3, 473):
                raise RuntimeError("CLIP score shape changed")
            gpu_ms.append(float(start_event.elapsed_time(end_event)))

    device_index = torch.device(args.device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    result = {
        "schema": "boxfusion.clip_vocab_gate_runtime.v1",
        "mode": "runtime_only_no_gt_no_evaluator",
        "gt_access": False,
        "evaluator_access": False,
        "training": False,
        "sidecar": os.fspath(args.sidecar.resolve()),
        "checkpoint": os.fspath(args.checkpoint.resolve()),
        "class_features": os.fspath(args.class_features.resolve()),
        "scene_root": os.fspath(args.scene_root.resolve()),
        "track_count": len(tracks),
        "real_crop_count": len(tracks) * 3,
        "batch_size": 3,
        "gpu_repeats": args.repeats,
        "gpu_warmup": args.warmup,
        "model_load_once_ms": model_load_ms,
        "cpu_disk_decode_resize_crop_preprocess_batch3": _percentiles(disk_ms),
        "cpu_crop_preprocess_from_resized_rgb_batch3": _percentiles(
            crop_preprocess_ms
        ),
        "gpu_encode_normalize_score473_batch3": _percentiles(gpu_ms),
        "gpu_timing_excludes": [
            "checkpoint/model load",
            "text feature load",
            "disk image decode",
            "RGB resize/crop/preprocess",
            "host-to-device transfer",
        ],
        "deployment_contract": {
            "model_reused": True,
            "text_features_reused_on_gpu": True,
            "three_preprocessed_crops_resident_on_gpu": True,
            "precision": "native float32 inference (no autocast)",
            "not_an_end_to_end_fps_measurement": True,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "requested_device": args.device,
            "cuda_device_index": device_index,
            "cuda_device_name": torch.cuda.get_device_name(device_index),
            "cuda_device_capability": list(
                torch.cuda.get_device_capability(device_index)
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sidecar",
        type=Path,
        default=REPOSITORY_ROOT
        / "logs/scannet_cbest_raw_boxer_clip_vocab_shadow_score05"
        / "CLIP_VOCAB_SHADOW_FULL100.json",
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=REPOSITORY_ROOT / "upstream_clean/scannet_readme_frames",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "models/open_clip_pytorch_model.bin",
    )
    parser.add_argument(
        "--class-features",
        type=Path,
        default=REPOSITORY_ROOT / "data/class_features.pt",
    )
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--max-tracks", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"Saved: {args.output}")
    print(encoded, end="")


if __name__ == "__main__":
    main()
