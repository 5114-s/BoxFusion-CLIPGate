#!/usr/bin/env python3
"""Create a no-GT MobileSAM box-prompt runtime receipt.

The benchmark is intentionally independent of detection/evaluation code.  It
loads one frozen image and one frozen checkpoint, performs one shared image
encoding plus a batch of four box prompts, and writes all synchronized timing
samples to a create-only JSON receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time

import cv2
import numpy as np
import torch


SCHEMA = "boxfusion.mobilesam_boxprompt_runtime_receipt.v1"

# Four ordinary first-frame detector boxes, expressed in the frozen 960x960
# OWLv2 coordinate system.  They are runtime inputs only; no label is loaded.
BOXES_960 = np.asarray(
    [
        [536.72, 128.91, 700.78, 324.84],
        [669.38, 112.50, 958.12, 363.75],
        [82.27, 247.85, 150.23, 299.65],
        [252.66, 265.20, 347.34, 316.05],
    ],
    dtype=np.float32,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file(path: Path, name: str) -> Path:
    resolved = path.resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{name} must be a regular non-symlink file: {resolved}")
    return resolved


def _write_create_only(path: Path, payload: dict[str, object]) -> None:
    destination = path.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def benchmark(
    *,
    model_root: Path,
    checkpoint: Path,
    image_path: Path,
    output_path: Path,
    device: str,
    warmup: int,
    runs: int,
) -> dict[str, object]:
    model_root = model_root.resolve()
    checkpoint = _regular_file(checkpoint, "MobileSAM checkpoint")
    image_path = _regular_file(image_path, "benchmark RGB image")
    if not model_root.is_dir():
        raise ValueError(f"MobileSAM source root is not a directory: {model_root}")
    if warmup < 1 or runs < 2:
        raise ValueError("warmup must be >=1 and runs must be >=2")
    if device != "cuda":
        raise ValueError("the formal runtime receipt requires device=cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    sys.path.insert(0, os.fspath(model_root))
    from mobile_sam import SamPredictor, sam_model_registry  # noqa: PLC0415

    bgr = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"failed to decode benchmark image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
    boxes = BOXES_960.copy()
    boxes[:, [0, 2]] *= 640.0 / 960.0
    boxes[:, [1, 3]] *= 480.0 / 960.0

    model = sam_model_registry["vit_t"](checkpoint=os.fspath(checkpoint))
    model.to(device=device)
    model.eval()
    predictor = SamPredictor(model)
    prompt_boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device)

    encoder_ms: list[float] = []
    decoder_ms: list[float] = []
    total_ms: list[float] = []
    selected_iou_mean: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    for index in range(warmup + runs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        predictor.set_image(rgb)
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        transformed = predictor.transform.apply_boxes_torch(
            prompt_boxes, rgb.shape[:2]
        )
        masks, predicted_iou, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed,
            multimask_output=True,
        )
        best = torch.argmax(predicted_iou, dim=1)
        selected = masks[torch.arange(len(masks), device=device), best]
        selected_scores = predicted_iou[
            torch.arange(len(predicted_iou), device=device), best
        ]
        # Materialize the same outputs needed by masked RGB-D lifting.
        _ = selected.detach().cpu().numpy()
        torch.cuda.synchronize()
        finished = time.perf_counter()
        if index >= warmup:
            encoder_ms.append((encoded - started) * 1000.0)
            decoder_ms.append((finished - encoded) * 1000.0)
            total_ms.append((finished - started) * 1000.0)
            selected_iou_mean.append(float(selected_scores.mean().item()))

    source_files = {
        "build_sam": model_root / "mobile_sam" / "build_sam.py",
        "predictor": model_root / "mobile_sam" / "predictor.py",
        "tiny_vit": (
            model_root
            / "mobile_sam"
            / "modeling"
            / "tiny_vit_sam.py"
        ),
    }
    source_sha256 = {
        name: _sha256(_regular_file(path, f"MobileSAM {name} source"))
        for name, path in source_files.items()
    }
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "gt_access": False,
        "training": False,
        "optimizer": False,
        "labels_loaded": False,
        "clip_access": False,
        "purpose": "runtime_only_not_accuracy_evidence",
        "model": {
            "registry_key": "vit_t",
            "checkpoint": os.fspath(checkpoint),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint),
            "source_root": os.fspath(model_root),
            "source_sha256": source_sha256,
            "parameter_count": int(sum(item.numel() for item in model.parameters())),
        },
        "input": {
            "image": os.fspath(image_path),
            "image_sha256": _sha256(image_path),
            "resized_hw": [480, 640],
            "box_count": int(len(boxes)),
            "boxes_xyxy_640x480": boxes.tolist(),
            "multimask_output": True,
            "mask_choice": "maximum_frozen_predicted_iou_per_box",
        },
        "runtime": {
            "device": device,
            "gpu_name": gpu.name,
            "gpu_total_memory_bytes": int(gpu.total_memory),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "python_version": platform.python_version(),
            "warmup_runs": warmup,
            "measured_runs": runs,
            "encoder_ms": encoder_ms,
            "decoder_and_host_mask_ms": decoder_ms,
            "total_ms": total_ms,
            "total_mean_ms": float(np.mean(total_ms)),
            "total_p50_ms": _quantile(total_ms, 0.50),
            "total_p95_ms": _quantile(total_ms, 0.95),
            "total_max_ms": float(np.max(total_ms)),
            "encoder_mean_ms": float(np.mean(encoder_ms)),
            "decoder_and_host_mask_mean_ms": float(np.mean(decoder_ms)),
            "selected_predicted_iou_mean": float(np.mean(selected_iou_mean)),
            "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "gap25_at_30fps_budget_ms": 25.0 / 30.0 * 1000.0,
            "every_measurement_within_gap25_budget": bool(
                np.all(np.asarray(total_ms) < (25.0 / 30.0 * 1000.0))
            ),
        },
    }
    _write_create_only(output_path, payload)
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = benchmark(
        model_root=args.model_root,
        checkpoint=args.checkpoint,
        image_path=args.image,
        output_path=args.output,
        device=args.device,
        warmup=args.warmup,
        runs=args.runs,
    )
    print(json.dumps(payload["runtime"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
