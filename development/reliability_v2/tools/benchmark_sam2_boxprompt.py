#!/usr/bin/env python3
"""Create a no-GT SAM2.1 box-prompt runtime receipt.

This is a hardware/provider feasibility benchmark only.  It takes the sealed
FastSAM tight boxes from one F0 frame, computes one shared SAM2 image embedding,
and refines all prompts in one batched decoder call.  It never reads native
predictions, labels, ground truth, CLIP features, or evaluator output.
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
from typing import Any

import cv2
import numpy as np
import torch


SCHEMA = "boxfusion.sam2_boxprompt_runtime_receipt.v1"
IMAGE_HW = (480, 640)
DEFAULT_MODEL_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/Grounded-SAM-2"
)
DEFAULT_CHECKPOINT = DEFAULT_MODEL_ROOT / "checkpoints/sam2.1_hiera_large.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_F0_SIDECAR = Path(
    "logs/scannet_fastsam_f0_full200_score05/scenes/scene0568_00.json"
)
DEFAULT_OUTPUT = Path(
    "logs/scannet_n0_sam2_runtime/sam2_hiera_l_f0_top16_rtx3090_receipt.json"
)


class SAM2BenchmarkError(ValueError):
    """Raised when the no-GT runtime benchmark contract is violated."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    try:
        source = path.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise SAM2BenchmarkError(f"missing {label}: {path}") from error
    if source.is_symlink() or not source.is_file():
        raise SAM2BenchmarkError(f"{label} must resolve to a regular file: {path}")
    return source


def _read_json(path: Path, label: str) -> dict[str, Any]:
    source = _regular_file(path, label)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SAM2BenchmarkError(f"invalid {label}: {source}") from error
    if not isinstance(value, dict):
        raise SAM2BenchmarkError(f"{label} must contain one JSON object")
    return value


def _load_f0_prompts(
    sidecar_path: Path, *, frame_ordinal: int, prompt_count: int
) -> tuple[Path, str, np.ndarray, dict[str, Any]]:
    if frame_ordinal < 0 or prompt_count < 1 or prompt_count > 16:
        raise SAM2BenchmarkError("frame_ordinal/prompt_count is outside the frozen range")
    sidecar = _read_json(sidecar_path, "F0 sidecar")
    if (
        sidecar.get("schema") != "boxfusion.scannet_fastsam_f0_full200.scene.v1"
        or sidecar.get("protocol_id")
        != "F0-frozen-FastSAM-x-residual-automatic-mask-shadow-full200"
        or sidecar.get("complete") is not True
    ):
        raise SAM2BenchmarkError("F0 sidecar contract differs")
    frames = sidecar.get("frames")
    if not isinstance(frames, list) or frame_ordinal >= len(frames):
        raise SAM2BenchmarkError("F0 frame ordinal is absent")
    frame = frames[frame_ordinal]
    if (
        not isinstance(frame, dict)
        or frame.get("frame_ordinal") != frame_ordinal
        or frame.get("successful") is not True
    ):
        raise SAM2BenchmarkError("F0 benchmark frame is not a successful exact ordinal")
    candidates = frame.get("funnel", {}).get("candidates")
    if not isinstance(candidates, list) or len(candidates) < prompt_count:
        raise SAM2BenchmarkError("F0 benchmark frame has too few selected candidates")
    boxes: list[np.ndarray] = []
    source_rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates[:prompt_count]):
        if not isinstance(candidate, dict) or candidate.get("rank") != rank:
            raise SAM2BenchmarkError("F0 candidate rank/order differs")
        box = np.asarray(candidate.get("tight_box_xyxy"), dtype=np.float32)
        if (
            box.shape != (4,)
            or not np.isfinite(box).all()
            or box[0] < 0.0
            or box[1] < 0.0
            or box[2] > IMAGE_HW[1]
            or box[3] > IMAGE_HW[0]
            or box[2] <= box[0]
            or box[3] <= box[1]
        ):
            raise SAM2BenchmarkError("F0 prompt box is invalid")
        boxes.append(box)
        source_rows.append(
            {
                "rank": rank,
                "raw_index": int(candidate["raw_index"]),
                "mask_sha256": str(candidate["mask_sha256"]),
                "tight_box_xyxy": box.astype(float).tolist(),
            }
        )
    inputs = frame.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("rgb_path"), str):
        raise SAM2BenchmarkError("F0 RGB input seal is absent")
    image_path = _regular_file(Path(inputs["rgb_path"]), "F0 RGB image")
    image_sha = str(inputs.get("rgb_sha256"))
    if _sha256(image_path) != image_sha:
        raise SAM2BenchmarkError("F0 RGB image hash differs")
    metadata = {
        "scene_id": sidecar.get("scene_id"),
        "scene_index": sidecar.get("scene_index"),
        "frame_id": frame.get("frame_id"),
        "frame_ordinal": frame_ordinal,
        "f0_sidecar": os.fspath(_regular_file(sidecar_path, "F0 sidecar")),
        "f0_sidecar_sha256": _sha256(_regular_file(sidecar_path, "F0 sidecar")),
        "f0_sources": source_rows,
    }
    return image_path, image_sha, np.stack(boxes), metadata


def _select_best_masks(
    masks: np.ndarray, scores: np.ndarray, *, prompt_count: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_array = np.asarray(masks)
    score_array = np.asarray(scores, dtype=np.float64)
    if prompt_count == 1 and mask_array.ndim == 3:
        mask_array = mask_array[None, ...]
    if prompt_count == 1 and score_array.ndim == 1:
        score_array = score_array[None, ...]
    if (
        mask_array.ndim != 4
        or mask_array.shape[0] != prompt_count
        or mask_array.shape[2:] != IMAGE_HW
        or score_array.shape != mask_array.shape[:2]
        or mask_array.shape[1] < 1
        or not np.isfinite(score_array).all()
    ):
        raise SAM2BenchmarkError(
            f"SAM2 output shape differs: masks={mask_array.shape}, scores={score_array.shape}"
        )
    best = np.argmax(score_array, axis=1).astype(np.int64)
    selected = mask_array[np.arange(prompt_count), best].astype(bool, copy=False)
    selected_scores = score_array[np.arange(prompt_count), best]
    return np.ascontiguousarray(selected), selected_scores, best


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _create_only_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"refusing to overwrite SAM2 receipt: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def benchmark(
    *,
    model_root: Path,
    checkpoint: Path,
    config_name: str,
    f0_sidecar: Path,
    frame_ordinal: int,
    prompt_count: int,
    output: Path,
    device: str,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    if device != "cuda" or not torch.cuda.is_available():
        raise SAM2BenchmarkError("formal SAM2 runtime benchmark requires CUDA")
    if warmup < 2 or runs < 3:
        raise SAM2BenchmarkError("warmup must be >=2 and runs must be >=3")
    root = model_root.resolve(strict=True)
    if not root.is_dir():
        raise SAM2BenchmarkError("SAM2 source root is not a directory")
    checkpoint_source = _regular_file(checkpoint, "SAM2 checkpoint")
    config_source = _regular_file(root / "sam2" / config_name, "SAM2 config")
    image_path, image_sha, boxes, f0_metadata = _load_f0_prompts(
        f0_sidecar, frame_ordinal=frame_ordinal, prompt_count=prompt_count
    )
    bgr = cv2.imread(os.fspath(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SAM2BenchmarkError(f"could not decode F0 RGB image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (IMAGE_HW[1], IMAGE_HW[0]), interpolation=cv2.INTER_LINEAR)
    if rgb.shape != (*IMAGE_HW, 3) or rgb.dtype != np.uint8:
        raise SAM2BenchmarkError("resized SAM2 RGB input differs")

    sys.path.insert(0, os.fspath(root))
    from sam2.build_sam import build_sam2  # noqa: PLC0415
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: PLC0415

    model = build_sam2(
        config_file=config_name,
        ckpt_path=os.fspath(checkpoint_source),
        device=device,
        mode="eval",
        apply_postprocessing=True,
    )
    predictor = SAM2ImagePredictor(model)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    allocated_after_load = int(torch.cuda.memory_allocated())

    encoder_ms: list[float] = []
    decoder_host_ms: list[float] = []
    total_ms: list[float] = []
    selected_score_mean: list[float] = []
    selected_pixel_count_mean: list[float] = []
    last_mask_hash = ""
    last_best: list[int] = []
    with torch.inference_mode():
        for index in range(warmup + runs):
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.set_image(rgb)
            torch.cuda.synchronize()
            encoded = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                masks, scores, _ = predictor.predict(
                    box=boxes,
                    multimask_output=True,
                    return_logits=False,
                )
            selected, selected_scores, best = _select_best_masks(
                masks, scores, prompt_count=prompt_count
            )
            packed = np.packbits(selected.reshape(prompt_count, -1), axis=1, bitorder="little")
            last_mask_hash = hashlib.sha256(packed.tobytes(order="C")).hexdigest()
            last_best = best.astype(int).tolist()
            torch.cuda.synchronize()
            finished = time.perf_counter()
            if index >= warmup:
                encoder_ms.append((encoded - started) * 1000.0)
                decoder_host_ms.append((finished - encoded) * 1000.0)
                total_ms.append((finished - started) * 1000.0)
                selected_score_mean.append(float(np.mean(selected_scores)))
                selected_pixel_count_mean.append(float(np.mean(selected.sum(axis=(1, 2)))))

    key_sources = {
        "build_sam": root / "sam2/build_sam.py",
        "image_predictor": root / "sam2/sam2_image_predictor.py",
        "transforms": root / "sam2/utils/transforms.py",
        "model_base": root / "sam2/modeling/sam2_base.py",
        "config": config_source,
    }
    source_sha256 = {
        name: _sha256(_regular_file(path, f"SAM2 {name}"))
        for name, path in key_sources.items()
    }
    gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
    gap25_ms = 25.0 / 30.0 * 1000.0
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "purpose": "provider_runtime_only_not_accuracy_or_ap_evidence",
        "gt_access": False,
        "native_prediction_access": False,
        "evaluator_access": False,
        "labels_loaded": False,
        "clip_access": False,
        "training": False,
        "optimizer": False,
        "online_learning": False,
        "model": {
            "family": "SAM2.1",
            "variant": "Hiera-L",
            "checkpoint_requested_path": os.fspath(checkpoint),
            "checkpoint_resolved_path": os.fspath(checkpoint_source),
            "checkpoint_bytes": checkpoint_source.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint_source),
            "config_name": config_name,
            "source_root": os.fspath(root),
            "source_sha256": source_sha256,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "inference_dtype": "bfloat16_autocast",
            "postprocessing": True,
        },
        "input": {
            **f0_metadata,
            "image": os.fspath(image_path),
            "image_sha256": image_sha,
            "resized_hw": list(IMAGE_HW),
            "prompt_count": prompt_count,
            "boxes_xyxy": boxes.astype(float).tolist(),
            "multimask_output": True,
            "mask_choice": "maximum_frozen_predicted_iou_tie_lowest_index",
            "output_mask_sha256": last_mask_hash,
            "selected_hypothesis_indices": last_best,
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
            "decoder_and_host_mask_ms": decoder_host_ms,
            "total_ms": total_ms,
            "encoder_mean_ms": float(np.mean(encoder_ms)),
            "decoder_and_host_mask_mean_ms": float(np.mean(decoder_host_ms)),
            "total_mean_ms": float(np.mean(total_ms)),
            "total_p50_ms": _quantile(total_ms, 0.50),
            "total_p95_ms": _quantile(total_ms, 0.95),
            "total_max_ms": float(np.max(total_ms)),
            "amortized_mean_ms_per_raw_frame_gap25": float(np.mean(total_ms) / 25.0),
            "selected_predicted_iou_mean": float(np.mean(selected_score_mean)),
            "selected_mask_pixel_count_mean": float(np.mean(selected_pixel_count_mean)),
            "allocated_after_model_load_bytes": allocated_after_load,
            "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "gap25_at_30fps_budget_ms": gap25_ms,
            "all_measured_runs_within_gap25_budget": bool(
                np.all(np.asarray(total_ms, dtype=np.float64) < gap25_ms)
            ),
        },
    }
    _create_only_json(output, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--f0-sidecar", type=Path, default=DEFAULT_F0_SIDECAR)
    parser.add_argument("--frame-ordinal", type=int, default=48)
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = benchmark(
        model_root=args.model_root,
        checkpoint=args.checkpoint,
        config_name=args.config,
        f0_sidecar=args.f0_sidecar,
        frame_ordinal=args.frame_ordinal,
        prompt_count=args.prompt_count,
        output=args.output,
        device=args.device,
        warmup=args.warmup,
        runs=args.runs,
    )
    print(json.dumps(result["runtime"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
