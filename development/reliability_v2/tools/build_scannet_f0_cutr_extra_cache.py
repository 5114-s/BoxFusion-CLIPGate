#!/usr/bin/env python3
"""Build the frozen score-0.5 CuTR keyframe cache needed by F0 full200.

This is a cache-only extraction of the released BoxFusion CuTR boundary.  It
stops before CLIP, association, fusion, terminal boxes, annotations, or any
evaluator.  The first 100 paper scenes already have a sealed v2 cache; this
tool is intended for the deterministic extra-100 extension only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml


SOURCE_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev"
).resolve()
if os.fspath(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SOURCE_ROOT))

from boxfusion.box_manager import BoxManager  # noqa: E402
from boxfusion.cubify_transformer import make_cubify_transformer  # noqa: E402
from boxfusion.preprocessor import Augmentor, Preprocessor  # noqa: E402
from boxfusion.proposal_cache import SCHEMA, build_proposal_cache  # noqa: E402
from tools.utils import get_dataset, move_input_to_current_device  # noqa: E402


EXPECTED_SCHEMA = "boxfusion.cutr_postfilter_cache.v2"
EXPECTED_SCORE_THRESHOLD = 0.5
EXPECTED_GAP = 25
EXPECTED_SOURCE_CONFIG = SOURCE_ROOT / "config/scannet_cutr_paired_scorefix.yaml"
DEFAULT_MODEL = Path("/data/ZhaoX/BoxFusion/models/cutr_rgbd.pth")
DEFAULT_RAW_ROOT = Path("/extra/ZhaoX/scannet_data/scans")
DEFAULT_SCENE_LIST = Path(
    "/data/ZhaoX/BoxFusion/evaluation/data_util/meta_data/"
    "scannetv2_val_f0_full200.txt"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/ZhaoX/BoxFusion/cache/f0_fastsam_full200/"
    "cutr-score05-gap25-v2-extra100"
)
SCHEMA_RECEIPT = "boxfusion.f0_cutr_extra_cache_build.v1"


class BuildError(RuntimeError):
    """Raised when the frozen cache-only contract cannot be honored."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_scenes(path: Path, expected_count: int) -> list[str]:
    if path.is_symlink() or not path.is_file():
        raise BuildError(f"scene list must be a regular file: {path}")
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_count or len(set(rows)) != len(rows):
        raise BuildError(
            f"expected {expected_count} unique scenes, found {len(rows)}"
        )
    return rows


def _load_frozen_config(output_root: Path) -> dict[str, Any]:
    with EXPECTED_SOURCE_CONFIG.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if (
        cfg.get("dataset") != "scannet"
        or float(cfg["detection"]["score_thresh"]) != EXPECTED_SCORE_THRESHOLD
        or int(cfg["data"]["gap"]) != EXPECTED_GAP
        or cfg["lifting"]["proposal_cache"]["mode"] != "record"
    ):
        raise BuildError("released score-0.5 CuTR configuration changed")
    cfg = copy.deepcopy(cfg)
    cfg["lifting"]["proposal_cache"]["root"] = os.fspath(output_root.parent)
    cfg["lifting"]["proposal_cache"]["namespace"] = output_root.name
    # No downstream stage is executed, but keep every upstream detector/filter
    # field byte-for-byte identical to the released score-0.5 configuration.
    return cfg


def _load_model(checkpoint_path: Path, device: torch.device):
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise BuildError(f"CuTR checkpoint must be a regular file: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    width = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
    model = make_cubify_transformer(dimension=width, depth_model=True).eval()
    model.load_state_dict(checkpoint)
    model.requires_grad_(False)
    return model.to(device)


def _cache_inputs(sample: Mapping[str, Any], image: np.ndarray) -> dict[str, Any]:
    return {
        "image": image,
        "depth": sample["wide"]["depth"][-1],
        "image_K": sample["sensor_info"].wide.image.K[-1],
        "depth_K": sample["sensor_info"].gt.depth.K[-1],
        "camera_to_world": sample["sensor_info"].gt.RT[-1],
    }


def _process_scene(
    *,
    scene: str,
    cfg_template: Mapping[str, Any],
    raw_root: Path,
    output_root: Path,
    model: Any,
    augmentor: Any,
    preprocessor: Any,
    sparse_reader: bool,
) -> dict[str, Any]:
    scene_cache_root = output_root / scene
    manifest_path = scene_cache_root / "manifest.json"
    if manifest_path.is_file():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if value.get("schema") != EXPECTED_SCHEMA or value.get("scene_id") != scene:
            raise BuildError(f"invalid existing cache manifest: {manifest_path}")
        return {
            "scene_id": scene,
            "status": "already_complete",
            "manifest": os.fspath(manifest_path.resolve()),
            "manifest_sha256": _sha256(manifest_path),
            "record_count": int(value["record_count"]),
            "proposal_count": int(value["proposal_count"]),
            "wall_seconds": 0.0,
        }
    if scene_cache_root.exists() or scene_cache_root.is_symlink():
        raise BuildError(f"refusing partial/orphan cache scene: {scene_cache_root}")

    cfg = copy.deepcopy(dict(cfg_template))
    cfg["data"]["datadir"] = os.fspath(raw_root / scene)
    dataset = get_dataset(cfg)
    dataset.load_arkit_depth = True
    dataset_length = len(dataset)
    if dataset_length <= 0:
        raise BuildError(f"empty ScanNet scene: {scene}")
    box_manager = BoxManager(cfg)
    cache = build_proposal_cache(cfg, device=model.pixel_mean.device)
    if cache is None or not cache.is_record or SCHEMA != EXPECTED_SCHEMA:
        raise BuildError("CuTR v2 proposal cache is unavailable")
    cache.bind_scene(scene, dataset_length=dataset_length, gap=EXPECTED_GAP)

    started = time.perf_counter()
    processed = 0
    proposal_count = 0
    # The released demo exits once fewer than ``gap`` future rows remain.  Its
    # sealed v2 schedule therefore contains multiples of 25 strictly before
    # ``dataset_length - gap`` (for example 0..1625 for length 1651).  Bound
    # the iterable to that same prefix; the dataset itself otherwise indexes
    # one row past the end.
    terminal_prefix = max(dataset_length - EXPECTED_GAP, 0)
    scheduled_frame_ids = tuple(range(0, terminal_prefix, EXPECTED_GAP))

    if sparse_reader:
        # ``ScannetDataset.__iter__`` eagerly decodes every raw frame.  The
        # released demo nevertheless performs all proposal work only at
        # gap-25 keyframes, and skipped iterator rows have no mutable/RNG side
        # effects.  Filter its already-resolved path/pose tables to that exact
        # schedule so we do not decode the other 24/25 frames.
        for role in ("img_files", "depth_paths", "poses"):
            values = getattr(dataset, role, None)
            if not isinstance(values, (list, tuple)) or len(values) != dataset_length:
                raise BuildError(f"ScannetDataset {role} cannot be sparsely scheduled")
            setattr(dataset, role, [values[index] for index in scheduled_frame_ids])
        dataset.frame_ids = range(len(scheduled_frame_ids))
        dataset.num_frames = len(scheduled_frame_ids)
        scheduled_samples = zip(scheduled_frame_ids, dataset)
    else:
        scheduled_samples = (
            (frame_id, sample)
            for frame_id, sample in enumerate(
                itertools.islice(dataset, terminal_prefix)
            )
            if frame_id % EXPECTED_GAP == 0
        )

    for frame_id, sample in scheduled_samples:
        if not isinstance(sample.get("meta"), dict):
            raise BuildError(f"ScannetDataset metadata is invalid: {scene}/{frame_id}")
        # Sparse iteration numbers decoded rows 0..K-1; restore the raw frame
        # identity that dense iteration supplied to downstream packaging.
        sample["meta"]["timestamp"] = frame_id
        image = np.moveaxis(sample["wide"]["image"][-1].numpy(), 0, -1)
        packaged = augmentor.package(sample)
        packaged = move_input_to_current_device(packaged, model.pixel_mean)
        packaged = preprocessor.preprocess([packaged])
        attempt = "primary"
        with torch.inference_mode():
            instances = model(packaged)[0]
        instances = instances[
            instances.scores >= float(cfg["detection"]["score_thresh"])
        ]
        if cfg["detection"]["uv_bound"]:
            keep = box_manager.check_uv_bounds(
                instances.pred_proj_xy,
                image.shape[1],
                image.shape[0],
                ratio=cfg["detection"]["uv_bound_value"],
            )
            instances = instances[keep]
        if cfg["detection"]["floor_mask"]:
            floor = box_manager.check_floor_mask(
                instances.pred_boxes_3d.tensor,
                ratio=cfg["detection"]["floor_ratio"],
            )
            instances = instances[~floor]
        if len(instances) == 0 and frame_id == 0:
            attempt = "retry"
            with torch.inference_mode():
                instances = model(packaged)[0]
            instances = instances[
                instances.scores
                >= float(cfg["detection"]["score_thresh"] / 4.0)
            ]
            if cfg["detection"]["uv_bound"]:
                keep = box_manager.check_uv_bounds(
                    instances.pred_proj_xy,
                    image.shape[1],
                    image.shape[0],
                    ratio=cfg["detection"]["uv_bound_value"],
                )
                instances = instances[keep]
        instances = cache.record(
            scene,
            frame_id,
            instances,
            attempt_id=attempt,
            inputs=_cache_inputs(sample, image),
        )
        processed += 1
        proposal_count += len(instances)

    prediction_receipt = output_root.parent / "_cache_only_receipts" / f"{scene}.json"
    _atomic_json(
        prediction_receipt,
        {
            "schema": SCHEMA_RECEIPT,
            "scene_id": scene,
            "terminal_prediction_generated": False,
            "record_count": processed,
            "proposal_count": proposal_count,
        },
    )
    manifest_path = cache.finalize(scene, prediction_path=prediction_receipt)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "scene_id": scene,
        "status": "built",
        "manifest": os.fspath(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "record_count": int(manifest["record_count"]),
        "proposal_count": int(manifest["proposal_count"]),
        "wall_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--expected-scene-count", type=int, default=200)
    parser.add_argument("--skip-prefix-scenes", type=int, default=100)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--dense-reader",
        action="store_true",
        help="audit-only legacy reader that decodes skipped non-keyframes",
    )
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise BuildError("invalid shard specification")
    rows = _read_scenes(args.scene_list, args.expected_scene_count)
    if not 0 <= args.skip_prefix_scenes <= len(rows):
        raise BuildError("invalid skip-prefix-scenes")
    selected = rows[args.skip_prefix_scenes :]
    selected = selected[args.shard_index :: args.num_shards]
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise BuildError("max-scenes must be positive")
        selected = selected[: args.max_scenes]
    plan = {
        "schema": SCHEMA_RECEIPT,
        "mode": "plan_only" if args.plan_only else "build",
        "scene_count": len(selected),
        "scene_order": selected,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "score_threshold": EXPECTED_SCORE_THRESHOLD,
        "gap": EXPECTED_GAP,
        "reader": "dense_audit" if args.dense_reader else "sparse_gap25",
        "source_config": os.fspath(EXPECTED_SOURCE_CONFIG),
        "source_config_sha256": _sha256(EXPECTED_SOURCE_CONFIG),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "scene_list_sha256": _sha256(args.scene_list),
        "output_root": os.fspath(args.output_root),
    }
    if args.plan_only:
        print(json.dumps(plan, sort_keys=True))
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)
    augmentor = Augmentor(("wide/image", "wide/depth"))
    preprocessor = Preprocessor()
    cfg = _load_frozen_config(args.output_root)
    fingerprint = hashlib.sha256(
        json.dumps(plan, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    os.environ["BOXFUSION_PROPOSAL_CACHE_PRODUCER_FINGERPRINT"] = fingerprint
    scenes = []
    for index, scene in enumerate(selected, 1):
        row = _process_scene(
            scene=scene,
            cfg_template=cfg,
            raw_root=args.raw_root,
            output_root=args.output_root,
            model=model,
            augmentor=augmentor,
            preprocessor=preprocessor,
            sparse_reader=not args.dense_reader,
        )
        scenes.append(row)
        print(
            f"[{index}/{len(selected)}] {scene}: {row['status']} "
            f"frames={row['record_count']} proposals={row['proposal_count']} "
            f"wall={row['wall_seconds']:.1f}s",
            flush=True,
        )
    receipt = {
        **plan,
        "mode": "build",
        "producer_fingerprint": fingerprint,
        "scenes": scenes,
        "totals": {
            "scene_count": len(scenes),
            "record_count": sum(row["record_count"] for row in scenes),
            "proposal_count": sum(row["proposal_count"] for row in scenes),
            "wall_seconds": sum(row["wall_seconds"] for row in scenes),
        },
    }
    shard_receipt = (
        args.output_root.parent
        / "_build_receipts"
        / f"shard{args.shard_index}-of-{args.num_shards}.json"
    )
    if shard_receipt.exists() or shard_receipt.is_symlink():
        raise BuildError(f"refusing to overwrite shard receipt: {shard_receipt}")
    _atomic_json(shard_receipt, receipt)
    print(f"Saved: {shard_receipt.resolve()}", flush=True)


if __name__ == "__main__":
    main()
