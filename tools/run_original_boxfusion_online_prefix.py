#!/usr/bin/env python3
"""Export causal 25/50/75/100% readouts from frozen original BoxFusion.

The frozen source receives only a read-only callback after each keyframe.  The
callback serializes the current native records and never mutates mapper state.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCENE_LIST = ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt"
FRAME_ROOT = ROOT / "upstream_clean/scannet_readme_frames"
NATIVE_SCHEDULE = (
    ROOT
    / "development/final_controls/experiments/mvsr_multiview_audit_20261012/native_prefix"
)
FRACTIONS = (0.25, 0.50, 0.75)
ORIGINAL_TERMINAL = (
    ROOT
    / "reports/original_boxfusion_vs_final_novis_calrv1_efficiency100_paired_20261008"
    / "receipts/original"
)


class StopAfterPrefix(Exception):
    """Stop a scene immediately after its last requested causal prefix."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_models(demo, repository: Path):
    import torch

    checkpoint = torch.load(
        repository / "models/cutr_rgbd.pth",
        map_location="cpu",
        weights_only=False,
    )["model"]
    dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
    model = demo.make_cubify_transformer(
        dimension=dimension, depth_model=True
    ).eval().cuda()
    model.load_state_dict(checkpoint)
    del checkpoint
    clip_model, preprocess = demo.load_clip(
        str(repository / "models/open_clip_pytorch_model.bin")
    )
    categories = np.genfromtxt(
        repository / "data/panoptic_categories_nomerge.txt",
        delimiter="\n",
        dtype=str,
    )
    features = torch.load(
        repository / "data/class_features.pt", weights_only=False
    ).cuda()
    return model, clip_model, preprocess, categories, features


def checkpoint_frames(scene: str) -> tuple[dict[int, str], list[int]]:
    path = NATIVE_SCHEDULE / f"{scene}.npz"
    with np.load(path, allow_pickle=False) as values:
        frame_ids = np.asarray(values["frame_ids"], dtype=np.int64)
    if not len(frame_ids) or np.any(np.diff(frame_ids) <= 0):
        raise RuntimeError(f"invalid keyframe schedule: {path}")
    targets: dict[int, str] = {}
    for fraction in FRACTIONS:
        ordinal = min(len(frame_ids) - 1, int(math.ceil(len(frame_ids) * fraction)) - 1)
        targets[int(frame_ids[ordinal])] = f"p{int(round(100 * fraction))}"
    return targets, frame_ids.tolist()


def save_snapshot(demo, path: Path, instances) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if instances is None or len(instances) == 0:
        corners = np.empty((0, 8, 3), dtype=np.float32)
        scores = np.empty((0,), dtype=np.float32)
    else:
        corners = instances.pred_boxes_3d.corners.detach().cpu().numpy()
        scores = instances.scores.detach().cpu().numpy()
        corners, valid = demo.post_process(corners, return_mask=True)
        scores = scores[valid]
    payload = [[(0, corners[index], float(scores[index])) for index in range(len(corners))]]
    temporary = path.with_suffix(path.suffix + ".tmp")
    demo.save_box(payload, str(temporary))
    temporary.replace(path)
    return int(len(corners))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, required=True)
    parser.add_argument(
        "--scene",
        action="append",
        dest="selected_scenes",
        help="Optional explicit scene subset; may be repeated.",
    )
    args = parser.parse_args()
    if not 0 <= args.worker_index < args.worker_count:
        raise ValueError("invalid worker index/count")

    source = args.source_root.resolve()
    output = args.output.resolve()
    official_scenes = [row for row in SCENE_LIST.read_text().split() if row]
    if args.selected_scenes:
        unknown = sorted(set(args.selected_scenes) - set(official_scenes))
        if unknown:
            raise ValueError(f"unknown scenes: {unknown}")
        scenes = list(dict.fromkeys(args.selected_scenes))
    else:
        scenes = official_scenes[args.worker_index :: args.worker_count]
    sys.path.insert(0, str(source))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device_index)
    import demo

    model, clip_model, preprocess, categories, features = load_models(demo, ROOT)
    augmentor = demo.Augmentor(("wide/image", "wide/depth"))
    preprocessor = demo.Preprocessor()
    base_config = yaml.safe_load((source / "config/scannet.yaml").read_text())
    base_config.setdefault("association", {}).setdefault("appearance_gate", {})[
        "enabled"
    ] = False

    worker_log = output / "logs" / f"worker{args.worker_index}.log"
    worker_log.parent.mkdir(parents=True, exist_ok=True)
    with worker_log.open("a", buffering=1) as progress:
        for position, scene in enumerate(scenes, 1):
            receipt = output / "receipts" / f"{scene}.json"
            terminal_source = (
                ORIGINAL_TERMINAL / scene / "native" / f"{scene}_boxes.pkl"
            )
            terminal_destination = (
                output
                / "predictions/boxfusion_p100"
                / f"{scene}_boxes.pkl"
            )
            if not terminal_source.is_file():
                raise FileNotFoundError(terminal_source)
            terminal_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(terminal_source, terminal_destination)
            expected = [
                output / "predictions" / f"boxfusion_p{percent}" / f"{scene}_boxes.pkl"
                for percent in (25, 50, 75, 100)
            ]
            if receipt.is_file() and all(path.is_file() for path in expected):
                print(f"SKIP {position}/{len(scenes)} {scene}", file=progress)
                continue

            targets, frame_ids = checkpoint_frames(scene)
            cfg = yaml.safe_load(yaml.safe_dump(base_config, sort_keys=False))
            cfg["data"]["datadir"] = str((FRAME_ROOT / scene / "frames").resolve())
            unused_terminal_dir = output / "unused_terminal"
            unused_terminal_dir.mkdir(parents=True, exist_ok=True)
            cfg["data"]["output_dir"] = str(unused_terminal_dir)
            cfg["vis"]["rerun"] = False
            cfg["eval"] = True

            raw_dataset = demo.get_dataset(cfg)
            raw_dataset.load_arkit_depth = True
            counts: dict[str, int] = {}

            def export(frame_id: int, instances) -> None:
                tag = targets.get(int(frame_id))
                if tag is None:
                    return
                destination = output / "predictions" / f"boxfusion_{tag}" / f"{scene}_boxes.pkl"
                counts[tag] = save_snapshot(demo, destination, instances)
                if tag == "p75":
                    raise StopAfterPrefix

            scene_log = output / "logs" / "scenes" / f"{scene}.log"
            scene_log.parent.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            with scene_log.open("w") as handle, contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                try:
                    demo.run(
                        cfg,
                        model,
                        raw_dataset,
                        clip_model,
                        preprocess,
                        categories,
                        features,
                        augmentor,
                        preprocessor,
                        score_thresh=cfg["detection"]["score_thresh"],
                        gap=cfg["data"]["gap"],
                        re_vis=False,
                        checkpoint_callback=export,
                    )
                except SystemExit as error:
                    if error.code not in (None, 0):
                        raise
                except StopAfterPrefix:
                    pass
            elapsed = time.perf_counter() - started
            for tag in ("p25", "p50", "p75"):
                if tag not in counts:
                    raise RuntimeError(f"{scene}: checkpoint {tag} was not exported")
            write_json(
                receipt,
                {
                    "schema": "boxfusion.original.online_prefix.scene.v1",
                    "scene": scene,
                    "source_demo_sha256": sha256(source / "demo.py"),
                    "source_config_sha256": sha256(source / "config/scannet.yaml"),
                    "keyframe_count": len(frame_ids),
                    "last_keyframe_id": frame_ids[-1],
                    "checkpoint_frames": {tag: frame for frame, tag in targets.items()},
                    "checkpoint_box_counts": counts,
                    "terminal_prediction_source": str(terminal_source),
                    "elapsed_seconds": elapsed,
                    "future_observations_used_at_checkpoint": False,
                    "checkpoint_callback_mutates_state": False,
                },
            )
            print(
                f"DONE {position}/{len(scenes)} {scene} {elapsed:.1f}s {counts}",
                file=progress,
            )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
