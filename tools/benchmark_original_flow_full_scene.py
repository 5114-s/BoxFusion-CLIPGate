#!/usr/bin/env python3
"""Time the current full model using the released BoxFusion frame loop."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()

    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "runtime.json"
    if receipt_path.exists():
        raise RuntimeError(f"refusing to overwrite runtime receipt: {receipt_path}")

    frames = (
        ROOT / "upstream_clean" / "scannet_readme_frames"
        / args.scene / "frames"
    ).resolve()
    if not frames.is_dir():
        raise FileNotFoundError(frames)

    cfg = yaml.safe_load(args.config.resolve().read_text(encoding="utf-8"))
    cfg["data"]["datadir"] = str(frames)
    cfg["data"]["output_dir"] = str(output / "native")
    (output / "native").mkdir(parents=True, exist_ok=True)
    cfg["vis"]["rerun"] = False
    cfg["eval"] = True
    cfg.setdefault("association", {}).setdefault("appearance_gate", {})[
        "enabled"
    ] = False
    cfg["association"].setdefault("pvq_ar", {})["enabled"] = False
    cfg["lifting"]["proposal_cache"]["mode"] = "disabled"
    cfg["lifting"]["boxer"]["diagnostics_dir"] = str(
        output / "native_boxer_diagnostics"
    )
    online = cfg.setdefault("online_candidate_map", {})
    online["enabled"] = True
    online["output_root"] = str(output / "full")
    online["diagnostics_root"] = str(output / "full_diagnostics")
    online["write_every_keyframe"] = False
    online["asynchronous"] = True
    online.setdefault("provider", {})["device"] = "cuda:0"
    online["provider"]["diagnostics_root"] = str(
        output / "provider_diagnostics"
    )
    online.setdefault("state", {}).setdefault("m1p", {})[
        "use_children"
    ] = False
    cfg["runtime_benchmark"] = {
        "synchronize_cuda": True,
        "prepare_all_raw_frames": True,
        "receipt_path": str(receipt_path),
    }
    (output / "effective_config.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8"
    )

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device_index)
    import demo

    checkpoint = torch.load(
        ROOT / "models/cutr_rgbd.pth", map_location="cpu", weights_only=False
    )["model"]
    dimension = checkpoint["backbone.0.patch_embed.proj.weight"].shape[0]
    model = demo.make_cubify_transformer(
        dimension=dimension, depth_model=True
    ).eval().cuda()
    model.load_state_dict(checkpoint)
    del checkpoint
    clip_model, preprocess = demo.load_clip(
        str(ROOT / "models/open_clip_pytorch_model.bin")
    )
    categories = np.genfromtxt(
        ROOT / "data/panoptic_categories_nomerge.txt", delimiter="\n", dtype=str
    )
    features = torch.load(
        ROOT / "data/class_features.pt", weights_only=False
    ).cuda()
    raw_dataset = demo.get_dataset(cfg)
    raw_dataset.load_arkit_depth = True
    dataset = CountedStream(raw_dataset)
    augmentor = demo.Augmentor(("wide/image", "wide/depth"))
    preprocessor = demo.Preprocessor()

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

    if not receipt_path.is_file():
        raise RuntimeError("demo did not write a synchronized runtime receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if int(receipt["raw_frames"]) != dataset.consumed:
        raise RuntimeError(
            f"frame count mismatch: receipt={receipt['raw_frames']} "
            f"stream={dataset.consumed}"
        )
    if receipt.get("prepare_all_raw_frames") is not True:
        raise RuntimeError("original per-frame preparation was not enabled")
    receipt.update({
        "arm": "full",
        "scene": args.scene,
        "gpu_count": 1,
        "gap": int(cfg["data"]["gap"]),
        "children_enabled": False,
        "scene_cache_prewarmed": False,
        "original_frame_loop": True,
        "config_sha256": sha256(args.config.resolve()),
        "demo_sha256": sha256(ROOT / "demo.py"),
        "online_map_sha256": sha256(ROOT / "boxfusion/online_candidate_map.py"),
        "online_runtime_sha256": sha256(
            ROOT / "boxfusion/online_candidate_runtime.py"
        ),
    })
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
