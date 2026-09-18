#!/usr/bin/env python3
"""Time one ScanNet scene with a frozen official-code snapshot."""
from __future__ import annotations

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--repository-root', type=Path, required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    source = args.source_root.resolve()
    root = args.repository_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(source))

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_device(args.device_index)
    import demo

    cfg = yaml.safe_load((source / 'config/scannet.yaml').read_text())
    cfg['data']['datadir'] = str(
        root / 'upstream_clean/scannet_readme_frames' / args.scene / 'frames')
    cfg['data']['output_dir'] = str(output / 'predictions')
    (output / 'predictions').mkdir(exist_ok=True)
    cfg['vis']['rerun'] = False
    cfg['eval'] = True
    if 'appearance_gate' in cfg.get('association', {}):
        cfg['association']['appearance_gate']['enabled'] = False
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    cold_start = time.perf_counter()
    checkpoint = torch.load(root / 'models/cutr_rgbd.pth', map_location='cpu',
                            weights_only=False)['model']
    dimension = checkpoint['backbone.0.patch_embed.proj.weight'].shape[0]
    model = demo.make_cubify_transformer(
        dimension=dimension, depth_model=True).eval().cuda()
    model.load_state_dict(checkpoint)
    del checkpoint
    clip_model, preprocess = demo.load_clip(
        str(root / 'models/open_clip_pytorch_model.bin'))
    categories = np.genfromtxt(root / 'data/panoptic_categories_nomerge.txt',
                               delimiter='\n', dtype=str)
    features = torch.load(root / 'data/class_features.pt',
                          weights_only=False).cuda()
    raw_dataset = demo.get_dataset(cfg)
    raw_dataset.load_arkit_depth = True
    dataset = CountedStream(raw_dataset)
    augmentor = demo.Augmentor(('wide/image', 'wide/depth'))
    preprocessor = demo.Preprocessor()
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        demo.run(cfg, model, dataset, clip_model, preprocess, categories,
                 features, augmentor, preprocessor,
                 score_thresh=cfg['detection']['score_thresh'],
                 gap=cfg['data']['gap'], re_vis=False)
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    torch.cuda.synchronize()
    finished = time.perf_counter()
    run_seconds = finished - loaded
    result = {
        'schema': 'boxfusion.paper_matched.official_scene.v1',
        'scene': args.scene,
        'source_root': str(source),
        'configured_gap': int(cfg['data']['gap']),
        'consumed_raw_frames': dataset.consumed,
        'model_load_seconds': loaded - cold_start,
        'run_seconds': run_seconds,
        'raw_fps': dataset.consumed / run_seconds,
        'peak_allocated_bytes': int(torch.cuda.max_memory_allocated()),
        'device': torch.cuda.get_device_name(args.device_index),
        'torch': torch.__version__,
        'model_load_excluded': True,
        'visualization': False,
        'proposal_replay': False,
        'output_serialization_included': True,
    }
    (output / 'runtime.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
