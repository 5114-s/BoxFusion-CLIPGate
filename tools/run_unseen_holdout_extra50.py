#!/usr/bin/env python3
"""Frozen-constant unseen-holdout runner (pre-registered, see protocol.json).

Phase A: native demo.run per scene with the frozen t05 config (paths only).
Phase B: one integrated_online batch (M1-P + M2 nativelogit + M5 dual +
strict-online M1-A, all constants from the module defaults / env of the
frozen recipe) over the same scenes.

No constant may be changed after results are seen. Resumable per scene.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/unseen_holdout_extra50_20260916'
SCANS = Path('/extra/ZhaoX/scannet_data/scans')
PYTHON = '/home/admin1/miniconda3/envs/boxfusion2/bin/python'
sys.path.insert(0, str(ROOT))


def link_scene(scene: str) -> Path:
    """<root>/<scene>/frames/{color,depth,pose,intrinsic} symlinks to /extra.

    The root segment is named 'scannet_input': BoxFusion.__init__ dispatches
    on the substring 'scannet' in basedir, so the path must keep it.
    """
    frames = OUT / 'scannet_input' / scene / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    for sub in ('color', 'depth', 'pose', 'intrinsic'):
        target = frames / sub
        source = SCANS / scene / sub
        if not source.is_dir():
            raise FileNotFoundError(source)
        if target.is_symlink() or target.exists():
            continue
        target.symlink_to(source)
    return frames.parent


def phase_native(scenes, device):
    import torch
    import yaml
    import demo

    native_dir = OUT / 'native'
    diag = OUT / 'native_diag'
    (native_dir).mkdir(exist_ok=True)
    (diag / 't05_boxer').mkdir(parents=True, exist_ok=True)
    cfg0 = yaml.safe_load(
        (ROOT / 'config/scannet_t05_boxer_kfmap_score05.yaml').read_text())

    checkpoint = torch.load(ROOT / 'models/cutr_rgbd.pth', map_location='cpu',
                            weights_only=False)['model']
    dimension = checkpoint['backbone.0.patch_embed.proj.weight'].shape[0]
    model = demo.make_cubify_transformer(dimension=dimension,
                                         depth_model=True).eval().cuda()
    model.load_state_dict(checkpoint)
    del checkpoint
    clip_model, preprocess = demo.load_clip(
        str(ROOT / 'models/open_clip_pytorch_model.bin'))
    categories = np.genfromtxt(ROOT / 'data/panoptic_categories_nomerge.txt',
                               delimiter='\n', dtype=str)
    features = torch.load(ROOT / 'data/class_features.pt',
                          weights_only=False).cuda()

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

    for i, scene in enumerate(scenes, 1):
        out_pkl = native_dir / f'{scene}_boxes.pkl'
        nms = diag / f'{scene}_pvq_nms.jsonl'
        if out_pkl.is_file() and nms.is_file():
            print(f'[{i}/{len(scenes)}] {scene} native CACHED', flush=True)
            continue
        scene_root = link_scene(scene)
        cfg = json.loads(json.dumps(cfg0))
        cfg['data']['datadir'] = str(scene_root / 'frames')
        cfg['data']['output_dir'] = str(native_dir)
        cfg['lifting']['proposal_cache']['mode'] = 'disabled'
        cfg['lifting']['boxer']['diagnostics_dir'] = \
            str(diag / 't05_boxer' / scene)
        cfg['association']['pvq_ar']['diagnostics_dir'] = str(diag)
        cfg['vis']['rerun'] = False
        dataset = demo.get_dataset(cfg)
        dataset.load_arkit_depth = True
        stream = CountedStream(dataset)
        augmentor = demo.Augmentor(('wide/image', 'wide/depth'))
        preprocessor = demo.Preprocessor()
        t0 = time.perf_counter()
        try:
            demo.run(cfg, model, stream, clip_model, preprocess, categories,
                     features, augmentor, preprocessor, score_thresh=.5,
                     gap=25, re_vis=False)
        except SystemExit as error:
            if error.code not in (None, 0):
                raise
        assert out_pkl.is_file(), scene
        print(f'[{i}/{len(scenes)}] {scene} native '
              f'{time.perf_counter() - t0:.1f}s frames={stream.consumed}',
              flush=True)
        (OUT / 'native_progress.json').write_text(json.dumps(
            {'done': i, 'last': scene}))
        del dataset, stream
        gc.collect()
    del model, clip_model, preprocess, features
    gc.collect()
    torch.cuda.empty_cache()


def phase_integrated(scenes, device):
    persistent = OUT / 'persistent'
    m1 = OUT / 'm1_only'
    traces = OUT / 'traces'
    for d in (persistent, m1, traces):
        d.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update({
        'CUDA_VISIBLE_DEVICES': str(device),
        'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'PYTHONUNBUFFERED': '1', 'HF_HUB_OFFLINE': '1',
        'M2_MODE': 'nativelogit', 'M2_EXCLUSIVE': '1',
        'M5_OFF': '0', 'M5_CH2_OFF': '0', 'CAUSAL_TAU': '0.5',
        'M5_SCORE_THR': '1.0',
        'CAUSAL_NAT': str(OUT / 'native'),
        'CAUSAL_KFD': str(OUT / 'native_diag'),
        'CAUSAL_OUT': str(persistent),
        'CAUSAL_M1_OUT': str(m1),
        'M1A_ONLINE': '1', 'M1A_TOPM': '300',
        'M1A_MAX_ACTIVE': '4096', 'M1A_MAX_BIRTHS': '640',
    })
    scene_file = OUT / 'integrated_scenes.txt'
    native = OUT / 'native'
    remaining = [s for s in scenes
                 if (native / f'{s}_boxes.pkl').is_file()
                 and not (persistent / f'{s}_boxes.pkl').is_file()]
    if not remaining:
        print('integrated: all cached', flush=True)
        return
    scene_file.write_text('\n'.join(remaining) + '\n')
    result = subprocess.run(
        [PYTHON, 'tools/integrated_online.py', '--batch', str(scene_file),
         '--score-view', 'persistent'],
        cwd=ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f'integrated batch failed: {result.returncode}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('native', 'integrated', 'both'),
                        default='both')
    parser.add_argument('--device', default='1')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()
    protocol = json.loads((OUT / 'protocol.json').read_text())
    scenes = protocol['scenes']
    if args.limit:
        scenes = scenes[:args.limit]
    if args.phase in ('native', 'both'):
        phase_native(scenes, args.device)
    if args.phase in ('integrated', 'both'):
        phase_integrated(scenes, args.device)


if __name__ == '__main__':
    main()
