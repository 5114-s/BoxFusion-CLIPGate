#!/usr/bin/env python3
"""One-scene, one-GPU timing for the final M1-P+A+M2 route.

The native prefix is executed once with proposal replay disabled.  All
ablations reuse that exact native result and prefix.  Explicit model loading is
reported separately; warm component FPS includes image I/O, live inference,
fusion/finalization, output serialization, and the common terminal semantic
readout.  It excludes evaluation and visualization.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_prefix(out: Path, scene: str, frames: int) -> Path:
    source = ROOT / 'upstream_clean/scannet_readme_frames' / scene / 'frames'
    folder = out / 'scannet_input' / scene / 'frames'
    folder.mkdir(parents=True, exist_ok=True)
    for sub, suffix in [('color', '.jpg'), ('depth', '.png'), ('pose', '.txt')]:
        destination = folder / sub
        destination.mkdir(exist_ok=True)
        for fid in range(frames):
            src = source / sub / f'{fid}{suffix}'
            if not src.is_file():
                raise FileNotFoundError(src)
            target = destination / src.name
            if not target.exists():
                target.symlink_to(src)
    intrinsic = folder / 'intrinsic'
    if not intrinsic.exists():
        intrinsic.symlink_to(source / 'intrinsic', target_is_directory=True)
    return folder


def semantic_time(out: Path, scene: str, arm: str, prediction_dir: Path,
                  native_dir: Path) -> dict:
    sys.path.insert(0, str(ROOT / 'tools'))
    import tools.run_scannet_semantic_table as semantic

    semantic.OUT = out / 'semantic' / arm
    semantic.OUT.mkdir(parents=True, exist_ok=True)
    semantic.FRAMES = out / 'scannet_input'
    semantic.ARMS = {
        'native': str(native_dir),
        'M1': 'paired reconstruction',
        'M1_M2': str(prediction_dir),
    }
    import torch
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    semantic.classify([scene], 16)
    torch.cuda.synchronize()
    cold_wall = time.perf_counter() - wall_start
    timing_path = semantic.OUT / 'semantic_cache' / f'{scene}.timing.json'
    timing = json.loads(timing_path.read_text())
    timing['cold_wall_seconds'] = cold_wall
    timing['warm_seconds'] = (
        float(timing['semantic_stage_seconds']) - float(timing['startup_seconds']))
    return timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=300)
    parser.add_argument('--scene', default='scene0011_01')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--final-only', action='store_true',
                        help='run only the final M1-P+A+M2 arm')
    args = parser.parse_args()
    out = args.output_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scene = args.scene
    folder = prepare_prefix(out, scene, args.frames)

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.cuda.set_device(args.device_index)
    device = torch.cuda.get_device_name(args.device_index)

    # Native live run: same protocol as the previous 17.19 FPS benchmark.
    native_dir = out / 'native'
    native_dir.mkdir(exist_ok=True)
    cfg = yaml.safe_load((ROOT / 'config/scannet_t05_boxer_kfmap_score05.yaml').read_text())
    cfg['data']['datadir'] = str(folder)
    cfg['data']['output_dir'] = str(native_dir)
    cfg['lifting']['proposal_cache']['mode'] = 'disabled'
    cfg['lifting']['boxer']['diagnostics_dir'] = str(out / 'boxer_diagnostics')
    cfg['association']['pvq_ar']['diagnostics_dir'] = str(out / 'native_diagnostics')
    cfg['vis']['rerun'] = False
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))

    cold_start = time.perf_counter()
    import demo
    checkpoint = torch.load(ROOT / 'models/cutr_rgbd.pth', map_location='cpu',
                            weights_only=False)['model']
    dimension = checkpoint['backbone.0.patch_embed.proj.weight'].shape[0]
    model = demo.make_cubify_transformer(
        dimension=dimension, depth_model=True).eval().cuda()
    model.load_state_dict(checkpoint)
    del checkpoint
    clip_model, preprocess = demo.load_clip(
        str(ROOT / 'models/open_clip_pytorch_model.bin'))
    categories = np.genfromtxt(ROOT / 'data/panoptic_categories_nomerge.txt',
                               delimiter='\n', dtype=str)
    features = torch.load(ROOT / 'data/class_features.pt',
                          weights_only=False).cuda()
    raw_dataset = demo.get_dataset(cfg)
    raw_dataset.load_arkit_depth = True
    dataset = CountedStream(raw_dataset)
    augmentor = demo.Augmentor(('wide/image', 'wide/depth'))
    preprocessor = demo.Preprocessor()
    torch.cuda.synchronize()
    native_loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        demo.run(cfg, model, dataset, clip_model, preprocess, categories,
                 features, augmentor, preprocessor, score_thresh=.5, gap=25,
                 re_vis=False)
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    torch.cuda.synchronize()
    native_done = time.perf_counter()
    native_peak = int(torch.cuda.max_memory_allocated())
    consumed = dataset.consumed
    del model, clip_model, preprocess, features, dataset.dataset
    gc.collect()
    torch.cuda.empty_cache()

    # Freeze the provider to the exact prefix actually consumed by native.
    for fid in range(consumed, args.frames):
        for sub, suffix in [('color', '.jpg'), ('depth', '.png'), ('pose', '.txt')]:
            path = folder / sub / f'{fid}{suffix}'
            if path.exists() or path.is_symlink():
                path.unlink()
    scene_root = out / 'scannet_input' / scene
    for sub in ('color', 'depth', 'pose', 'intrinsic'):
        target = scene_root / sub
        if not target.exists():
            target.symlink_to(folder / sub, target_is_directory=True)
    keyframes = len(range(0, consumed, 25))
    native_pkl = native_dir / f'{scene}_boxes.pkl'
    nms_jsonl = out / 'native_diagnostics' / f'{scene}_pvq_nms.jsonl'

    os.environ.update(M2_MODE='nativelogit', M2_EXCLUSIVE='1',
                      CAUSAL_TAU='0.5', M5_OFF='1')
    import tools.integrated_online as integrated
    integrated.SCANS = str(out / 'scannet_input')
    integrated.M2_MODE = 'nativelogit'
    integrated.M2_EXCLUSIVE = True
    integrated.TAU = .5
    integrated.M5_OFF = True

    # M1-P timing.  Its paired pre-M2 output and timing checkpoint are used;
    # the post-checkpoint M2 work is deliberately excluded from this arm.
    p_load_start = p_loaded = None
    p_peak = None
    p_timing = None
    p_out = out / 'm1p' / f'{scene}_boxes.pkl'
    if not args.final_only:
        p_load_start = time.perf_counter()
        p_model, p_adapter = integrated.load_models(m1a_online=False)
        torch.cuda.synchronize()
        p_loaded = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        integrated.process_scene(
            scene, str(native_pkl), str(nms_jsonl),
            str(out / 'm1p_unused_m2' / f'{scene}_boxes.pkl'), p_model, p_adapter,
            gap=25, score_view='persistent', m1_out_pkl=str(p_out),
            m1a_online=False, timing_out_json=str(out / 'timing_m1p.json'))
        p_peak = int(torch.cuda.max_memory_allocated())
        p_timing = json.loads((out / 'timing_m1p.json').read_text())
        del p_model, p_adapter
        gc.collect()
        torch.cuda.empty_cache()

    # Final route: one detector forward exposes ordinary proposals and dense
    # anchors; Boxer lifts their union once per keyframe.
    final_load_start = time.perf_counter()
    final_model, final_adapter = integrated.load_models(m1a_online=True)
    torch.cuda.synchronize()
    final_loaded = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pa_out = out / 'm1pa' / f'{scene}_boxes.pkl'
    final_out = out / 'm1pa_m2' / f'{scene}_boxes.pkl'
    integrated.process_scene(
        scene, str(native_pkl), str(nms_jsonl), str(final_out),
        final_model, final_adapter, gap=25, score_view='persistent',
        m1_out_pkl=str(pa_out), m1a_online=True,
        timing_out_json=str(out / 'timing_final.json'))
    final_peak = int(torch.cuda.max_memory_allocated())
    final_timing = json.loads((out / 'timing_final.json').read_text())
    del final_model, final_adapter
    gc.collect()
    torch.cuda.empty_cache()

    # Common semantic readout.  M1-P+A and +M2 have identical geometry, so
    # their crops are byte-identical and one measurement applies to both.
    semantic = {'m1pa': semantic_time(
        out, scene, 'm1pa', pa_out.parent, native_dir)}
    if not args.final_only:
        semantic['base'] = semantic_time(
            out, scene, 'base', native_dir, native_dir)
        semantic['m1p'] = semantic_time(
            out, scene, 'm1p', p_out.parent, native_dir)

    native_run = native_done - native_loaded
    if args.final_only:
        stage = {'m1pa_m2': float(final_timing['total_seconds'])}
        semantic_warm = {'m1pa_m2': semantic['m1pa']['warm_seconds']}
    else:
        stage = {
            'base': 0.0,
            'm1p': float(p_timing['cumulative_through_m1_seconds']),
            'm1pa': float(final_timing['cumulative_through_m1_seconds']),
            'm1pa_m2': float(final_timing['total_seconds']),
        }
        semantic_warm = {
            'base': semantic['base']['warm_seconds'],
            'm1p': semantic['m1p']['warm_seconds'],
            'm1pa': semantic['m1pa']['warm_seconds'],
            'm1pa_m2': semantic['m1pa']['warm_seconds'],
        }
    arms = {}
    for name in stage:
        seconds = native_run + stage[name] + semantic_warm[name]
        arms[name] = {
            'native_seconds': native_run,
            'extension_seconds': stage[name],
            'semantic_readout_seconds': semantic_warm[name],
            'warm_end_to_end_seconds': seconds,
            'raw_fps': consumed / seconds,
            'effective_ms_per_raw_frame': 1000.0 * seconds / consumed,
            'extension_ms_per_keyframe': 1000.0 * stage[name] / keyframes,
        }

    with native_pkl.open('rb') as handle:
        native_rows = len(pickle.load(handle)[0])
    result = {
        'schema': 'boxfusion.final_route_runtime.v1',
        'scene': scene,
        'configured_raw_frames': args.frames,
        'consumed_raw_frames': consumed,
        'keyframes': keyframes,
        'gap': 25,
        'device': device,
        'torch': torch.__version__,
        'timing_scope': {
            'included': ['raw-frame image/depth I/O', 'live native CuTR/CLIP/Boxer/fusion',
                         'live WeDetect-Uni forward', 'live single-pool Boxer lift',
                         'M1-P finalization', 'strict-causal M1-A state update',
                         'M2 scoring', 'prediction serialization',
                         'common terminal semantic readout'],
            'excluded': ['explicit model/checkpoint initialization', 'visualization',
                         'metric evaluation'],
            'native_first_forward_warmup_included': True,
            'proposal_replay': False,
        },
        'model_load_seconds': {
            'native': native_loaded - cold_start,
            'm1p': (p_loaded - p_load_start) if p_loaded is not None else None,
            'm1pa_m2': final_loaded - final_load_start,
            'semantic_base': (semantic.get('base') or {}).get('startup_seconds'),
            'semantic_m1p': (semantic.get('m1p') or {}).get('startup_seconds'),
            'semantic_m1pa_and_m2': semantic['m1pa']['startup_seconds'],
        },
        'peak_allocated_bytes': {
            'native': native_peak, 'm1p': p_peak, 'm1pa_m2': final_peak,
            'semantic': max(int(v['clip_peak_allocated_bytes']) for v in semantic.values()),
        },
        'rows': {
            'base': native_rows,
            'm1p': (len(pickle.load(open(p_out, 'rb'))[0])
                    if p_out.is_file() else None),
            'm1pa': len(pickle.load(open(pa_out, 'rb'))[0]),
            'm1pa_m2': len(pickle.load(open(final_out, 'rb'))[0]),
        },
        'arms': arms,
        'm1p_stage': p_timing,
        'final_stage': final_timing,
        'semantic': semantic,
        'integrity': {
            'native_sha256': sha256(native_pkl),
            'm1p_sha256': sha256(p_out) if p_out.is_file() else None,
            'm1pa_sha256': sha256(pa_out),
            'm1pa_m2_sha256': sha256(final_out),
            'm1pa_m2_geometry_paired': all(
                np.array_equal(a[1], b[1])
                for a, b in zip(pickle.load(open(pa_out, 'rb'))[0],
                                pickle.load(open(final_out, 'rb'))[0])),
        },
        'limits': [
            'One fixed development-scene prefix; throughput is not a scene-distribution confidence interval.',
            'M1-A births are causal and bounded; the current M1-P finalization and M2 readout occur at scene end.',
            'M1-P+A and +M2 share one semantic timing because their geometry and crops are identical.',
        ],
    }
    (out / 'runtime.json').write_text(json.dumps(result, ensure_ascii=False,
                                                  indent=2) + '\n')
    print(json.dumps({'arms': arms, 'rows': result['rows'],
                      'device': device}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
