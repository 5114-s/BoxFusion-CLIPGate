"""Bounded RGB-only clip search; outputs are proposals, never 3D ground truth."""
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/original_motion_clip_20260914'
CKPT = Path('/data/ZhaoX/OVM3D-Dett/boxfusion_b6_selective_boxer_dev/models/yoloe-11s-seg-pf.pt')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['prepare', 'infer'], required=True)
    args = p.parse_args()
    if args.phase == 'prepare':
        audit = json.loads((ROOT / 'reports/original_dynamic_admission_20260914/audit.json').read_text())
        frames = []
        for dataset in ['scannet', 'ca1m']:
            for row in audit[dataset]:
                scene = row['scene']
                folder = (ROOT / 'data/scannet_val_rgbfix' / scene / 'frames/color'
                          if dataset == 'scannet' else Path('/extra/ZhaoX/boxfusion_ca1m') / scene / 'rgb')
                paths = sorted(folder.glob('*.jpg' if dataset == 'scannet' else '*.png'), key=lambda x: int(x.stem))
                for q in [.1, .5, .9]:
                    f = paths[round((len(paths)-1)*q)]
                    frames.append({'dataset': dataset, 'scene': scene, 'frame': int(f.stem), 'path': str(f)})
        protocol = {'purpose': 'Locate people in original dataset RGB; not native detector evaluation or dynamic AP.',
                    'sampling': 'Fixed 10%, 50%, 90% frames of each original evaluation sequence; do not treat a negative sample as absence over the complete sequence.',
                    'checkpoint': str(CKPT), 'checkpoint_sha256': hashlib.sha256(CKPT.read_bytes()).hexdigest(),
                    'frozen_settings': {'conf': .25, 'iou': .70, 'imgsz': 640, 'max_det': 64, 'agnostic_nms': True},
                    'batch_size': 8, 'frames': frames}
        (OUT / 'sparse_screen_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
        print('prepared', len(frames), 'frames', flush=True)
        return
    import numpy as np
    from ultralytics import YOLOE
    protocol = json.loads((OUT / 'sparse_screen_protocol.json').read_text())
    assert hashlib.sha256(CKPT.read_bytes()).hexdigest() == protocol['checkpoint_sha256']
    model = YOLOE(str(CKPT))
    frames = protocol['frames']
    records = []
    start = time.perf_counter()
    output = OUT / 'sparse_screen.jsonl'
    with output.open('w') as f:
        for offset in range(0, len(frames), protocol['batch_size']):
            batch = frames[offset:offset+protocol['batch_size']]
            results = model.predict(source=[x['path'] for x in batch], device='cuda:0',
                                    verbose=False, **protocol['frozen_settings'])
            assert len(results) == len(batch)
            for entry, result in zip(batch, results):
                people = []
                if result.boxes is not None:
                    for box, score, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                        name = result.names[int(cls)]
                        if name.lower() in ['person', 'people', 'human', 'man', 'woman']:
                            people.append({'label': name, 'score': float(score), 'box2d': box.tolist()})
                row = dict(entry, people=people, source_sha256=hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest())
                records.append(row)
                f.write(json.dumps(row) + '\n')
            f.flush()
            if offset % 80 == 0:
                print('screened', len(records), '/', len(frames), 'person_frames', sum(bool(x['people']) for x in records), flush=True)
    summary = {'frames': len(records), 'seconds_excluding_model_load': time.perf_counter()-start,
               'person_frames': sum(bool(x['people']) for x in records),
               'person_scenes': sorted({(x['dataset'], x['scene']) for x in records if x['people']}),
               'limits': 'Sparse detector-based retrieval, not exhaustive motion screening, human labels, or 3D GT.'}
    (OUT / 'sparse_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
