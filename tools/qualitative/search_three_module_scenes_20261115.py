#!/usr/bin/env python3
"""Read-only prediction search; saves a reproducible qualitative-selection audit."""
from pathlib import Path
import json
import numpy as np
import build_framework_scene0030_real_assets as source

ROOT = source.ROOT
OUT = ROOT / 'reports/three_module_scene_search_20261115'
LABELS = {3:'cabinet',4:'bed',5:'chair',6:'sofa',7:'table',8:'door',9:'window',10:'bookshelf',
          11:'picture',12:'counter',14:'desk',16:'curtain',24:'refrigerator',28:'shower curtain',
          33:'toilet',34:'sink',36:'bathtub',39:'other furniture'}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, failures = [], []
    for path in sorted(source.RECEIPT.parent.iterdir()):
        scene = path.name
        if not path.is_dir() or not scene.startswith('scene'): continue
        try:
            matrix = source.common.alignment(scene)
            gt_path = ROOT/'evaluation/data_util/scannet_train_detection_data'/f'{scene}_bbox.npy'
            ga = np.load(gt_path)
            gt = ga[:,None,:3] + (source.common.CORNER_CODES[None]-.5)*ga[:,None,3:6]
            native = source.common.read_predictions(path/'native'/f'{scene}_boxes.pkl', matrix)
            full = source.common.read_predictions(path/'recar'/f'{scene}_boxes.pkl', matrix)
            diag = json.loads((path/'online_diagnostics'/f'{scene}.json').read_text())
            counts = diag['state']['terminal_counts']
            n, p, a = [counts[k] for k in ['native','m1p','m1a']]
            assert len(full['boxes']) == n+p+a and len(native['boxes']) == n
            assert np.array_equal(full['raw'][:n], native['raw'])
            ov = source.common.iou_matrix(full['boxes'], gt)
            nm = ov[:n].max(0) if n else np.zeros(len(gt))
            missed = np.flatnonzero(nm < .15)
            pairs = []
            for pg in missed:
                if not p or not a: continue
                pi = n + int(ov[n:n+p,pg].argmax())
                for vg in missed:
                    if pg == vg: continue
                    vi = n+p+int(ov[n+p:,vg].argmax())
                    if min(ov[pi,pg],ov[vi,vg]) < .25: continue
                    pairs.append(dict(ptr_gt=int(pg),var_gt=int(vg),ptr_index=pi,var_index=vi,
                                      ptr_class=LABELS.get(int(ga[pg,-1]),str(ga[pg,-1])),
                                      var_class=LABELS.get(int(ga[vg,-1]),str(ga[vg,-1])),
                                      ptr_iou=float(ov[pi,pg]),var_iou=float(ov[vi,vg]),
                                      ptr_native_max=float(nm[pg]),var_native_max=float(nm[vg]),
                                      minimum_iou=float(min(ov[pi,pg],ov[vi,vg]))))
            pairs.sort(key=lambda r: (-r['minimum_iou'],-r['ptr_iou']-r['var_iou']))
            meta = (source.common.RAW/scene/f'{scene}.txt').read_text()
            scene_type = next((s.split('=',1)[1].strip() for s in meta.splitlines() if s.startswith('sceneType')), 'unknown')
            row = dict(scene=scene, scene_type=scene_type, counts=counts, pairs=pairs,
                       native_sha256=native['sha256'],full_sha256=full['sha256'],
                       native_cache_exists=source.NATIVE.with_name(scene+'.npz').exists(),
                       evidence_cache_exists=source.EVIDENCE.with_name(scene+'.npz').exists(),
                       hsr_readout_exists=source.READOUT.with_name(scene+'.npz').exists())
            records.append(row)
        except Exception as exc:
            failures.append(dict(scene=scene,error=repr(exc)))
    records.sort(key=lambda r: (not bool(r['pairs']), -(r['pairs'][0]['minimum_iou'] if r['pairs'] else 0)))
    source.save_json(OUT/'archived_scene_scan.json', dict(
        purpose='GT-assisted qualitative scene search; not evaluation or hyperparameter selection',
        native_missed_threshold=.15, candidate_pair_min_iou=.25, scenes=records,failures=failures))
    for r in records:
        if r['pairs']:
            print(json.dumps(dict(scene=r['scene'],type=r['scene_type'],pair=r['pairs'][0]),ensure_ascii=False))
    print('SCANNED',len(records),'FAILURES',len(failures),'OFFICES WITHOUT PAIRS',
          [r['scene'] for r in records if 'office' in r['scene_type'].lower() and not r['pairs']])

if __name__ == '__main__': main()
