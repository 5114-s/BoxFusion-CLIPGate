"""Dynamic-benchmark metrics: stale-box rate & retirement, M5-off vs M5-on."""
import os, sys, json, pickle, glob
import numpy as np
sys.path.insert(0, '/data/ZhaoX/BoxFusion/tools')
def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0); lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = float(ov[0]*ov[1]*ov[2])
    return inter/float(np.maximum(np.prod(hi1-lo1)+np.prod(hi2-lo2)-inter, 1e-9))
man = json.load(open('/data/ZhaoX/BoxFusion/data_dyn/manifest.json'))
rows = []
for scene, m in man.items():
    corners = [np.asarray(c).reshape(8, 3) for c in m['removed']]
    outs = {}
    for tag, d in (('m5off', 'scannet_dyn_m5off'), ('m5on', 'scannet_dyn_m5on')):
        p = f'/data/ZhaoX/BoxFusion/results/{d}/{scene}_boxes.pkl'
        outs[tag] = pickle.load(open(p, 'rb'))[0] if os.path.exists(p) else []
    for k, gc in enumerate(corners):
        r = {'scene': scene, 'obj': k}
        for tag in ('m5off', 'm5on'):
            best_s, best_i = 0.0, -1
            for i, row in enumerate(outs[tag]):
                v = aabb_iou(np.asarray(row[1], float), gc)
                if v >= 0.25 and float(row[2]) > best_s:
                    best_s, best_i = float(row[2]), i
            r[tag+'_score'] = best_s
            r[tag+'_present'] = best_i >= 0
        rows.append(r)
off = np.array([r['m5off_score'] for r in rows])
on = np.array([r['m5on_score'] for r in rows])
STALE = 0.3
print(f'removed objects: {len(rows)}')
print(f'M5-OFF (immortal map): stale(rate@score>{STALE}) = {(off>STALE).mean()*100:.0f}%  median score of matching box = {np.median(off):.2f}')
print(f'M5-ON  (retirement):    stale(rate@score>{STALE}) = {(on>STALE).mean()*100:.0f}%  median score of matching box = {np.median(on):.2f}')
print(f'retired (off stale & on clean): {((off>STALE)&(on<=STALE)).sum()} / {max((off>STALE).sum(),1)}')
json.dump(rows, open('/data/ZhaoX/BoxFusion/results/dyn_metrics.json', 'w'), indent=1)
