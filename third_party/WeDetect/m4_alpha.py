"""M4 first instance: per-scene alpha self-calibration for M2 consensus boost.

Replays causal-v2 M1 exactly, then at scene end computes every row's support,
derives the scene's separation statistic (P90-P50 of supports), and writes
variant score sets:
  - fixed a=0.6 (control, must reproduce 41.28)
  - adaptive a_scene = clip(2*(P90-P50), 0.2, 0.9)
A priori rule constants; no test-set search.
"""
import os, sys, glob, json, pickle
import numpy as np

sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')

SCANS = '/extra/ZhaoX/scannet_data/scans'
WDCACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
NATIVE = '/data/ZhaoX/BoxFusion/results/scannet_t05_boxer_kfmap_score05'
KFDIAG = '/data/ZhaoX/BoxFusion/diagnostics/kfmap_score05'
CACHE = '/data/ZhaoX/BoxFusion/results/m4_alpha_supports.pkl'

# ---- reuse causal v2 M1 machinery by importing its module-level functions ----
import importlib.util
spec = importlib.util.spec_from_file_location(
    'cv2mod', '/data/ZhaoX/BoxFusion/third_party/WeDetect/causal_m1m2_v2.py')
# v2 executes on import (script style); instead replicate the needed pieces here.

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0)
    lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = ov[0]*ov[1]*ov[2]
    v1 = np.prod(hi1-lo1); v2 = np.prod(hi2-lo2)
    return inter/max(v1+v2-inter, 1e-9)

EDGES = (0.3, 0.5, 0.7, 1.0)
TABLE = (0.05, 0.10, 0.25, 0.40, 0.50)

def price(m):
    e = float((m.max(0)-m.min(0)).max())
    for b, s in zip(EDGES, TABLE):
        if e < b:
            return s
    return TABLE[-1]

def project_xyxy(corners_w, pose, K, W=1296, H=968):
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners_w.T).T + Rt[:3, 3]
    if (c[:, 2] < 0.1).any():
        return None
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1, x2, y2 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H or (x2-x1) < 2 or (y2-y1) < 2:
        return None
    return np.array([max(0, x1), max(0, y1), min(W, x2), min(H, y2)])

def iou2d_vec(box, props):
    x1 = np.maximum(box[0], props[:, 0]); y1 = np.maximum(box[1], props[:, 1])
    x2 = np.minimum(box[2], props[:, 2]); y2 = np.minimum(box[3], props[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    ua = (box[2]-box[0])*(box[3]-box[1]) + (props[:,2]-props[:,0])*(props[:,3]-props[:,1]) - inter
    return inter / np.maximum(ua, 1e-9)

DEDUP, CAP, TTL, SELF_NMS = 0.25, 12, 10, 0.50

data = {}
if os.path.exists(CACHE):
    data = pickle.load(open(CACHE, 'rb'))
    print(f'loaded support cache: {len(data)} scenes')
else:
    for pf in sorted(glob.glob(f'{NATIVE}/scene*_boxes.pkl')):
        scene = os.path.basename(pf).replace('_boxes.pkl', '')
        d = pickle.load(open(pf, 'rb'))
        rows = [tuple(r) for r in d[0]]
        # ---- M1 (causal v2 replica) ----
        zw = np.load(f'{WDCACHE}/{scene}.npz')
        cands = [(int(fr), np.asarray(c, float), float(s), 1)
                 for c, s, fr in zip(zw['corners_raw'], zw['scores'], zw['frame_ids'])]
        try:
            for line in open(f'{KFDIAG}/{scene}_pvq_nms.jsonl'):
                r = json.loads(line)
                cands.append((int(r['keyframe_id']), np.asarray(r['child_corners_world'], float),
                              float(r['child_score']), 0))
        except FileNotFoundError:
            pass
        births = []
        if cands:
            frames_sorted = sorted({f for f, *_ in cands})
            ordinal = {f: i for i, f in enumerate(frames_sorted)}
            cands.sort(key=lambda x: x[0])
            receipts = []
            by_f = {}
            for c in cands:
                by_f.setdefault(c[0], []).append(c)
            for f in sorted(by_f):
                t = ordinal[f]
                receipts = [r for r in receipts if t - r['last_ord'] <= TTL]
                for fr, c, s, is_wd in by_f[f]:
                    best, best_v = None, 0.0
                    for r in receipts:
                        rc = r['obs'][-1]
                        if np.linalg.norm(rc.mean(0) - c.mean(0)) > 0.50:
                            continue
                        v = aabb_iou(c, rc)
                        if v >= 0.10 and v > best_v:
                            best_v, best = v, r
                    if best is not None:
                        best['obs'].append(c); best['frames'].add(t)
                        best['last_ord'] = t; best['scores'].append(s); best['n_wd'] += is_wd
                    else:
                        receipts.append(dict(obs=[c], frames={t}, last_ord=t, scores=[s],
                                             n_wd=is_wd))
            native_final = [np.asarray(r[1], float) for r in rows]
            cand_births = []
            for r in receipts:
                min_views = 3 if 2*r['n_wd'] > len(r['obs']) else 2
                if len(r['frames']) < min_views:
                    continue
                obs = r['obs']
                best_j, best_s = 0, -1.0
                for j, a in enumerate(obs):
                    ssum = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                    if ssum > best_s:
                        best_s, best_j = ssum, j
                cand_births.append((float(np.mean(r['scores'])), obs[best_j]))
            cand_births.sort(key=lambda x: -x[0])
            for strength, m in cand_births:
                if len(births) >= CAP:
                    break
                if any(aabb_iou(m, nb) >= DEDUP for nb in native_final):
                    continue
                if any(aabb_iou(m, km) >= SELF_NMS for km in births):
                    continue
                births.append(m)
        all_rows = list(rows) + [(0, m, price(m)) for m in births]
        # ---- supports ----
        by_frame = {}
        for b, fr in zip(zw['boxes2d'], zw['frame_ids']):
            by_frame.setdefault(int(fr), []).append(b)
        by_frame = {k: np.array(v) for k, v in by_frame.items()}
        K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
        pose_cache = {}
        sups = []
        for cls, corners, s in all_rows:
            cc = np.asarray(corners, float)
            best = 0.0
            for fr, props in by_frame.items():
                if fr not in pose_cache:
                    p = np.loadtxt(f'{SCANS}/{scene}/pose/{fr}.txt').reshape(4, 4)
                    pose_cache[fr] = p if np.isfinite(p).all() else None
                pose = pose_cache[fr]
                if pose is None:
                    continue
                bb = project_xyxy(cc, pose, K)
                if bb is None:
                    continue
                best = max(best, float(iou2d_vec(bb, props).max()))
            sups.append(best)
        data[scene] = {'corners': [r[1] for r in all_rows],
                       'raw_scores': [float(r[2]) for r in all_rows],
                       'supports': sups}
    pickle.dump(data, open(CACHE, 'wb'))
    print(f'computed supports for {len(data)} scenes')

# ---- variants ----
def write_variant(name, alpha_fn, tau=0.5):
    out = f'/data/ZhaoX/BoxFusion/results/m4_{name}'
    os.makedirs(out, exist_ok=True)
    alphas = []
    for scene, dd in data.items():
        sups = np.array(dd['supports'])
        a = alpha_fn(sups)
        alphas.append(a)
        rows = [(0, c, min(0.99, s + a * max(0.0, sp - tau)))
                for c, s, sp in zip(dd['corners'], dd['raw_scores'], dd['supports'])]
        pickle.dump([rows], open(f'{out}/{scene}_boxes.pkl', 'wb'))
    print(f'{name}: alpha median={np.median(alphas):.2f} min={min(alphas):.2f} max={max(alphas):.2f}')

write_variant('fixed06', lambda s: 0.6)
write_variant('adaptive', lambda s: float(np.clip(2.0*(np.percentile(s, 90) - np.percentile(s, 50)), 0.2, 0.9)))
