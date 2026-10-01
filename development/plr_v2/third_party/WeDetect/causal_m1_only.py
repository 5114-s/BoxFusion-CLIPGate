"""Causal online re-verification of M1 (recovery+pricing) + M2 (consensus).

Strictly online semantics on top of the kfmap run (sanity-gated base, 35.04):
  - M1: funnel confirms at the EARLIEST keyframe where min_views distinct past
    frames are held (medoid from observations so far, dedup vs map-so-far at
    gate 0.25, self-NMS vs births confirmed so far, cap 12 first-come),
    pricing by max-edge table (0.3/0.5/0.7/1.0 -> 0.05/0.10/0.25/0.40/0.50).
  - M2: add-mode consensus boost (alpha=0.6, tau=0.5) with support over all
    keyframes (all in the past at scene end -> causal).
Candidates: WeDetect lifted cache + THIS run's NMS-observer child events.
Native rows: THIS run's output pkls (not the sealed Cbest).
"""
import os, sys, glob, json, pickle
import numpy as np

SCANS = '/extra/ZhaoX/scannet_data/scans'
WDCACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
NATIVE = '/data/ZhaoX/BoxFusion/results/scannet_t05_boxer_kfmap_score05'
KFDIAG = '/data/ZhaoX/BoxFusion/diagnostics/kfmap_score05'
OUT = os.environ.get("CAUSAL_OUT", "/data/ZhaoX/BoxFusion/results/causal_m1m2")
os.makedirs(OUT, exist_ok=True)

DEDUP = 0.25
CAP = 12
TTL = 10
SELF_NMS = 0.50
ALPHA, TAU = float(os.environ.get("CAUSAL_ALPHA", "0.6")), 0.5
EDGES = (0.3, 0.5, 0.7, 1.0)
TABLE = (0.05, 0.10, 0.25, 0.40, 0.50)

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0)
    lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = ov[0]*ov[1]*ov[2]
    v1 = np.prod(hi1-lo1); v2 = np.prod(hi2-lo2)
    return inter/max(v1+v2-inter, 1e-9)

def price(medoid):
    e = float((medoid.max(0)-medoid.min(0)).max())
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

total_births = 0
for pf in sorted(glob.glob(f'{NATIVE}/scene*_boxes.pkl')):
    scene = os.path.basename(pf).replace('_boxes.pkl', '')
    d = pickle.load(open(pf, 'rb'))
    rows = [list(r) for r in d[0]]

    # map timeline from kfmap records: init_id -> (kf, corners), keep latest <= t
    timeline = []   # (kf, {id: corners})
    cur = {}
    for line in open(f'{KFDIAG}/{scene}_pvq_kfmap.jsonl'):
        r = json.loads(line)
        if r.get('type') != 'kf_map':
            continue
        for i, cid in enumerate(r['init_ids']):
            cur[int(cid)] = np.asarray(r['boxes'][i], float)
        timeline.append((int(r['keyframe_id']), dict(cur)))

    def map_at(f):
        snap = {}
        for kf, mp in timeline:
            if kf <= f:
                snap = mp
            else:
                break
        return list(snap.values())

    # candidates: WeDetect cache + child events, all causal
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
    if cands:
        frames_sorted = sorted({f for f, *_ in cands})
        ordinal = {f: i for i, f in enumerate(frames_sorted)}
        cands.sort(key=lambda x: x[0])

        receipts = []   # dict(obs, frames, last_ord, scores, n_wd, confirmed)
        births = []
        by_f = {}
        for c in cands:
            by_f.setdefault(c[0], []).append(c)
        for f in sorted(by_f):
            t = ordinal[f]
            receipts = [r for r in receipts if t - r['last_ord'] <= TTL]
            mnow = map_at(f)
            for fr, c, s, is_wd in by_f[f]:
                if any(aabb_iou(c, nb) >= DEDUP for nb in mnow):
                    continue
                best, best_v = None, 0.0
                for r in receipts:
                    if r['confirmed']:
                        continue
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
                                         n_wd=is_wd, confirmed=False))
            # confirm at earliest eligibility
            for r in receipts:
                if r['confirmed']:
                    continue
                min_views = 3 if 2*r['n_wd'] > len(r['obs']) else 2
                if len(r['frames']) < min_views:
                    continue
                obs = r['obs']
                best_j, best_s = 0, -1.0
                for j, a in enumerate(obs):
                    ssum = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                    if ssum > best_s:
                        best_s, best_j = ssum, j
                medoid = obs[best_j]
                r['confirmed'] = True
                if any(aabb_iou(medoid, m) >= SELF_NMS for m in births):
                    continue
                if len(births) >= CAP:
                    continue
                births.append(medoid)

    total_births += len(births)
    new_rows = [(0, m, price(m)) for m in births]

    # M2 consensus over all rows (native + births), support over all keyframes
    zw2 = np.load(f'{WDCACHE}/{scene}.npz')
    by_frame = {}
    for b, fr in zip(zw2['boxes2d'], zw2['frame_ids']):
        by_frame.setdefault(int(fr), []).append(b)
    by_frame = {k: np.array(v) for k, v in by_frame.items()}
    K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
    pose_cache = {}
    all_rows = [tuple(r) for r in rows] + new_rows
    out_rows = []
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
        ns = min(0.99, float(s) + ALPHA * max(0.0, best - TAU))
        out_rows.append((cls, corners, ns))
    out_sc = [out_rows] + [[(det[0], det[1], det[2]) for det in sc] for sc in d[1:]]
    pickle.dump(out_sc, open(f'{OUT}/{scene}_boxes.pkl', 'wb'))

print(f'causal_m1m2: {total_births} births total -> {OUT}')
