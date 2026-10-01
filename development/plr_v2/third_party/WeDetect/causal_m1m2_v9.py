"""v9: EXACT offline algorithm (confirm_and_append semantics), run as scene-end
post-processing of the causal kfmap run. All inputs are past at scene end:
proposals (per-keyframe WeDetect cache), child events (this run's NMS observer),
native map (this run's final output), frame-order processing. Then M2 on top.
If this reproduces ~42.9, the online causal number IS the offline number."""
import os, sys, glob, json, pickle
import numpy as np

SCANS = '/extra/ZhaoX/scannet_data/scans'
WDCACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
NATIVE = '/data/ZhaoX/BoxFusion/results/scannet_t05_boxer_kfmap_score05'
KFDIAG = '/data/ZhaoX/BoxFusion/diagnostics/kfmap_score05'
OUT = '/data/ZhaoX/BoxFusion/results/causal_m1m2_v9'
os.makedirs(OUT, exist_ok=True)

DEDUP = 0.25
CAP = 12
ALPHA, TAU = 0.6, 0.5
EDGES = (0.3, 0.5, 0.7, 1.0)
TABLE = (0.05, 0.10, 0.25, 0.40, 0.50)

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0)
    lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = ov[0]*ov[1]*ov[2]
    v1 = np.prod(hi1-lo1); v2 = np.prod(hi2-lo2)
    return inter/max(v1+v2-inter, 1e-9)

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

total = 0
for pf in sorted(glob.glob(f'{NATIVE}/scene*_boxes.pkl')):
    scene = os.path.basename(pf).replace('_boxes.pkl', '')
    d = pickle.load(open(pf, 'rb'))
    rows = [tuple(r) for r in d[0]]
    native = [np.asarray(r[1], float) for r in rows]

    zw = np.load(f'{WDCACHE}/{scene}.npz')
    corners, scores, fids = zw['corners_raw'], zw['scores'], zw['frame_ids']
    n_wd = len(corners)
    child_c, child_s, child_f = [], [], []
    try:
        for line in open(f'{KFDIAG}/{scene}_pvq_nms.jsonl'):
            r = json.loads(line)
            child_c.append(r['child_corners_world'])
            child_s.append(r['child_score'])
            child_f.append(r['keyframe_id'])
    except FileNotFoundError:
        pass
    all_c = list(corners) + child_c
    all_s = list(scores) + child_s
    all_f = list(fids) + child_f
    all_c = [np.asarray(c, float) for c in all_c]
    all_f = [int(f) for f in all_f]

    # ===== EXACT offline confirm_and_append algorithm =====
    births_kept = []
    if all_c:
        order = np.argsort(np.array(all_f), kind='stable')
        uniq, ordinal_of = np.unique(np.array(all_f), return_inverse=True)
        receipts = []
        for i in order:
            c, s, f = all_c[i], float(all_s[i]), int(ordinal_of[i])
            cc = c.mean(0)
            if any(aabb_iou(c, nb) >= DEDUP for nb in native):
                continue
            best, best_iou = None, 0.0
            for r in receipts:
                if f - r['last_ord'] > 10:
                    continue
                rc = r['obs'][-1]
                if np.linalg.norm(rc.mean(0) - cc) > 0.50:
                    continue
                v = aabb_iou(c, rc)
                if v >= 0.10 and v > best_iou:
                    best_iou, best = v, r
            src_wd = int(i < n_wd)
            if best is not None:
                best['obs'].append(c); best['frames'].add(f)
                best['last_ord'] = f; best['scores'].append(s); best['n_wd'] += src_wd
            else:
                receipts.append(dict(obs=[c], frames={f}, last_ord=f, scores=[s], n_wd=src_wd))
        births = []
        for r in receipts:
            min_views = 3 if 2 * r['n_wd'] > len(r['obs']) else 2
            if len(r['frames']) < min_views:
                continue
            obs = r['obs']
            best_j, best_s = 0, -1.0
            for j, a in enumerate(obs):
                ssum = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                if ssum > best_s:
                    best_s, best_j = ssum, j
            births.append((float(np.mean(r['scores'])), obs[best_j]))
        births.sort(key=lambda x: -x[0])
        kept = []
        for strength, m in births:
            if any(aabb_iou(m, km) >= 0.50 for _, km in kept):
                continue
            kept.append((strength, m))
            if len(kept) >= CAP:
                break
        births_kept = kept
    total += len(births_kept)
    all_rows = list(rows) + [(0, m, price(m)) for _, m in births_kept]

    # ===== M2 (add alpha=0.6 tau=0.5, support over all keyframes) =====
    by_frame = {}
    for b, fr in zip(zw['boxes2d'], zw['frame_ids']):
        by_frame.setdefault(int(fr), []).append(b)
    by_frame = {k: np.array(v) for k, v in by_frame.items()}
    K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
    pose_cache = {}
    out_rows = []
    for cls, corners_, s in all_rows:
        cc = np.asarray(corners_, float)
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
        out_rows.append((cls, corners_, ns))
    out_sc = [out_rows] + [[(det[0], det[1], det[2]) for det in sc] for sc in d[1:]]
    pickle.dump(out_sc, open(f'{OUT}/{scene}_boxes.pkl', 'wb'))

print(f'v9 (exact offline algorithm on causal run): {total} births -> {OUT}')
