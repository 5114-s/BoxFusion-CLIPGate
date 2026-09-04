"""Birth-MVF: refine confirmed births by multi-view projection-IoU optimization.

Same confirmation as the cap12 union config; replaces medoid geometry with
PFO-style optimization of center+size (yaw from medoid ring, kept fixed).
Observed 2D target per view: WeDetect boxes2d (wd rows) or projected 3D corners (child rows).
Accept refinement only if it improves the objective.
Outputs: uniform-0.10 and size-bucket scored variants.
"""
import os, sys, glob, pickle, json
import numpy as np

sys.path.insert(0, '/data/ZhaoX/BoxFusion')
UNION = '/data/ZhaoX/BoxFusion/results/union_recycle_cache'
WDCACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
CBEST = '/data/ZhaoX/BoxFusion/results/v2_rescore_prefix_only'
OUTBASE = '/data/ZhaoX/BoxFusion/results'
SCANS = '/extra/ZhaoX/scannet_data/scans'
CAP = 12

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0); lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = ov[0]*ov[1]*ov[2]
    v1 = np.prod(hi1-lo1); v2 = np.prod(hi2-lo2)
    return inter/max(v1+v2-inter, 1e-9)

def hull_ring(xy):
    pts = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in xy))
    if len(pts) < 3:
        return None
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]

def decompose(corners):
    ring = hull_ring(corners[:, :2])
    if ring is None or len(ring) < 4:
        return None
    ring = np.array(ring)
    e1 = ring[1] - ring[0]
    l = float(np.linalg.norm(e1)); w = float(np.linalg.norm(ring[2] - ring[1]))
    h = float(corners[:, 2].max() - corners[:, 2].min())
    cx = float(ring[:, 0].mean()); cy = float(ring[:, 1].mean())
    cz = float((corners[:, 2].max() + corners[:, 2].min()) / 2)
    yaw = float(np.arctan2(e1[1], e1[0]))
    return np.array([cx, cy, cz]), np.array([l, w, h]), yaw

def rebuild(center, size, yaw):
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return center[None] + (signs * 0.5 * size) @ R.T

def project(corners, pose, K):
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners.T).T + Rt[:3, 3]
    if (c[:, 2] < 0.05).any():
        return None
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return np.array([uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()])

def iou2d(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/max(ua, 1e-9)

def optimize_birth(center, size, yaw, views, iters=4, n=150):
    """views: list of (pose, K, target_xyxy). Random search over (c, s)."""
    def objective(cc, ss):
        corners = rebuild(cc, ss, yaw)
        vals = []
        for pose, K, tgt in views:
            p = project(corners, pose, K)
            if p is None:
                continue
            vals.append(iou2d(p, tgt))
        return float(np.mean(vals)) if vals else 0.0
    best_c, best_s = center.copy(), size.copy()
    best_v = objective(best_c, best_s)
    scale = np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])
    rng = np.random.RandomState(0)
    for it in range(iters):
        cand = np.tile(np.concatenate([best_c, best_s]), (n, 1))
        cand += rng.randn(n, 6) * scale[None]
        cand[:, 3:] = np.maximum(cand[:, 3:], 0.05)
        vals = np.array([objective(c[:3], c[3:]) for c in cand])
        j = int(np.argmax(vals))
        if vals[j] > best_v:
            best_v = vals[j]; best_c, best_s = cand[j, :3].copy(), cand[j, 3:].copy()
        scale *= 0.6
    return best_c, best_s, best_v

def size_score(ext):
    if ext < 0.3: return 0.05
    if ext < 0.5: return 0.10
    if ext < 0.7: return 0.25
    if ext < 1.0: return 0.40
    return 0.50

if __name__ == '__main__':
    files = sorted(glob.glob(f'{UNION}/scene*.npz'))
    out_u = f'{OUTBASE}/bmvf_u010'
    out_s = f'{OUTBASE}/bmvf_size'
    os.makedirs(out_u, exist_ok=True); os.makedirs(out_s, exist_ok=True)
    n_ref = n_med = 0
    for sf in files:
        scene = os.path.basename(sf).replace('.npz', '')
        z = np.load(sf)
        corners, scores, fids, b2d = z['corners_raw'], z['scores'], z['frame_ids'], z['boxes2d']
        zw = np.load(f'{WDCACHE}/{scene}.npz')
        n_wd = len(zw['corners_raw'])
        K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
        pose_cache = {}
        def get_pose(f):
            if f not in pose_cache:
                p = np.loadtxt(f'{SCANS}/{scene}/pose/{f}.txt').reshape(4, 4)
                pose_cache[f] = p if np.isfinite(p).all() else None
            return pose_cache[f]
        d = pickle.load(open(f'{CBEST}/{scene}_boxes.pkl', 'rb'))
        native = [np.asarray(det[1], float) for sc in d for det in sc]
        births = []
        if len(corners):
            order = np.argsort(fids, kind='stable')
            uniq, ordinal_of = np.unique(fids, return_inverse=True)
            receipts = []
            for i in order:
                c, s, f = corners[i], float(scores[i]), int(fids[i])
                if any(aabb_iou(c, nb) >= 0.10 for nb in native):
                    continue
                cc = c.mean(0)
                best, best_iou = None, 0.0
                for r in receipts:
                    if ordinal_of[i] - r['last_ord'] > 10:
                        continue
                    rc = r['obs'][-1][0]
                    if np.linalg.norm(rc.mean(0) - cc) > 0.50:
                        continue
                    v = aabb_iou(c, rc)
                    if v >= 0.10 and v > best_iou:
                        best_iou, best = v, r
                if best is not None:
                    best['obs'].append((c, i)); best['frames'].add(int(f)); best['last_ord'] = ordinal_of[i]
                    best['scores'].append(s)
                else:
                    receipts.append(dict(obs=[(c, i)], frames={int(f)}, last_ord=ordinal_of[i], scores=[s]))
            for r in receipts:
                if len(r['frames']) < 3:
                    continue
                obs = [c for c, _ in r['obs']]
                best_j, best_s = 0, -1.0
                for j, a in enumerate(obs):
                    ssum = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                    if ssum > best_s:
                        best_s, best_j = ssum, j
                medoid = obs[best_j]
                births.append((float(np.mean(r['scores'])), medoid, r['obs']))
            births.sort(key=lambda x: -x[0])
            kept = []
            for strength, m, obs in births:
                if any(aabb_iou(m, km) >= 0.50 for _, km, _ in kept):
                    continue
                kept.append((strength, m, obs))
                if len(kept) >= CAP:
                    break
            # refine each kept birth
            rows_u, rows_s = [], []
            for strength, medoid, obs in kept:
                dec = decompose(medoid)
                use_medoid = True
                if dec is not None:
                    center, size, yaw = dec
                    views = []
                    ok = True
                    for c, i in obs:
                        f = int(fids[i])
                        pose = get_pose(f)
                        if pose is None:
                            continue
                        if i < n_wd:
                            tgt = b2d[i]
                            if not np.any(tgt):
                                p = project(c, pose, K)
                                if p is None: continue
                                tgt = p
                        else:
                            p = project(c, pose, K)
                            if p is None: continue
                            tgt = p
                        views.append((pose, K, tgt))
                    if len(views) >= 2:
                        nc, ns, v = optimize_birth(center, size, yaw, views)
                        med_iou = float(np.mean([iou2d(project(medoid, p, K), t) for p, K, t in views
                                                 if project(medoid, p, K) is not None]))
                        if v > med_iou + 1e-4:
                            medoid = rebuild(nc, ns, yaw)
                            use_medoid = False
                            n_ref += 1
                        else:
                            n_med += 1
                    else:
                        n_med += 1
                ext = float((medoid.max(0) - medoid.min(0)).max())
                rows_u.append((0, medoid, 0.10))
                rows_s.append((0, medoid, size_score(ext)))
            out_sc = [[(det[0], det[1], det[2]) for det in sc] for sc in d]
            pickle.dump([sc + ([] if not rows_u or ix else []) for ix, sc in enumerate(out_sc)] if False else
                        [out_sc[0] + rows_u] + out_sc[1:], open(f'{out_u}/{scene}_boxes.pkl', 'wb'))
            out_sc2 = [[(det[0], det[1], det[2]) for det in sc] for sc in d]
            pickle.dump([out_sc2[0] + rows_s] + out_sc2[1:], open(f'{out_s}/{scene}_boxes.pkl', 'wb'))
        else:
            for o in (out_u, out_s):
                out_sc = [[(det[0], det[1], det[2]) for det in sc] for sc in d]
                pickle.dump(out_sc, open(f'{o}/{scene}_boxes.pkl', 'wb'))
        print(scene, flush=True)
    print(f'refined: {n_ref}, kept-medoid: {n_med}')
