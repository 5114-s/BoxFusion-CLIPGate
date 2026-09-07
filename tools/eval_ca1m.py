"""Class-agnostic AP15/25/50 evaluator for CA-1M.

GT: <scene>/derived_train_gt_boxes.npy  (N,8,3) world-frame corners
    (falls back to instances.json OBBs -> corners if npy missing)
Pred: <pred_root>/<scene>_boxes.pkl rows (cls, corners8x3, score), world frame.
Protocol: per-scene greedy match by score at IoU threshold; pooled PR curve;
AP = sum of precision at each TP / total GT (continuous AP).
"""
import os, sys, glob, json, pickle, argparse
import numpy as np

def _poly_area(pts):
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    return 0.5*abs(sum(x[i]*y[(i+1)%len(pts)] - x[(i+1)%len(pts)]*y[i] for i in range(len(pts))))

def _intersect(p1, p2, a, b):
    d1 = (b[0]-a[0])*(p1[1]-a[1]) - (b[1]-a[1])*(p1[0]-a[0])
    d2 = (b[0]-a[0])*(p2[1]-a[1]) - (b[1]-a[1])*(p2[0]-a[0])
    t = d1/(d1-d2)
    return (p1[0]+t*(p2[0]-p1[0]), p1[1]+t*(p2[1]-p1[1]))

def _clip(subject, clip):
    out = subject
    for i in range(len(clip)):
        a, b = clip[i], clip[(i+1) % len(clip)]
        inp, out = out, []
        if not inp:
            break
        for j in range(len(inp)):
            cur, prev = inp[j], inp[j-1]
            c_in = (b[0]-a[0])*(cur[1]-a[1]) - (b[1]-a[1])*(cur[0]-a[0]) >= -1e-9
            p_in = (b[0]-a[0])*(prev[1]-a[1]) - (b[1]-a[1])*(prev[0]-a[0]) >= -1e-9
            if c_in:
                if not p_in:
                    out.append(_intersect(prev, cur, a, b))
                out.append(cur)
            elif p_in:
                out.append(_intersect(prev, cur, a, b))
    return out

def _hull_ring(xy):
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

def box_iou(c1, c2):
    p1 = _hull_ring(c1[:, :2]); p2 = _hull_ring(c2[:, :2])
    if p1 is None or p2 is None:
        return 0.0
    z1 = (c1[:, 2].min(), c1[:, 2].max()); z2 = (c2[:, 2].min(), c2[:, 2].max())
    zov = max(0.0, min(z1[1], z2[1]) - max(z1[0], z2[0]))
    h1 = z1[1]-z1[0]; h2 = z2[1]-z2[0]
    inter_poly = _clip(p1, p2)
    inter = (_poly_area(inter_poly) if len(inter_poly) >= 3 else 0.0) * zov
    union = _poly_area(p1)*h1 + _poly_area(p2)*h2 - inter
    return inter/max(union, 1e-9)

def load_gt(scene_dir):
    npy = f'{scene_dir}/after_filter_boxes.npy'
    if os.path.exists(npy):
        return np.load(npy).astype(np.float64)
    npy = f'{scene_dir}/derived_train_gt_boxes.npy'
    if os.path.exists(npy):
        return np.load(npy).astype(np.float64)
    boxes = []
    for inst in json.load(open(f'{scene_dir}/instances.json')):
        pos = np.array(inst['position']); sc = np.array(inst['scale']); R = np.array(inst['R'])
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
        boxes.append(pos[None] + signs * (0.5*sc) @ R.T)
    return np.array(boxes) if boxes else np.zeros((0, 8, 3))

def evaluate(data_root, pred_root, scenes):
    gts, preds = {}, {}
    skipped = []
    for sc in scenes:
        if not os.path.exists(f'{data_root}/{sc}/instances.json'):
            skipped.append(sc)
            continue
        gts[sc] = load_gt(f'{data_root}/{sc}')
        p = f'{pred_root}/{sc}_boxes.pkl'
        if os.path.exists(p):
            d = pickle.load(open(p, 'rb'))
            preds[sc] = d[0]
        else:
            preds[sc] = []
    if skipped:
        print(f'(skipped {len(skipped)} scenes without GT: {skipped[:5]}...)')
    out = []
    for thr in (0.15, 0.25, 0.50):
        entries = []      # (score, is_tp) pooled
        total_gt = 0
        for sc in gts:
            gt = gts[sc]
            total_gt += len(gt)
            rows = sorted(preds[sc], key=lambda r: -float(r[2]))
            taken = [False]*len(gt)
            for r in rows:
                cc = np.asarray(r[1], float)
                best, bj = 0.0, -1
                for j, g in enumerate(gt):
                    v = box_iou(cc, g)
                    if v > best:
                        best, bj = v, j
                if best >= thr and bj >= 0 and not taken[bj]:
                    taken[bj] = True
                    entries.append((float(r[2]), 1))
                else:
                    entries.append((float(r[2]), 0))
        entries.sort(key=lambda e: -e[0])
        tp_cum = 0
        ap = 0.0
        for i, (s, tp) in enumerate(entries):
            if tp:
                tp_cum += 1
                ap += tp_cum / (i + 1)
        out.append(ap/max(total_gt, 1))
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/extra/ZhaoX/boxfusion_ca1m')
    ap.add_argument('--pred-root', required=True)
    ap.add_argument('--scenes', help='optional scene list file; default = all in data root with pkl')
    args = ap.parse_args()
    if args.scenes:
        scenes = [l.strip() for l in open(args.scenes) if l.strip()]
    else:
        scenes = sorted(os.path.basename(p).replace('_boxes.pkl', '')
                        for p in glob.glob(f'{args.pred_root}/*_boxes.pkl'))
    aps = evaluate(args.data_root, args.pred_root, scenes)
    print(f'CA-1M class-agnostic ({len(scenes)} scenes): AP15={aps[0]*100:.2f} AP25={aps[1]*100:.2f} AP50={aps[2]*100:.2f}')
