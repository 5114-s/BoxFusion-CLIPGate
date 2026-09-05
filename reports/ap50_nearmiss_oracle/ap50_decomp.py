"""AP50 failure decomposition for final outputs: TP / near-miss / far-FP / duplicate,
plus geometry error anatomy of near-misses (tight/loose/offset)."""
import os, sys, glob, pickle
import numpy as np

os.chdir('/data/ZhaoX/BoxFusion/evaluation')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/evaluation')

def _ring_ok(poly):
    x = [p[0] for p in poly]; y = [p[1] for p in poly]
    s = sum(x[i]*y[(i+1)%len(poly)] - x[(i+1)%len(poly)]*y[i] for i in range(len(poly)))
    return s > 0

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

def load_axis_align(scene):
    meta = f'/extra/ZhaoX/scannet_data/scans/{scene}/{scene}.txt'
    for line in open(meta):
        if 'axisAlignment' in line:
            m = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            return np.array(m).reshape(4, 4)
    raise RuntimeError(f'no axisAlignment in {meta}')

from torch.utils.data import DataLoader
from utils.ap_helper import parse_groundtruths
from utils.utils import flip_axis_to_camera, obb_to_aabb_corners, reorganize_obb_to_aabb
from data_util.dataset import ScannetDetectionDataset
from data_util.model_util_scannet import ScannetDatasetConfig
from torch.utils.data._utils.collate import default_collate

DC = ScannetDatasetConfig()
CONFIG_DICT = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
               'use_old_type_nms': False, 'cls_nms': True, 'per_class_proposal': True,
               'conf_thresh': 0.5, 'dataset_config': DC}
dataset = ScannetDetectionDataset(split_set='val', num_points=40000, use_color=False,
                                  use_height=True, augment=False,
                                  data_path='./data_util/scannet_train_detection_data')
name2idx = {n: i for i, n in enumerate(dataset.scan_names)}

def pred_to_gt_frame(corners, T):
    b = np.asarray(corners, dtype=np.float64)[None]
    b = np.transpose(T[None, :3, :3] @ np.transpose(b, (0, 2, 1)), (0, 2, 1)) + T[None, :3, 3]
    b = flip_axis_to_camera(b)
    b = obb_to_aabb_corners(b)
    b = reorganize_obb_to_aabb(b)
    return b[0]

def vol(c):
    e = c.max(0) - c.min(0)
    return float(np.prod(e))

def analyze(result_dir, label):
    stats = dict(tp=0, dup=0, near=0, far=0, near_tight=0, near_loose=0, near_offset=0,
                near_shape=0, gt_total=0)
    near_ratio_sum, near_cdist_sum = [], []
    for pf in sorted(glob.glob(f'{result_dir}/scene*_boxes.pkl')):
        scene = os.path.basename(pf).replace('_boxes.pkl', '')
        d = pickle.load(open(pf, 'rb'))
        preds = d[0]
        batch = default_collate([dataset[name2idx[scene]]])
        end_points = {'model': 'scannet'}
        end_points.update(batch)
        gt_map = parse_groundtruths(end_points, CONFIG_DICT)[0]
        gt = [np.asarray(j[1], dtype=np.float64) for j in gt_map]
        stats['gt_total'] += len(gt)
        T = load_axis_align(scene)
        pc = [pred_to_gt_frame(p[1], T) for p in preds]
        sc = [float(p[2]) for p in preds]
        order = np.argsort(sc)[::-1]
        taken = [False]*len(gt)
        ious = {}
        for i in order:
            best, bj = 0.0, -1
            for j, g in enumerate(gt):
                v = box_iou(pc[i], g)
                if v > best:
                    best, bj = v, j
            ious[int(i)] = (best, bj)
        for i in order:
            best, bj = ious[int(i)]
            if best >= 0.5 and not taken[bj]:
                taken[bj] = True
                stats['tp'] += 1
            elif best >= 0.5:
                stats['dup'] += 1
            elif best >= 0.25:
                stats['near'] += 1
                g = gt[bj]
                vr = vol(pc[i])/max(vol(g), 1e-6)
                cd = float(np.linalg.norm((pc[i].max(0)+pc[i].min(0))/2 - (g.max(0)+g.min(0))/2))
                near_ratio_sum.append(vr); near_cdist_sum.append(cd)
                if vr < 0.8: stats['near_tight'] += 1
                elif vr > 1.25: stats['near_loose'] += 1
                elif cd > 0.30: stats['near_offset'] += 1
                else: stats['near_shape'] += 1
            else:
                stats['far'] += 1
    n = stats['near']
    print(f'== {label} ({result_dir})')
    print(f"  preds: TP={stats['tp']} DUP={stats['dup']} NEAR(0.25-0.5)={stats['far'] if False else stats['near']} FAR(<0.25)={stats['far']} | GT total={stats['gt_total']}")
    print(f"  near-miss anatomy: tight(<0.8x vol)={stats['near_tight']} loose(>1.25x)={stats['near_loose']} offset(>0.3m)={stats['near_offset']} shape/other={stats['near_shape']}")
    if near_ratio_sum:
        print(f"  near-miss vol ratio median={np.median(near_ratio_sum):.2f}, center-dist median={np.median(near_cdist_sum):.2f} m")
        print(f"  NEAR/TP = {n/max(stats['tp'],1):.2f}  (upper bound on AP50 gain if all near-misses were fixed and re-ranked: TP {stats['tp']}->{stats['tp']+n})")
    return stats

s1 = analyze("/data/ZhaoX/BoxFusion/results/v2_rescore_prefix_only", "Cbest base (34.99)")
s2 = analyze("/data/ZhaoX/BoxFusion/results/dedup25_size", "M1 final (39.18)")
#s2 = analyze('/data/ZhaoX/BoxFusion/results/consensus_a60t50', 'Final M1+M2 (42.95)')
