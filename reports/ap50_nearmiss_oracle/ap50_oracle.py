"""Oracle for 2D-guided extent calibration: can WeDetect 2D evidence fix the 344 near-misses?
Measures per near-miss box: support (2D evidence exists?), IoU gap to 0.5, and directional
signal (best WeDetect 2D box tighter/looser than the 3D projection). TP boxes as control."""
import os, sys, glob, pickle
import numpy as np

os.chdir('/data/ZhaoX/BoxFusion/evaluation')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/evaluation')

SCANS = '/extra/ZhaoX/scannet_data/scans'
WDCACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
FINAL = '/data/ZhaoX/BoxFusion/results/dedup25_size'

# ---- IoU machinery (from ap50_decomp, GT-frame AABB vs GT OBB) ----
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
    meta = f'{SCANS}/{scene}/{scene}.txt'
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

# ---- projection machinery (from consensus_rerank) ----
def project_xyxy(corners_w, pose, K, W, H):
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

def vol(c):
    e = c.max(0) - c.min(0)
    return float(np.prod(e))

# ---- oracle walk ----
buckets = {k: [] for k in ('TP', 'NEAR_tight', 'NEAR_loose', 'NEAR_other')}
for pf in sorted(glob.glob(f'{FINAL}/scene*_boxes.pkl')):
    scene = os.path.basename(pf).replace('_boxes.pkl', '')
    d = pickle.load(open(pf, 'rb'))
    rows = d[0]
    batch = default_collate([dataset[name2idx[scene]]])
    end_points = {'model': 'scannet'}; end_points.update(batch)
    gt_map = parse_groundtruths(end_points, CONFIG_DICT)[0]
    gt = [np.asarray(j[1], dtype=np.float64) for j in gt_map]
    T = load_axis_align(scene)
    pc = [pred_to_gt_frame(p[1], T) for p in rows]
    sc = [float(p[2]) for p in rows]
    order = np.argsort(sc)[::-1]
    ious = {}
    for i in order:
        best, bj = 0.0, -1
        for j, g in enumerate(gt):
            v = box_iou(pc[i], g)
            if v > best:
                best, bj = v, j
        ious[int(i)] = (best, bj)
    taken = [False]*len(gt)
    cls = {}
    for i in order:
        best, bj = ious[int(i)]
        if best >= 0.5 and not taken[bj]:
            taken[bj] = True; cls[int(i)] = 'TP'
        elif best >= 0.25:
            g = gt[bj]; vr = vol(pc[i])/max(vol(g), 1e-6)
            cls[int(i)] = 'NEAR_tight' if vr < 0.8 else ('NEAR_loose' if vr > 1.25 else 'NEAR_other')
        else:
            cls[int(i)] = 'FAR'
    # 2D evidence for every box
    zw = np.load(f'{WDCACHE}/{scene}.npz')
    b2d, fids = zw['boxes2d'], zw['frame_ids']
    by_frame = {}
    for b, fr in zip(b2d, fids):
        by_frame.setdefault(int(fr), []).append(b)
    by_frame = {k: np.array(v) for k, v in by_frame.items()}
    K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
    pose_cache = {}
    for i in range(len(rows)):
        if cls[i] not in buckets:
            continue
        corners = np.asarray(rows[i][1], float)
        sup, area_ratio = 0.0, None
        for fr, props in by_frame.items():
            if fr not in pose_cache:
                p = np.loadtxt(f'{SCANS}/{scene}/pose/{fr}.txt').reshape(4, 4)
                pose_cache[fr] = p if np.isfinite(p).all() else None
            pose = pose_cache[fr]
            if pose is None:
                continue
            bb = project_xyxy(corners, pose, K, 1296, 968)
            if bb is None:
                continue
            iv = iou2d_vec(bb, props)
            k = int(np.argmax(iv)) if len(iv) else -1
            if k >= 0 and iv[k] > sup:
                sup = float(iv[k])
                pa = (props[k,2]-props[k,0])*(props[k,3]-props[k,1])
                ba = max((bb[2]-bb[0])*(bb[3]-bb[1]), 1e-9)
                area_ratio = pa/ba
        buckets[cls[i]].append((sup, area_ratio, ious[i][0]))

def summarize(name, arr):
    a = np.array([(s, ar if ar is not None else np.nan, i) for s, ar, i in arr])
    if not len(a):
        return
    sup = a[:, 0]; ar = a[:, 1]; iou = a[:, 2]
    print(f'== {name}  n={len(a)}')
    print(f'   support: >=0.5: {(sup>=0.5).mean()*100:.0f}%   0.3-0.5: {((sup>=0.3)&(sup<0.5)).mean()*100:.0f}%   <0.3: {(sup<0.3).mean()*100:.0f}%')
    print(f'   IoU dist: 0.45-0.50: {(iou>=0.45).mean()*100:.0f}%   0.35-0.45: {((iou>=0.35)&(iou<0.45)).mean()*100:.0f}%   <0.35: {(iou<0.35).mean()*100:.0f}%')
    m = np.isfinite(ar)
    if m.any():
        print(f'   2D/proj area ratio @best-view: median={np.median(ar[m]):.2f}   <0.85 (shrink evidence): {(ar[m]<0.85).mean()*100:.0f}%   >1.15 (expand evidence): {(ar[m]>1.15).mean()*100:.0f}%')

for k in ('TP', 'NEAR_loose', 'NEAR_tight', 'NEAR_other'):
    summarize(k, buckets[k])
