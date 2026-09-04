"""PVQ-AR shadow ambiguity-event autopsy: GT labeling + gate-2 + native/PVQ correctness."""
import os, sys, json, glob
import numpy as np

os.chdir('/data/ZhaoX/BoxFusion/evaluation')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/evaluation')

# ---------- rotated-box IoU (yaw-only OBBs) via convex polygon clipping ----------
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
    """CCW convex hull of 2D points; robust to any corner ordering."""
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

# ---------- load events ----------
EVENTS = []
for f in sorted(glob.glob('/data/ZhaoX/BoxFusion/diagnostics/pvq_ar_shadow_score05/*_pvq_ar.jsonl')):
    for line in open(f):
        r = json.loads(line)
        if r.get('type') == 'ambiguity_event':
            EVENTS.append(r)
print(f'events: {len(EVENTS)}')

# sanity: unit-test the rotated IoU on synthetic boxes with known values
def _synbox(cx=0, cy=0, cz=0, dx=1, dy=1, dz=1, yaw=0):
    a, b = np.cos(yaw), np.sin(yaw)
    local = np.array([[-dx/2, -dy/2], [dx/2, -dy/2], [dx/2, dy/2], [-dx/2, dy/2]])
    R = np.array([[a, -b], [b, a]])
    ring = local @ R.T + np.array([cx, cy])
    z0, z1 = cz - dz/2, cz + dz/2
    return np.vstack([np.column_stack([ring, np.full(4, z0)]),
                      np.column_stack([ring, np.full(4, z1)])])
t1 = box_iou(_synbox(), _synbox())                       # identical -> 1
t2 = box_iou(_synbox(), _synbox(cx=10))                  # disjoint -> 0
t3 = box_iou(_synbox(), _synbox(cx=0.5))                 # half x-overlap -> 1/3
t4 = box_iou(_synbox(), _synbox(yaw=np.pi/2))            # rotated square same footprint -> 1
print(f'IoU unit tests: identical={t1:.4f} disjoint={t2:.4f} half={t3:.4f} (expect 0.3333) yaw90={t4:.4f}')
assert abs(t1-1) < 1e-6 and t2 < 1e-9 and abs(t3-1/3) < 1e-3 and abs(t4-1) < 1e-6, 'IoU implementation broken'

# ---------- load GT ----------
from torch.utils.data import DataLoader
from utils.ap_helper import parse_groundtruths  # must precede data_util.dataset (primes the 'utils' namespace pkg)
from utils.utils import flip_axis_to_camera, obb_to_aabb_corners, reorganize_obb_to_aabb
from data_util.dataset import ScannetDetectionDataset
from data_util.model_util_scannet import ScannetDatasetConfig

def load_axis_align(scene):
    meta = f'/extra/ZhaoX/scannet_data/scans/{scene}/{scene}.txt'
    for line in open(meta):
        if 'axisAlignment' in line:
            m = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            return np.array(m).reshape(4, 4)
    raise RuntimeError(f'no axisAlignment in {meta}')

def pred_corners_to_gt_frame(corners, T):
    b = np.asarray(corners, dtype=np.float64)[None]                       # (1,8,3)
    b = np.transpose(T[None, :3, :3] @ np.transpose(b, (0, 2, 1)), (0, 2, 1)) + T[None, :3, 3]
    b = flip_axis_to_camera(b)
    b = obb_to_aabb_corners(b)
    b = reorganize_obb_to_aabb(b)
    return b[0]
from torch.utils.data._utils.collate import default_collate

DC = ScannetDatasetConfig()
CONFIG_DICT = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
               'use_old_type_nms': False, 'cls_nms': True, 'per_class_proposal': True,
               'conf_thresh': 0.5, 'dataset_config': DC}
dataset = ScannetDetectionDataset(split_set='val', num_points=40000, use_color=False,
                                  use_height=True, augment=False,
                                  data_path='./data_util/scannet_train_detection_data')
name2idx = {n: i for i, n in enumerate(dataset.scan_names)}
scenes_needed = sorted({e['scene_id'] for e in EVENTS})
print(f'scenes with events: {len(scenes_needed)}')

GT = {}
for s in scenes_needed:
    batch = default_collate([dataset[name2idx[s]]])
    end_points = {'model': 'scannet'}
    end_points.update(batch)
    gt_map = parse_groundtruths(end_points, CONFIG_DICT)[0]
    GT[s] = np.array([np.asarray(j[1], dtype=np.float64) for j in gt_map])
    print(f'  {s}: {len(GT[s])} GT boxes', flush=True)

# ---------- label events ----------
MATCH_THR = 0.10
ALIGN = {s: load_axis_align(s) for s in scenes_needed}
# guard: transformed final predictions must match GT at plausible rates
import pickle as _pkl
_s = scenes_needed[0]
_gt = GT[_s]
_d = _pkl.load(open(f'/data/ZhaoX/BoxFusion/results/v2_rescore_prefix_only/{_s}_boxes.pkl', 'rb'))
_preds = [pred_corners_to_gt_frame(np.asarray(det[1]), ALIGN[_s]) for sc in _d for det in sc]
_bst = [max(box_iou(p, g) for g in _gt) for p in _preds]
_rate15 = float(np.mean([b >= 0.15 for b in _bst]))
print(f'frame guard ({_s}): preds={len(_preds)}, match@0.15={_rate15:.2f}')
assert _rate15 > 0.2, 'coordinate transform chain still wrong'

rows = []
for e in EVENTS:
    pc = pred_corners_to_gt_frame(np.array(e['proposal_corners_world']), ALIGN[e['scene_id']])
    gts = GT[e['scene_id']]
    ious_p = [box_iou(pc, g) for g in gts]
    prop_gt = int(np.argmax(ious_p)) if ious_p and max(ious_p) >= MATCH_THR else None
    cand_gt = []
    for c in e['candidates']:
        cc = pred_corners_to_gt_frame(np.array(c['corners_world']), ALIGN[e['scene_id']])
        ious_c = [box_iou(cc, g) for g in gts]
        cand_gt.append(int(np.argmax(ious_c)) if ious_c and max(ious_c) >= MATCH_THR else None)
    chosen = e.get('chosen', 0)
    rows.append(dict(scene=e['scene_id'], frame=e['frame_id'],
                     prop_gt=prop_gt, prop_iou=max(ious_p) if ious_p else 0.0,
                     cand_gt=cand_gt, chosen=chosen,
                     reason=e.get('reason'), n_gt=len(gts)))

# ---------- aggregate ----------
n = len(rows)
has_gt = [r for r in rows if r['prop_gt'] is not None]
native_ok = [r for r in has_gt if r['cand_gt'][0] == r['prop_gt']]
pvq_ok = [r for r in has_gt if r['cand_gt'][r['chosen']] == r['prop_gt']]
in_set = [r for r in has_gt if r['prop_gt'] in r['cand_gt']]
fixable = [r for r in has_gt if r['cand_gt'][0] != r['prop_gt'] and r['prop_gt'] in r['cand_gt']]
pvq_fixed = [r for r in fixable if r['cand_gt'][r['chosen']] == r['prop_gt']]
pvq_breaks = [r for r in native_ok if r['cand_gt'][r['chosen']] != r['prop_gt']]

print()
print('================ AUTOPSY SUMMARY ================')
print(f'events total:                  {n}')
print(f'proposal matches GT (>={MATCH_THR}): {len(has_gt)}  ({100*len(has_gt)/n:.1f}%)')
print(f'proposal FP (<{MATCH_THR}):          {n-len(has_gt)}')
print(f'native top-1 correct:          {len(native_ok)}/{len(has_gt)} ({100*len(native_ok)/max(1,len(has_gt)):.1f}%)')
print(f'PVQ hypothetical correct:      {len(pvq_ok)}/{len(has_gt)} ({100*len(pvq_ok)/max(1,len(has_gt)):.1f}%)')
print(f'GATE2 correct in choice set:   {len(in_set)+ (n-len(has_gt))}/{n} ({100*(len(in_set)+n-len(has_gt))/n:.1f}%)  [FPs count as null-available]')
print(f'   strict (GT proposals only): {len(in_set)}/{len(has_gt)} ({100*len(in_set)/max(1,len(has_gt)):.1f}%)')
print(f'native wrong & fixable in set: {len(fixable)}')
print(f'PVQ fixes among fixable:       {len(pvq_fixed)}/{len(fixable)}')
print(f'PVQ breaks correct native:     {len(pvq_breaks)}  <-- damage in active mode')
print()
print('per-reason breakdown:')
import collections
for reason, cnt in collections.Counter(r['reason'] for r in rows).items():
    sub = [r for r in rows if r['reason'] == reason and r['prop_gt'] is not None]
    okn = sum(1 for r in sub if r['cand_gt'][0] == r['prop_gt'])
    okp = sum(1 for r in sub if r['cand_gt'][r['chosen']] == r['prop_gt'])
    print(f'  {reason}: n={cnt}, gt-matched={len(sub)}, native_ok={okn}, pvq_ok={okp}')

json.dump(rows, open('/tmp/pvq_autopsy_rows.json', 'w'), indent=1)
print('\nrows saved: /tmp/pvq_autopsy_rows.json')
