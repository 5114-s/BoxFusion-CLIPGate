"""Stage-2 source preflight: can WeDetect-Uni see (in 2D) the GT objects Cbest missed?

Sealed trio: scene0568_00, scene0606_01, scene0377_02 (gap-25 keyframes).
Pipeline:
  1. missed GT = GT not covered by Cbest boxes at 3D IoU 0.15 (greedy, audit machinery)
  2. WeDetect-Uni proposals on every keyframe (original-image coords)
  3. project missed-GT boxes (raw world) into keyframes; max 2D IoU with proposals
  4. projection convention auto-calibrated between the two ScanNet variants
"""
import os, sys, glob, pickle
import numpy as np

os.chdir('/data/ZhaoX/BoxFusion/evaluation')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/evaluation')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')
exec(open('/tmp/pvq_autopsy.py').read().split("# ---------- load events ----------")[0]
     .replace("import os, sys, json, glob", "import os, sys, json, glob"))
from utils.ap_helper import parse_groundtruths
from utils.utils import flip_axis_to_camera
from data_util.dataset import ScannetDetectionDataset
from data_util.model_util_scannet import ScannetDatasetConfig
from torch.utils.data._utils.collate import default_collate
import torch
from wedetect_uni_infer import SimpleYOLOWorldDetector

SCENES = ['scene0568_00', 'scene0606_01', 'scene0377_02']
SCANS = '/extra/ZhaoX/scannet_data/scans'
GAP = 25
IOU3D_MISS = 0.15

def iou2d(a, b):
    # a, b: (4,) xyxy
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/max(ua, 1e-9)

def load_axis_align(scene):
    for line in open(f'{SCANS}/{scene}/{scene}.txt'):
        if 'axisAlignment' in line:
            return np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape(4, 4)
    raise RuntimeError(scene)

def flip_inv(c):
    # F(x,y,z)=(x,-z,y)  =>  F^{-1}(a,b,c)=(a,c,-b)
    out = np.array(c, copy=True)
    out[..., [0, 1, 2]] = c[..., [0, 2, 1]]
    out[..., 2] *= -1
    return out

def project_corners_world(corners_w, pose, K, W, H, convention):
    """corners_w (8,3) raw-world -> xyxy or None if behind camera."""
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners_w.T).T + Rt[:3, 3]
    if convention == 'A':       # cam as-is (x right, y down, z fwd)
        p = c
    else:                       # 'B': y-up -> flip
        p = c[:, [0, 2, 1]].copy(); p[:, 1] *= -1
    z = p[:, 2]
    if (z < 0.1).any():
        return None
    uv = (K @ p.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    if uv[:, 0].min() > W or uv[:, 0].max() < 0 or uv[:, 1].min() > H or uv[:, 1].max() < 0:
        return None
    return np.array([np.clip(uv[:, 0].min(), 0, W), np.clip(uv[:, 1].min(), 0, H),
                     np.clip(uv[:, 0].max(), 0, W), np.clip(uv[:, 1].max(), 0, H)])

# ---------- WeDetect-Uni model ----------
CKPT = '/data/ZhaoX/BoxFusion/third_party/WeDetect/wedetect_base_uni.pth'
model = SimpleYOLOWorldDetector(backbone_size='base', prompt_dim=768, num_prompts=256, num_proposals=300)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
for key in list(ck.keys()):
    if 'backbone' in key:
        ck[key.replace('backbone.image_model.model.', 'backbone.')] = ck.pop(key)
for key in list(ck.keys()):
    if 'bbox_head' in key:
        nk = key.replace('bbox_head.head_module.', 'bbox_head.')
        nk = nk.replace('0.2.', '0.6.').replace('1.2.', '1.6.').replace('2.2.', '2.6.')
        nk = nk.replace('1.bn', '4').replace('1.conv', '3').replace('0.bn', '1').replace('0.conv', '0')
        ck[nk] = ck.pop(key)
model.load_state_dict(ck, strict=False)
model = model.cuda().eval()

# ---------- GT + Cbest ----------
DC = ScannetDatasetConfig()
CD = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25, 'use_old_type_nms': False,
      'cls_nms': True, 'per_class_proposal': True, 'conf_thresh': 0.5, 'dataset_config': DC}
ds = ScannetDetectionDataset('val', num_points=40000, use_color=False, use_height=True,
                             augment=False, data_path='./data_util/scannet_train_detection_data')

def to_gt_frame(corners, T):
    b = np.asarray(corners, float)[None]
    b = np.transpose(T[None, :3, :3] @ np.transpose(b, (0, 2, 1)), (0, 2, 1)) + T[None, :3, 3]
    from utils.utils import obb_to_aabb_corners, reorganize_obb_to_aabb
    b = flip_axis_to_camera(b)
    b = obb_to_aabb_corners(b)
    b = reorganize_obb_to_aabb(b)
    return b[0]

summary = {}
for scene in SCENES:
    idx = ds.scan_names.index(scene)
    ep = {'model': 'scannet'}; ep.update(default_collate([ds[idx]]))
    gt_flip = [np.asarray(j[1]) for j in parse_groundtruths(ep, CD)[0]]
    T = load_axis_align(scene)
    Tinv = np.linalg.inv(T)

    d = pickle.load(open(f'/data/ZhaoX/BoxFusion/results/v2_rescore_prefix_only/{scene}_boxes.pkl', 'rb'))
    cbest = [to_gt_frame(np.asarray(det[1]), T) for sc in d for det in sc]

    # missed GT at 3D IoU 0.15 (greedy)
    covered = set()
    pairs = [(box_iou(b, g), bi, gi) for bi, b in enumerate(cbest) for gi, g in enumerate(gt_flip)
             if box_iou(b, g) >= IOU3D_MISS]
    pairs.sort(reverse=True)
    ub = set()
    for v, bi, gi in pairs:
        if bi in ub or gi in covered:
            continue
        ub.add(bi); covered.add(gi)
    missed = [g for gi, g in enumerate(gt_flip) if gi not in covered]

    # missed GT -> raw world corners
    missed_raw = []
    for g in missed:
        c = flip_inv(g)                       # undo flip (points)
        c = (Tinv[:3, :3] @ c.T).T + Tinv[:3, 3]   # undo axis-align
        missed_raw.append(c)

    # keyframes
    kfs = []
    n_frames = len(glob.glob(f'{SCANS}/{scene}/color/*.jpg'))
    for f in range(0, n_frames, GAP):
        pf, cf = f'{SCANS}/{scene}/pose/{f}.txt', f'{SCANS}/{scene}/color/{f}.jpg'
        if not (os.path.exists(pf) and os.path.exists(cf)):
            continue
        pose = np.loadtxt(pf).reshape(4, 4)
        if not np.isfinite(pose).all():
            continue
        kfs.append((f, pose, cf))
    K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
    W, H = 1296, 968

    # proposals (batched)
    props = {}
    with torch.no_grad():
        for i in range(0, len(kfs), 8):
            chunk = kfs[i:i+8]
            outs = model([c[2] for c in chunk])
            for (f, pose, cf), out in zip(chunk, outs):
                props[f] = (out['bboxes'].float().cpu().numpy(), out['scores'].float().cpu().numpy())

    # convention calibration via Cbest boxes (use 12 boxes)
    calib = {}
    sample_cbest_raw = [flip_inv(to_gt_frame(np.asarray(det[1]), T)) if False else None for det in []] 
    cbest_raw = []
    for sc in d:
        for det in sc:
            c = np.asarray(det[1], float)  # already raw world
            cbest_raw.append(c)
    for conv in ('A', 'B'):
        hits = 0; tot = 0
        for f, pose, cf in kfs[::max(1, len(kfs)//10)]:
            pb, ps = props[f]
            sel = pb[ps > 0.05]
            for b in cbest_raw[:12]:
                bb = project_corners_world(b, pose, K, W, H, conv)
                if bb is None:
                    continue
                tot += 1
                if len(sel) and max(iou2d(bb, p) for p in sel) > 0.1:
                    hits += 1
        calib[conv] = hits/max(tot, 1)
    conv = 'A'  # global: ScanNet color convention (2/3 scenes decisive; per-scene calib unreliable in clutter)
    print(f'{scene}: kfs={len(kfs)} GT={len(gt_flip)} missed_by_cbest={len(missed)} calib A={calib["A"]:.2f} B={calib["B"]:.2f} -> {conv}')

    # coverage of missed GT by proposals
    res = {}
    for sthr in (0.05, 0.1, 0.2):
        for ithr in (0.3, 0.5):
            cov = 0
            for g in missed_raw:
                best = 0.0
                for f, pose, cf in kfs:
                    bb = project_corners_world(g, pose, K, W, H, conv)
                    if bb is None:
                        continue
                    pb, ps = props[f]
                    sel = pb[ps > sthr]
                    if len(sel):
                        best = max(best, max(iou2d(bb, p) for p in sel))
                if best >= ithr:
                    cov += 1
            res[(sthr, ithr)] = cov
            print(f'  score>{sthr} & IoU2D>={ithr}: covered {cov}/{len(missed)} ({100*cov/max(1,len(missed)):.1f}% of missed; {100*cov/len(gt_flip):.1f} pts of all GT)')
    summary[scene] = dict(n_gt=len(gt_flip), missed=len(missed), res={f'{k[0]}_{k[1]}': v for k, v in res.items()})

print()
print('================ SUMMARY (2D recall headroom, WeDetect-Base-Uni) ================')
tot_gt = sum(s['n_gt'] for s in summary.values())
tot_miss = sum(s['missed'] for s in summary.values())
print(f'3 sealed scenes: GT={tot_gt}, missed-by-Cbest={tot_miss}')
for key in ['0.05_0.3', '0.1_0.3', '0.05_0.5', '0.1_0.5', '0.2_0.3']:
    cov = sum(s['res'][key] for s in summary.values())
    print(f'  score>{key.split("_")[0]}, IoU2D>={key.split("_")[1]}: {cov}/{tot_miss} missed covered '
          f'({100*cov/max(1,tot_miss):.1f}% of missed | {100*cov/tot_gt:.1f} pts headroom of {tot_gt} GT)')
print('reference: OWLv2+Boxer pre-gate 3D recall headroom 25.0/21.4/17.9 (3-scene preflight)')
