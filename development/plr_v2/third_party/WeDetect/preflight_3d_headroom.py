"""Stage-2 3D preflight (Script B): WeDetect-Uni -> frozen BoxerNet -> 3D recall headroom.

Clean process: must NOT import evaluation/ utils (Boxer owns top-level 'utils').
Missed-GT inputs from Script A (/tmp/wedetect_preflight_gt/<scene>.npz).
"""
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, '/data/ZhaoX/BoxFusion')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')

SCENES = ['scene0568_00', 'scene0606_01', 'scene0377_02']
SCANS = '/extra/ZhaoX/scannet_data/scans'
GAP = 25
SCORE_LIFT = 0.05
TOPK_PER_FRAME = 150

# ---------- geometry (inlined, no evaluation imports) ----------
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

def _poly_area(pts):
    x = [p[0] for p in pts]; y = [p[1] for p in pts]
    return 0.5*abs(sum(x[i]*y[(i+1) % len(pts)] - x[(i+1) % len(pts)]*y[i] for i in range(len(pts))))

def _isect(p1, p2, a, b):
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
            ci = (b[0]-a[0])*(cur[1]-a[1]) - (b[1]-a[1])*(cur[0]-a[0]) >= -1e-9
            pi = (b[0]-a[0])*(prev[1]-a[1]) - (b[1]-a[1])*(prev[0]-a[0]) >= -1e-9
            if ci:
                if not pi:
                    out.append(_isect(prev, cur, a, b))
                out.append(cur)
            elif pi:
                out.append(_isect(prev, cur, a, b))
    return out

def box_iou(c1, c2):
    # evaluator-comparable: axis-align both first (preds are OBB, GT already AABB)
    def aabb(c):
        lo, hi = c.min(0), c.max(0)
        signs = np.array([[sx, sy, sz] for sx in (0, 1) for sy in (0, 1) for sz in (0, 1)], float)
        return lo + signs * (hi - lo)
    c1, c2 = aabb(c1), aabb(c2)
    p1 = _hull_ring(c1[:, :2]); p2 = _hull_ring(c2[:, :2])
    if p1 is None or p2 is None:
        return 0.0
    z1 = (c1[:, 2].min(), c1[:, 2].max()); z2 = (c2[:, 2].min(), c2[:, 2].max())
    zov = max(0.0, min(z1[1], z2[1]) - max(z1[0], z2[0]))
    h1 = z1[1]-z1[0]; h2 = z2[1]-z2[0]
    ip = _clip(p1, p2)
    inter = (_poly_area(ip) if len(ip) >= 3 else 0.0) * zov
    union = _poly_area(p1)*h1 + _poly_area(p2)*h2 - inter
    return inter/max(union, 1e-9)

def to_gt_frame(corners, T):
    """raw-world corners -> GT frame (axis-align + flip). AABB-comparable."""
    b = np.asarray(corners, float)
    b = (T[:3, :3] @ b.T).T + T[:3, 3]
    b = b[:, [0, 2, 1]].copy(); b[:, 1] *= -1
    return b

def obb_to_corners(center, extents, R):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return center[None] + signs * (0.5 * extents) @ R.T

# ---------- WeDetect-Uni ----------
CKPT_W = '/data/ZhaoX/BoxFusion/third_party/WeDetect/wedetect_base_uni.pth'
wmodel = SimpleYOLOWorldDetector = None
from wedetect_uni_infer import SimpleYOLOWorldDetector
wmodel = SimpleYOLOWorldDetector(backbone_size='base', prompt_dim=768, num_prompts=256, num_proposals=300)
ck = torch.load(CKPT_W, map_location='cpu', weights_only=False)
for key in list(ck.keys()):
    if 'backbone' in key:
        ck[key.replace('backbone.image_model.model.', 'backbone.')] = ck.pop(key)
for key in list(ck.keys()):
    if 'bbox_head' in key:
        nk = key.replace('bbox_head.head_module.', 'bbox_head.')
        nk = nk.replace('0.2.', '0.6.').replace('1.2.', '1.6.').replace('2.2.', '2.6.')
        nk = nk.replace('1.bn', '4').replace('1.conv', '3').replace('0.bn', '1').replace('0.conv', '0')
        ck[nk] = ck.pop(key)
wmodel.load_state_dict(ck, strict=False)
wmodel = wmodel.cuda().eval()

# ---------- Boxer adapter ----------
from boxfusion.boxer_lifter import build_lifting_adapter
cfg = {"lifting": {"backend": "boxer", "boxer": {
    "mode": "observer", "apply_stage": "post_filter",
    "official_root": "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer",
    "checkpoint": "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt",
    "expected_commit": "1f86542dc342a4b1d474c87c97c5d1d6566d9148",
    "checkpoint_sha256": "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f",
    "dinov3_sha256": "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
    "precision": "bfloat16", "use_sdp": True, "sdp_samples": 10000, "seed": 0,
    "diagnostics_dir": "/tmp/wedetect_preflight_diag", "cache_image_features": True}}}
adapter = build_lifting_adapter(cfg, device="cuda", code_root="/data/ZhaoX/BoxFusion")

from PIL import Image

summary = {}
for scene in SCENES:
    z = np.load(f'/tmp/wedetect_preflight_gt/{scene}.npz')
    missed_gt = z['missed']          # (M,8,3) GT-frame
    T = z['T']
    n_gt = len(z['gt'])

    K_color = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
    K_depth = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_depth.txt')[:3, :3]
    n_frames = len(glob.glob(f'{SCANS}/{scene}/color/*.jpg'))
    kfs = []
    for f in range(0, n_frames, GAP):
        pf, cf, df = f'{SCANS}/{scene}/pose/{f}.txt', f'{SCANS}/{scene}/color/{f}.jpg', f'{SCANS}/{scene}/depth/{f}.png'
        if not all(os.path.exists(p) for p in (pf, cf, df)):
            continue
        pose = np.loadtxt(pf).reshape(4, 4)
        if not np.isfinite(pose).all():
            continue
        kfs.append((f, pose, cf, df))

    lifted = []
    for f, pose, cf, df in kfs:
        with torch.no_grad():
            out = wmodel([cf])[0]
        pb = out['bboxes'].float().cpu().numpy()
        ps = out['scores'].float().cpu().numpy()
        sel = ps >= SCORE_LIFT
        pb, ps = pb[sel], ps[sel]
        if len(pb) == 0:
            continue
        if len(pb) > TOPK_PER_FRAME:
            order = np.argsort(-ps)[:TOPK_PER_FRAME]
            pb, ps = pb[order], ps[order]
        rgb = np.asarray(Image.open(cf).convert('RGB'))
        depth = np.asarray(Image.open(df)).astype(np.float32) / 1000.0
        datum, meta = adapter._make_datum(
            image=rgb, depth=depth, boxes_xyxy=torch.from_numpy(pb).float(),
            image_K=K_color, depth_K=K_depth, camera_to_world=pose,
            scene_id=scene, frame_id=int(f))
        outp, _, _ = adapter.forward_raw_with_feature_cache(
            datum, scene_id=scene, frame_id=int(f),
            encoder_input_sha256=meta['encoder_input_sha256'])
        obbs = outp['obbs_pr_w'][0]
        centers = obbs.bb3_center_world.float().cpu().numpy()
        extents = obbs.bb3_diagonal.float().cpu().numpy()
        rots = obbs.T_world_object.R.float().cpu().numpy()
        for i in range(len(centers)):
            corners_raw = obb_to_corners(centers[i], np.abs(extents[i]) + 1e-6, rots[i])
            lifted.append((float(ps[i]), to_gt_frame(corners_raw, T)))
        print(f'  {scene} f{f}: +{len(pb)} (total {len(lifted)})', flush=True)

    print(f'{scene}: GT={n_gt} missed={len(missed_gt)} lifted={len(lifted)}')
    res = {}
    for sthr in (0.05, 0.1, 0.2):
        cand = [c for s, c in lifted if s > sthr]
        for t in (0.15, 0.25, 0.5):
            cov = sum(1 for g in missed_gt if cand and max(box_iou(g, c) for c in cand) >= t)
            res[(sthr, t)] = cov
            print(f'  score>{sthr} IoU3D>={t}: {cov}/{len(missed_gt)}')
    summary[scene] = dict(n_gt=n_gt, missed=len(missed_gt),
                          res={f'{k[0]}_{k[1]}': v for k, v in res.items()})

print()
print('========== 3D RECALL HEADROOM (WeDetect-Uni + frozen BoxerNet) ==========')
tot_gt = sum(s['n_gt'] for s in summary.values())
tot_miss = sum(s['missed'] for s in summary.values())
print(f'trio: GT={tot_gt}, missed-by-Cbest={tot_miss}')
for key in ['0.05_0.15', '0.05_0.25', '0.05_0.5', '0.1_0.15', '0.1_0.25', '0.1_0.5', '0.2_0.15', '0.2_0.25', '0.2_0.5']:
    st, it = key.split('_')
    cov = sum(s['res'][key] for s in summary.values())
    print(f'  score>{st}, IoU3D>={it}: {cov}/{tot_miss} -> headroom {100*cov/tot_gt:.1f} pts')
print('reference OWLv2+Boxer (pre-gate): 25.0 / 21.4 / 17.9')
