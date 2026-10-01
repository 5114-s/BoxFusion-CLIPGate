"""M5 negative-evidence retirement: full held-out workflow.

Channel 1 (cross-detector invisibility): in-view-but-unsupported ratio over WeDetect keyframes.
Channel 2 (empty interior): depth-backprojected core (0.75x extent) contains no points.
Rules applied on top of causal_v9_a80 (the A-tuned/B-verified paper config).

Phases: stats(all 100) -> A-half sweep -> B-half verify -> full-100 sealed confirm.
"""
import os, sys, glob, pickle, subprocess, shutil, json
import numpy as np
from PIL import Image

SCANS = '/extra/ZhaoX/scannet_data/scans'
WDC = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
BASE = '/data/ZhaoX/BoxFusion/results/causal_v9_a80'
ROOT = '/data/ZhaoX/BoxFusion'
PY = '/home/admin1/miniconda3/envs/boxfusion2/bin/python'
STATS = f'{ROOT}/results/m5_stats.pkl'
A = [l.strip() for l in open(f'{ROOT}/results/heldout_split_A.txt') if l.strip()]
B = [l.strip() for l in open(f'{ROOT}/results/heldout_split_B.txt') if l.strip()]

def project(corners_w, pose, K, W=1296, H=968):
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners_w.T).T + Rt[:3, 3]
    if (c[:, 2] < 0.1).any():
        return None
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1, x2, y2 = uv[:,0].min(), uv[:,1].min(), uv[:,0].max(), uv[:,1].max()
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H or (x2-x1) < 2 or (y2-y1) < 2:
        return None
    return np.array([max(0,x1), max(0,y1), min(W,x2), min(H,y2)])

def iou2d(box, props):
    if not len(props):
        return 0.0
    x1 = np.maximum(box[0], props[:,0]); y1 = np.maximum(box[1], props[:,1])
    x2 = np.minimum(box[2], props[:,2]); y2 = np.minimum(box[3], props[:,3])
    inter = np.maximum(0, x2-x1)*np.maximum(0, y2-y1)
    ua = (box[2]-box[0])*(box[3]-box[1]) + (props[:,2]-props[:,0])*(props[:,3]-props[:,1]) - inter
    return float((inter/np.maximum(ua,1e-9)).max())

# ---------- phase 1: stats ----------
if os.path.exists(STATS):
    stats = pickle.load(open(STATS, 'rb'))
    print(f'loaded stats: {len(stats)} scenes')
else:
    stats = {}
    for pf in sorted(glob.glob(f'{BASE}/scene*_boxes.pkl')):
        scene = os.path.basename(pf).replace('_boxes.pkl', '')
        d = pickle.load(open(pf, 'rb'))
        rows = d[0]
        zw = np.load(f'{WDC}/{scene}.npz')
        by_frame = {}
        for b, fr in zip(zw['boxes2d'], zw['frame_ids']):
            by_frame.setdefault(int(fr), []).append(b)
        by_frame = {k: np.array(v) for k, v in by_frame.items()}
        K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
        Kd = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_depth.txt')[:3, :3]
        pose_cache = {}
        world_pts = []
        for fr in sorted(by_frame)[::4]:
            p = np.loadtxt(f'{SCANS}/{scene}/pose/{fr}.txt').reshape(4, 4)
            pose_cache[fr] = p if np.isfinite(p).all() else None
            pose = pose_cache[fr]
            dp = f'{SCANS}/{scene}/depth/{fr}.png'
            if pose is None or not os.path.exists(dp):
                continue
            depth = np.asarray(Image.open(dp), dtype=np.float64) / 1000.0
            Hh, Ww = depth.shape
            ys, xs = np.mgrid[0:Hh:8, 0:Ww:8]
            z = depth[ys, xs].ravel(); xs_ = xs.ravel(); ys_ = ys.ravel()
            m = (z > 0.2) & (z < 6.0)
            if not m.any():
                continue
            x = (xs_[m] + 0.5) * z[m] / Kd[0,0]
            y = (ys_[m] + 0.5) * z[m] / Kd[1,1]
            cam = np.stack([x, y, z[m]], 1)
            world_pts.append((pose[:3,:3] @ cam.T).T + pose[:3,3])
        P = np.concatenate(world_pts) if world_pts else np.zeros((0, 3))
        recs = []
        for cls, corners, s in rows:
            cc = np.asarray(corners, float)
            inv, uns = 0, 0
            for fr, props in by_frame.items():
                if fr not in pose_cache:
                    p = np.loadtxt(f'{SCANS}/{scene}/pose/{fr}.txt').reshape(4, 4)
                    pose_cache[fr] = p if np.isfinite(p).all() else None
                pose = pose_cache[fr]
                if pose is None:
                    continue
                bb = project(cc, pose, K)
                if bb is None:
                    continue
                inv += 1
                if iou2d(bb, props) < 0.30:
                    uns += 1
            lo, hi = cc.min(0), cc.max(0)
            ctr = (lo+hi)/2; ext = hi-lo
            lo2, hi2 = ctr-ext*0.75, ctr+ext*0.75
            core = int(((P >= lo2) & (P <= hi2)).all(1).sum()) if len(P) else 0
            recs.append(dict(inv=inv, uns=uns, core=core))
        stats[scene] = recs
        print(f'stats {scene}: {len(recs)} rows', flush=True)
    pickle.dump(stats, open(STATS, 'wb'))

# ---------- phase 2: variants + A sweep ----------
def apply_rule(name, ch1_thr=None, use_core=False, mode='remove', demote=0.3, min_inv=5):
    full = f'{ROOT}/results/m5_{name}'
    shutil.rmtree(full, ignore_errors=True)
    os.makedirs(full)
    for scene, recs in stats.items():
        d = pickle.load(open(f'{BASE}/{scene}_boxes.pkl', 'rb'))
        rows = list(d[0])
        assert len(rows) == len(recs)
        out = []
        for r, st in zip(rows, recs):
            kill = st['inv'] >= min_inv and (
                (ch1_thr is not None and st['uns']/max(st['inv'],1) >= ch1_thr) or
                (use_core and st['core'] == 0))
            if kill and mode == 'remove':
                continue
            if kill and mode == 'demote':
                r = (r[0], r[1], float(r[2]) * demote)
            out.append(r)
        d[0] = out
        pickle.dump(d, open(f'{full}/{scene}_boxes.pkl', 'wb'))
    return full

def eval_dir(pred_root, tag):
    r = subprocess.run(
        [PY, 'eval_scannet.py', '--dataset', 'scannet', '--data_path', '/extra/ZhaoX/scannet_data/scans',
         '--num_point', '40000', '--cluster_sampling', 'seed_fps', '--use_3d_nms', '--use_cls_nms',
         '--per_class_proposal', '--num_workers', '0', '--gpu', '0', '--pred_root', pred_root],
        capture_output=True, text=True, cwd=f'{ROOT}/evaluation',
        env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
                 MPLCONFIGDIR='/tmp/mplm5'))
    import re
    vals = [float(m) for m in re.findall(r'eval mAP: ([0-9.]+)', r.stdout)]
    print(f'{tag}: {vals}', flush=True)
    return vals

def subset_dir(src, scenes, name):
    dst = f'{ROOT}/results/{name}'
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst)
    for sc in scenes:
        shutil.copy(f'{src}/{sc}_boxes.pkl', f'{dst}/{sc}_boxes.pkl')
    return dst

VARIANTS = [
    ('r1_ch1_80_rm',  dict(ch1_thr=0.80, mode='remove')),
    ('r2_ch1_80_dm',  dict(ch1_thr=0.80, mode='demote')),
    ('r3_core_rm',    dict(use_core=True, mode='remove')),
    ('r4_joint_rm',   dict(ch1_thr=0.80, use_core=True, mode='remove')),
    ('r5_joint_dm',   dict(ch1_thr=0.80, use_core=True, mode='demote')),
    ('r6_joint90_rm', dict(ch1_thr=0.90, use_core=True, mode='remove')),
]
baseA = subset_dir(BASE, A, 'm5_baseA')
res = {}
for name, kw in VARIANTS:
    full = apply_rule(name, **kw)
    a_dir = subset_dir(full, A, f'm5_{name}_A')
    res[name] = eval_dir(a_dir, f'A {name}')
res['BASE'] = eval_dir(baseA, 'A baseline(no M5)')
best = max([k for k in res if k != 'BASE'], key=lambda k: res[k][0])
print(f'M5 BEST on A: {best} -> {res[best]}  (baseline {res["BASE"]})', flush=True)

# ---------- phase 3: B verify ----------
kw = dict(VARIANTS)[best]
full = f'{ROOT}/results/m5_{best}'
b_best = eval_dir(subset_dir(full, B, 'm5_best_B'), f'B {best}')
b_base = eval_dir(subset_dir(BASE, B, 'm5_baseB'), 'B baseline')
print(f'B VERIFY: best={b_best} vs baseline={b_base}', flush=True)

# ---------- phase 4: full-100 sealed ----------
r = subprocess.run(['bash', f'{ROOT}/scripts/eval_scannet_official100_real_score.sh', f'm5_{best}'],
                   capture_output=True, text=True)
import re
vals = [float(m) for m in re.findall(r'eval mAP: ([0-9.]+)', r.stdout)]
print(f'FULL100 SEALED m5_{best}: {vals}  (current 42.78/38.71/18.94)', flush=True)
print('M5 DONE', flush=True)
