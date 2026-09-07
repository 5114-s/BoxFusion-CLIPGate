"""CA-1M gate diagnostics: support-vs-confidence agreement (M4 gate), and
size-gated M5 simulation on A-half native rows."""
import os, sys, glob, json, pickle
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, '/data/ZhaoX/BoxFusion')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/tools')
from eval_ca1m import box_iou, load_gt
from wedetect_uni_infer import SimpleYOLOWorldDetector

CA = '/extra/ZhaoX/boxfusion_ca1m'
CKPT_W = '/data/ZhaoX/BoxFusion/third_party/WeDetect/wedetect_base_uni.pth'
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

def project(corners_w, pose, K, W, H):
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners_w.T).T + Rt[:3, :3].dot(np.zeros(3)) + Rt[:3, 3]
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

A = [l.strip() for l in open('/data/ZhaoX/BoxFusion/results/ca1m_split_A.txt') if l.strip()]
gate_fracs = []
demote_tp = {'cur': [0,0], 'size': [0,0]}     # rule -> [demoted, demoted&GT]
for scene in A:
    root = f'{CA}/{scene}'
    poses = np.load(f'{root}/all_poses.npy')
    K = np.loadtxt(f'{root}/K_rgb.txt').reshape(3,3)[:3,:3]
    w0, h0 = Image.open(f'{root}/rgb/0.png').size
    frames = {}
    for f in range(0, len(poses), 20):
        cf = f'{root}/rgb/{f}.png'
        if not os.path.exists(cf) or not np.isfinite(poses[f]).all():
            continue
        with torch.no_grad():
            out = wmodel([cf])[0]
        pb = out['bboxes'].float().cpu().numpy(); ps = out['scores'].float().cpu().numpy()
        pb = pb[ps >= 0.05]
        if len(pb):
            frames[f] = pb
    rows = pickle.load(open(f'/data/ZhaoX/BoxFusion/results/ca1m_prod/{scene}_boxes.pkl','rb'))[0]
    gt = load_gt(root)
    sups, scores = [], []
    for cls, corners, s in rows:
        cc = np.asarray(corners, float)
        best = 0.0
        for f, props in frames.items():
            bb = project(cc, poses[f], K, w0, h0)
            if bb is not None:
                best = max(best, iou2d(bb, props))
        sups.append(best); scores.append(float(s))
    sups = np.array(sups); scores = np.array(scores)
    if len(sups) >= 8:
        top = scores >= np.percentile(scores, 75)
        gate_fracs.append((sups[top] >= 0.5).mean())
    # M5 demotion simulation
    for i, (cls, corners, s) in enumerate(rows):
        cc = np.asarray(corners, float)
        sup = sups[i]
        g = max((box_iou(cc, gbox) for gbox in gt), default=0.0)
        size = float((cc.max(0)-cc.min(0)).max())
        # current rule approximation (support-based part; depth channel excluded)
        if sup < 0.30:
            demote_tp['cur'][0] += 1
            demote_tp['cur'][1] += (g >= 0.15)
        if sup < 0.30 and size >= 0.5:
            demote_tp['size'][0] += 1
            demote_tp['size'][1] += (g >= 0.15)
gf = np.array(gate_fracs)
print(f'CA-1M gate statistic (A-half, {len(gf)} scenes): median={np.median(gf):.2f} p25={np.percentile(gf,25):.2f}  [ScanNet: 1.00]')
for k, (d, tp) in demote_tp.items():
    print(f'M5 rule {k}: would-demote {d}, of which GT-matched {tp} ({tp/max(d,1)*100:.0f}%)')
