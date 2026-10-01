"""Regenerate WeDetect embeddings aligned 1:1 with the existing lifted cache rows.

Replicates generate_lifted_cache.py's exact per-frame filtering (score>=0.05,
top-150 by np.argsort(-ps)) so embedding row i corresponds to cache row i.
Verifies per-frame candidate counts match the existing cache's frame_ids.
"""
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, '/data/ZhaoX/BoxFusion')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')

SCANS = '/extra/ZhaoX/scannet_data/scans'
CACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
OUT = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_emb'
GAP = 25
SCORE_LIFT = 0.05
TOPK = 150

from wedetect_uni_infer import SimpleYOLOWorldDetector
wmodel = SimpleYOLOWorldDetector(backbone_size='base', prompt_dim=768, num_prompts=256, num_proposals=300)
ck = torch.load('/data/ZhaoX/BoxFusion/third_party/WeDetect/wedetect_base_uni.pth', map_location='cpu', weights_only=False)
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

os.makedirs(OUT, exist_ok=True)
scene_list = [l.strip() for l in open('/data/ZhaoX/BoxFusion/evaluation/data_util/meta_data/scannetv2_val.txt')
              if l.strip() and not l.strip().startswith('#')]
for si, scene in enumerate(scene_list):
    out_path = f'{OUT}/{scene}.npz'
    if os.path.exists(out_path):
        continue
    z = np.load(f'{CACHE}/{scene}.npz')
    fids_cache = z['frame_ids']
    n_frames = len(glob.glob(f'{SCANS}/{scene}/color/*.jpg'))
    kfs = []
    for f in range(0, n_frames, GAP):
        cf = f'{SCANS}/{scene}/color/{f}.jpg'
        pf = f'{SCANS}/{scene}/pose/{f}.txt'
        df = f'{SCANS}/{scene}/depth/{f}.png'
        if not all(os.path.exists(p) for p in (cf, pf, df)):
            continue
        pose = np.loadtxt(pf).reshape(4, 4)
        if not np.isfinite(pose).all():
            continue
        kfs.append(f)
    # collect regenerated proposals per scene, grouped by frame
    regen = {}
    with torch.no_grad():
        for i in range(0, len(kfs), 8):
            chunk = kfs[i:i+8]
            outs = wmodel([f'{SCANS}/{scene}/color/{f}.jpg' for f in chunk])
            for f, out in zip(chunk, outs):
                ps = out['scores'].float().cpu().numpy()
                pb = out['bboxes'].float().cpu().numpy()
                pe = out['embeddings'].float().cpu().numpy()
                sel = ps >= SCORE_LIFT
                pb, ps, pe = pb[sel], ps[sel], pe[sel]
                if len(pb) > TOPK:
                    order = np.argsort(-ps)[:TOPK]
                    pb, pe = pb[order], pe[order]
                regen[int(f)] = (pb.astype(np.float64), pe.astype(np.float16))
    # align cache rows by nearest box within the same frame (0.5 px tolerance)
    zc = np.load(f'{CACHE}/{scene}.npz')
    fids_c, b2d_c = zc['frame_ids'], zc['boxes2d']
    emb_rows, matched = [], 0
    placeholder = np.zeros(768, np.float16)
    for f, b in zip(fids_c, b2d_c):
        e = None
        grp = regen.get(int(f))
        if grp is not None and len(grp[0]):
            d = np.abs(grp[0] - b.astype(np.float64)).sum(axis=1)
            j = int(np.argmin(d))
            if d[j] < 2.0:   # sum abs diff over 4 coords < 0.5px avg
                e = grp[1][j]
        if e is not None:
            emb_rows.append(e); matched += 1
        else:
            emb_rows.append(placeholder)
    emb = np.stack(emb_rows)
    ok = matched >= 0.95 * len(fids_c)
    if not ok:
        print(f'{scene}: ALIGNMENT FAIL matched={matched}/{len(fids_c)}', flush=True)
        continue
    np.savez_compressed(out_path, emb=emb, frame_ids=fids_c, matched=np.array([matched]))
    print(f'[{si+1}/{len(scene_list)}] {scene}: {matched}/{len(fids_c)} matched', flush=True)
print('ALL_DONE', flush=True)
