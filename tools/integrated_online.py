"""Integrated online M1/M2/M5 post-processor (Plan A+).

For one scene, immediately after demo.py (observers on) finishes:
  1. LIVE WeDetect-Uni forward on every gap-25 keyframe (frozen weights) -> 2D proposals
  2. LIVE BoxerNet lift (shared engine, feature cache) -> 3D candidate corners
  3. Children from the run's own NMS observer log (M1b)
  4. Funnel + scene-end finalize (v9 semantics: per-observation dedup vs final map,
     medoid, cap, size pricing)                                    (M1c/M1d)
  5. Consensus support over live keyframe proposals, boost alpha=0.8 tau=0.5 (M2)
  6. Negative-evidence retirement: ch1 in-view-unsupported, ch2 depth-empty core (M5)
Everything is past data at scene end. No offline caches are read except the run's own logs.

Usage:
  python tools/integrated_online.py --scene scene0568_00 \
      --native-pkl results/<exp>/scene0568_00_boxes.pkl \
      --nms-jsonl diagnostics/<exp>/scene0568_00_pvq_nms.jsonl \
      --out-pkl results/integrated/scene0568_00_boxes.pkl
"""
import os, sys, glob, json, pickle, argparse, time
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, '/data/ZhaoX/BoxFusion')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')

SCANS = '/extra/ZhaoX/scannet_data/scans'
GAP = 25
SCORE_LIFT = 0.05
TOPK_PER_FRAME = 150
DEDUP, CAP, TTL, SELF_NMS = 0.25, 12, 10, 0.50
ALPHA, TAU = 0.8, 0.5
EDGES = (0.3, 0.5, 0.7, 1.0)
TABLE = (0.05, 0.10, 0.25, 0.40, 0.50)
M5_UNS_THR, M5_MIN_INV, M5_DEMOTE = 0.80, 5, 0.3

def obb_to_corners(center, extents, R):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return center[None] + signs * (0.5 * extents) @ R.T

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0); lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = float(ov[0]*ov[1]*ov[2])
    return inter/float(np.maximum(np.prod(hi1-lo1)+np.prod(hi2-lo2)-inter, 1e-9))

def price(m):
    e = float((m.max(0)-m.min(0)).max())
    for b, s in zip(EDGES, TABLE):
        if e < b:
            return s
    return TABLE[-1]

def project_xyxy(corners_w, pose, K, W=1296, H=968):
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

def load_models():
    from wedetect_uni_infer import SimpleYOLOWorldDetector
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
    from boxfusion.boxer_lifter import build_lifting_adapter
    cfg = {"lifting": {"backend": "boxer", "boxer": {
        "mode": "observer", "apply_stage": "post_filter",
        "official_root": "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer",
        "checkpoint": "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer/ckpts/boxernet_hw960in2x6d768-c88128f8.ckpt",
        "expected_commit": "1f86542dc342a4b1d474c87c97c5d1d6566d9148",
        "checkpoint_sha256": "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f",
        "dinov3_sha256": "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea",
        "precision": "bfloat16", "use_sdp": True, "sdp_samples": 10000, "seed": 0,
        "cache_image_features": True,
        "diagnostics_dir": "/tmp/integrated_online_diag"}}}
    adapter = build_lifting_adapter(cfg, device="cuda", code_root="/data/ZhaoX/BoxFusion")
    return wmodel, adapter

def process_scene(scene, native_pkl, nms_jsonl, out_pkl, wmodel, adapter):
    t0 = time.time()
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
    # ---- 1+2: live WeDetect + lift on every keyframe (in stream order) ----
    corners_all, scores_all, fids_all, b2d_all = [], [], [], []
    world_pts = []          # M5 ch2: depth backprojection, every 4th kf, 8px step
    for ki, (f, pose, cf, df) in enumerate(kfs):
        try:
            with torch.no_grad():
                out = wmodel([cf])[0]
            pb = out['bboxes'].float().cpu().numpy()
            ps = out['scores'].float().cpu().numpy()
            sel = ps >= SCORE_LIFT
            pb, ps = pb[sel], ps[sel]
            if len(pb) > TOPK_PER_FRAME:
                order = np.argsort(-ps)[:TOPK_PER_FRAME]
                pb, ps = pb[order], ps[order]
            if len(pb):
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
                    corners_all.append(obb_to_corners(centers[i], np.abs(extents[i])+1e-6, rots[i]))
                    scores_all.append(float(ps[i]))
                    fids_all.append(int(f))
                    b2d_all.append(pb[i])
            if ki % 4 == 0:                      # depth points for M5 ch2
                depth = np.asarray(Image.open(df)).astype(np.float64) / 1000.0
                Hh, Ww = depth.shape
                ys, xs = np.mgrid[0:Hh:8, 0:Ww:8]
                z = depth[ys, xs].ravel(); xs_ = xs.ravel(); ys_ = ys.ravel()
                m = (z > 0.2) & (z < 6.0)
                if m.any():
                    x = (xs_[m]+0.5)*z[m]/K_depth[0,0]
                    y = (ys_[m]+0.5)*z[m]/K_depth[1,1]
                    cam = np.stack([x, y, z[m]], 1)
                    world_pts.append((pose[:3,:3] @ cam.T).T + pose[:3,3])
        except Exception as e:
            print(f'  {scene} f{f} ERROR: {e}', flush=True)
    P = np.concatenate(world_pts) if world_pts else np.zeros((0, 3))
    by_frame = {}
    for b, fr in zip(b2d_all, fids_all):
        by_frame.setdefault(int(fr), []).append(b)
    by_frame = {k: np.array(v) for k, v in by_frame.items()}
    pose_cache = {f: p for f, p, _, _ in kfs}
    t_fwd = time.time() - t0

    # ---- 3: children from this run's NMS log ----
    cands = [(int(fr), np.asarray(c, float), float(s), 1)
             for c, s, fr in zip(corners_all, scores_all, fids_all)]
    if os.path.exists(nms_jsonl):
        for line in open(nms_jsonl):
            r = json.loads(line)
            cands.append((int(r['keyframe_id']), np.asarray(r['child_corners_world'], float),
                          float(r['child_score']), 0))
    d = pickle.load(open(native_pkl, 'rb'))
    rows = [tuple(r) for r in d[0]]
    native = [np.asarray(r[1], float) for r in rows]
    n_wd = len(corners_all)

    # ---- 4: funnel, v9 semantics (dedup per-observation vs final map) ----
    births = []
    if cands:
        all_c = [c for _, c, _, _ in cands]
        all_s = [s for _, _, s, _ in cands]
        all_f = [f for f, *_ in cands]
        order = np.argsort(np.array(all_f), kind='stable')
        uniq, ordinal_of = np.unique(np.array(all_f), return_inverse=True)
        receipts = []
        for i in order:
            c, s, f = all_c[i], float(all_s[i]), int(ordinal_of[i])
            if any(aabb_iou(c, nb) >= DEDUP for nb in native):
                continue
            best, best_v = None, 0.0
            for r in receipts:
                if f - r['last_ord'] > TTL:
                    continue
                rc = r['obs'][-1]
                if np.linalg.norm(rc.mean(0) - c.mean(0)) > 0.50:
                    continue
                v = aabb_iou(c, rc)
                if v >= 0.10 and v > best_v:
                    best_v, best = v, r
            src_wd = int(i < n_wd)
            if best is not None:
                best['obs'].append(c); best['frames'].add(f)
                best['last_ord'] = f; best['scores'].append(s); best['n_wd'] += src_wd
            else:
                receipts.append(dict(obs=[c], frames={f}, last_ord=f, scores=[s], n_wd=src_wd))
        cand_b = []
        for r in receipts:
            mv = 3 if 2*r['n_wd'] > len(r['obs']) else 2
            if len(r['frames']) < mv:
                continue
            obs = r['obs']
            bj, bs = 0, -1.0
            for j, a in enumerate(obs):
                ss = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                if ss > bs:
                    bs, bj = ss, j
            cand_b.append((float(np.mean(r['scores'])), obs[bj]))
        cand_b.sort(key=lambda x: -x[0])
        kept = []
        for strength, m in cand_b:
            if any(aabb_iou(m, km) >= SELF_NMS for _, km in kept):
                continue
            kept.append((strength, m))
            if len(kept) >= CAP:
                break
        births = kept
    all_rows = list(rows) + [(0, m, price(m)) for _, m in births]

    # ---- 5+6: M2 support and M5 dual-channel, per row ----
    out_rows = []
    n_demoted = 0
    for cls, corners_, s in all_rows:
        cc = np.asarray(corners_, float)
        sup, inv, uns, core = 0.0, 0, 0, 0
        for fr, props in by_frame.items():
            pose = pose_cache.get(fr)
            if pose is None:
                continue
            bb = project_xyxy(cc, pose, K_color)
            if bb is None:
                continue
            inv += 1
            v = iou2d(bb, props)
            sup = max(sup, v)
            if v < 0.30:
                uns += 1
        lo, hi = cc.min(0), cc.max(0)
        ctr = (lo+hi)/2; ext = hi-lo
        lo2, hi2 = ctr-ext*0.75, ctr+ext*0.75
        core = int(((P >= lo2) & (P <= hi2)).all(1).sum()) if len(P) else 0
        ns = min(0.99, float(s) + ALPHA * max(0.0, sup - TAU))          # M2
        if inv >= M5_MIN_INV and (uns/max(inv,1) >= M5_UNS_THR or core == 0):   # M5
            ns = ns * M5_DEMOTE
            n_demoted += 1
        out_rows.append((cls, corners_, ns))
    out_sc = [out_rows] + [[(det[0], det[1], det[2]) for det in sc] for sc in d[1:]]
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    pickle.dump(out_sc, open(out_pkl, 'wb'))
    dt = time.time() - t0
    print(f'{scene}: kfs={len(kfs)} lifted={len(corners_all)} births={len(births)} '
          f'rows={len(out_rows)} demoted={n_demoted} | fwd {t_fwd:.1f}s total {dt:.1f}s', flush=True)
    return dt, len(kfs)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene')
    ap.add_argument('--native-pkl')
    ap.add_argument('--nms-jsonl')
    ap.add_argument('--out-pkl')
    ap.add_argument('--batch', help='scene list file; native/kfmap run dirs fixed')
    args = ap.parse_args()
    wmodel, adapter = load_models()
    if args.batch:
        NAT = os.environ.get('CAUSAL_NAT', '/data/ZhaoX/BoxFusion/results/scannet_t05_boxer_kfmap_score05')
        KFD = os.environ.get('CAUSAL_KFD', '/data/ZhaoX/BoxFusion/diagnostics/kfmap_score05')
        OUTD = os.environ.get("CAUSAL_OUT", "/data/ZhaoX/BoxFusion/results/integrated_100_live")
        os.makedirs(OUTD, exist_ok=True)
        scenes = [l.strip() for l in open(args.batch) if l.strip()]
        for sc in scenes:
            process_scene(sc, f'{NAT}/{sc}_boxes.pkl', f'{KFD}/{sc}_pvq_nms.jsonl',
                          f'{OUTD}/{sc}_boxes.pkl', wmodel, adapter)
        print('BATCH_DONE', flush=True)
    else:
        process_scene(args.scene, args.native_pkl, args.nms_jsonl, args.out_pkl, wmodel, adapter)
