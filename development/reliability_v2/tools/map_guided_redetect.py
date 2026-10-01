"""Map-guided budgeted local re-detection for CA-1M.

Uses the online object map to identify unexplained regions, crops and
zooms those regions, re-runs WeDetect, maps new proposals back to the
original image, lifts via Boxer, and feeds through M1's causal funnel.

Map feedback policy:
  - Score each keyframe's "unexplained regions" using depth backprojection
  - Budget: K crops per keyframe (default 3)
  - Skip regions already covered by high-confidence map boxes
  - Stop re-checking a region after N consecutive failures (default 2)
"""
import os, sys, glob, json, pickle, argparse, time
import numpy as np
import torch
from PIL import Image
from collections import defaultdict

sys.path.insert(0, '/data/ZhaoX/BoxFusion')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')

CA = '/extra/ZhaoX/boxfusion_ca1m'
GAP = 20
BUDGET_K = 3          # max crops per keyframe
STOP_N = 2            # consecutive failures before abandoning a region
SCORE_LIFT = 0.05
DEDUP, CAP, TTL, SELF_NMS = 0.25, 12, 10, 0.50
EDGES = (0.3, 0.5, 0.7, 1.0)
TABLE = (0.05, 0.10, 0.25, 0.40, 0.50)

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0); lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = float(ov[0]*ov[1]*ov[2])
    return inter/float(np.maximum(np.prod(hi1-lo1)+np.prod(hi2-lo2)-inter, 1e-9))

def price(m):
    e = float((m.max(0)-m.min(0)).max())
    for b, s in zip(EDGES, TABLE):
        if e < b: return s
    return TABLE[-1]

def load_models():
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
        "diagnostics_dir": "/tmp/redetect_diag"}}}
    adapter = build_lifting_adapter(cfg, device="cuda", code_root="/data/ZhaoX/BoxFusion")
    return wmodel, adapter

def obb_to_corners(center, extents, R):
    signs = np.array([[sx, sy, sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], float)
    return center[None] + signs * (0.5*extents) @ R.T

def find_unexplained_regions(depth, K, pose, Kd, map_boxes_world, H=384, W=512):
    """Find K most promising unexplained regions using depth + map state."""
    # Backproject a subsample of depth points to world
    ys, xs = np.mgrid[0:H:4, 0:W:4]
    z = depth[ys, xs].astype(np.float64) / 1000.0
    m = (z > 0.2) & (z < 5.0)
    if not m.any():
        return []
    x = (xs[m]+0.5)*z[m]/Kd[0,0]; y = (ys[m]+0.5)*z[m]/Kd[1,1]
    cam = np.stack([x, y, z[m]], 1)
    world = (pose[:3,:3] @ cam.T).T + pose[:3,3]
    # Find clusters of points NOT covered by any map box
    uncovered = []
    for p in world:
        covered = False
        for mb in map_boxes_world:
            lo, hi = mb.min(0), mb.max(0)
            if np.all(p >= lo - 0.1) and np.all(p <= hi + 0.1):
                covered = True; break
        if not covered:
            uncovered.append(p)
    if len(uncovered) < 10:
        return []
    uncovered = np.array(uncovered)
    # Simple grid clustering in XY
    from collections import Counter
    grid = Counter()
    for p in uncovered:
        grid[(int(p[0]//0.5), int(p[1]//0.5))] += 1
    # Top K clusters by point count
    top = sorted(grid.items(), key=lambda x: -x[1])[:K]
    regions = []
    for (gx, gy), count in top:
        center_world = np.array([gx*0.5+0.25, gy*0.5+0.25, uncovered[:,2].mean()])
        regions.append(center_world)
    return regions

def project_point(world_pt, pose, K, W, H):
    Rt = np.linalg.inv(pose)
    c = Rt[:3,:3] @ world_pt + Rt[:3,3]
    if c[2] < 0.1: return None
    uv = K @ c
    x, y = int(uv[0]/uv[2]), int(uv[1]/uv[2])
    if 0 <= x < W and 0 <= y < H: return (x, y)
    return None

def process_scene(scene, native_pkl, nms_jsonl, out_pkl, wmodel, adapter,
                  mode='map_guided', budget_k=BUDGET_K):
    """mode: 'map_guided' or 'uniform' (for control comparison)"""
    t0 = time.time()
    root = f'{CA}/{scene}'
    poses_all = np.load(f'{root}/all_poses.npy')
    K_color = np.loadtxt(f'{root}/K_rgb.txt').reshape(3,3)[:3,:3]
    K_depth = np.loadtxt(f'{root}/K_depth.txt').reshape(3,3)[:3,:3]
    w0, h0 = Image.open(f'{root}/rgb/0.png').size

    d = pickle.load(open(native_pkl, 'rb'))
    rows = [tuple(r) for r in d[0]]
    native = [np.asarray(r[1], float) for r in rows]

    # Map state tracking
    map_boxes = list(native)  # starts with native, grows as births confirmed
    region_fail_count = defaultdict(int)

    # Candidate collection (same format as integrated_online)
    cands = []  # (frame_id, corners_3d, score, is_wd)

    # Standard WeDetect pass on every keyframe (same as before)
    kf_list = list(range(0, len(poses_all), GAP))
    for f in kf_list:
        cf = f'{root}/rgb/{f}.png'
        df = f'{root}/depth/{f}.png'
        if not all(os.path.exists(p) for p in (cf, df)):
            continue
        pose = poses_all[f]
        if not np.isfinite(pose).all():
            continue
        # Standard full-image detection
        with torch.no_grad():
            out = wmodel([cf])[0]
        pb = out['bboxes'].float().cpu().numpy()
        ps = out['scores'].float().cpu().numpy()
        sel = ps >= SCORE_LIFT
        pb, ps = pb[sel], ps[sel]
        if len(pb) > 150:
            order = np.argsort(-ps)[:150]
            pb, ps = pb[order], ps[order]
        rgb = np.asarray(Image.open(cf).convert('RGB'))
        depth = np.asarray(Image.open(df)).astype(np.float32) / 1000.0
        if len(pb):
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
                cands.append((int(f), obb_to_corners(centers[i], np.abs(extents[i])+1e-6, rots[i]), float(ps[i]), 1))

        # --- Map-guided re-detection ---
        depth_raw = np.asarray(Image.open(df))
        if mode == 'map_guided':
            regions = find_unexplained_regions(depth_raw, budget_k, pose, K_depth, map_boxes, H=depth_raw.shape[0], W=depth_raw.shape[1])
        else:
            # uniform control: random crops
            rng = np.random.RandomState(f)
            regions = []
            for _ in range(budget_k):
                cx, cy = rng.randint(100, w0-100), rng.randint(100, h0-100)
                regions.append(np.array([cx, cy, 0]))  # dummy world coords, use as pixel centers

        for region in regions:
            if mode == 'map_guided':
                center_px = project_point(region, pose, K_color, w0, h0)
                if center_px is None:
                    continue
                cx, cy = center_px
            else:
                cx, cy = int(region[0]), int(region[1])

            # Skip if too close to edge
            if cx < 50 or cx >= w0-50 or cy < 50 or cy >= h0-50:
                continue

            # Check failure count for this region
            region_key = (f // (GAP*5), cx//100, cy//100)  # coarse spatial-temporal key
            if region_fail_count[region_key] >= STOP_N:
                continue

            # Crop 2x zoom
            crop_w, crop_h = w0//2, h0//2
            x1 = max(0, cx - crop_w//2); y1 = max(0, cy - crop_h//2)
            x2 = min(w0, x1+crop_w); y2 = min(h0, y1+crop_h)
            if x2-x1 < 100 or y2-y1 < 100:
                continue

            img = Image.open(cf).convert('RGB')
            crop = img.crop((x1,y1,x2,y2)).resize((w0,h0), Image.BICUBIC)
            crop.save('/tmp/redetect_crop.png')

            with torch.no_grad():
                out2 = wmodel(['/tmp/redetect_crop.png'])[0]
            pb2 = out2['bboxes'].float().cpu().numpy()
            ps2 = out2['scores'].float().cpu().numpy()
            sel2 = ps2 >= SCORE_LIFT
            pb2, ps2 = pb2[sel2], ps2[sel2]

            found_new = False
            if len(pb2):
                # Map crop coords back to original
                orig_boxes = np.zeros_like(pb2)
                for i in range(len(pb2)):
                    bx1, by1, bx2, by2 = pb2[i]
                    orig_boxes[i] = [
                        x1 + bx1*(x2-x1)/w0, y1 + by1*(y2-y1)/h0,
                        x1 + bx2*(x2-x1)/w0, y1 + by2*(y2-y1)/h0]

                # Filter: only keep detections NOT overlapping existing full-image detections
                new_boxes = []
                new_scores = []
                for i in range(len(orig_boxes)):
                    ob = orig_boxes[i]
                    overlaps_existing = False
                    for fb in pb:
                        fx1, fy1, fx2, fy2 = fb
                        ix1 = max(ob[0], fx1); iy1 = max(ob[1], fy1)
                        ix2 = min(ob[2], fx2); iy2 = min(ob[3], fy2)
                        if ix2 > ix1 and iy2 > iy1:
                            iou = (ix2-ix1)*(iy2-iy1) / max((ob[2]-ob[0])*(ob[3]-ob[1]) + (fx2-fx1)*(fy2-fy1) - (ix2-ix1)*(iy2-iy1), 1e-9)
                            if iou > 0.3:
                                overlaps_existing = True; break
                    if not overlaps_existing:
                        new_boxes.append(ob)
                        new_scores.append(ps2[i])
                        found_new = True

                if new_boxes:
                    new_boxes = np.array(new_boxes)
                    # Lift via Boxer
                    datum2, meta2 = adapter._make_datum(
                        image=rgb, depth=depth, boxes_xyxy=torch.from_numpy(new_boxes).float(),
                        image_K=K_color, depth_K=K_depth, camera_to_world=pose,
                        scene_id=scene, frame_id=int(f))
                    outp2, _, _ = adapter.forward_raw_with_feature_cache(
                        datum2, scene_id=scene, frame_id=int(f)+10000,  # different cache key
                        encoder_input_sha256=meta2['encoder_input_sha256'])
                    obbs2 = outp2['obbs_pr_w'][0]
                    centers2 = obbs2.bb3_center_world.float().cpu().numpy()
                    extents2 = obbs2.bb3_diagonal.float().cpu().numpy()
                    rots2 = obbs2.T_world_object.R.float().cpu().numpy()
                    for i in range(len(centers2)):
                        cands.append((int(f), obb_to_corners(centers2[i], np.abs(extents2[i])+1e-6, rots2[i]), float(new_scores[i]), 1))

            if not found_new:
                region_fail_count[region_key] += 1

    # NMS-child events
    if os.path.exists(nms_jsonl):
        for line in open(nms_jsonl):
            r = json.loads(line)
            cands.append((int(r['keyframe_id']), np.asarray(r['child_corners_world'], float),
                          float(r['child_score']), 0))
    n_wd = sum(1 for c in cands if c[3] == 1)

    # --- M1 causal funnel (same as integrated_online) ---
    births = []
    if cands:
        all_f = [c[0] for c in cands]
        order = np.argsort(np.array(all_f), kind='stable')
        uniq, ordinal_of = np.unique(np.array(all_f), return_inverse=True)
        receipts = []
        for i in order:
            c, s, f = cands[i][1], float(cands[i][2]), int(ordinal_of[i])
            cc = c.mean(0)
            if any(aabb_iou(c, nb) >= DEDUP for nb in native):
                continue
            best, best_v = None, 0.0
            for r in receipts:
                if f - r['last_ord'] > TTL: continue
                rc = r['obs'][-1]
                if np.linalg.norm(rc.mean(0)-cc) > 0.50: continue
                v = aabb_iou(c, rc)
                if v >= 0.10 and v > best_v: best_v, best = v, r
            src_wd = int(i < n_wd)
            if best is not None:
                best['obs'].append(c); best['frames'].add(f)
                best['last_ord'] = f; best['scores'].append(s); best['n_wd'] += src_wd
            else:
                receipts.append(dict(obs=[c], frames={f}, last_ord=f, scores=[s], n_wd=src_wd))
        cand_b = []
        for r in receipts:
            mv = 3 if 2*r['n_wd'] > len(r['obs']) else 2
            if len(r['frames']) < mv: continue
            obs = r['obs']
            bj, bs = 0, -1.0
            for j, a in enumerate(obs):
                ss = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                if ss > bs: bs, bj = ss, j
            cand_b.append((float(np.mean(r['scores'])), obs[bj]))
        cand_b.sort(key=lambda x: -x[0])
        kept = []
        for strength, m in cand_b:
            if any(aabb_iou(m, km) >= SELF_NMS for _, km in kept): continue
            kept.append((strength, m))
            if len(kept) >= CAP: break
        births = kept

    all_rows = list(rows) + [(0, m, price(m)) for _, m in births]
    out_sc = [all_rows] + [[(det[0],det[1],det[2]) for det in sc] for sc in d[1:]]
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    pickle.dump(out_sc, open(out_pkl, 'wb'))
    dt = time.time() - t0
    n_new = len(cands) - n_wd - (len(cands) - n_wd - sum(1 for c in cands if c[3]==0))
    print(f'{scene}: cands={len(cands)} (wd={n_wd} child={len(cands)-n_wd}) births={len(births)} rows={len(all_rows)} | {dt:.1f}s', flush=True)
    return dt

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene')
    ap.add_argument('--native-pkl')
    ap.add_argument('--nms-jsonl')
    ap.add_argument('--out-pkl')
    ap.add_argument('--mode', default='map_guided', choices=['map_guided', 'uniform'])
    ap.add_argument('--budget', type=int, default=BUDGET_K)
    args = ap.parse_args()
    wmodel, adapter = load_models()
    process_scene(args.scene, args.native_pkl, args.nms_jsonl, args.out_pkl, wmodel, adapter,
                  mode=args.mode, budget_k=args.budget)
