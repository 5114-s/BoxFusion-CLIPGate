"""Map-guided budgeted local re-detection v2 — streaming scheduler.

Fixes from v1:
  - Streaming: per-keyframe processing with pending-candidate queue
  - No future info: uses map-so-far (not final map) for region selection
  - Correct depth backprojection (subtracts principal point)
  - 3D instance tracking (world coords, not image coords)
  - Confirmation-aware budget: 1 exploration + 2 confirmation slots
  - Same-frame single-count (no double-counting from overlapping crops)
  - Original ≥3/2 threshold, Boxer, dedup, pricing — all unchanged
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
SCORE_LIFT = 0.05
DEDUP, CAP, TTL, SELF_NMS = 0.25, 12, 10, 0.50
EDGES = (0.3, 0.5, 0.7, 1.0)
TABLE = (0.05, 0.10, 0.25, 0.40, 0.50)
MAX_PENDING = 30        # max pending candidates in queue
PENDING_TTL = 10         # keyframes before pending candidate expires

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
        "diagnostics_dir": "/tmp/redetect_v2"}}}
    adapter = build_lifting_adapter(cfg, device="cuda", code_root="/data/ZhaoX/BoxFusion")
    return wmodel, adapter

def obb_to_corners(center, extents, R):
    signs = np.array([[sx, sy, sz] for sx in (-1,1) for sy in (-1,1) for sz in (-1,1)], float)
    return center[None] + signs * (0.5*extents) @ R.T

def backproject_depth(depth_raw, K_depth, pose, stride=4):
    """Correct depth backprojection with principal point subtraction."""
    H, W = depth_raw.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    z = depth_raw[ys, xs].astype(np.float64) / 1000.0
    m = (z > 0.2) & (z < 5.0)
    if not m.any():
        return np.zeros((0, 3))
    # Subtract principal point (cx, cy) from pixel coordinates
    x_cam = ((xs[m] + 0.5) - K_depth[0, 2]) * z[m] / K_depth[0, 0]
    y_cam = ((ys[m] + 0.5) - K_depth[1, 2]) * z[m] / K_depth[1, 1]
    cam = np.stack([x_cam, y_cam, z[m]], 1)
    world = (pose[:3, :3] @ cam.T).T + pose[:3, 3]
    return world

def find_unexplained_v2(depth_raw, K_depth, pose, map_boxes_world, K_regions=1):
    """Find top-K unexplained regions using correctly backprojected depth."""
    pts = backproject_depth(depth_raw, K_depth, pose, stride=4)
    if len(pts) < 10:
        return []
    uncovered = []
    for p in pts:
        covered = False
        for mb in map_boxes_world:
            lo, hi = mb.min(0), mb.max(0)
            if np.all(p >= lo - 0.15) and np.all(p <= hi + 0.15):
                covered = True; break
        if not covered:
            uncovered.append(p)
    if len(uncovered) < 15:
        return []
    uncovered = np.array(uncovered)
    # Cluster in XY (0.5m grid)
    from collections import Counter
    grid = Counter()
    for p in uncovered:
        grid[(int(p[0] // 0.5), int(p[1] // 0.5))] += 1
    top = sorted(grid.items(), key=lambda x: -x[1])[:K_regions]
    regions = []
    for (gx, gy), count in top:
        mask = (uncovered[:, 0] >= gx * 0.5) & (uncovered[:, 0] < (gx + 1) * 0.5) & \
               (uncovered[:, 1] >= gy * 0.5) & (uncovered[:, 1] < (gy + 1) * 0.5)
        cluster_pts = uncovered[mask]
        if len(cluster_pts) >= 10:
            center = cluster_pts.mean(0)
            regions.append(center)
    return regions

def project_to_image(world_pt, pose, K, W, H):
    Rt = np.linalg.inv(pose)
    c = Rt[:3, :3] @ world_pt + Rt[:3, 3]
    if c[2] < 0.1: return None
    uv = K @ c
    x, y = int(uv[0] / uv[2]), int(uv[1] / uv[2])
    if 0 <= x < W and 0 <= y < H: return (x, y)
    return None

def detect_and_lift(wmodel, adapter, rgb, depth, boxes_2d, K_color, K_depth, pose, scene, frame_id):
    """Run Boxer lifting on 2D boxes, return 3D corners list."""
    if len(boxes_2d) == 0:
        return []
    datum, meta = adapter._make_datum(
        image=rgb, depth=depth, boxes_xyxy=torch.from_numpy(boxes_2d).float(),
        image_K=K_color, depth_K=K_depth, camera_to_world=pose,
        scene_id=scene, frame_id=int(frame_id))
    outp, _, _ = adapter.forward_raw_with_feature_cache(
        datum, scene_id=scene, frame_id=int(frame_id),
        encoder_input_sha256=meta['encoder_input_sha256'])
    obbs = outp['obbs_pr_w'][0]
    centers = obbs.bb3_center_world.float().cpu().numpy()
    extents = obbs.bb3_diagonal.float().cpu().numpy()
    rots = obbs.T_world_object.R.float().cpu().numpy()
    return [obb_to_corners(centers[i], np.abs(extents[i]) + 1e-6, rots[i])
            for i in range(len(centers))]

def process_scene(scene, native_pkl, nms_jsonl, out_pkl, wmodel, adapter,
                  mode='confirm_aware', budget=3, diag_out=None):
    """
    mode: 'none' (A), 'explore_only' (B), 'confirm_aware' (C)
    """
    t0 = time.time()
    root = f'{CA}/{scene}'
    poses_all = np.load(f'{root}/all_poses.npy')
    K_color = np.loadtxt(f'{root}/K_rgb.txt').reshape(3, 3)[:3, :3]
    K_depth = np.loadtxt(f'{root}/K_depth.txt').reshape(3, 3)[:3, :3]
    img0 = Image.open(f'{root}/rgb/0.png')
    w0, h0 = img0.size

    d = pickle.load(open(native_pkl, 'rb'))
    native_rows = [tuple(r) for r in d[0]]
    native_corners = [np.asarray(r[1], float) for r in native_rows]

    # Streaming state
    map_so_far = list(native_corners)     # starts with native, grows
    pending = []                           # list of dict(id, corners, frames=set, scores=[], n_wd, born_kf, last_kf)
    next_id = 0
    births = []
    # Funnel receipts (for child stream, processed per-frame)
    child_receipts = []
    # Diagnostics
    diag = defaultdict(int)
    conversion = []  # track candidate lifecycle

    kf_list = list(range(0, len(poses_all), GAP))
    kf_ordinal = {f: i for i, f in enumerate(kf_list)}

    for kf_idx, f in enumerate(kf_list):
        cf = f'{root}/rgb/{f}.png'
        df = f'{root}/depth/{f}.png'
        if not all(os.path.exists(p) for p in (cf, df)):
            continue
        pose = poses_all[f]
        if not np.isfinite(pose).all():
            continue
        rgb = np.asarray(Image.open(cf).convert('RGB'))
        depth = np.asarray(Image.open(df)).astype(np.float32) / 1000.0
        depth_raw = np.asarray(Image.open(df))
        frame_ord = kf_idx

        # --- 1. Full-image detection ---
        with torch.no_grad():
            out = wmodel([cf])[0]
        pb_full = out['bboxes'].float().cpu().numpy()
        ps_full = out['scores'].float().cpu().numpy()
        sel = ps_full >= SCORE_LIFT
        pb_full, ps_full = pb_full[sel], ps_full[sel]
        if len(pb_full) > 150:
            order = np.argsort(-ps_full)[:150]
            pb_full, ps_full = pb_full[order], ps_full[order]

        corners_full = []
        if len(pb_full):
            corners_full = detect_and_lift(wmodel, adapter, rgb, depth, pb_full,
                                            K_color, K_depth, pose, scene, f)

        # --- 2. Associate full-image detections with map/pending (M1 funnel logic) ---
        for i, c3d in enumerate(corners_full):
            if any(aabb_iou(c3d, nb) >= DEDUP for nb in native_corners):
                continue
            cc = c3d.mean(0)
            best, best_v = None, 0.0
            for p in pending:
                if p['confirmed']: continue
                rc = p['corners'][-1]
                if np.linalg.norm(rc.mean(0) - cc) > 0.50: continue
                v = aabb_iou(c3d, rc)
                if v >= 0.10 and v > best_v: best_v, best = v, p
            if best is not None:
                if frame_ord not in best['frame_set']:  # same-frame single-count
                    best['corners'].append(c3d)
                    best['frames'].add(frame_ord)
                    best['frame_set'].add(frame_ord)
                    best['scores'].append(float(ps_full[i]))
                    best['n_wd'] += 1
                    best['last_kf'] = frame_ord
            else:
                pending.append(dict(id=next_id, corners=[c3d], frames={frame_ord},
                                    frame_set={frame_ord}, scores=[float(ps_full[i])],
                                    n_wd=1, confirmed=False, born_kf=frame_ord,
                                    last_kf=frame_ord))
                conversion.append(dict(id=next_id, born_kf=frame_ord, source='full',
                                       n_frames=1, confirmed=False))
                next_id += 1
                diag['new_pending_full'] += 1

        # --- 3. Budget allocation for local re-detection ---
        if mode != 'none' and budget > 0:
            explore_slots = budget
            confirm_slots = 0
            if mode == 'confirm_aware':
                explore_slots = 1
                confirm_slots = budget - 1

            # --- 3a. Confirmation slots: pick pending candidates that are:
            #   - not confirmed
            #   - visible in current frame
            #   - closest to confirmation (most frames accumulated)
            #   - not expired
            confirm_targets = []
            if confirm_slots > 0:
                candidates = [p for p in pending
                              if not p['confirmed']
                              and frame_ord - p['last_kf'] <= PENDING_TTL
                              and len(p['frames']) >= 1]
                # Sort by frames accumulated (descending) — closest to confirmation first
                candidates.sort(key=lambda p: -len(p['frames']))
                for p in candidates[:confirm_slots]:
                    center_world = p['corners'][-1].mean(0)
                    px = project_to_image(center_world, pose, K_color, w0, h0)
                    if px is not None:
                        confirm_targets.append((p, px))
                        diag['confirm_slot_used'] += 1

            # --- 3b. Exploration slots: find unexplained regions
            explore_targets = []
            if explore_slots > 0:
                regions = find_unexplained_v2(depth_raw, K_depth, pose, map_so_far, explore_slots)
                for region in regions:
                    px = project_to_image(region, pose, K_color, w0, h0)
                    if px is not None:
                        explore_targets.append(px)
                        diag['explore_slot_used'] += 1

            # Unused confirm slots become explore slots
            remaining_explore = explore_slots + (confirm_slots - len(confirm_targets))
            if remaining_explore > len(explore_targets):
                extra_regions = find_unexplained_v2(depth_raw, K_depth, pose, map_so_far,
                                                     remaining_explore - len(explore_targets))
                for r in extra_regions:
                    px = project_to_image(r, pose, K_color, w0, h0)
                    if px is not None:
                        explore_targets.append(px)

            # --- 3c. Execute re-detections ---
            all_targets = [(None, px) for px in explore_targets] + confirm_targets
            for pend_obj, (cx, cy) in all_targets:
                if cx is None or cx < 50 or cx >= w0 - 50 or cy < 50 or cy >= h0 - 50:
                    continue
                crop_w, crop_h = w0 // 2, h0 // 2
                x1 = max(0, cx - crop_w // 2); y1 = max(0, cy - crop_h // 2)
                x2 = min(w0, x1 + crop_w); y2 = min(h0, y1 + crop_h)
                if x2 - x1 < 100 or y2 - y1 < 100:
                    continue
                img = Image.open(cf).convert('RGB')
                crop = img.crop((x1, y1, x2, y2)).resize((w0, h0), Image.BICUBIC)
                crop.save('/tmp/v2_crop.png')
                with torch.no_grad():
                    out2 = wmodel(['/tmp/v2_crop.png'])[0]
                pb2 = out2['bboxes'].float().cpu().numpy()
                ps2 = out2['scores'].float().cpu().numpy()
                sel2 = ps2 >= SCORE_LIFT
                pb2, ps2 = pb2[sel2], ps2[sel2]
                if len(pb2) == 0:
                    continue

                # Map crop coords back
                orig_boxes = np.zeros_like(pb2)
                for j in range(len(pb2)):
                    bx1, by1, bx2, by2 = pb2[j]
                    orig_boxes[j] = [x1 + bx1 * (x2 - x1) / w0, y1 + by1 * (y2 - y1) / h0,
                                     x1 + bx2 * (x2 - x1) / w0, y1 + by2 * (y2 - y1) / h0]

                # For confirmation targets: check if detection covers the target
                if pend_obj is not None:
                    # Check if any crop detection covers the pending candidate's center
                    found = False
                    for j in range(len(orig_boxes)):
                        ob = orig_boxes[j]
                        if ob[0] <= cx <= ob[2] and ob[1] <= cy <= ob[3]:
                            found = True
                            # Lift just this box
                            box_3d = detect_and_lift(wmodel, adapter, rgb, depth,
                                                     orig_boxes[j:j+1], K_color, K_depth, pose,
                                                     scene, f + 10000)
                            if box_3d:
                                c3d = box_3d[0]
                                if frame_ord not in pend_obj['frame_set']:
                                    pend_obj['corners'].append(c3d)
                                    pend_obj['frames'].add(frame_ord)
                                    pend_obj['frame_set'].add(frame_ord)
                                    pend_obj['scores'].append(float(ps2[j]))
                                    pend_obj['n_wd'] += 1
                                    pend_obj['last_kf'] = frame_ord
                                    diag['confirm_obs_added'] += 1
                            break
                    if not found:
                        diag['confirm_miss'] += 1
                else:
                    # Exploration: find NEW detections not in full-image results
                    new_boxes = []
                    new_scores = []
                    for j in range(len(orig_boxes)):
                        ob = orig_boxes[j]
                        overlaps = False
                        for fb in pb_full:
                            ix1 = max(ob[0], fb[0]); iy1 = max(ob[1], fb[1])
                            ix2 = min(ob[2], fb[2]); iy2 = min(ob[3], fb[3])
                            if ix2 > ix1 and iy2 > iy1:
                                inter = (ix2 - ix1) * (iy2 - iy1)
                                area1 = (ob[2] - ob[0]) * (ob[3] - ob[1])
                                area2 = (fb[2] - fb[0]) * (fb[3] - fb[1])
                                if inter / max(area1 + area2 - inter, 1e-9) > 0.3:
                                    overlaps = True; break
                        if not overlaps:
                            new_boxes.append(ob)
                            new_scores.append(ps2[j])
                    if new_boxes:
                        new_boxes = np.array(new_boxes)
                        boxes_3d = detect_and_lift(wmodel, adapter, rgb, depth, new_boxes,
                                                   K_color, K_depth, pose, scene, f + 20000)
                        for c3d in boxes_3d:
                            if any(aabb_iou(c3d, nb) >= DEDUP for nb in native_corners):
                                continue
                            cc = c3d.mean(0)
                            best, best_v = None, 0.0
                            for p in pending:
                                if p['confirmed']: continue
                                rc = p['corners'][-1]
                                if np.linalg.norm(rc.mean(0) - cc) > 0.50: continue
                                v = aabb_iou(c3d, rc)
                                if v >= 0.10 and v > best_v: best_v, best = v, p
                            if best is not None:
                                if frame_ord not in best['frame_set']:
                                    best['corners'].append(c3d)
                                    best['frames'].add(frame_ord)
                                    best['frame_set'].add(frame_ord)
                                    best['scores'].append(float(new_scores[0]))
                                    best['n_wd'] += 1
                                    best['last_kf'] = frame_ord
                            else:
                                pending.append(dict(id=next_id, corners=[c3d], frames={frame_ord},
                                                    frame_set={frame_ord}, scores=[float(new_scores[0])],
                                                    n_wd=1, confirmed=False, born_kf=frame_ord,
                                                    last_kf=frame_ord))
                                conversion.append(dict(id=next_id, born_kf=frame_ord, source='crop',
                                                       n_frames=1, confirmed=False))
                                next_id += 1
                                diag['new_pending_crop'] += 1

        # --- 4. Check confirmations ---
        for p in pending:
            if p['confirmed']: continue
            mv = 3 if 2 * p['n_wd'] > len(p['corners']) else 2
            if len(p['frames']) >= mv:
                obs = p['corners']
                bj, bs = 0, -1.0
                for j, a in enumerate(obs):
                    ss = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                    if ss > bs: bs, bj = ss, j
                medoid = obs[bj]
                p['confirmed'] = True
                strength = float(np.mean(p['scores']))
                if not any(aabb_iou(medoid, mb) >= SELF_NMS for mb in map_so_far):
                    if len(births) < CAP:
                        births.append(medoid)
                        map_so_far.append(medoid)
                        diag['births_confirmed'] += 1
                        for c in conversion:
                            if c['id'] == p['id']:
                                c['confirmed'] = True
                                c['n_frames'] = len(p['frames'])
                                break

        # --- 5. Expire old pending ---
        pending = [p for p in pending
                   if not p['confirmed'] and frame_ord - p['last_kf'] <= PENDING_TTL]
        if len(pending) > MAX_PENDING:
            pending.sort(key=lambda p: -len(p['frames']))
            pending = pending[:MAX_PENDING]

    # NMS-child events (processed at end, same as integrated_online)
    if os.path.exists(nms_jsonl):
        cands_child = []
        for line in open(nms_jsonl):
            r = json.loads(line)
            cands_child.append((int(r['keyframe_id']), np.asarray(r['child_corners_world'], float),
                                float(r['child_score'])))
        if cands_child:
            all_f = [c[0] for c in cands_child]
            order = np.argsort(np.array(all_f), kind='stable')
            uniq, ordinal_of = np.unique(np.array(all_f), return_inverse=True)
            receipts = []
            for i in order:
                c, s, fo = cands_child[i][1], float(cands_child[i][2]), int(ordinal_of[i])
                cc = c.mean(0)
                if any(aabb_iou(c, nb) >= DEDUP for nb in native_corners): continue
                best, best_v = None, 0.0
                for r in receipts:
                    if fo - r['last_ord'] > TTL: continue
                    rc = r['obs'][-1]
                    if np.linalg.norm(rc.mean(0) - cc) > 0.50: continue
                    v = aabb_iou(c, rc)
                    if v >= 0.10 and v > best_v: best_v, best = v, r
                if best is not None:
                    best['obs'].append(c); best['frames'].add(fo)
                    best['last_ord'] = fo; best['scores'].append(s)
                else:
                    receipts.append(dict(obs=[c], frames={fo}, last_ord=fo, scores=[s]))
            for r in receipts:
                if len(r['frames']) < 2: continue
                obs = r['obs']
                bj, bs = 0, -1.0
                for j, a in enumerate(obs):
                    ss = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                    if ss > bs: bs, bj = ss, j
                medoid = obs[bj]
                if not any(aabb_iou(medoid, mb) >= SELF_NMS for mb in map_so_far):
                    if len(births) < CAP:
                        births.append(medoid)
                        map_so_far.append(medoid)
                        diag['births_from_child'] += 1

    all_rows = list(native_rows) + [(0, m, price(m)) for m in births]
    out_sc = [all_rows] + [[(det[0], det[1], det[2]) for det in sc] for sc in d[1:]]
    os.makedirs(os.path.dirname(out_pkl), exist_ok=True)
    pickle.dump(out_sc, open(out_pkl, 'wb'))

    dt = time.time() - t0
    print(f'{scene}: births={len(births)} rows={len(all_rows)} | {dt:.1f}s', flush=True)
    print(f'  diag: {dict(diag)}', flush=True)
    # Print conversion chain
    if diag_out:
        json.dump(dict(diag=dict(diag), conversion=conversion,
                       births=len(births), rows=len(all_rows)),
                  open(diag_out, 'w'), indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene')
    ap.add_argument('--native-pkl')
    ap.add_argument('--nms-jsonl')
    ap.add_argument('--out-pkl')
    ap.add_argument('--mode', default='confirm_aware', choices=['none', 'explore_only', 'confirm_aware'])
    ap.add_argument('--budget', type=int, default=3)
    ap.add_argument('--diag-out')
    args = ap.parse_args()
    wmodel, adapter = load_models()
    process_scene(args.scene, args.native_pkl, args.nms_jsonl, args.out_pkl,
                  wmodel, adapter, mode=args.mode, budget=args.budget,
                  diag_out=args.diag_out)
