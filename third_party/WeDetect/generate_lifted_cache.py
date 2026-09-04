"""Generate sealed per-view WeDetect-Uni + BoxerNet lifted proposals for all 100 scenes.

Output per scene: /data/ZhaoX/BoxFusion/results/wedetect_lifted_cache/<scene>.npz
  corners_raw (N,8,3)  raw-world OBB corners (pipeline frame)
  scores     (N,)      WeDetect objectness score
  frame_ids  (N,)      keyframe id
  boxes2d    (N,4)     xyxy in native color coords
Clean process: no evaluation/ imports (Boxer owns top-level 'utils').
"""
import os, sys, glob
import numpy as np
import torch

sys.path.insert(0, '/data/ZhaoX/BoxFusion')
sys.path.insert(0, '/data/ZhaoX/BoxFusion/third_party/WeDetect')

SCANS = '/extra/ZhaoX/scannet_data/scans'
OUT = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
GAP = 25
SCORE_LIFT = 0.05
TOPK_PER_FRAME = 150

def obb_to_corners(center, extents, R):
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return center[None] + signs * (0.5 * extents) @ R.T

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
    "diagnostics_dir": "/tmp/wedetect_lifted_diag"}}}
adapter = build_lifting_adapter(cfg, device="cuda", code_root="/data/ZhaoX/BoxFusion")

from PIL import Image
os.makedirs(OUT, exist_ok=True)

scene_list = [l.strip() for l in open('/data/ZhaoX/BoxFusion/evaluation/data_util/meta_data/scannetv2_val.txt')
              if l.strip() and not l.strip().startswith('#')]
print(f'scenes: {len(scene_list)}', flush=True)

for si, scene in enumerate(scene_list):
    out_path = f'{OUT}/{scene}.npz'
    if os.path.exists(out_path):
        print(f'[{si+1}] {scene} cached, skip', flush=True)
        continue
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
    corners_all, scores_all, fids_all, b2d_all = [], [], [], []
    for f, pose, cf, df in kfs:
        try:
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
                corners_all.append(obb_to_corners(centers[i], np.abs(extents[i]) + 1e-6, rots[i]))
                scores_all.append(float(ps[i]))
                fids_all.append(int(f))
                b2d_all.append(pb[i])
        except Exception as e:
            print(f'  {scene} f{f} ERROR: {e}', flush=True)
            continue
    np.savez_compressed(out_path,
                        corners_raw=np.array(corners_all, dtype=np.float32) if corners_all else np.zeros((0, 8, 3), np.float32),
                        scores=np.array(scores_all, dtype=np.float32),
                        frame_ids=np.array(fids_all, dtype=np.int64),
                        boxes2d=np.array(b2d_all, dtype=np.float32))
    print(f'[{si+1}/{len(scene_list)}] {scene}: {len(corners_all)} lifted, {len(kfs)} kfs', flush=True)

print('ALL_DONE', flush=True)
