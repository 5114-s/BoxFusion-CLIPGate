"""Build simulated object-removal benchmark (dynamic ScanNet-Dyn).

For N scenes: pick GT objects well-visible before T (40% of frames); for every
frame >= T, inpaint the object's projected 2D region in color AND depth (sensor-
level removal). Unmodified files are symlinked. Writes manifest.json per scene.
"""
import os, sys, glob, json, shutil
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, '/data/ZhaoX/BoxFusion/evaluation')
os.chdir('/data/ZhaoX/BoxFusion/evaluation')
SCANS = '/extra/ZhaoX/scannet_data/scans'
SRC_ROOT = '/data/ZhaoX/BoxFusion/upstream_clean/scannet_readme_frames'
DST_ROOT = '/data/ZhaoX/BoxFusion/data_dyn'
GT_ROOT = '/data/ZhaoX/BoxFusion/data_util/scannet_train_detection_data'
os.makedirs(DST_ROOT, exist_ok=True)
N_SCENES = 30

def project_bbox(corners_w, pose, K, W, H):
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners_w.T).T + Rt[:3, 3]
    if (c[:, 2] < 0.1).any():
        return None
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1, x2, y2 = uv[:,0].min(), uv[:,1].min(), uv[:,0].max(), uv[:,1].max()
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H:
        return None
    return (int(max(0, x1)), int(max(0, y1)), int(min(W, x2)), int(min(H, y2)))

manifest_all = {}
scenes = sorted(os.path.basename(p).replace('_boxes.pkl', '')
                for p in glob.glob('/data/ZhaoX/BoxFusion/results/scannet_t05_boxer_kfmap_score05/scene*_boxes.pkl'))[:N_SCENES]
for scene in scenes:
    src = f'{SRC_ROOT}/{scene}/frames'
    dst = f'{DST_ROOT}/{scene}/frames'
    if os.path.exists(dst):
        continue
    # GT boxes (world, same frame as poses): reuse evaluation GT via corners in scans GT... use axisAligned GT in world of pipeline:
    # pipeline world = raw scan world (poses as-is); GT txt boxes are in axis-aligned frame; invert.
    T = None
    for line in open(f'{SCANS}/{scene}/{scene}.txt'):
        if 'axisAlignment' in line:
            T = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape(4, 4)
    if T is None:
        continue
    Tinv = np.linalg.inv(T)
    from utils.ap_helper import parse_groundtruths
    from data_util.dataset import ScannetDetectionDataset
    from data_util.model_util_scannet import ScannetDatasetConfig
    from torch.utils.data._utils.collate import default_collate
    if '_ds' not in globals():
        global _ds, _n2i, _CFG
        _ds = ScannetDetectionDataset(split_set='val', num_points=40000, use_color=False,
                                      use_height=True, augment=False,
                                      data_path='/data/ZhaoX/BoxFusion/evaluation/data_util/scannet_train_detection_data')
        _n2i = {n: i for i, n in enumerate(_ds.scan_names)}
        _CFG = {'remove_empty_box': True, 'use_3d_nms': True, 'nms_iou': 0.25,
                'use_old_type_nms': False, 'cls_nms': True, 'per_class_proposal': True,
                'conf_thresh': 0.5, 'dataset_config': ScannetDatasetConfig()}
    batch = default_collate([_ds[_n2i[scene]]])
    ep = {'model': 'scannet'}; ep.update(batch)
    gt_list = parse_groundtruths(ep, _CFG)[0]
    gt_corners = []
    for j in gt_list:
        cc = np.asarray(j[1], float)
        cw = (Tinv[:3, :3] @ cc.T).T + Tinv[:3, 3]
        gt_corners.append(cw)
    if not gt_corners:
        continue
    K = np.loadtxt(f'{src}/intrinsic/intrinsic_color.txt')[:3, :3]
    color_files = sorted(glob.glob(f'{src}/color/*.jpg'), key=lambda p: int(os.path.basename(p)[:-4]))
    n = len(color_files)
    T_frame = int(n * 0.4)
    # pick GT objects visible in >=8 keyframes before T
    visibility = {i: 0 for i in range(len(gt_corners))}
    for f in range(0, T_frame, 25):
        pf = f'{src}/pose/{f}.txt'
        if not os.path.exists(pf):
            continue
        pose = np.loadtxt(pf).reshape(4, 4)
        img = Image.open(f'{src}/color/{f}.jpg')
        W0, H0 = img.size
        for i, cc in enumerate(gt_corners):
            bb = project_bbox(cc, pose, K, W0, H0)
            area = (bb[2]-bb[0])*(bb[3]-bb[1]) if bb else 0
            if bb and area > 500:
                visibility[i] += 1
    removable = [i for i, v in visibility.items() if v >= 8][:2]
    if not removable:
        continue
    # build dst with symlinks
    for sub in ('color', 'depth', 'pose', 'intrinsic'):
        os.makedirs(f'{dst}/{sub}', exist_ok=True)
    for f_ in glob.glob(f'{src}/intrinsic/*'):
        os.symlink(f_, f'{dst}/intrinsic/{os.path.basename(f_)}')
    depth_dir = f'{src}/depth'
    Kd = np.loadtxt(f'{src}/intrinsic/intrinsic_depth.txt')[:3, :3]
    masked = 0
    for cf in color_files:
        f = int(os.path.basename(cf)[:-4])
        df = f'{src}/depth/{f}.png'
        pf = f'{src}/pose/{f}.txt'
        if not os.path.exists(df) or not os.path.exists(pf):
            continue
        os.symlink(pf, f'{dst}/pose/{f}.txt')
        if f < T_frame:
            os.symlink(cf, f'{dst}/color/{f}.jpg')
            os.symlink(df, f'{dst}/depth/{f}.png')
            continue
        pose = np.loadtxt(pf).reshape(4, 4)
        img = np.asarray(Image.open(cf).convert('RGB'))
        dep = np.asarray(Image.open(df))
        H0, W0 = img.shape[:2]
        mask = np.zeros((H0, W0), np.uint8)
        for i in removable:
            bb = project_bbox(gt_corners[i], pose, K, W0, H0)
            if bb is None:
                continue
            x1, y1, x2, y2 = bb
            mx = int((x2-x1)*0.15); my = int((y2-y1)*0.15)
            mask[max(0,y1-my):min(H0,y2+my), max(0,x1-mx):min(W0,x2+mx)] = 255
        if mask.any():
            img2 = cv2.inpaint(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), mask, 5, cv2.INPAINT_TELEA)
            dep2 = cv2.inpaint(dep, mask, 5, cv2.INPAINT_NS)
            Image.fromarray(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)).save(f'{dst}/color/{f}.jpg', quality=92)
            Image.fromarray(dep2).save(f'{dst}/depth/{f}.png')
            masked += 1
        else:
            os.symlink(cf, f'{dst}/color/{f}.jpg')
            os.symlink(df, f'{dst}/depth/{f}.png')
    manifest_all[scene] = dict(T_frame=T_frame, removed=[list(map(float, gt_corners[i].reshape(-1))) for i in removable],
                               removed_ids=removable, n_masked=masked)
    print(f'{scene}: T={T_frame}, removed {len(removable)} objs, masked {masked} frames', flush=True)

json.dump(manifest_all, open(f'{DST_ROOT}/manifest.json', 'w'))
print(f'DYN_BENCH_BUILT: {len(manifest_all)} scenes')
