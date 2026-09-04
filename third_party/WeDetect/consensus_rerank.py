"""Module-2 v1: asymmetric cross-detector consensus re-ranking of NATIVE boxes.

support(native) = max over keyframes of max 2D-IoU with cached WeDetect proposals.
score' = score + alpha * max(0, support - tau)   (boost-only, cap 0.99)
Applied on top of the dedup25_size configuration (native rows only).
"""
import os, sys, glob, pickle
import numpy as np

SCANS = '/extra/ZhaoX/scannet_data/scans'
WDCACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
CBEST = '/data/ZhaoX/BoxFusion/results/v2_rescore_prefix_only'
FINAL = '/data/ZhaoX/BoxFusion/results/dedup25_size'
OUTBASE = '/data/ZhaoX/BoxFusion/results'

def project_xyxy(corners_w, pose, K, W, H):
    Rt = np.linalg.inv(pose)
    c = (Rt[:3, :3] @ corners_w.T).T + Rt[:3, 3]
    if (c[:, 2] < 0.1).any():
        return None
    uv = (K @ c.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    x1, y1, x2, y2 = uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max()
    if x2 <= 0 or y2 <= 0 or x1 >= W or y1 >= H or (x2-x1) < 2 or (y2-y1) < 2:
        return None
    return np.array([max(0, x1), max(0, y1), min(W, x2), min(H, y2)])

def iou2d_vec(box, props):
    x1 = np.maximum(box[0], props[:, 0]); y1 = np.maximum(box[1], props[:, 1])
    x2 = np.minimum(box[2], props[:, 2]); y2 = np.minimum(box[3], props[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    ua = (box[2]-box[0])*(box[3]-box[1]) + (props[:,2]-props[:,0])*(props[:,3]-props[:,1]) - inter
    return inter / np.maximum(ua, 1e-9)

MODE = 'add'

def run(alpha, tau, tag, mode='add'):
    global MODE
    MODE = mode
    out_dir = f'{OUTBASE}/consensus_{tag}'
    os.makedirs(out_dir, exist_ok=True)
    n_boost = 0
    for f in sorted(glob.glob(f'{FINAL}/scene*_boxes.pkl')):
        scene = os.path.basename(f).replace('_boxes.pkl', '')
        d = pickle.load(open(f, 'rb'))
        dref = pickle.load(open(f'{CBEST}/{scene}_boxes.pkl', 'rb'))
        n_native0 = len(dref[0])
        zw = np.load(f'{WDCACHE}/{scene}.npz')
        b2d, fids = zw['boxes2d'], zw['frame_ids']
        by_frame = {}
        for b, fr in zip(b2d, fids):
            by_frame.setdefault(int(fr), []).append(b)
        by_frame = {k: np.array(v) for k, v in by_frame.items()}
        K = np.loadtxt(f'{SCANS}/{scene}/intrinsic/intrinsic_color.txt')[:3, :3]
        W, H = 1296, 968
        pose_cache = {}
        rows = list(d[0])
        for i in range(n_native0):
            corners = np.asarray(rows[i][1], float)
            best = 0.0
            for fr, props in by_frame.items():
                if fr not in pose_cache:
                    p = np.loadtxt(f'{SCANS}/{scene}/pose/{fr}.txt').reshape(4, 4)
                    pose_cache[fr] = p if np.isfinite(p).all() else None
                pose = pose_cache[fr]
                if pose is None:
                    continue
                bb = project_xyxy(corners, pose, K, W, H)
                if bb is None:
                    continue
                best = max(best, float(iou2d_vec(bb, props).max()))
            s = float(rows[i][2])
            if MODE == 'pure':
                ns = 0.5 + 0.5 * best if best > tau else s
            elif MODE == 'mix':
                ns = 0.5 * best + 0.5 * s if best > tau else s
            else:
                ns = min(0.99, s + alpha * max(0.0, best - tau))
            if ns != s:
                rows[i] = (rows[i][0], rows[i][1], ns)
                n_boost += 1
        out_sc = [rows] + [[(det[0], det[1], det[2]) for det in sc] for sc in d[1:]]
        pickle.dump(out_sc, open(f'{out_dir}/{scene}_boxes.pkl', 'wb'))
    print(f'consensus_{tag}: boosted {n_boost} native boxes')

if __name__ == '__main__':
    run(0.8, 0.5, 'a80t50')
    run(0.0, 0.5, 'pure', mode='pure')
    run(0.0, 0.5, 'mix', mode='mix')
