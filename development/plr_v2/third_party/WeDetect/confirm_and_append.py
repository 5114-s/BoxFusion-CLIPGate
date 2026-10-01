"""Stage-2 offline birth: causal 3-view confirmation of WeDetect+Boxer lifted proposals.

Protocol (birth-v2 contract + Stream3Dv2-lite lessons):
  - past-only: proposals processed in frame order, receipts only ever hold earlier frames
  - association: AABB IoU >= 0.10 AND center distance <= 0.50 m, TTL 10 keyframes
  - confirmation: >=3 observations from >=3 distinct frames; medoid geometry
  - native dedup: proposal skipped if AABB IoU >= 0.10 with any Cbest native box
  - per-scene cap (default 6); appended at a low score from the proven band
Outputs variant prediction dirs ready for the sealed real-score evaluator.
"""
import os, sys, glob, pickle, argparse
import numpy as np

sys.path.insert(0, '/data/ZhaoX/BoxFusion')

CACHE = os.environ.get('RECYCLE_CACHE', '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache')
OUTTAG = os.environ.get('RECYCLE_TAG', 'wedetect')
CBEST = '/data/ZhaoX/BoxFusion/results/v2_rescore_prefix_only'   # Cbest 1,788, real scores
OUTBASE = '/data/ZhaoX/BoxFusion/results'

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0)
    lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = ov[0]*ov[1]*ov[2]
    v1 = np.prod(hi1-lo1); v2 = np.prod(hi2-lo2)
    return inter/max(v1+v2-inter, 1e-9)

def run_variant(score_assign, cap, tag, scene_files, native_score=False):
    out_dir = f'{OUTBASE}/{OUTTAG}_birth_{tag}'
    os.makedirs(out_dir, exist_ok=True)
    total_app = 0
    zw_root = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
    for sf in scene_files:
        scene = os.path.basename(sf).replace('.npz', '')
        z = np.load(sf)
        _n_wd = len(np.load(f'{zw_root}/{scene}.npz')['corners_raw'])
        corners, scores, fids = z['corners_raw'], z['scores'], z['frame_ids']
        d = pickle.load(open(f'{CBEST}/{scene}_boxes.pkl', 'rb'))
        native = [np.asarray(det[1], float) for sc in d for det in sc]
        new_rows_per_scene = []

        if len(corners):
            order = np.argsort(fids, kind='stable')
            uniq, ordinal_of = np.unique(fids, return_inverse=True)
            receipts = []   # each: dict(frames=set, obs=[corners...], last_ord, scores=[])
            for i in order:
                c, s, f = corners[i], float(scores[i]), int(ordinal_of[i])
                cc = c.mean(0)
                # native dedup
                if any(aabb_iou(c, nb) >= float(os.environ.get('RECYCLE_DEDUP', '0.10')) for nb in native):
                    continue
                # associate to open receipts (past-only by construction)
                best, best_iou = None, 0.0
                for r in receipts:
                    if f - r['last_ord'] > 10:
                        continue
                    rc = r['obs'][-1]
                    if np.linalg.norm(rc.mean(0) - cc) > 0.50:
                        continue
                    v = aabb_iou(c, rc)
                    if v >= 0.10 and v > best_iou:
                        best_iou, best = v, r
                src_wd = int(i < _n_wd)
                if best is not None:
                    best['obs'].append(c); best['frames'].add(f)
                    best['last_ord'] = f; best['scores'].append(s); best['n_wd'] += src_wd
                else:
                    receipts.append(dict(obs=[c], frames={f}, last_ord=f, scores=[s], n_wd=src_wd))
            # confirm
            births = []
            for r in receipts:
                min_views = 3 if 2 * r['n_wd'] > len(r['obs']) else 2
                if len(r['frames']) < min_views:
                    continue
                obs = r['obs']
                # medoid: max pairwise-IoU sum
                best_j, best_s = 0, -1.0
                for j, a in enumerate(obs):
                    ssum = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                    if ssum > best_s:
                        best_s, best_j = ssum, j
                medoid = obs[best_j]
                strength = float(np.mean(r['scores']))
                births.append((strength, medoid))
            births.sort(key=lambda x: -x[0])
            # self-NMS among births
            kept = []
            for strength, m in births:
                if any(aabb_iou(m, km) >= 0.50 for _, km in kept):
                    continue
                kept.append((strength, m))
                if len(kept) >= cap:
                    break
            if native_score:
                new_rows_per_scene = [(0, m, float(strength)) for strength, m in kept]
            else:
                new_rows_per_scene = [(0, m, score_assign) for _, m in kept]

        # write output: native rows byte-identical + appended suffix
        out_sc = [[(det[0], det[1], det[2]) for det in sc] for sc in d]
        if new_rows_per_scene:
            out_sc[0].extend(new_rows_per_scene)
        total_app += len(new_rows_per_scene)
        pickle.dump(out_sc, open(f'{out_dir}/{scene}_boxes.pkl', 'wb'))
    print(f'variant {tag}: appended {total_app} boxes total -> {out_dir}')
    return out_dir

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset', type=int, default=0, help='0 = all scenes')
    args = ap.parse_args()
    scene_files = sorted(glob.glob(f'{CACHE}/scene*.npz'))
    if args.subset:
        scene_files = scene_files[:args.subset]
    print(f'scenes: {len(scene_files)}')
    import os as _os
    if _os.environ.get('RECYCLE_NATIVE') == '1':
        run_variant(0.0, 6, 'native_cap6', scene_files, native_score=True)
    else:
        for score in (0.1,):
            run_variant(score, int(__import__('os').environ.get('RECYCLE_CAP', '6')), f's{int(score*100):03d}_cap6', scene_files)
