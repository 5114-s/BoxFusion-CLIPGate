"""Union-pool causal confirmation with cross-view embedding consistency.

Replicates the cap12 union configuration exactly, plus:
  - gate mode: reject receipts whose embedded observations have mean pairwise
    cosine < RECYCLE_EMB_GATE (only when >=2 embedded observations exist)
  - disc mode: score = 0.10 + 0.45 * ramp(mean_cos) for embedded receipts
Union pool rows [0..Nw) = WeDetect lifted (have embeddings), [Nw..) = children.
"""
import os, sys, glob, pickle
import numpy as np

sys.path.insert(0, '/data/ZhaoX/BoxFusion')
UNION = '/data/ZhaoX/BoxFusion/results/union_recycle_cache'
EMB = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_emb'
WDCACHE = '/data/ZhaoX/BoxFusion/results/wedetect_lifted_cache'
CBEST = '/data/ZhaoX/BoxFusion/results/v2_rescore_prefix_only'
OUTBASE = '/data/ZhaoX/BoxFusion/results'

MODE = os.environ.get('EMB_MODE', 'gate')            # gate | disc | none
GATE = float(os.environ.get('EMB_GATE', '0.5'))
CAP = int(os.environ.get('RECYCLE_CAP', '12'))
TAG = os.environ.get('RECYCLE_TAG', f'emb_{MODE}{int(GATE*100)}')

def aabb_iou(c1, c2):
    lo1, hi1 = c1.min(0), c1.max(0)
    lo2, hi2 = c2.min(0), c2.max(0)
    ov = np.maximum(0, np.minimum(hi1, hi2) - np.maximum(lo1, lo2))
    inter = ov[0]*ov[1]*ov[2]
    v1 = np.prod(hi1-lo1); v2 = np.prod(hi2-lo2)
    return inter/max(v1+v2-inter, 1e-9)

def run(scene_files):
    out_dir = f'{OUTBASE}/{TAG}_birth_s010_cap6'
    os.makedirs(out_dir, exist_ok=True)
    total = kept_total = 0
    stats = dict(receipts=0, confirmed=0, gated_out=0, raised=0)
    for sf in scene_files:
        scene = os.path.basename(sf).replace('.npz', '')
        z = np.load(sf)
        corners, scores, fids = z['corners_raw'], z['scores'], z['frame_ids']
        emb = None
        zw = np.load(f'{WDCACHE}/{scene}.npz')
        n_wd = len(zw['corners_raw'])
        if os.path.exists(f'{EMB}/{scene}.npz'):
            ze = np.load(f'{EMB}/{scene}.npz')
            if len(ze['emb']) == n_wd:
                emb = ze['emb'].astype(np.float32)
        d = pickle.load(open(f'{CBEST}/{scene}_boxes.pkl', 'rb'))
        native = [np.asarray(det[1], float) for sc in d for det in sc]
        new_rows = []
        if len(corners):
            order = np.argsort(fids, kind='stable')
            uniq, ordinal_of = np.unique(fids, return_inverse=True)
            receipts = []
            for i in order:
                c, s, f = corners[i], float(scores[i]), int(ordinal_of[i])
                e = emb[i] if (emb is not None and i < n_wd) else None
                if any(aabb_iou(c, nb) >= 0.10 for nb in native):
                    continue
                cc = c.mean(0)
                best, best_iou = None, 0.0
                for r in receipts:
                    if f - r['last_ord'] > 10:
                        continue
                    rc = r['obs'][-1][0]
                    if np.linalg.norm(rc.mean(0) - cc) > 0.50:
                        continue
                    v = aabb_iou(c, rc)
                    if v >= 0.10 and v > best_iou:
                        best_iou, best = v, r
                if best is not None:
                    best['obs'].append((c, e)); best['frames'].add(f)
                    best['last_ord'] = f; best['scores'].append(s)
                else:
                    receipts.append(dict(obs=[(c, e)], frames={f}, last_ord=f, scores=[s]))
            stats['receipts'] += len(receipts)
            births = []
            for r in receipts:
                if len(r['frames']) < 3:
                    continue
                emb_obs = [e for _, e in r['obs'] if e is not None]
                mean_cos = None
                if len(emb_obs) >= 2:
                    E = np.stack(emb_obs)
                    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
                    sims = (E @ E.T)[np.triu_indices(len(E), 1)]
                    mean_cos = float(sims.mean())
                if MODE == 'gate' and mean_cos is not None and mean_cos < GATE:
                    stats['gated_out'] += 1
                    continue
                obs = [c for c, _ in r['obs']]
                best_j, best_s = 0, -1.0
                for j, a in enumerate(obs):
                    ssum = sum(aabb_iou(a, b) for k, b in enumerate(obs) if k != j)
                    if ssum > best_s:
                        best_s, best_j = ssum, j
                score_out = 0.10
                if MODE == 'disc' and mean_cos is not None:
                    score_out = 0.10 + 0.45 * max(0.0, min(1.0, (mean_cos - 0.5) / 0.5))
                    stats['raised'] += 1
                births.append((float(np.mean(r['scores'])), obs[best_j], score_out))
                stats['confirmed'] += 1
            births.sort(key=lambda x: -x[0])
            kept = []
            for strength, m, sc in births:
                if any(aabb_iou(m, km) >= 0.50 for _, km, _ in kept):
                    continue
                kept.append((strength, m, sc))
                if len(kept) >= CAP:
                    break
            new_rows = [(0, m, sc) for _, m, sc in kept]
        out_sc = [[(det[0], det[1], det[2]) for det in sc] for sc in d]
        out_sc[0].extend(new_rows)
        kept_total += len(new_rows)
        pickle.dump(out_sc, open(f'{out_dir}/{scene}_boxes.pkl', 'wb'))
    print(f'{TAG}: appended {kept_total} | {stats}')
    return out_dir

if __name__ == '__main__':
    files = sorted(glob.glob(f'{UNION}/scene*.npz'))
    print(f'scenes: {len(files)}, mode={MODE}, gate={GATE}, cap={CAP}')
    run(files)
