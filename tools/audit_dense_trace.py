#!/usr/bin/env python3
"""Check endpoint comparisons; never label them as causal track histories."""
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from diagnose_dense_instance_separation import CAPTURE, GT_ROOT, ROOT, aabb, iou_matrix_aabb, iou_smaller_matrix

BASE = ROOT / 'reports/dense_instance_diag_20260914'
M1 = ROOT / 'results/ca1m_thr15_m1_full107'
M2 = ROOT / 'results/ca1m_thr15_m1_m2nl_full107'


def load_rows(path):
    with path.open('rb') as stream:
        rows = pickle.load(stream)[0]
    return np.asarray([r[1] for r in rows]).reshape(-1, 8, 3)


def classify(row):
    c, m, f = row['best_cand_iou'], row['best_m1_iou'], row['best_m2_iou']
    if m < .25:
        detail = ('one_qualifying_frame' if row['n_keyframes'] == 1 else
                  'multiple_frames_max_score_below_0.3' if row['cand_score_max'] < .3
                  else 'other_multiple_frames')
        return 'm1_below_0.25/' + detail
    if m >= .5 and f < .5:
        return 'm1_ge_0.5_final_below_0.5'
    if f >= .25:
        detail = ('lower_over_0.05' if f < c - .05 else
                  'lower_up_to_0.05' if f < c else 'equal' if f == c else 'higher')
        return 'final_ge_0.25/' + detail
    return 'other'


def main():
    original = json.loads((BASE / 'trace_lost_targets.json').read_text())
    expected = {(r['scene'], r['gt']) for r in original['lost']}
    scenes = json.loads((CAPTURE / 'protocol.json').read_text())['scenes']
    counts, subset, rows = Counter(), Counter(), []
    geom_equal, crossing = 0, []
    hashes = {}
    for scene in scenes:
        paths = [CAPTURE / f'{scene}.npz', GT_ROOT / scene / 'after_filter_boxes.npy',
                 M1 / f'{scene}_boxes.pkl', M2 / f'{scene}_boxes.pkl']
        for path in paths:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(paths[0], allow_pickle=False) as data:
            cand, frames, scores = data['corners'], data['frame_ids'], data['scores']
        gt = np.load(paths[1], allow_pickle=False)
        f1, f2 = load_rows(paths[2]), load_rows(paths[3])
        geom_equal += int(np.array_equal(f1, f2))
        g = aabb(gt)
        gc, gm, gf = [iou_matrix_aabb(*g, *aabb(x)) for x in (cand, f1, f2)]
        dense = iou_smaller_matrix(*g, *g) >= .25
        np.fill_diagonal(dense, False)
        for gi in np.flatnonzero(dense.any(axis=1)):
            c = float(gc[gi].max()) if len(cand) else 0.
            m = float(gm[gi].max()) if len(f1) else 0.
            f = float(gf[gi].max()) if len(f2) else 0.
            if c < .3 or f >= .5:
                continue
            ci, fi = int(gc[gi].argmax()), int(gf[gi].argmax()) if len(f2) else None
            qualifying = gc[gi] >= .3
            row = dict(scene=scene, gt=int(gi), best_cand_iou=c, best_m1_iou=m,
                       best_m2_iou=f, n_keyframes=len(np.unique(frames[qualifying])),
                       cand_score_max=float(scores[qualifying].max()),
                       best_candidate_index=ci, best_candidate_frame=int(frames[ci]),
                       best_final_row_index=fi, lineage_link_verified=False)
            row['coverage_group'] = classify(row)
            counts[row['coverage_group']] += 1
            if f < .3:
                subset[row['coverage_group']] += 1
            row['best_final_row_also_covers_dense_partner'] = bool(
                fi is not None and np.any(dense[gi] & (gf[:, fi] >= .3)))
            if row['coverage_group'] == 'final_ge_0.25/lower_over_0.05':
                counts['lower_over_0.05_with_partner_overlap'] += int(
                    row['best_final_row_also_covers_dense_partner'])
            rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()}
            if classify(rounded) != row['coverage_group']:
                crossing.append(dict(scene=scene, gt=int(gi),
                                     exact=row['coverage_group'], rounded=classify(rounded)))
            rows.append(row)
    assert {(r['scene'], r['gt']) for r in rows} == expected
    assert len(rows) == 3005 and sum(subset.values()) == 1490
    # Exact counterexample: the old 'improved' bucket also includes small losses.
    assert classify(dict(best_cand_iou=.4, best_m1_iou=.38, best_m2_iou=.38)) == 'final_ge_0.25/lower_up_to_0.05'
    result = dict(
        expanded_population=dict(candidate_min=.3, final_max_exclusive=.5, count=len(rows)),
        original_population=dict(candidate_min=.3, final_max_exclusive=.3, count=sum(subset.values()),
                                 endpoint_groups=dict(subset)),
        expanded_endpoint_groups=dict(counts),
        m1_m2_geometry_rowwise_exact_equal_scenes=geom_equal,
        rounding_changed_classification=crossing,
        verified_candidate_to_track_lineage_links=0,
        lineage_note='The inspected trace script reads endpoint arrays only; argmax indices are not ancestry.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        rows=rows,
    )
    out = BASE / 'audit'
    out.mkdir(exist_ok=True)
    (out / 'trace_review.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'trace_review_inputs.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
