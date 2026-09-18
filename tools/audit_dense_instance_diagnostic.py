#!/usr/bin/env python3
"""Audit dense-scene coverage; does not infer detector misses or merge history."""
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from diagnose_dense_instance_separation import (
    CAPTURE, FINAL, GT_ROOT, ROOT, aabb, iou_matrix_aabb,
    iou_smaller_matrix,
)

OUT = ROOT / 'reports/dense_instance_diag_20260914/audit'
THRESHOLDS = (.15, .25, .30, .50)


def pair_capacity(matches, pairs):
    """Maximum matching cardinality for each two-GT/all-row bipartite graph."""
    counts = matches.sum(axis=1)
    result = np.zeros(len(pairs), dtype=np.int8)
    for k, (i, j) in enumerate(pairs):
        if not counts[i] or not counts[j]:
            result[k] = int(bool(counts[i] or counts[j]))
        elif counts[i] == counts[j] == 1 and np.any(matches[i] & matches[j]):
            result[k] = 1
        else:
            result[k] = 2
    return result


def final_pair_categories(matches, pairs):
    capacity = pair_capacity(matches, pairs)
    covered = matches.any(axis=1)
    both = covered[pairs[:, 0]] & covered[pairs[:, 1]]
    out = dict(two_distinct_rows_possible=int((capacity == 2).sum()),
               single_shared_row_only=int((both & (capacity == 1)).sum()),
               at_least_one_gt_uncovered=int((~both).sum()))
    assert sum(out.values()) == len(pairs)
    return out


def fingerprint(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    # Meaningful edge cases: a shared row does not prevent two-row assignment;
    # a single shared row does, and an empty candidate pool covers neither GT.
    pair = np.array([[0, 1]])
    assert pair_capacity(np.array([[1, 1], [1, 0]], bool), pair).tolist() == [2]
    assert pair_capacity(np.ones((2, 1), bool), pair).tolist() == [1]
    assert pair_capacity(np.zeros((2, 0), bool), pair).tolist() == [0]
    protocol = json.loads((CAPTURE / 'protocol.json').read_text())
    scenes = protocol['scenes']
    assert len(scenes) == len(set(scenes)) == 107
    aggregates = {str(t): Counter() for t in THRESHOLDS}
    totals, old_reproduction = Counter(), Counter()
    rows, hashes = [], {}
    for scene in scenes:
        paths = [CAPTURE / f'{scene}.npz', GT_ROOT / scene / 'after_filter_boxes.npy',
                 FINAL / f'{scene}_boxes.pkl']
        for path in paths:
            hashes[str(path)] = fingerprint(path)
        with np.load(paths[0], allow_pickle=False) as data:
            cand = data['corners']
        gt = np.load(paths[1], allow_pickle=False)
        with paths[2].open('rb') as stream:
            final_rows = pickle.load(stream)[0]
        final = np.asarray([r[1] for r in final_rows]).reshape(-1, 8, 3)
        for boxes in (gt, cand, final):
            assert boxes.ndim == 3 and boxes.shape[1:] == (8, 3)
            assert np.isfinite(boxes).all()
        gc = iou_matrix_aabb(*aabb(gt), *aabb(cand))
        gf = iou_matrix_aabb(*aabb(gt), *aabb(final))
        dense = iou_smaller_matrix(*aabb(gt), *aabb(gt)) >= .25
        np.fill_diagonal(dense, False)
        dense_ids = np.flatnonzero(dense.any(axis=1))
        pairs = np.argwhere(np.triu(dense, k=1))
        selected_pairs = np.asarray(sorted({tuple(sorted((int(i), int(np.argmax(dense[i])))))
                                            for i in dense_ids}), dtype=int).reshape(-1, 2)
        local = dict(scene=scene, gt=len(gt), dense_gt=len(dense_ids),
                     all_dense_pairs=len(pairs), old_selected_pairs=len(selected_pairs),
                     candidate_rows=len(cand), final_rows=len(final))
        totals.update({k: v for k, v in local.items() if k != 'scene'})
        local['thresholds'] = {}
        for threshold in THRESHOLDS:
            cm, fm = gc >= threshold, gf >= threshold
            covered, final_covered = cm.any(axis=1), fm.any(axis=1)
            # Unique among ALL scene GT, not merely the first selected partner.
            unique_columns = cm.sum(axis=0) == 1
            exclusive = cm[:, unique_columns].any(axis=1)
            stats = dict(
                unique_gt_no_candidate=int((~covered[dense_ids]).sum()),
                unique_gt_shared_only=int((covered[dense_ids] & ~exclusive[dense_ids]).sum()),
                unique_gt_has_exclusive=int(exclusive[dense_ids].sum()),
                candidate_yes_final_yes=int((covered[dense_ids] & final_covered[dense_ids]).sum()),
                candidate_yes_final_no=int((covered[dense_ids] & ~final_covered[dense_ids]).sum()),
                candidate_no_final_yes=int((~covered[dense_ids] & final_covered[dense_ids]).sum()),
                candidate_no_final_no=int((~covered[dense_ids] & ~final_covered[dense_ids]).sum()),
            )
            assert sum(stats[k] for k in ('unique_gt_no_candidate', 'unique_gt_shared_only',
                                          'unique_gt_has_exclusive')) == len(dense_ids)
            assert sum(v for k, v in stats.items() if k.startswith('candidate_')) == len(dense_ids)
            stats.update({'final_pair_' + k: v for k, v in final_pair_categories(fm, pairs).items()})
            cp, fp = pair_capacity(cm, pairs), pair_capacity(fm, pairs)
            stats['candidate_two_rows_final_less_than_two'] = int(((cp == 2) & (fp < 2)).sum())
            aggregates[str(threshold)].update(stats)
            local['thresholds'][str(threshold)] = stats
        for i, j in selected_pairs:
            mi, mj = gc[i] >= .3, gc[j] >= .3
            for has, alone in ((mi.any(), (mi & ~mj).any()), (mj.any(), (mj & ~mi).any())):
                key = 'no_candidate' if not has else ('exclusive' if alone else 'shared_only')
                old_reproduction[key] += 1
            fi, fj = gf[i] >= .5, gf[j] >= .5
            if (fi & fj).any():
                old_reproduction['called_merged'] += 1
                if pair_capacity(np.stack((fi, fj)), pair)[0] == 2:
                    old_reproduction['called_merged_but_two_rows_available'] += 1
        rows.append(local)
    old = json.loads((OUT.parent / 'dense_diag_stats.json').read_text())['stats']
    for corrected, original in [('no_candidate', 'gt_no_candidate'),
                                ('exclusive', 'gt_cand_only_alone'),
                                ('shared_only', 'gt_cand_shared'), ('called_merged', 'pair_merged')]:
        assert old_reproduction[corrected] == old[original], (corrected, old_reproduction)
    for path, expected in hashes.items():
        assert fingerprint(Path(path)) == expected, path
    result = dict(
        scope='Frozen WeDetect+Boxer branch only; no native CuTR candidate stream or event lineage.',
        comparison='3D AABB coverage, not 2D proposal existence, causal attribution, or AP.',
        capture_protocol={k: protocol[k] for k in ('gap', 'score_min', 'topk_per_frame')},
        totals=dict(totals), thresholds={k: dict(v) for k, v in aggregates.items()},
        original_pair_role_counts=dict(old_reproduction), per_scene=rows,
        verification=dict(all_107_scenes=True, original_counts_reproduced=True,
                          empty_pools_retained=True, input_hashes_unchanged=True,
                          bipartite_matching_edge_cases_passed=True),
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'corrected_stats.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'input_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_scene'}, indent=2))


if __name__ == '__main__':
    main()
