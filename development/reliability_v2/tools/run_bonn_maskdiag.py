#!/usr/bin/env python3
"""Independent causal replay of single-detection gaps; cached data only."""
from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_bonn_fourarm import OUT, error_at, input_hashes, load_inputs, replay, summarize_errors

ARM_NAMES = {'A': 'A_latest', 'HOLD': 'HOLD', 'C': 'C_cvel', 'D': 'D_maware'}


def gap_positions(kfs, cands, refs, min_history=3):
    """Select without looking at error: three past observations + recovery."""
    return [(t, kfs[i+1]) for i, t in enumerate(kfs[:-1])
            if t in refs and t in cands and kfs[i+1] in cands
            and sum(k in cands for k in kfs[:i]) >= min_history]


def run_cases(kfs, ts, cands, refs, sanity):
    baseline = replay(kfs, ts, cands)
    rows = []
    for masked, following in gap_positions(kfs, cands, refs):
        outputs = replay(kfs, ts, cands, masked_frames={masked})
        last = max(k for k in kfs if k < masked and k in cands)
        recovery_ref = refs.get(following)
        row = {'mask': masked, 'recovery_frame': following,
               'observation_age_at_gap_s': ts[masked] - ts[last],
               'time_between_received_observations_s': ts[following] - ts[last],
               'size_filter_pass': bool(sanity[masked]['person_sane']),
               'recovery_reference_available': recovery_ref is not None,
               'recovery_reference_size_filter_pass': bool(sanity.get(following, {}).get('person_sane')),
               'gap_err': {}, 'gap_output_present': {}, 'recovery_next': {},
               'recovery_change_vs_unmasked_m': {}}
        for alias, arm in ARM_NAMES.items():
            row['gap_output_present'][alias] = masked in outputs[arm]
            row['gap_err'][alias] = error_at(outputs[arm], masked, refs[masked])
            new = error_at(outputs[arm], following, recovery_ref)
            old = error_at(baseline[arm], following, recovery_ref)
            row['recovery_next'][alias] = new
            row['recovery_change_vs_unmasked_m'][alias] = new-old if old is not None and new is not None else None
        rows.append(row)
    return rows


def summarize_cases(rows, filter_recovery=False):
    arms = tuple(ARM_NAMES)
    recovery_rows = [r for r in rows if r['recovery_reference_available'] and
                     (not filter_recovery or r['recovery_reference_size_filter_pass'])]
    paired = [r for r in rows if r['gap_err']['C'] is not None and r['gap_err']['D'] is not None]
    wins = sum(r['gap_err']['D'] < r['gap_err']['C']-1e-9 for r in paired)
    ties = sum(abs(r['gap_err']['D']-r['gap_err']['C']) <= 1e-9 for r in paired)
    majority = bool(rows) and len(paired) == len(rows) and wins > len(rows)/2
    return {'n_positions': len(rows), 'positions': [r['mask'] for r in rows],
            'gap': summarize_errors([r['gap_err'] for r in rows], arms),
            'recovery': summarize_errors([r['recovery_next'] for r in recovery_rows], arms),
            'gap_output_coverage': {a: {'outputs': sum(r['gap_output_present'][a] for r in rows),
                                       'total': len(rows)} for a in arms},
            'recovery_reference_count': len(recovery_rows),
            'recovery_reference_missing_or_filtered': len(rows)-len(recovery_rows),
            'D_beats_C_positions': wins, 'D_ties_C_positions': ties,
            'D_loses_C_positions': len(paired)-wins-ties, 'paired_positions': len(paired),
            'stop_rule_verdict': ('INSUFFICIENT' if not rows else
                                 'CONTINUE_DIAGNOSTIC' if majority else 'STOP')}


def main():
    _, ts, cands, kfs, refs, sanity = load_inputs()
    rows = run_cases(kfs, ts, cands, refs, sanity)
    filtered = [r for r in rows if r['size_filter_pass']]
    hashes = input_hashes()
    hashes['tools/run_bonn_maskdiag.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result = {'summary': summarize_cases(filtered, filter_recovery=True),
              'summary_all_references': summarize_cases(rows), 'rows': rows,
              'input_sha256': hashes,
              'protocol': {
                  'metric': 'Euclidean 3D centre error, metres',
                  'minimum_actual_past_observations': 3,
                  'mask': 'Whole candidate record removed before any state update',
                  'replay': 'Independent causal replay from start for each case',
                  'recovery': 'Next exported keyframe with an original candidate',
                  'stop_rule': 'D beats C on more than half the positions, with full paired coverage',
                  'scope': 'Controlled candidate dropout, not real occlusion or sensor simulation',
                  'references': 'VLM-box/depth proxies; size filter is not a membership test',
                  'guard': 'No reference supplied to trackers; no parameter sweep or model forward'}}
    (OUT/'maskdiag_results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'filtered': result['summary'], 'all': result['summary_all_references']}, indent=1))


if __name__ == '__main__':
    main()
