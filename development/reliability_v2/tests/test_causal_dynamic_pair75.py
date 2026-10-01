from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from tools.eval_causal_dynamic_pair75 import covered_gt, coverage_delta
from tools.run_causal_dynamic_pair75 import ARMS, ROOT, arm_config


def test_configs_only_change_artifact_paths_and_explicit_ablation_switches():
    base = yaml.safe_load((ROOT / 'config/scannet_dyn_causal_dynamic_active.yaml').read_text())
    before = deepcopy(base)
    for arm in ARMS:
        cfg = arm_config(base, Path('/tmp/pair-test'), arm)
        assert cfg['causal_dynamic_branch']['mode'] == ('disabled' if arm == 'native_off' else 'active')
        assert cfg['dynamic_objects']['miss_lifecycle_updates'] == (arm != 'no_miss_retirement')
        cfg['data']['output_dir'] = base['data']['output_dir']
        cfg['lifting']['boxer']['diagnostics_dir'] = base['lifting']['boxer']['diagnostics_dir']
        for name in ('events_root', 'current_output_root', 'mode'):
            cfg['causal_dynamic_branch'][name] = base['causal_dynamic_branch'][name]
        cfg['dynamic_objects']['enabled'] = base['dynamic_objects']['enabled']
        del cfg['dynamic_objects']['miss_lifecycle_updates']
        assert cfg == base
    assert base == before


def test_stale_coverage_uses_strict_fixed_score_and_iou_and_handles_empty_predictions():
    box = np.asarray([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)])[None]
    assert covered_gt(box, [.31], box).tolist() == [True]
    assert covered_gt(box, [.30], box).tolist() == [False]
    assert covered_gt(np.empty((0, 8, 3)), [], box).tolist() == [False]
    assert covered_gt(box, [.9], box + 10).tolist() == [False]
    exact_quarter = box.copy().astype(float)
    exact_quarter[:, :, 0] *= .25
    assert covered_gt(exact_quarter, [.9], box).tolist() == [False]


def test_coverage_delta_does_not_confuse_net_reduction_with_removal_count():
    result = coverage_delta(np.asarray([True, True, False]), np.asarray([True, False, True]))
    assert result == {'reference_covered': 2, 'variant_covered': 2, 'lost': 1, 'gained': 1}
