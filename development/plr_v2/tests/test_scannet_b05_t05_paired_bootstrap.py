from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "scannet_b05_t05_paired_bootstrap.py"
SPEC = importlib.util.spec_from_file_location("b05_t05_bootstrap", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_identical_arms_have_exact_zero_observed_and_bootstrap_delta():
    scenes, ious, gt_counts = MODULE.synthetic_data(identical=True)
    result = MODULE.run_bootstrap(scenes, ious, gt_counts, 41, 123, 0)
    for metric in result["contrast_T05_minus_B05"].values():
        assert metric["observed_delta"] == 0.0
        assert metric["bootstrap_mean"] == 0.0
        assert metric["ci95_percentile"] == [0.0, 0.0]
        assert metric["positive_rate"] == 0.0


def test_positive_fixture_improves_at_all_thresholds():
    scenes, ious, gt_counts = MODULE.synthetic_data(identical=False)
    result = MODULE.run_bootstrap(scenes, ious, gt_counts, 101, 456, 0)
    for metric in result["contrast_T05_minus_B05"].values():
        assert metric["observed_delta"] > 0.0
        assert metric["positive_rate"] > 0.95


def test_strict_threshold_rejects_exact_equality():
    # At each corresponding threshold, exact equality must not be a TP.
    for threshold_index, threshold in enumerate(MODULE.THRESHOLDS):
        ious = [np.asarray([[threshold]], dtype=np.float64)]
        ap = MODULE.evaluate_sample(ious, [1], np.asarray([0], dtype=np.int64))
        assert ap[threshold_index] == 0.0


def test_duplicate_bootstrap_scene_copies_have_independent_gt_matching():
    # The same perfect detection in two sampled copies must match twice.
    ious = [np.asarray([[1.0]], dtype=np.float64)]
    sample = np.asarray([0, 0], dtype=np.int64)
    ap = MODULE.evaluate_sample(ious, [1], sample)
    expected = 2.0 / (2.0 + 1e-6)
    assert np.allclose(ap, expected, rtol=0.0, atol=1e-15)


def test_default_quicksort_tie_permutation_is_used_verbatim(monkeypatch):
    calls = []
    original = np.argsort

    def wrapped(values, *args, **kwargs):
        calls.append((values.copy(), args, dict(kwargs)))
        return original(values, *args, **kwargs)

    monkeypatch.setattr(MODULE.np, "argsort", wrapped)
    ious = [np.zeros((24, 1), dtype=np.float64)]
    MODULE.evaluate_sample(ious, [1], np.asarray([0], dtype=np.int64))
    assert len(calls) == 1
    values, args, kwargs = calls[0]
    assert np.array_equal(values, -np.ones(24, dtype=np.float64))
    assert args == ()
    assert kwargs == {}


def test_voc_ap_matches_evaluator_recurrence():
    tp = np.asarray([1.0, 0.0, 1.0, 0.0])
    fp = 1.0 - tp
    observed = MODULE.voc_ap(tp, fp, 3)
    # Hand-computed envelope: recall increments 1/3 at precision 1 and 2/3.
    expected = (1.0 / (3.0 + 1e-6)) + (1.0 / (3.0 + 1e-6)) * (2.0 / 3.0)
    assert np.isclose(observed, expected, rtol=0.0, atol=1e-15)


def test_prediction_set_audit_rejects_missing_and_extra(tmp_path):
    scenes = ["scene0000_00", "scene0001_00"]
    payload = [[]]
    with (tmp_path / "scene0000_00_boxes.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    with (tmp_path / "scene9999_00_boxes.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    try:
        MODULE.audit_prediction_set(scenes, "B05", tmp_path)
    except ValueError as error:
        message = str(error)
        assert "scene0001_00_boxes.pkl" in message
        assert "scene9999_00_boxes.pkl" in message
    else:
        raise AssertionError("prediction-set mismatch was accepted")


def test_queue_baseline_treatment_aliases(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--baseline",
            "/tmp/b05_alias_test",
            "--treatment",
            "/tmp/t05_alias_test",
        ],
    )
    args = MODULE.parse_args()
    assert args.b05 == Path("/tmp/b05_alias_test")
    assert args.t05 == Path("/tmp/t05_alias_test")
