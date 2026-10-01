import json
import pickle

import numpy as np
import pytest

from tools import integrated_online


def _dual_rows():
    corners_a = np.arange(24, dtype=np.float64).reshape(8, 3)
    corners_b = corners_a + 100.0
    return [
        (1, corners_a, 0.8, 0.24),
        (0, corners_b, 0.1, 0.1),
    ]


def _state_rows():
    return [
        {
            'row_index': 0,
            'source': 'native',
            'persistent_score': 0.8,
            'current_score': 0.24,
            'current_state': 'retired',
            'm5_retired': True,
            'm5_reasons': ['support_lost'],
        },
        {
            'row_index': 1,
            'source': 'birth',
            'persistent_score': 0.1,
            'current_score': 0.1,
            'current_state': 'active',
            'm5_retired': False,
            'm5_reasons': [],
        },
    ]


def test_static_default_keeps_legacy_rows_and_persistent_scores(tmp_path):
    untouched = [(3, np.zeros((8, 3)), 0.42)]
    payload = [[], untouched]
    output = tmp_path / 'scene_boxes.pkl'

    integrated_online._write_dual_score_output(
        payload,
        _dual_rows(),
        _state_rows(),
        str(output),
        scene='scene_test',
    )

    with output.open('rb') as handle:
        written = pickle.load(handle)
    assert all(len(row) == 3 for row in written[0])
    assert [row[2] for row in written[0]] == pytest.approx([0.8, 0.1])
    assert written[1][0][2] == pytest.approx(0.42)

    sidecar = json.loads(
        (tmp_path / 'scene_boxes.pkl.dual_state.json').read_text(encoding='utf-8')
    )
    assert sidecar['schema'] == integrated_online.DUAL_STATE_SCHEMA
    assert sidecar['scene_id'] == 'scene_test'
    assert sidecar['selected_score_view'] == 'persistent'
    assert sidecar['static_score_view'] == 'persistent'
    assert sidecar['dynamic_score_view'] == 'current'
    assert sidecar['m5_dataset_conditioned'] is False
    assert sidecar['rows'][0]['persistent_score'] == pytest.approx(0.8)
    assert sidecar['rows'][0]['current_score'] == pytest.approx(0.24)
    assert sidecar['rows'][0]['current_state'] == 'retired'


def test_dynamic_view_is_explicit_and_preserves_both_sidecar_scores(tmp_path):
    output = tmp_path / 'dynamic_boxes.pkl'
    integrated_online._write_dual_score_output(
        [[]],
        _dual_rows(),
        _state_rows(),
        str(output),
        scene='scene_test',
        score_view='current',
    )

    with output.open('rb') as handle:
        written = pickle.load(handle)
    assert [row[2] for row in written[0]] == pytest.approx([0.24, 0.1])
    sidecar = json.loads(
        (tmp_path / 'dynamic_boxes.pkl.dual_state.json').read_text(encoding='utf-8')
    )
    assert sidecar['selected_score_view'] == 'current'
    assert sidecar['rows'][0]['persistent_score'] == pytest.approx(0.8)
    assert sidecar['rows'][0]['current_score'] == pytest.approx(0.24)


def test_one_dual_state_materializes_both_evaluator_pickles(tmp_path):
    persistent = tmp_path / 'static' / 'scene_boxes.pkl'
    current = tmp_path / 'dynamic' / 'scene_boxes.pkl'

    written = integrated_online._write_requested_score_outputs(
        [[]],
        _dual_rows(),
        _state_rows(),
        str(persistent),
        scene='scene_test',
        score_view='persistent',
        persistent_out_pkl=str(persistent),
        current_out_pkl=str(current),
    )

    assert set(written) == {'primary', 'persistent', 'current'}
    assert written['primary'] == written['persistent']
    for output in (persistent, current):
        with output.open('rb') as handle:
            rows = pickle.load(handle)[0]
        assert all(len(row) == 3 for row in rows)
    with persistent.open('rb') as handle:
        persistent_rows = pickle.load(handle)[0]
    with current.open('rb') as handle:
        current_rows = pickle.load(handle)[0]
    assert [row[2] for row in persistent_rows] == pytest.approx([0.8, 0.1])
    assert [row[2] for row in current_rows] == pytest.approx([0.24, 0.1])

    persistent_state = json.loads(
        (tmp_path / 'static' / 'scene_boxes.pkl.dual_state.json').read_text(
            encoding='utf-8')
    )
    current_state = json.loads(
        (tmp_path / 'dynamic' / 'scene_boxes.pkl.dual_state.json').read_text(
            encoding='utf-8')
    )
    assert persistent_state['selected_score_view'] == 'persistent'
    assert current_state['selected_score_view'] == 'current'
    assert persistent_state['rows'] == current_state['rows']


def test_dual_output_rejects_invalid_view_or_misaligned_rows(tmp_path):
    with pytest.raises(ValueError, match='score_view'):
        integrated_online._normalise_score_view('dataset_specific')
    with pytest.raises(ValueError, match='identical length'):
        integrated_online._write_dual_score_output(
            [[]], _dual_rows(), _state_rows()[:1], str(tmp_path / 'bad.pkl')
        )
    shared = tmp_path / 'shared.pkl'
    with pytest.raises(ValueError, match='both'):
        integrated_online._write_requested_score_outputs(
            [[]],
            _dual_rows(),
            _state_rows(),
            str(shared),
            score_view='persistent',
            current_out_pkl=str(shared),
        )


def test_m1_only_output_keeps_pre_m2_scores_and_payload_tail(tmp_path):
    corners_a = np.arange(24, dtype=np.float64).reshape(8, 3)
    corners_b = corners_a + 100.0
    payload_tail = [(7, corners_b + 100.0, 0.37)]
    payload = [[], payload_tail]
    all_rows = [
        (1, corners_a, 0.61, True),
        (0, corners_b, 0.05, False),
    ]
    output = tmp_path / 'm1' / 'scene_boxes.pkl'

    integrated_online._write_m1_only_output(payload, all_rows, str(output))

    with output.open('rb') as handle:
        written = pickle.load(handle)
    assert all(len(row) == 3 for row in written[0])
    assert [row[2] for row in written[0]] == pytest.approx([0.61, 0.05])
    assert written[0][0][0] == 1
    assert written[0][1][0] == 0
    np.testing.assert_array_equal(written[0][0][1], corners_a)
    assert written[1][0][2] == pytest.approx(0.37)
