from dataclasses import FrozenInstanceError, replace
import math

import numpy as np
import pytest

from boxfusion import puf_lite as puf_module
from boxfusion.puf_lite import (
    PUFCandidateInput,
    SCHEMA,
    compute_puf_lite,
)


def _candidate(proposal_id, track_id, intersection, proposal_count, **extra):
    return {
        "proposal_id": proposal_id,
        "track_id": track_id,
        "intersection": intersection,
        "proposal_voxel_count": proposal_count,
        **extra,
    }


def test_joint_normalization_formula_gate_margin_and_entropy():
    result = compute_puf_lite(
        [
            _candidate(1, 10, 20, 40, track_voxel_count=30, centroid_distance=2.5),
            _candidate(1, 11, 10, 40),
            _candidate(2, 20, 5, 20),
        ]
    )
    assert not result.diagnostics.fail_open
    assert result.schema == SCHEMA
    assert result.diagnostics.lambda_null == 0.4
    assert result.diagnostics.birth_enabled is False
    assert result.diagnostics.max_normalization_error <= 1e-12

    first, second = result.proposals
    # Proposal 1: L=(.5,.25), Z=1.15.
    assert first.beta_null == pytest.approx(0.4 / 1.15)
    assert [item.spatial_likelihood for item in first.candidates] == pytest.approx(
        [0.5, 0.25]
    )
    assert [item.beta for item in first.candidates] == pytest.approx(
        [0.5 / 1.15, 0.25 / 1.15]
    )
    assert first.argmax_track_id == 10
    assert first.argmax_beta == pytest.approx(0.5 / 1.15)
    assert first.paper_gate is True
    assert first.birth_enabled is False
    assert first.margin == pytest.approx((0.5 - 0.4) / 1.15)
    probabilities = [0.4 / 1.15, 0.5 / 1.15, 0.25 / 1.15]
    expected_entropy = -sum(p * math.log(p) for p in probabilities) / math.log(3)
    assert first.normalized_entropy == pytest.approx(expected_entropy)
    assert first.candidates[0].track_voxel_count == 30
    assert first.candidates[0].centroid_distance == 2.5
    assert first.normalization_error == pytest.approx(
        abs(math.fsum(probabilities) - 1.0)
    )

    # Proposal 2: the null state wins and the paper gate abstains; no birth is
    # produced even though this is the original paper's birth branch.
    assert second.beta_null == pytest.approx(0.4 / 0.65)
    assert second.argmax_track_id == 20
    assert second.paper_gate is False
    assert second.margin == pytest.approx((0.25 - 0.4) / 0.65)
    assert second.birth_enabled is False


def test_paper_gate_includes_exact_half_boundary():
    result = compute_puf_lite([_candidate(1, 9, 2, 5)])
    proposal = result.proposals[0]
    assert proposal.beta_null == pytest.approx(0.5)
    assert proposal.candidates[0].beta == pytest.approx(0.5)
    assert proposal.paper_gate is True
    assert proposal.margin == pytest.approx(0.0)


def test_all_candidates_are_jointly_normalized_and_order_is_deterministic():
    records = [
        _candidate(8, 90, 10, 20),
        _candidate(3, 30, 5, 20),
        _candidate(8, 70, 10, 20),
        _candidate(3, 20, 10, 20),
    ]
    forward = compute_puf_lite(records)
    reverse = compute_puf_lite(list(reversed(records)))
    assert forward == reverse
    assert [item.proposal_id for item in forward.proposals] == [3, 8]
    assert [item.track_id for item in forward.proposals[0].candidates] == [20, 30]
    assert [item.track_id for item in forward.proposals[1].candidates] == [70, 90]
    # Exact candidate tie is resolved by stable track ID, not input order.
    assert forward.proposals[1].argmax_track_id == 70


def test_frozen_input_dataclass_and_numpy_scalar_inputs_are_supported():
    result = compute_puf_lite(
        (
            PUFCandidateInput(
                proposal_id=np.int64(4),
                track_id=np.int32(7),
                intersection=np.int16(8),
                proposal_voxel_count=np.int64(16),
                track_voxel_count=np.int64(20),
                centroid_distance=np.float32(1.25),
            ),
        )
    )
    assert not result.diagnostics.fail_open
    assert result.proposals[0].candidates[0].spatial_likelihood == 0.5


def test_empty_sequence_is_valid_and_does_not_force_an_association():
    result = compute_puf_lite(())
    assert result.proposals == ()
    assert result.diagnostics.code == "ok"
    assert not result.diagnostics.fail_open
    assert result.diagnostics.proposal_count == 0


def test_explicit_universe_preserves_null_only_proposal_rows():
    result = compute_puf_lite(
        [_candidate(5, 9, 8, 16)],
        proposal_ids=[5, 3],
    )
    assert not result.diagnostics.fail_open
    assert [item.proposal_id for item in result.proposals] == [3, 5]
    null_only = result.proposals[0]
    assert null_only.beta_null == 1.0
    assert null_only.candidates == ()
    assert null_only.argmax_track_id is None
    assert null_only.argmax_beta == 0.0
    assert null_only.paper_gate is False
    assert null_only.margin is None
    assert null_only.normalized_entropy == 0.0
    assert null_only.normalization_error == 0.0
    assert null_only.birth_enabled is False
    assert result.diagnostics.proposal_count == 2


def test_explicit_empty_universe_is_valid_when_candidates_are_empty():
    result = compute_puf_lite([], proposal_ids=())
    assert result.proposals == ()
    assert not result.diagnostics.fail_open
    assert result.diagnostics.proposal_count == 0


@pytest.mark.parametrize(
    ("proposal_ids", "records", "code"),
    [
        (iter(()), [], "invalid_proposal_universe"),
        ([True], [], "invalid_proposal_id"),
        ([1, 1], [], "duplicate_proposal_id"),
        (list(range(65)), [], "proposal_cap"),
        ([1], [_candidate(2, 9, 8, 16)], "candidate_outside_proposal_universe"),
        ([], [_candidate(2, 9, 8, 16)], "candidate_outside_proposal_universe"),
    ],
)
def test_invalid_or_misaligned_explicit_universe_fails_open(
    proposal_ids, records, code
):
    result = compute_puf_lite(records, proposal_ids=proposal_ids)
    assert result.proposals == ()
    assert result.diagnostics.fail_open
    assert result.diagnostics.code == code


@pytest.mark.parametrize(
    ("records", "code"),
    [
        (iter(()), "invalid_candidate_sequence"),
        ([_candidate(True, 1, 1, 1)], "invalid_proposal_id"),
        ([_candidate(1, -1, 1, 1)], "invalid_track_id"),
        ([_candidate(1, 1, 0, 1)], "invalid_intersection"),
        ([_candidate(1, 1, 2, 1)], "invalid_intersection"),
        ([_candidate(1, 1, 1, 0)], "invalid_proposal_voxel_count"),
        ([_candidate(1, 1, 2, 2, track_voxel_count=1)], "intersection_exceeds_track"),
        ([_candidate(1, 1, 1, 1, centroid_distance=float("nan"))], "invalid_centroid_distance"),
        ([_candidate(1, 1, 1, 1, centroid_distance=-0.1)], "invalid_centroid_distance"),
        ([{"proposal_id": 1}], "missing_track_id"),
    ],
)
def test_malformed_input_fails_open_without_any_association(records, code):
    result = compute_puf_lite(records)
    assert result.proposals == ()
    assert result.diagnostics.fail_open
    assert result.diagnostics.code == code
    assert result.diagnostics.birth_enabled is False


def test_duplicate_and_inconsistent_proposal_records_fail_open():
    duplicate = compute_puf_lite(
        [_candidate(1, 2, 4, 10), _candidate(1, 2, 5, 10)]
    )
    assert duplicate.proposals == ()
    assert duplicate.diagnostics.code == "duplicate_candidate_pair"

    inconsistent = compute_puf_lite(
        [_candidate(1, 2, 4, 10), _candidate(1, 3, 4, 11)]
    )
    assert inconsistent.proposals == ()
    assert inconsistent.diagnostics.code == "inconsistent_proposal_voxel_count"


def test_hard_caps_fail_open_before_a_counterfactual_can_escape():
    per_proposal = compute_puf_lite(
        [_candidate(1, track_id, 1, 16) for track_id in range(9)]
    )
    assert per_proposal.proposals == ()
    assert per_proposal.diagnostics.code == "candidate_per_proposal_cap"

    proposal_cap = compute_puf_lite(
        [_candidate(proposal_id, 1, 1, 16) for proposal_id in range(65)]
    )
    assert proposal_cap.proposals == ()
    assert proposal_cap.diagnostics.code == "proposal_cap"

    total_cap = compute_puf_lite(
        [_candidate(index // 8, index % 8, 1, 16) for index in range(513)]
    )
    assert total_cap.proposals == ()
    assert total_cap.diagnostics.code == "total_candidate_cap"


def test_results_are_read_only_and_do_not_alias_mapping_inputs():
    record = _candidate(1, 2, 8, 16)
    result = compute_puf_lite([record])
    record["intersection"] = 1
    assert result.proposals[0].candidates[0].intersection == 8
    with pytest.raises(FrozenInstanceError):
        result.proposals[0].beta_null = 0.0
    with pytest.raises(TypeError):
        result.proposals[0].candidates[0] = None


def test_unexpected_record_failure_is_defensively_fail_open():
    class Hostile:
        @property
        def proposal_id(self):
            raise RuntimeError("do not escape")

    result = compute_puf_lite([Hostile()])
    assert result.proposals == ()
    assert result.diagnostics.fail_open
    assert result.diagnostics.code == "exception:RuntimeError"


def test_normalization_error_above_frozen_tolerance_fails_open(monkeypatch):
    original = puf_module._score_proposal

    def _bad_score(proposal_id, candidates):
        return replace(
            original(proposal_id, candidates),
            normalization_error=1.000001e-12,
        )

    monkeypatch.setattr(puf_module, "_score_proposal", _bad_score)
    result = compute_puf_lite([_candidate(1, 2, 8, 16)])
    assert result.proposals == ()
    assert result.diagnostics.fail_open is True
    assert result.diagnostics.code == "normalization_error"
    assert result.diagnostics.max_normalization_error == pytest.approx(
        1.000001e-12
    )


def test_normalization_error_exactly_at_frozen_tolerance_is_allowed(monkeypatch):
    original = puf_module._score_proposal

    def _boundary_score(proposal_id, candidates):
        return replace(
            original(proposal_id, candidates),
            normalization_error=1e-12,
        )

    monkeypatch.setattr(puf_module, "_score_proposal", _boundary_score)
    result = compute_puf_lite([_candidate(1, 2, 8, 16)])
    assert result.diagnostics.fail_open is False
    assert len(result.proposals) == 1
    assert result.diagnostics.max_normalization_error == 1e-12
