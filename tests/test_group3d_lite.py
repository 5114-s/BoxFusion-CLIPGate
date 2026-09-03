import unittest

import numpy as np

from boxfusion import group3d_lite as fast
from boxfusion import group3d_lite_oracle as oracle
from boxfusion.group3d_lite import (
    MAX_CANDIDATES,
    MAX_PROPOSALS,
    MAX_TRACKS,
    MAX_UNION_VOXELS_PER_TRACK,
    MAX_VIEWS_PER_TRACK,
    MAX_VOXELS_PER_VIEW,
    extract_prepared_pair_evidence,
    match_prepared,
    match_prepared_proposals,
    match_voxels,
    prepare_proposals,
    prepare_track_snapshot,
    update_touched_prepared_tracks,
    update_prepared_track_snapshot,
)


def voxels(n, offset=(0, 0, 0)):
    """A deterministic compact grid with exactly n unique integer voxels."""
    base = np.stack((np.arange(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)), axis=1)
    return base + np.asarray(offset, dtype=np.int64)


def prop(identifier, points, score=1.0):
    return {"id": identifier, "score": score, "voxels": points}


def track(identifier, *views):
    return {"id": identifier, "views": list(views)}


class Group3DLiteTests(unittest.TestCase):
    def match(self, ps, ts, mask=None):
        return match_voxels(ps, ts, np.ones(len(ts), dtype=bool) if mask is None else mask)

    def test_exact_thresholds_accept(self):
        # Both fragments have >=16 voxels and I is exactly eight.
        proposal = np.vstack([voxels(8), voxels(8, (100, 0, 0))])
        result = self.match([prop(1, proposal)], [track(9, voxels(20))])
        self.assertEqual([(a.proposal_id, a.track_id) for a in result.associations], [(1, 9)])
        self.assertEqual(result.associations[0].intersection, 8)

    def test_signed_integer_nx3_and_negative_coordinates(self):
        negative = voxels(16, (-80, -7, -11)).astype(np.int16)
        result = self.match(
            [prop(1, negative)],
            [track(9, negative.astype(np.int32))],
        )
        self.assertEqual(
            [(association.proposal_id, association.track_id)
             for association in result.associations],
            [(1, 9)],
        )
        unsigned = negative.astype(np.uint64)
        rejected = self.match([prop(2, unsigned)], [track(9, negative)])
        self.assertEqual(
            rejected.diagnostics.fragment_errors,
            ((2, "non_integer_voxels_proposal"),),
        )
        wrong_shape = np.zeros((16, 4), dtype=np.int64)
        rejected = self.match([prop(3, wrong_shape)], [track(9, negative)])
        self.assertEqual(
            rejected.diagnostics.fragment_errors,
            ((3, "invalid_voxels_proposal"),),
        )

    def test_intersection_and_jaccard_boundaries(self):
        # I=7 abstains; I=8 with 16-vs-64 gives J=8/72 >= .10 but min C=.125 fails.
        p7 = np.vstack([voxels(7), voxels(9, (100, 0, 0))])
        self.assertFalse(self.match([prop(1, p7)], [track(2, np.vstack([voxels(7), voxels(9, (200, 0, 0))]))]).associations)
        p8 = np.vstack([voxels(8), voxels(8, (100, 0, 0))])
        self.assertFalse(self.match([prop(1, p8)], [track(2, voxels(64))]).associations)
        # exact J=.10 and containment thresholds: 20-vs-24 with I=4 would miss I;
        # 40-vs-48 with I=8 has J=.10 but min C=.1667 and max C=.20 < .40.
        p40 = np.vstack([voxels(8), voxels(32, (100, 0, 0))])
        self.assertFalse(self.match([prop(1, p40)], [track(2, np.vstack([voxels(8), voxels(40, (200, 0, 0))]))]).associations)

    def test_containment_boundaries(self):
        # I=12, |Vp|=80, |Vt|=30: Cp=.15 and Ct=.40 exactly.
        p80 = voxels(80)
        t30 = np.vstack([voxels(12), voxels(18, (1000, 0, 0))])
        self.assertTrue(self.match([prop(1, p80)], [track(2, t30)]).associations)
        # Just below min containment: 12 / 81 < .15.
        self.assertFalse(self.match([prop(1, voxels(81))], [track(2, t30)]).associations)
        # Just below max containment: 12 / 31 < .40 (the other is exactly .15).
        p31 = np.vstack([voxels(12), voxels(19, (2000, 0, 0))])
        t80 = np.vstack([voxels(12), voxels(68, (3000, 0, 0))])
        self.assertFalse(self.match([prop(1, p31)], [track(2, t80)]).associations)

    def test_aabb_expand_two(self):
        # A raw gap of four voxels intersects after both AABBs expand by two;
        # I=0 ultimately rejects it.
        result = self.match([prop(1, voxels(16))], [track(2, voxels(16, (19, 0, 0)))])
        self.assertEqual(result.diagnostics.aabb_pairs, 1)
        # A raw gap of five does not intersect after each AABB is expanded.
        result = self.match([prop(1, voxels(16))], [track(2, voxels(16, (20, 0, 0)))])
        self.assertEqual(result.diagnostics.aabb_pairs, 0)

    def test_mutual_best_conflict(self):
        p1, p2 = prop(1, voxels(16)), prop(2, voxels(16), .9)
        result = self.match([p1, p2], [track(7, voxels(16))])
        self.assertEqual([(a.proposal_id, a.track_id) for a in result.associations], [])
        # Equal best edges conflict and runner-up margin also prevents an arbitrary choice.

    def test_output_is_one_to_one(self):
        proposals = [prop(i, voxels(16, (i * 1000, -i, 0))) for i in range(3)]
        tracks = [track(100 + i, voxels(16, (i * 1000, -i, 0))) for i in range(3)]
        result = self.match(proposals, tracks)
        proposal_ids = [association.proposal_id for association in result.associations]
        track_ids = [association.track_id for association in result.associations]
        self.assertEqual(len(result.associations), 3)
        self.assertEqual(len(proposal_ids), len(set(proposal_ids)))
        self.assertEqual(len(track_ids), len(set(track_ids)))

    def test_runner_up_margin_both_sides(self):
        # Proposal has alternatives D=1 vs D=30/32=.9375: insufficient margin.
        result = self.match([prop(1, voxels(16))], [track(1, voxels(16)), track(2, voxels(16))])
        self.assertFalse(result.associations)
        # Track has alternatives with D=1 vs 30/32: mutual best rejected on track margin.
        result = self.match([prop(1, voxels(16)), prop(2, voxels(16))], [track(1, voxels(16))])
        self.assertFalse(result.associations)

    def test_runner_up_exact_margin_accepts(self):
        # D=1.0 vs D=.95: the frozen >= .05 boundary is inclusive.
        alternate = np.vstack([voxels(19), voxels(1, (100, 0, 0))])
        result = self.match([prop(1, voxels(20))], [track(1, voxels(20)), track(2, alternate)])
        self.assertEqual([(a.proposal_id, a.track_id) for a in result.associations], [(1, 1)])
        # The same inclusive boundary must also hold for the track-side runner-up.
        result = self.match([prop(1, voxels(20)), prop(2, alternate)], [track(1, voxels(20))])
        self.assertEqual([(a.proposal_id, a.track_id) for a in result.associations], [(1, 1)])

    def test_tie_and_order_invariance(self):
        ps = [prop(4, voxels(16), .5), prop(3, voxels(16, (100, 0, 0)), .5)]
        ts = [track(8, voxels(16, (100, 0, 0))), track(7, voxels(16))]
        a = self.match(ps, ts)
        b = self.match(list(reversed(ps)), list(reversed(ts)))
        self.assertEqual(a.associations, b.associations)
        self.assertEqual([(x.proposal_id, x.track_id) for x in a.associations], [(3, 8), (4, 7)])
        tied = self.match([prop(1, voxels(16))], [track(9, voxels(16)), track(8, voxels(16))])
        self.assertFalse(tied.associations)  # stable IDs rank the tie; margin prevents arbitrary match.

    def test_score_selection_is_fragment_independent(self):
        ps = [prop(i, voxels(16), score=float(100 - i)) for i in range(MAX_PROPOSALS + 1)]
        ps[-1]["voxels"] = np.zeros((0, 3), dtype=np.int64)  # low-score invalid, unselected
        result = self.match(ps, [track(1, voxels(16))])
        self.assertFalse(result.diagnostics.fail_open)
        self.assertNotIn(MAX_PROPOSALS, result.diagnostics.selected_proposal_ids)
        self.assertEqual(len(result.diagnostics.selected_proposal_ids), MAX_PROPOSALS)

    def test_past_only_is_mask_expressed(self):
        result = self.match([prop(1, voxels(16))], [track(2, voxels(16))], np.array([False]))
        self.assertFalse(result.associations)
        self.assertEqual(result.diagnostics.skipped_track_ids, (2,))

    def test_nan_overflow_and_caps_fail_open(self):
        nan_score = self.match([prop(1, voxels(16), float("nan"))], [track(2, voxels(16))])
        self.assertTrue(nan_score.diagnostics.fail_open)
        bad_voxels = self.match([prop(1, np.ones((16, 3), dtype=np.float64))], [track(2, voxels(16))])
        self.assertFalse(bad_voxels.associations)  # invalid fragment abstains
        overflow = self.match([prop(1, np.full((16, 3), 1 << 60, dtype=np.int64))], [track(2, voxels(16))])
        self.assertFalse(overflow.associations)
        self.assertEqual(overflow.diagnostics.fragment_errors, ((1, "coordinate_overflow_proposal"),))
        bad_mask = match_voxels([prop(1, voxels(16))], [track(2, voxels(16))], np.array([1], dtype=np.int64))
        self.assertTrue(bad_mask.diagnostics.fail_open)
        self.assertEqual(bad_mask.diagnostics.code, "invalid_eligibility_mask")
        too_many_tracks = [track(i, voxels(16, (i * 100, 0, 0))) for i in range(MAX_TRACKS + 1)]
        self.assertTrue(self.match([prop(1, voxels(16))], too_many_tracks).diagnostics.fail_open)
        many_views = [voxels(16, (i * 100, 0, 0)) for i in range(MAX_VIEWS_PER_TRACK + 1)]
        self.assertTrue(self.match([prop(1, voxels(16))], [track(1, *many_views)]).diagnostics.fail_open)
        too_many_view_voxels = np.zeros((MAX_VOXELS_PER_VIEW + 1, 3), dtype=np.int64)
        self.assertFalse(self.match([prop(1, too_many_view_voxels)], [track(1, voxels(16))]).associations)
        union_views = [voxels(MAX_VOXELS_PER_VIEW, (i * 1000, 0, 0)) for i in range(3)]
        self.assertTrue(self.match([prop(1, voxels(16))], [track(1, *union_views)]).diagnostics.fail_open)

    def test_candidate_cap_is_eight_nearest(self):
        # All tracks pass AABB, but only nearest eight are tested. IDs 0..7 win tie distance.
        tracks = [track(i, voxels(16)) for i in range(MAX_CANDIDATES + 1)]
        result = self.match([prop(99, voxels(16))], tracks)
        self.assertEqual(result.diagnostics.candidate_pairs, MAX_CANDIDATES)
        self.assertFalse(result.associations)  # tied runner-ups bar acceptance

    def test_pair_evidence_keeps_positive_overlap_below_match_threshold(self):
        proposal_voxels = voxels(16)
        # Exactly one shared voxel: valid prepared fragments and an AABB broad-
        # phase candidate, but well below Group3D's intersection threshold 8.
        track_voxels = np.vstack(
            [proposal_voxels[:1], voxels(15, (100, 0, 0))]
        )
        proposals = [prop(7, proposal_voxels)]
        tracks = [track(9, track_voxels)]
        snapshot = prepare_track_snapshot(tracks).snapshot
        batch = prepare_proposals(proposals).batch
        native_before = match_prepared_proposals(
            batch, snapshot, np.ones(1, dtype=bool)
        )

        result = extract_prepared_pair_evidence(
            batch, snapshot, np.ones(1, dtype=bool)
        )

        self.assertFalse(result.diagnostics.fail_open)
        self.assertEqual(result.diagnostics.aabb_pairs, 1)
        self.assertEqual(result.diagnostics.candidate_pairs, 1)
        self.assertEqual(result.diagnostics.positive_intersection_pairs, 1)
        self.assertEqual(len(result.proposals), 1)
        evidence = result.proposals[0].candidates[0]
        self.assertEqual((evidence.proposal_id, evidence.track_id), (7, 9))
        self.assertEqual(evidence.intersection, 1)
        self.assertEqual(evidence.proposal_voxel_count, 16)
        self.assertEqual(evidence.track_voxel_count, 16)
        self.assertEqual(evidence.proposal_containment, 1.0 / 16.0)
        self.assertEqual(evidence.track_containment, 1.0 / 16.0)
        self.assertFalse(native_before.associations)
        self.assertEqual(
            match_prepared_proposals(batch, snapshot, np.ones(1, dtype=bool)),
            native_before,
        )

    def test_pair_evidence_counts_containments_and_voxel_distance(self):
        proposal_voxels = voxels(16)
        track_voxels = voxels(20)
        snapshot = prepare_track_snapshot([track(40, track_voxels)]).snapshot
        batch = prepare_proposals([prop(3, proposal_voxels)]).batch

        result = extract_prepared_pair_evidence(
            batch, snapshot, np.ones(1, dtype=bool)
        )
        evidence = result.proposals[0].candidates[0]

        self.assertEqual(evidence.intersection, 16)
        self.assertEqual(evidence.proposal_voxel_count, 16)
        self.assertEqual(evidence.track_voxel_count, 20)
        self.assertEqual(evidence.proposal_containment, 1.0)
        self.assertEqual(evidence.track_containment, 0.8)
        self.assertEqual(evidence.centroid_distance_voxels, 2.0)

    def test_pair_evidence_is_bounded_deterministic_and_eligibility_aware(self):
        proposals = [prop(99, voxels(16))]
        tracks = [track(index, voxels(16)) for index in range(MAX_CANDIDATES + 1)]
        snapshot = prepare_track_snapshot(tracks).snapshot
        batch = prepare_proposals(proposals).batch
        all_eligible = np.ones(len(tracks), dtype=bool)
        native_before = match_prepared_proposals(batch, snapshot, all_eligible)

        first = extract_prepared_pair_evidence(batch, snapshot, all_eligible)
        second = extract_prepared_pair_evidence(batch, snapshot, all_eligible)

        self.assertEqual(first.proposals, second.proposals)
        first_diagnostics = dict(vars(first.diagnostics))
        second_diagnostics = dict(vars(second.diagnostics))
        first_diagnostics.pop("elapsed_ms")
        second_diagnostics.pop("elapsed_ms")
        self.assertEqual(first_diagnostics, second_diagnostics)
        self.assertGreaterEqual(first.diagnostics.elapsed_ms, 0.0)
        self.assertGreaterEqual(second.diagnostics.elapsed_ms, 0.0)
        self.assertEqual(first.diagnostics.aabb_pairs, MAX_CANDIDATES + 1)
        self.assertEqual(first.diagnostics.candidate_pairs, MAX_CANDIDATES)
        self.assertEqual(first.diagnostics.positive_intersection_pairs, MAX_CANDIDATES)
        self.assertEqual(
            [item.track_id for item in first.proposals[0].candidates],
            list(range(MAX_CANDIDATES)),
        )
        self.assertEqual(
            match_prepared_proposals(batch, snapshot, all_eligible), native_before
        )
        self.assertFalse(batch.proposals[0].voxels.flags.writeable)
        self.assertFalse(snapshot.tracks[0].voxels.flags.writeable)

        only_last = np.zeros(len(tracks), dtype=bool)
        only_last[-1] = True
        eligible = extract_prepared_pair_evidence(batch, snapshot, only_last)
        self.assertEqual(
            [item.track_id for item in eligible.proposals[0].candidates],
            [MAX_CANDIDATES],
        )
        self.assertEqual(
            eligible.diagnostics.skipped_track_ids,
            tuple(range(MAX_CANDIDATES)),
        )

    def test_pair_evidence_keeps_empty_rows_and_fails_open_structurally(self):
        proposal_voxels = voxels(16)
        # The expanded AABBs overlap, but the voxel sets do not.
        track_voxels = voxels(16, (16, 0, 0))
        snapshot = prepare_track_snapshot([track(2, track_voxels)]).snapshot
        batch = prepare_proposals([prop(1, proposal_voxels)]).batch

        no_overlap = extract_prepared_pair_evidence(
            batch, snapshot, np.ones(1, dtype=bool)
        )
        self.assertFalse(no_overlap.diagnostics.fail_open)
        self.assertEqual(no_overlap.diagnostics.aabb_pairs, 1)
        self.assertEqual(no_overlap.diagnostics.candidate_pairs, 1)
        self.assertEqual(no_overlap.diagnostics.positive_intersection_pairs, 0)
        self.assertEqual(no_overlap.proposals[0].proposal_id, 1)
        self.assertEqual(no_overlap.proposals[0].candidates, ())

        bad_mask = extract_prepared_pair_evidence(
            batch, snapshot, np.ones(1, dtype=np.int64)
        )
        self.assertTrue(bad_mask.diagnostics.fail_open)
        self.assertEqual(bad_mask.diagnostics.code, "invalid_eligibility_mask")
        self.assertEqual(bad_mask.proposals, ())

        bad_snapshot = extract_prepared_pair_evidence(
            batch, object(), np.zeros(0, dtype=bool)
        )
        self.assertTrue(bad_snapshot.diagnostics.fail_open)
        self.assertEqual(bad_snapshot.diagnostics.code, "invalid_prepared_snapshot")
        self.assertEqual(bad_snapshot.proposals, ())

    def test_pair_evidence_elapsed_covers_success_and_fail_open_paths(self):
        snapshot = prepare_track_snapshot([track(2, voxels(16))]).snapshot
        batch = prepare_proposals([prop(1, voxels(16))]).batch

        original_clock = fast.time.perf_counter_ns
        try:
            ticks = iter((1_000_000, 3_500_000))
            fast.time.perf_counter_ns = lambda: next(ticks)
            success = extract_prepared_pair_evidence(
                batch, snapshot, np.ones(1, dtype=bool)
            )
            self.assertEqual(success.diagnostics.elapsed_ms, 2.5)

            ticks = iter((10_000_000, 11_250_000))
            fast.time.perf_counter_ns = lambda: next(ticks)
            failure = extract_prepared_pair_evidence(
                batch, snapshot, np.ones(1, dtype=np.int64)
            )
            self.assertTrue(failure.diagnostics.fail_open)
            self.assertEqual(failure.diagnostics.elapsed_ms, 1.25)
        finally:
            fast.time.perf_counter_ns = original_clock

    def test_randomized_exact_parity_with_oracle(self):
        rng = np.random.default_rng(20260822)
        for _ in range(100):
            proposals = []
            for proposal_id in range(int(rng.integers(0, 14))):
                # Duplicates and a mix of sub-minimum fragments exercise set
                # canonicalization without relaxing any validity rules.
                points = rng.integers(-30, 31, size=(int(rng.integers(0, 45)), 3), dtype=np.int64)
                proposals.append(prop(proposal_id, points, float(rng.normal())))
            tracks = []
            for track_id in range(int(rng.integers(0, 20))):
                views = [rng.integers(-30, 31, size=(int(rng.integers(0, 45)), 3), dtype=np.int64)
                         for _ in range(int(rng.integers(1, 4)))]
                tracks.append(track(track_id, *views))
            mask = rng.integers(0, 2, size=len(tracks), dtype=np.int8).astype(bool)
            self.assertEqual(match_voxels(proposals, tracks, mask),
                             oracle.match_voxels(proposals, tracks, mask))

    def test_prepared_randomized_exact_parity_and_immutability(self):
        rng = np.random.default_rng(20260823)
        for _ in range(100):
            proposals = [prop(i, rng.integers(-20, 21, size=(int(rng.integers(0, 40)), 3), dtype=np.int64),
                              float(rng.normal())) for i in range(int(rng.integers(0, 14)))]
            tracks = []
            for track_id in range(int(rng.integers(0, 20))):
                views = [rng.integers(-20, 21, size=(int(rng.integers(0, 40)), 3), dtype=np.int64)
                         for _ in range(int(rng.integers(1, 4)))]
                tracks.append(track(track_id, *views))
            # One isolated anchor makes every randomized trial exercise a
            # non-empty accepted association, not merely equal abstentions.
            tracks.append(track(1000, voxels(16, (1000, 0, 0))))
            proposals.append(prop(999, voxels(16, (1000, 0, 0)), 1000.0))
            prepared = prepare_track_snapshot(tracks)
            self.assertFalse(prepared.diagnostics.fail_open)
            self.assertIsNotNone(prepared.snapshot)
            mask = np.concatenate([rng.integers(0, 2, size=len(tracks) - 1, dtype=np.int8).astype(bool), [True]])
            self.assertEqual(match_prepared(proposals, prepared.snapshot, mask),
                             oracle.match_voxels(proposals, tracks, mask))
            prepared_proposals = prepare_proposals(proposals)
            self.assertIsNotNone(prepared_proposals.batch)
            self.assertEqual(match_prepared_proposals(prepared_proposals.batch, prepared.snapshot, mask),
                             oracle.match_voxels(proposals, tracks, mask))
        immutable = prepare_track_snapshot([track(1, voxels(16))]).snapshot
        self.assertFalse(immutable.tracks[0].voxels.flags.writeable)
        self.assertEqual(len(immutable.tracks[0].keys), immutable.tracks[0].voxel_count)
        with self.assertRaises(ValueError):
            immutable.tracks[0].voxels[0, 0] = 1

    def test_prepared_fail_open_and_atomic_update(self):
        bad = prepare_track_snapshot([track(i, voxels(16)) for i in range(MAX_TRACKS + 1)])
        self.assertTrue(bad.diagnostics.fail_open)
        original_tracks = [track(1, voxels(16))]
        old = prepare_track_snapshot(original_tracks).snapshot
        updated_tracks = [track(1, voxels(16, (100, 0, 0))), track(2, voxels(16))]
        updated = update_prepared_track_snapshot(old, updated_tracks)
        self.assertFalse(updated.diagnostics.fail_open)
        np.testing.assert_array_equal(old.tracks[0].voxels, voxels(16))
        ps = [prop(9, voxels(16))]
        self.assertEqual(match_prepared(ps, updated.snapshot, np.ones(2, dtype=bool)),
                         oracle.match_voxels(ps, updated_tracks, np.ones(2, dtype=bool)))
        invalid = match_prepared(ps, object(), np.ones(0, dtype=bool))
        self.assertTrue(invalid.diagnostics.fail_open)

    def test_prepared_copies_inputs_seal_and_touched_update(self):
        input_track = voxels(16)
        input_proposal = voxels(16)
        tracks = [track(1, input_track)]
        prepared_tracks = prepare_track_snapshot(tracks).snapshot
        prepared_proposals = prepare_proposals([prop(2, input_proposal)]).batch
        input_track[:, 0] += 1000  # caller-owned source mutation cannot alter cache
        input_proposal[:, 0] += 1000
        result = match_prepared_proposals(prepared_proposals, prepared_tracks, np.ones(1, dtype=bool))
        self.assertEqual([(a.proposal_id, a.track_id) for a in result.associations], [(2, 1)])
        changed = update_touched_prepared_tracks(prepared_tracks, [track(1, voxels(16, (500, 0, 0)))])
        self.assertFalse(changed.diagnostics.fail_open)
        self.assertNotEqual(changed.snapshot.digest, prepared_tracks.digest)
        # Basic object.__setattr__ seal tampering is rejected; deeper object
        # internals are explicitly a trusted-snapshot boundary (documented).
        attacked = prepare_track_snapshot([track(1, voxels(16))]).snapshot
        object.__setattr__(attacked, "_seal", object())
        self.assertTrue(match_prepared_proposals(prepared_proposals, attacked, np.ones(1, dtype=bool)).diagnostics.fail_open)

    def test_public_policy_rebind_fast_and_oracle_is_inert(self):
        tracks = [track(1, voxels(16))]
        proposals = [prop(2, voxels(16))]
        snapshot = prepare_track_snapshot(tracks).snapshot
        batch = prepare_proposals(proposals).batch
        mask = np.ones(1, dtype=bool)
        baseline_fast_raw = match_voxels(proposals, tracks, mask)
        baseline_oracle_raw = oracle.match_voxels(proposals, tracks, mask)
        baseline_prepared = match_prepared_proposals(batch, snapshot, mask)
        attacks = {
            "VOXEL_SIZE_METERS": 999.0,
            "MAX_PROPOSALS": 0,
            "MAX_TRACKS": 0,
            "MAX_CANDIDATES": 0,
            "MAX_VIEWS_PER_TRACK": 0,
            "MAX_VOXELS_PER_VIEW": 0,
            "MAX_UNION_VOXELS_PER_TRACK": 0,
            "MIN_VOXELS": 1000,
            "MIN_INTERSECTION": 1000,
            "MIN_JACCARD": 2.0,
            "MIN_CONTAINMENT": 2.0,
            "MAX_CONTAINMENT": 2.0,
            "MIN_RUNNER_UP_MARGIN": 2.0,
        }
        saved = {
            module: {name: getattr(module, name) for name in attacks}
            for module in (fast, oracle)
        }
        try:
            for module in (fast, oracle):
                for name, value in attacks.items():
                    setattr(module, name, value)
            self.assertEqual(match_voxels(proposals, tracks, mask), baseline_fast_raw)
            self.assertEqual(
                oracle.match_voxels(proposals, tracks, mask), baseline_oracle_raw
            )
            self.assertEqual(
                match_prepared_proposals(batch, snapshot, mask), baseline_prepared
            )
            many_proposals = [
                prop(i, voxels(16, (i * 100, 0, 0)), score=float(100 - i))
                for i in range(65)
            ]
            self.assertEqual(
                len(match_voxels(many_proposals, [], np.zeros(0, dtype=bool))
                    .diagnostics.selected_proposal_ids),
                64,
            )
            self.assertEqual(
                len(oracle.match_voxels(many_proposals, [], np.zeros(0, dtype=bool))
                    .diagnostics.selected_proposal_ids),
                64,
            )
            too_many_tracks = [
                track(i, voxels(16, (i * 100, 0, 0))) for i in range(1025)
            ]
            self.assertTrue(
                prepare_track_snapshot(too_many_tracks).diagnostics.fail_open
            )
            self.assertTrue(
                match_voxels(
                    proposals, too_many_tracks, np.ones(1025, dtype=bool)
                ).diagnostics.fail_open
            )
            self.assertTrue(
                oracle.match_voxels(
                    proposals, too_many_tracks, np.ones(1025, dtype=bool)
                ).diagnostics.fail_open
            )
        finally:
            for module, values in saved.items():
                for name, value in values.items():
                    setattr(module, name, value)
        # The matcher reads only its sealed payload, never this public mirror.
        forged_valid = np.zeros_like(snapshot.valid)
        forged_valid.setflags(write=False)
        object.__setattr__(snapshot, "valid", forged_valid)
        self.assertEqual(
            match_prepared_proposals(batch, snapshot, mask), baseline_prepared
        )


if __name__ == "__main__":
    unittest.main()
