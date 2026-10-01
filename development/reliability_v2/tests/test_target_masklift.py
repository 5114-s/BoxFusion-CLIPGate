import unittest

import numpy as np

from boxfusion.target_masklift import (
    PastOnlyTargetTracker,
    TargetObservation,
    aabb_overlap,
    circular_medoid_yaw,
    fuse_three_view_points,
    quaternion_yaws_wxyz,
    robust_yaw_obb,
    signed_floor_voxels,
    yaw_medoid_wxyz,
)


def point_for_voxel(x, y=0, z=0):
    return np.asarray([[0.05 * x + 0.01, 0.05 * y + 0.01, 0.05 * z + 0.01]])


def quaternion(yaw_degrees):
    yaw = np.deg2rad(yaw_degrees)
    return np.asarray([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def observation(identifier, frame, lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0), group="chair"):
    return TargetObservation(identifier, frame, group, np.asarray(lower), np.asarray(upper))


class TargetMaskLiftVoxelTests(unittest.TestCase):
    def test_signed_floor_negative_coordinates(self):
        keys = signed_floor_voxels(
            np.asarray([[-1e-9, 0.0, 0.0499], [0.0, -0.0501, 0.05]])
        )
        np.testing.assert_array_equal(keys, [[-1, 0, 0], [0, -2, 1]])
        with self.assertRaises(ValueError):
            keys.setflags(write=True)

    def test_two_distinct_views_chebyshev_support(self):
        # Voxels 0 and 1 support one another. Voxel 8 occurs in only one view.
        result = fuse_three_view_points(
            [
                np.vstack([point_for_voxel(0), point_for_voxel(8)]),
                point_for_voxel(1, 1, 1),
                point_for_voxel(30),
            ]
        )
        np.testing.assert_array_equal(result.voxel_keys, [[0, 0, 0], [1, 1, 1]])
        np.testing.assert_array_equal(result.support_view_count, [2, 2])
        np.testing.assert_array_equal(result.support_view_matrix, [[True, True, False]] * 2)
        np.testing.assert_array_equal(result.exact_view_matrix, [[True, False, False], [False, True, False]])
        np.testing.assert_array_equal(result.input_point_counts, [2, 1, 1])
        np.testing.assert_array_equal(result.supported_point_counts, [1, 1, 0])
        np.testing.assert_array_equal(result.exact_supported_voxel_counts, [1, 1, 0])
        np.testing.assert_array_equal(result.neighborhood_supported_voxel_counts, [2, 2, 0])
        np.testing.assert_allclose(result.voxel_centers, [[0.025] * 3, [0.075] * 3])

    def test_many_points_in_one_view_do_not_count_as_cross_view_support(self):
        repeated = np.repeat(point_for_voxel(2), 20, axis=0)
        result = fuse_three_view_points(
            [repeated, point_for_voxel(20), np.empty((0, 3), dtype=np.float64)]
        )
        self.assertEqual(result.voxel_count, 0)
        self.assertEqual(result.point_count, 0)

    def test_output_is_invariant_to_point_order(self):
        views = [
            np.vstack([point_for_voxel(2), point_for_voxel(-1), point_for_voxel(1)]),
            np.vstack([point_for_voxel(0), point_for_voxel(2)]),
            point_for_voxel(20),
        ]
        left = fuse_three_view_points(views)
        right = fuse_three_view_points([value[::-1].copy() for value in views])
        for name in (
            "voxel_keys",
            "voxel_centers",
            "support_view_matrix",
            "exact_view_matrix",
            "supported_points",
            "supported_point_view_ids",
        ):
            np.testing.assert_array_equal(getattr(left, name), getattr(right, name))

    def test_strict_point_validation(self):
        with self.assertRaises(ValueError):
            fuse_three_view_points([np.zeros((1, 3)), np.zeros((1, 3))])
        with self.assertRaises(ValueError):
            fuse_three_view_points(
                [np.zeros((1, 3)), np.full((1, 3), np.nan), np.zeros((1, 3))]
            )
        with self.assertRaises(ValueError):
            signed_floor_voxels([[0.0, 0.0, 0.0]])


class TargetMaskLiftOBBTests(unittest.TestCase):
    def test_circular_medoid_crosses_pi_without_averaging_to_zero(self):
        yaws = quaternion_yaws_wxyz(
            np.stack([quaternion(179.0), quaternion(-179.0), quaternion(178.0)])
        )
        self.assertAlmostEqual(np.rad2deg(circular_medoid_yaw(yaws)), 179.0, places=9)
        self.assertAlmostEqual(np.rad2deg(yaw_medoid_wxyz(np.stack([quaternion(179.0), quaternion(-179.0), quaternion(178.0)]))), 179.0, places=9)
        # q and -q encode the same orientation.
        self.assertAlmostEqual(
            quaternion_yaws_wxyz(np.stack([quaternion(42.0), -quaternion(42.0)]))[0],
            quaternion_yaws_wxyz(np.stack([quaternion(42.0), -quaternion(42.0)]))[1],
        )

    def test_q02_q98_local_frame_obb_is_robust_to_two_outliers(self):
        x = np.repeat(np.linspace(-1.0, 1.0, 101), 3)
        local = np.stack([x, np.tile([-0.5, 0.0, 0.5], 101), np.zeros(len(x))], axis=1)
        local = np.vstack([local, [-100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        yaw = np.deg2rad(35.0)
        rotation = np.asarray(
            [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
        )
        world = local @ rotation.T + np.asarray([2.0, -1.0, 0.7])
        obb = robust_yaw_obb(world, yaw_rad=yaw)
        self.assertLess(obb.extent[0], 2.1)
        self.assertGreater(obb.extent[0], 1.8)
        self.assertAlmostEqual(obb.extent[1], 1.0, places=10)
        self.assertAlmostEqual(obb.center[2], 0.7, places=10)
        self.assertEqual(obb.corners.shape, (8, 3))
        with self.assertRaises(ValueError):
            obb.corners.setflags(write=True)

    def test_obb_can_take_exactly_three_raw_quaternions(self):
        points = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
        q = np.stack([quaternion(10.0), quaternion(12.0), quaternion(11.0)])
        obb = robust_yaw_obb(points, quaternions_wxyz=q)
        self.assertAlmostEqual(np.rad2deg(obb.yaw_rad), 11.0, places=9)
        with self.assertRaises(ValueError):
            robust_yaw_obb(points, yaw_rad=0.0, quaternions_wxyz=q)

    def test_aabb_iou_and_bidirectional_containment(self):
        result = aabb_overlap([0, 0, 0], [2, 2, 2], [1, 1, 1], [3, 3, 3])
        self.assertAlmostEqual(result.intersection_volume, 1.0)
        self.assertAlmostEqual(result.iou, 1.0 / 15.0)
        self.assertAlmostEqual(result.left_containment, 1.0 / 8.0)
        self.assertAlmostEqual(result.right_containment, 1.0 / 8.0)
        with self.assertRaises(ValueError):
            aabb_overlap([1, 0, 0], [0, 1, 1], [0, 0, 0], [1, 1, 1])


class PastOnlyTargetTrackerTests(unittest.TestCase):
    def test_current_frame_rows_cannot_match_one_another_and_third_frame_confirms(self):
        tracker = PastOnlyTargetTracker()
        first = tracker.update(10, [observation(2, 10), observation(1, 10)])
        self.assertEqual([(row.observation_id, row.track_id, row.action) for row in first.assignments], [(1, 0, "created"), (2, 1, "created")])
        self.assertFalse(first.newly_confirmed_tracks)

        second = tracker.update(20, [observation(3, 20)])
        # Identical candidate tracks tie; lower stable track id wins.
        self.assertEqual((second.assignments[0].track_id, second.assignments[0].action), (0, "matched"))
        third = tracker.update(30, [observation(4, 30)])
        self.assertEqual(len(third.newly_confirmed_tracks), 1)
        receipt = third.newly_confirmed_tracks[0]
        self.assertTrue(receipt.confirmed)
        self.assertEqual(receipt.evidence_frame_ids, (10, 20, 30))
        self.assertEqual(receipt.evidence_observation_ids, (1, 3, 4))

    def test_target_group_and_both_geometry_gates_are_required(self):
        tracker = PastOnlyTargetTracker()
        tracker.update(1, [observation(1, 1)])
        different_group = tracker.update(2, [observation(2, 2, group="table")])
        self.assertEqual(different_group.assignments[0].action, "created")
        # Shift .5 has center distance exactly .5 and positive IoU: inclusive.
        accepted = tracker.update(3, [observation(3, 3, lower=(0.5, 0, 0), upper=(1.5, 1, 1))])
        self.assertEqual((accepted.assignments[0].track_id, accepted.assignments[0].action), (0, "matched"))
        # Center distance just above .5 rejects despite IoU > .1.
        rejected = tracker.update(4, [observation(4, 4, lower=(1.0001, 0, 0), upper=(2.0001, 1, 1))])
        self.assertEqual(rejected.assignments[0].action, "created")

    def test_ttl_counts_valid_keyframe_calls_and_frame_ids_are_monotonic(self):
        tracker = PastOnlyTargetTracker()
        tracker.update(100, [observation(1, 100)])
        for frame in range(101, 111):
            result = tracker.update(frame, [])
            self.assertNotIn(0, result.retired_track_ids)
        result = tracker.update(111, [])
        self.assertEqual(result.retired_track_ids, (0,))
        with self.assertRaises(ValueError):
            tracker.update(111, [])

    def test_observation_ids_are_global_and_input_order_is_stable(self):
        left, right = PastOnlyTargetTracker(), PastOnlyTargetTracker()
        rows = [observation(20, 1, lower=(4, 0, 0), upper=(5, 1, 1)), observation(10, 1)]
        a = left.update(1, rows)
        b = right.update(1, list(reversed(rows)))
        self.assertEqual(
            [(row.observation_id, row.track_id) for row in a.assignments],
            [(row.observation_id, row.track_id) for row in b.assignments],
        )
        with self.assertRaises(ValueError):
            left.update(2, [observation(10, 2)])


if __name__ == "__main__":
    unittest.main()
