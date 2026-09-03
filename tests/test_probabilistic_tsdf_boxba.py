import math
import unittest

import numpy as np

from boxfusion import probabilistic_tsdf_boxba as route
from boxfusion.sam2_tsdf_mv3dis_shadow import LiftedMaskView


def _box(center, size, yaw=0.0):
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return np.asarray(center, dtype=np.float64)[None, :] + (
        signs * (np.asarray(size, dtype=np.float64)[None, :] * 0.5)
    ) @ rotation.T


def _points_and_keys():
    keys = np.asarray(
        [
            (x, y, z)
            for x in range(-5, 6)
            for y in range(-3, 4)
            for z in (40, 41)
        ],
        dtype=np.int64,
    )
    return (keys.astype(np.float64) + 0.5) * route.VOXEL_SIZE_M, keys


def _view(frame_id, *, good_mask=True, source_suffix=""):
    points, keys = _points_and_keys()
    mask = np.zeros((480, 640), dtype=np.bool_)
    if good_mask:
        mask[226:259, 298:348] = True
    else:
        mask[40:90, 40:90] = True
    depth = np.full((480, 640), 2.05, dtype=np.float64)
    intrinsic = np.asarray(
        [[180.0, 0.0, 320.0], [0.0, 180.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return LiftedMaskView(
        source_id=f"view-{frame_id}{source_suffix}",
        frame_id=frame_id,
        mask=mask,
        depth_m=depth,
        intrinsic=intrinsic,
        camera_to_world=np.eye(4, dtype=np.float64),
        points_world=points,
        voxel_keys=keys,
        support_pixel_count=max(int(np.count_nonzero(mask)), len(points)),
        uncapped_voxel_count=len(points),
    )


class ProbabilisticTSDFBoxBATests(unittest.TestCase):
    def setUp(self):
        self.baseline = _box([0.10, 0.025, 2.05], [0.68, 0.42, 0.08])
        self.fit_views = (_view(1), _view(2))

    def test_static_policy_freezes_requested_protocol(self):
        receipt = route.policy_receipt()
        config = receipt["config"]
        self.assertEqual(config["voxel_size_m"], 0.05)
        self.assertEqual(config["tsdf_truncation_m"], 0.10)
        self.assertEqual(config["beta_prior_alpha"], 1.0)
        self.assertEqual(config["beta_prior_beta"], 1.0)
        self.assertEqual(config["maximum_voxels"], 8192)
        self.assertEqual(config["minimum_consensus_voxels"], 16)
        self.assertEqual(config["search_layers"], 5)
        self.assertEqual(config["maximum_center_delta_m"], 0.20)
        self.assertEqual(config["minimum_size_ratio"], 0.70)
        self.assertEqual(config["maximum_size_ratio"], 1.40)
        self.assertAlmostEqual(config["maximum_yaw_delta_rad"], math.radians(20.0))
        self.assertEqual(config["heldout_minimum_loss_improvement"], 0.01)
        self.assertEqual(config["heldout_minimum_depth_containment"], 0.45)
        self.assertEqual(config["heldout_minimum_mask_iou"], 0.10)
        self.assertEqual(config["heldout_maximum_free_ratio"], 0.05)
        self.assertEqual(receipt["fit_views"], "ordered[:-1]")
        self.assertEqual(receipt["heldout_view"], "ordered[-1]")
        self.assertFalse(receipt["contracts"]["ground_truth_access"])
        self.assertTrue(receipt["contracts"]["past_only"])

        with self.assertRaises(route.PITSDFBoxBAError):
            route.PITSDFBoxBAConfig(voxel_size_m=0.10)

    def test_good_heldout_accepts_candidate_and_records_beta_tsdf(self):
        result = route.refine_causal_track(
            views=(*self.fit_views, _view(3)),
            boxer_corners_by_source={},
            baseline_corners=self.baseline,
        )

        self.assertTrue(result["accepted"], result["reason"])
        self.assertEqual(result["reason"], "accepted")
        np.testing.assert_allclose(
            result["output_corners"], result["fit_candidate_corners"], atol=0.0
        )
        self.assertGreaterEqual(len(result["consensus_points"]), 16)
        pi_tsdf = result["receipt"]["pi_tsdf"]
        self.assertLessEqual(pi_tsdf["union_voxel_count"], 8192)
        self.assertEqual(
            pi_tsdf["consensus_voxel_count"], len(result["consensus_points"])
        )
        self.assertGreaterEqual(pi_tsdf["posterior_min"], 0.60)
        self.assertEqual(len(pi_tsdf["state_sha256"]), 64)
        acceptance = result["receipt"]["acceptance"]
        self.assertGreaterEqual(acceptance["loss_improvement"], 0.01)
        self.assertTrue(all(acceptance["checks"].values()))
        self.assertEqual(len(result["receipt"]["boxba"]["layers"]), 5)
        self.assertLessEqual(result["receipt"]["boxba"]["center_delta_m"], 0.20 + 1e-12)
        ratios = np.asarray(result["receipt"]["boxba"]["size_ratios"])
        self.assertTrue(np.all(ratios >= 0.70 - 1e-12))
        self.assertTrue(np.all(ratios <= 1.40 + 1e-12))
        self.assertLessEqual(
            abs(result["receipt"]["boxba"]["yaw_delta_rad"]),
            math.radians(20.0) + 1e-12,
        )

    def test_heldout_content_cannot_change_fit_and_failure_rolls_back(self):
        good = route.refine_causal_track(
            views=(*self.fit_views, _view(3)),
            boxer_corners_by_source={
                # This entry belongs to held-out only and must be unreachable
                # from PI-TSDF construction and BoxBA fitting.
                "view-3": _box([3.0, 3.0, 3.0], [1.0, 1.0, 1.0]),
            },
            baseline_corners=self.baseline,
        )
        bad = route.refine_causal_track(
            views=(*self.fit_views, _view(3, good_mask=False)),
            boxer_corners_by_source={
                "view-3": _box([-3.0, -3.0, 1.0], [2.0, 2.0, 2.0]),
            },
            baseline_corners=self.baseline,
        )

        self.assertTrue(good["accepted"])
        self.assertFalse(bad["accepted"])
        np.testing.assert_array_equal(
            good["fit_candidate_corners"], bad["fit_candidate_corners"]
        )
        np.testing.assert_array_equal(
            good["consensus_points"], bad["consensus_points"]
        )
        self.assertEqual(
            good["receipt"]["boxba"]["fit_candidate_sha256"],
            bad["receipt"]["boxba"]["fit_candidate_sha256"],
        )
        np.testing.assert_array_equal(bad["output_corners"], self.baseline)
        self.assertNotEqual(bad["reason"], "accepted")
        self.assertFalse(
            bad["receipt"]["acceptance"]["checks"]["mask_iou_at_least_0.10"]
        )

    def test_requires_three_chronological_views(self):
        with self.assertRaises(route.PITSDFBoxBAError):
            route.refine_causal_track(
                views=self.fit_views,
                boxer_corners_by_source={},
                baseline_corners=self.baseline,
            )
        with self.assertRaises(route.PITSDFBoxBAError):
            route.refine_causal_track(
                views=(_view(2), _view(1), _view(3)),
                boxer_corners_by_source={},
                baseline_corners=self.baseline,
            )


if __name__ == "__main__":
    unittest.main()

