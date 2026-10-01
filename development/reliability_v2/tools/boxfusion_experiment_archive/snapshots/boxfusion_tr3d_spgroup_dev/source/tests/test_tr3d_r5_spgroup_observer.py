import unittest

import numpy as np

from boxfusion.spgroup_feature_cache import SPGroupFeatureSidecar, SCHEMA as FEATURE_SCHEMA
from boxfusion.spgroup_official_adapter import SPGroupFeatures
from boxfusion.spgroup_partition_cache import SPGroupPartition, SCHEMA as PARTITION_SCHEMA
from boxfusion.tr3d_r5_spgroup_observer import METRIC_NAMES, observe_pairs, points_in_yaw_box


class R5ObserverTest(unittest.TestCase):
    def setUp(self) -> None:
        vertices = np.asarray([
            [-0.4, -0.4, 0], [-0.3, 0.3, 0], [0.3, -0.3, 0], [0.4, 0.4, 0],
            [1.1, -0.2, 0], [1.2, 0.2, 0], [1.4, -0.2, 0], [1.4, 0.2, 0],
        ], dtype=np.float32)
        self.partition = SPGroupPartition(
            scene_id="scene0000_00", vertices_unaligned=vertices,
            vertices_aligned=vertices, colors=np.ones_like(vertices),
            faces=np.asarray([[0, 1, 2], [1, 2, 3], [4, 5, 6], [5, 6, 7]], dtype=np.int32),
            superpoint_ids=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
            axis_alignment=np.eye(4),
            metadata={"schema": PARTITION_SCHEMA, "scene_id": "scene0000_00", "ground_truth_access": False},
        )
        features = SPGroupFeatures(
            superpoint_ids=np.asarray([0, 1], dtype=np.int32),
            centers_aligned=np.asarray([[0, 0, 0], [1.3, 0, 0]], dtype=np.float32),
            embeddings=np.vstack((np.ones(390), -np.ones(390))).astype(np.float32),
            vote_offsets=np.zeros((2, 3), dtype=np.float32),
            vote_offset_std=np.asarray([[0.01] * 3, [0.02] * 3], dtype=np.float32),
            voxel_counts=np.asarray([4, 4], dtype=np.int32),
        )
        self.features = SPGroupFeatureSidecar(
            scene_id="scene0000_00", features=features,
            metadata={
                "schema": FEATURE_SCHEMA, "scene_id": "scene0000_00",
                "observer_only": True, "ground_truth_access": False,
                "clip_access": False, "semantic_head_used": False,
            },
        )

    def test_rotated_membership(self) -> None:
        mask = points_in_yaw_box(np.asarray([[0.0, 0.4, 0], [1.0, 0, 0]]), np.asarray([0, 0, 0, 1, .2, 1, np.pi / 2]))
        self.assertEqual(mask.tolist(), [True, False])

    def test_paired_metrics(self) -> None:
        anchor = np.asarray([[0, 0, 0, 1, 1, 1, 0]], dtype=np.float32)
        candidate = np.asarray([[0.5, 0, 0, 2, 1, 1, 0]], dtype=np.float32)
        result = observe_pairs(self.partition, self.features, anchor, candidate)
        self.assertEqual(result.metrics.shape, (1, 2, len(METRIC_NAMES)))
        self.assertGreater(result.metrics[0, 1, 1], result.metrics[0, 0, 1])


if __name__ == "__main__":
    unittest.main()
