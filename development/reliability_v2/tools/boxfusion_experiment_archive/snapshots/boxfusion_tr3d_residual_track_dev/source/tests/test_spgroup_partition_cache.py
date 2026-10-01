import tempfile
import unittest
from pathlib import Path

import numpy as np

from boxfusion.spgroup_partition_cache import (
    SCHEMA, SPGroupPartition, load_partition, write_partition,
)


class PartitionCacheTest(unittest.TestCase):
    def value(self) -> SPGroupPartition:
        vertices = np.asarray(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=np.float32
        )
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = (1, 2, 3)
        return SPGroupPartition(
            scene_id="scene0000_00",
            vertices_unaligned=vertices,
            vertices_aligned=vertices + np.asarray((1, 2, 3), dtype=np.float32),
            colors=np.ones((4, 3), dtype=np.float32),
            faces=np.asarray([[0, 1, 2]], dtype=np.int32),
            superpoint_ids=np.asarray([0, 0, 0, 1], dtype=np.int32),
            axis_alignment=transform,
            metadata={
                "schema": SCHEMA,
                "scene_id": "scene0000_00",
                "ground_truth_access": False,
            },
        )

    def test_round_trip_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as folder:
            path = Path(folder) / "value.npz"
            write_partition(path, self.value())
            loaded = load_partition(path)
            self.assertEqual(loaded.scene_id, "scene0000_00")
            self.assertEqual(loaded.superpoint_count, 2)
            with self.assertRaises(FileExistsError):
                write_partition(path, self.value())


if __name__ == "__main__":
    unittest.main()
