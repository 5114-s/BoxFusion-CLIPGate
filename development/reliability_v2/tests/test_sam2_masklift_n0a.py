import copy
import hashlib
import unittest

import numpy as np

from boxfusion import sam2_masklift_n0a as n0a

from boxfusion.sam2_masklift_n0a import (
    DEPTH_JUMP_M,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MASK_BITORDER,
    MASK_PACKED_BYTES,
    MAX_STORED_POINTS,
    MIN_AABB_EXTENT_M,
    N0AMaskLiftContractError,
    POLICY,
    VOXEL_SIZE_M,
    lift_sam2_mask,
)


def _identity():
    return {
        "source_id": "scene0000_00/frame-000123/rank-2/raw-7",
        "scene_id": "scene0000_00",
        "frame_ordinal": 123,
        "rank": 2,
        "raw_index": 7,
        "f0_opaque": {"provider": "frozen-fastsam-x", "tags": ["a", 3]},
    }


def _h0():
    return {
        "valid": True,
        "q02": [-1.0, -2.0, 0.50],
        "q98": [1.0, 2.0, 1.50],
        "center": [0.0, 0.0, 1.0],
        "extent": [2.0, 4.0, 1.0],
        "f0_geometry_field": {"must": "survive"},
    }


def _mask(y0=100, y1=120, x0=300, x1=320):
    mask = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _depth(value=1.0):
    return np.full((IMAGE_HEIGHT, IMAGE_WIDTH), value, dtype=np.float64)


def _intrinsics(fx=20.0, fy=None, cx=320.0, cy=240.0):
    if fy is None:
        fy = fx
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _lift(mask=None, depth=None, intrinsics=None, pose=None, identity=None, h0=None):
    return lift_sam2_mask(
        f0_source_identity=_identity() if identity is None else identity,
        selected_mask=_mask() if mask is None else mask,
        depth_m=_depth() if depth is None else depth,
        intrinsics=_intrinsics() if intrinsics is None else intrinsics,
        camera_to_world=np.eye(4, dtype=np.float64) if pose is None else pose,
        h0=_h0() if h0 is None else h0,
    )


def _raw_world_points(mask, depth, intrinsics, pose):
    """Reference backprojection for the one-pixel-eroded rectangular masks."""

    rows, cols = np.nonzero(mask)
    pixels = np.column_stack(
        (cols.astype(np.float64), rows.astype(np.float64), np.ones(len(rows)))
    )
    rays = pixels @ np.linalg.inv(intrinsics).T
    rays /= rays[:, 2:3]
    camera = rays * depth[rows, cols, None]
    return camera @ pose[:3, :3].T + pose[:3, 3]


class N0AMaskLiftGeometryTests(unittest.TestCase):
    def test_voxel_centroid_sum_preserves_original_row_order(self):
        # Rows from two voxels are interleaved, and their low-order bits are
        # deliberately non-monotonic.  Stable key sorting must preserve the
        # original within-voxel order rather than adding coordinates as sort
        # keys (which changes the first centroid by one float64 ULP here).
        base = np.float64(0.01)
        lo = np.nextafter(base, -np.inf)
        hi = np.nextafter(base, np.inf)
        first = np.asarray(
            [
                [hi, base, lo],
                [lo, hi, base],
                [base, lo, hi],
                [hi, lo, hi],
                [lo, base, lo],
            ],
            dtype=np.float64,
        )
        second = first.copy()
        second[:, 0] += np.float64(0.02)
        points = np.empty((10, 3), dtype=np.float64)
        points[::2] = first
        points[1::2] = second
        keys, centroids = n0a._voxel_centroids(points)
        np.testing.assert_array_equal(
            keys, np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
        )
        stable_grouped = np.concatenate((points[::2], points[1::2]), axis=0)
        expected = np.add.reduceat(stable_grouped, [0, 5], axis=0) / 5.0
        np.testing.assert_array_equal(centroids, expected)

    def test_boundary_packbits_lineage_and_output_inert_receipt(self):
        identity = _identity()
        h0 = _h0()
        identity_before = copy.deepcopy(identity)
        h0_before = copy.deepcopy(h0)
        mask = _mask()

        result = _lift(mask=mask, identity=identity, h0=h0)

        self.assertTrue(result.valid)
        self.assertIsNone(result.abstention_reason)
        self.assertEqual(result.mask_pixel_count, 20 * 20)
        # A one-pixel 8-connected boundary is removed from all four sides.
        self.assertEqual(result.interior_pixel_count, 18 * 18)
        self.assertEqual(result.support_pixel_count, 18 * 18)
        self.assertEqual(result.depth_jump_pixel_count, 0)
        self.assertEqual(result.voxel_count, 18 * 18)
        self.assertEqual(identity, identity_before)
        self.assertEqual(h0, h0_before)
        self.assertEqual(dict(result.f0_source_identity), identity_before)
        self.assertEqual(dict(result.h0_input), h0_before)
        np.testing.assert_allclose(result.h0.q02, h0["q02"])
        np.testing.assert_allclose(result.h0.q98, h0["q98"])

        expected_packed = np.packbits(mask.reshape(-1), bitorder=MASK_BITORDER)
        np.testing.assert_array_equal(result.mask_packbits, expected_packed)
        self.assertEqual(result.mask_packbits.shape, (MASK_PACKED_BYTES,))
        self.assertEqual(
            result.mask_sha256, hashlib.sha256(expected_packed.tobytes()).hexdigest()
        )
        expected_points_hash = hashlib.sha256()
        expected_points_hash.update(
            np.ascontiguousarray(result.points_world, dtype="<f8").tobytes()
        )
        expected_points_hash.update(
            np.ascontiguousarray(result.voxel_keys, dtype="<i8").tobytes()
        )
        self.assertEqual(
            result.points_and_voxel_keys_sha256, expected_points_hash.hexdigest()
        )

        receipt = result.to_receipt()
        self.assertEqual(receipt["f0_source_identity"], identity_before)
        self.assertEqual(receipt["h0_input"], h0_before)
        self.assertEqual(receipt["mask"]["bitorder"], "little")
        self.assertEqual(receipt["mask"]["packed_byte_count"], MASK_PACKED_BYTES)
        self.assertEqual(receipt["points"]["quantile_point_count"], 18 * 18)
        self.assertEqual(receipt["points"]["maximum_stored_point_count"], 2048)
        self.assertFalse(receipt["contracts"]["ground_truth_access"])
        self.assertFalse(receipt["contracts"]["semantic_or_clip_access"])
        self.assertFalse(receipt["contracts"]["history_or_state"])
        self.assertFalse(receipt["contracts"]["native_output_mutation"])
        self.assertTrue(POLICY["shadow_only"])

        for array in (
            result.mask_packbits,
            result.tight_box_xyxy,
            result.points_world,
            result.voxel_keys,
            result.h0.q02,
            result.hs.q02,
        ):
            with self.assertRaises(ValueError):
                array.setflags(write=True)

    def test_strict_depth_jump_rejects_both_four_neighbor_endpoints(self):
        mask = _mask()
        at_threshold = _depth()
        at_threshold[:, 310:] = 1.0 + DEPTH_JUMP_M
        exact = _lift(mask=mask, depth=at_threshold)
        self.assertEqual(exact.depth_jump_pixel_count, 0)
        self.assertEqual(exact.support_pixel_count, 18 * 18)

        above_threshold = _depth()
        above_threshold[:, 310:] = np.nextafter(
            1.0 + DEPTH_JUMP_M, np.inf
        )
        jumped = _lift(mask=mask, depth=above_threshold)
        # The vertical discontinuity has 18 interior rows and both adjacent
        # columns are rejected: 18 * 2 endpoints.
        self.assertEqual(jumped.depth_jump_pixel_count, 36)
        self.assertEqual(jumped.support_pixel_count, 18 * 18 - 36)
        self.assertTrue(jumped.valid)

    def test_signed_floor_voxels_and_all_point_float64_centroids(self):
        mask = _mask(y0=230, y1=250, x0=310, x1=330)
        intrinsic = _intrinsics(fx=100.0)
        result = _lift(mask=mask, intrinsics=intrinsic)
        self.assertTrue(result.valid)

        np.testing.assert_array_equal(
            np.floor(result.points_world / VOXEL_SIZE_M).astype(np.int64),
            result.voxel_keys,
        )
        self.assertTrue(np.any(result.voxel_keys[:, 0] == -1))
        self.assertTrue(np.any(result.voxel_keys[:, 1] == -1))

        interior = np.zeros_like(mask)
        interior[231:249, 311:329] = True
        raw = _raw_world_points(
            interior, _depth(), intrinsic, np.eye(4, dtype=np.float64)
        )
        raw_keys = np.floor(raw / VOXEL_SIZE_M).astype(np.int64)
        # This signed voxel contains several pixels; the sealed representative
        # must be their float64 mean rather than the first pixel.
        target_key = np.asarray([-5, -5, 50], dtype=np.int64)
        members = np.all(raw_keys == target_key, axis=1)
        self.assertGreater(int(np.count_nonzero(members)), 1)
        expected_centroid = raw[members].mean(axis=0, dtype=np.float64)
        output_row = np.flatnonzero(
            np.all(result.voxel_keys == target_key, axis=1)
        )
        self.assertEqual(len(output_row), 1)
        np.testing.assert_allclose(
            result.points_world[output_row[0]], expected_centroid, rtol=0.0, atol=1e-15
        )

    def test_cap_is_deterministic_but_hs_uses_all_centroids(self):
        mask = _mask(y0=100, y1=170, x0=200, x1=270)
        intrinsic = _intrinsics(fx=20.0)
        left = _lift(mask=mask, intrinsics=intrinsic)
        right = _lift(mask=mask.copy(), intrinsics=intrinsic.copy())

        expected_voxels = 68 * 68
        self.assertTrue(left.valid)
        self.assertEqual(left.voxel_count, expected_voxels)
        self.assertEqual(left.quantile_point_count, expected_voxels)
        self.assertEqual(left.stored_point_count, MAX_STORED_POINTS)
        self.assertEqual(left.result_sha256, right.result_sha256)
        np.testing.assert_array_equal(left.voxel_keys, right.voxel_keys)
        np.testing.assert_array_equal(left.points_world, right.points_world)
        self.assertEqual(
            left.points_and_voxel_keys_sha256,
            right.points_and_voxel_keys_sha256,
        )

        lex_order = np.lexsort(
            (left.voxel_keys[:, 2], left.voxel_keys[:, 1], left.voxel_keys[:, 0])
        )
        np.testing.assert_array_equal(lex_order, np.arange(MAX_STORED_POINTS))

        interior = np.zeros_like(mask)
        interior[101:169, 201:269] = True
        all_points = _raw_world_points(
            interior, _depth(), intrinsic, np.eye(4, dtype=np.float64)
        )
        # With 5 cm pixel spacing and 2 cm voxels, every support point is its
        # own centroid.  This independently checks that q02/q98 see all 4,624
        # centroids, not only the sealed 2,048-point sample.
        self.assertEqual(
            len(np.unique(np.floor(all_points / VOXEL_SIZE_M), axis=0)),
            expected_voxels,
        )
        raw_q02, raw_q98 = np.quantile(
            all_points, (0.02, 0.98), axis=0, method="linear"
        )
        expected_center = (raw_q02 + raw_q98) * 0.5
        expected_extent = np.maximum(raw_q98 - raw_q02, MIN_AABB_EXTENT_M)
        np.testing.assert_allclose(
            left.hs.q02, expected_center - expected_extent * 0.5, atol=1e-12
        )
        np.testing.assert_allclose(
            left.hs.q98, expected_center + expected_extent * 0.5, atol=1e-12
        )


class N0AMaskLiftAbstentionTests(unittest.TestCase):
    def test_fixed_mask_and_depth_sanity_checks_abstain(self):
        too_few = _mask(y0=100, y1=110, x0=300, x1=319)  # 190 pixels

        too_many = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=bool)
        too_many[:400, :400] = True

        short_side = _mask(y0=100, y1=110, x0=300, x1=320)  # 200 pixels
        excessive_aspect = _mask(y0=100, y1=116, x0=300, x1=412)

        poor_depth_mask = _mask()
        poor_depth = _depth()
        rows, cols = np.nonzero(poor_depth_mask)
        poor_depth[rows[:201], cols[:201]] = np.nan

        cases = (
            (too_few, _depth(), "mask_pixel_count_below_200"),
            (too_many, _depth(), "mask_pixel_count_above_122880"),
            (short_side, _depth(), "tight_box_side_below_16"),
            (excessive_aspect, _depth(), "tight_box_aspect_above_6"),
            (poor_depth_mask, poor_depth, "valid_depth_ratio_below_0_50"),
        )
        for mask, depth, reason in cases:
            with self.subTest(reason=reason):
                result = _lift(mask=mask, depth=depth)
                self.assertFalse(result.valid)
                self.assertEqual(result.abstention_reason, reason)
                self.assertFalse(result.hs.valid)
                self.assertIsNone(result.hs.q02)
                self.assertTrue(result.h0.valid)

    def test_fewer_than_sixteen_signed_voxels_abstains(self):
        mask = _mask(y0=232, y1=248, x0=312, x1=328)
        result = _lift(mask=mask, intrinsics=_intrinsics(fx=100_000.0))
        self.assertFalse(result.valid)
        self.assertEqual(result.abstention_reason, "fewer_than_16_unique_voxels")
        self.assertLess(result.voxel_count, 16)
        self.assertEqual(result.stored_point_count, result.voxel_count)
        self.assertEqual(result.quantile_point_count, result.voxel_count)


class N0AMaskLiftContractTests(unittest.TestCase):
    def test_invalid_mask_depth_shape_intrinsics_and_pose_are_fatal(self):
        bad_mask = _mask().astype(np.float64)
        bad_mask[100, 300] = 0.5
        cases = (
            {"mask": bad_mask},
            {"depth": np.ones((240, 320), dtype=np.float64)},
            {"intrinsics": np.asarray([[0.0, 0.0, 320.0], [0.0, 20.0, 240.0], [0.0, 0.0, 1.0]])},
            {"pose": np.diag([-1.0, 1.0, 1.0, 1.0])},
        )
        for kwargs in cases:
            with self.subTest(kwargs=tuple(kwargs)):
                with self.assertRaises(N0AMaskLiftContractError):
                    _lift(**kwargs)

    def test_source_identity_and_h0_require_finite_json_contracts(self):
        with self.assertRaises(N0AMaskLiftContractError):
            _lift(identity={"source_id": "", "rank": 0})
        with self.assertRaises(N0AMaskLiftContractError):
            _lift(identity={"source_id": "source", "bad": np.asarray([1])})
        bad_h0 = _h0()
        bad_h0["extent"] = [3.0, 4.0, 1.0]
        with self.assertRaises(N0AMaskLiftContractError):
            _lift(h0=bad_h0)


if __name__ == "__main__":
    unittest.main()
