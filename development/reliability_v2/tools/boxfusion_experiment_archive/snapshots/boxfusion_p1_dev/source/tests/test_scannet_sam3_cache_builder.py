import json
from pathlib import Path

import numpy as np
import pytest

from boxfusion.supplemental_proposals import (
    NpzProposalCache,
    StrictCacheProposalProvider,
    SupplementalProposal,
    proposal_cache_key,
)
from tools.build_scannet_sam3_proposal_cache import (
    deduplicate_proposals,
    discover_scannet_scene,
    load_scannet_rgb,
    load_runtime_rgb_manifest,
    load_staged_runtime_rgb,
    main,
    normalize_sam3_output,
    read_scene_list,
    scheduled_frame_indices,
    store_frame_proposals,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def _proposal(
    mask,
    *,
    score=0.8,
    label="chair",
    bbox=None,
):
    binary = np.asarray(mask, dtype=np.bool_)
    if bbox is None:
        rows, columns = np.nonzero(binary)
        bbox = [
            float(columns.min()),
            float(rows.min()),
            float(columns.max() + 1),
            float(rows.max() + 1),
        ]
    return SupplementalProposal(
        bbox=np.asarray(bbox, dtype=np.float32),
        score=score,
        mask=binary,
        label=label,
        feature=None,
    )


def _write_scene(
    root: Path,
    scene_id: str,
    *,
    frame_count: int = 1,
    height: int = 2,
    width: int = 3,
):
    import cv2

    frames = root / scene_id / "frames"
    for name in ("color", "depth", "pose"):
        (frames / name).mkdir(parents=True)
    for index in range(frame_count):
        # cv2.imwrite consumes BGR, matching the source ScanNet JPEG contract.
        bgr = np.empty((height, width, 3), dtype=np.uint8)
        bgr[...] = np.asarray([10 + index, 20, 30], dtype=np.uint8)
        assert cv2.imwrite(
            str(frames / "color" / f"{index}.jpg"),
            bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 100],
        )
        depth = np.full((height, width), 1000, dtype=np.uint16)
        assert cv2.imwrite(
            str(frames / "depth" / f"{index}.png"), depth
        )
        np.savetxt(frames / "pose" / f"{index}.txt", np.eye(4))


def test_scene_list_and_provider_schedule_are_strict(tmp_path):
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(
        "# fixed order\nscene0000_00\n\nscene0001_00\n",
        encoding="utf-8",
    )
    assert read_scene_list(scene_list) == [
        "scene0000_00",
        "scene0001_00",
    ]

    # The early exit in demo.py means the physical final frame is not
    # unconditionally appended. Provider calls occur every 5 keyframes.
    assert scheduled_frame_indices(
        1015, gap=25, proposal_interval=5
    ) == [0, 125, 250, 375, 500, 625, 750, 875]
    assert scheduled_frame_indices(
        126, gap=25, proposal_interval=5
    ) == [0]

    scene_list.write_text(
        "scene0000_00\nscene0000_00\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        read_scene_list(scene_list)
    with pytest.raises(ValueError, match="frame_count"):
        scheduled_frame_indices(0)


def test_scannet_rgb_matches_bgr_to_rgb_and_orientation(tmp_path):
    import cv2

    color_path = tmp_path / "0.jpg"
    depth_path = tmp_path / "0.png"
    bgr = np.empty((2, 3, 3), dtype=np.uint8)
    bgr[...] = np.asarray([10, 20, 30], dtype=np.uint8)
    assert cv2.imwrite(
        str(color_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 100]
    )
    assert cv2.imwrite(
        str(depth_path), np.ones((2, 3), dtype=np.uint16)
    )

    upright, orientation = load_scannet_rgb(
        color_path,
        depth_path,
        np.eye(4),
        configured_height=2,
        configured_width=3,
    )
    assert orientation == 0
    assert upright.shape == (2, 3, 3)
    np.testing.assert_allclose(
        upright[0, 0], np.asarray([30, 20, 10]), atol=1
    )

    left_pose = np.eye(4)
    left_pose[2, :3] = np.asarray([-1.0, 0.0, 0.0])
    rotated, orientation = load_scannet_rgb(
        color_path,
        depth_path,
        left_pose,
        configured_height=2,
        configured_width=3,
    )
    assert orientation == 1
    assert rotated.shape == (3, 2, 3)
    np.testing.assert_array_equal(rotated, np.rot90(upright, -1))


def test_normalize_and_deduplicate_sam3_outputs():
    masks = np.zeros((3, 1, 4, 5), dtype=np.float32)
    masks[0, 0, 0:2, 0:2] = 0.9
    masks[1, 0, 0:2, 0:2] = 0.8
    masks[2, 0, 3, 4] = 0.9
    output = {
        "boxes": FakeTensor(
            [
                [-2.0, -1.0, 3.0, 3.0],
                [0.0, 0.0, 2.0, 2.0],
                [4.0, 3.0, 5.0, 4.0],
            ]
        ),
        "scores": FakeTensor([0.8, 0.9, 0.95]),
        "masks_logits": FakeTensor(masks),
    }
    normalized = normalize_sam3_output(
        output,
        label="chair",
        image_shape=(4, 5),
        mask_threshold=0.5,
        min_mask_pixels=2,
        max_per_class=2,
    )
    assert len(normalized) == 2
    assert [item.score for item in normalized] == pytest.approx([0.9, 0.8])
    np.testing.assert_allclose(normalized[1].bbox, [0.0, 0.0, 3.0, 3.0])

    # The two proposals share the same instance mask across prompts. The
    # deterministic score ordering must retain only the higher-scoring row.
    duplicate = SupplementalProposal(
        bbox=normalized[0].bbox,
        score=0.95,
        mask=normalized[0].mask,
        label="furniture",
        feature=None,
    )
    disjoint_mask = np.zeros((4, 5), dtype=np.bool_)
    disjoint_mask[2:4, 3:5] = True
    disjoint = _proposal(disjoint_mask, score=0.7, label="table")
    kept = deduplicate_proposals(
        [normalized[0], duplicate, disjoint],
        duplicate_mask_iou=0.8,
        max_proposals=8,
    )
    assert [(item.label, item.score) for item in kept] == [
        ("furniture", 0.95),
        ("table", 0.7),
    ]


def test_builder_cache_round_trip_matches_strict_online_provider(tmp_path):
    namespace = "sam3-scannet-test-v1"
    scene_id = "scene0000_00"
    raw_frame_id = 125
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.bool_)
    mask[1:3, 1:4] = True
    values = [_proposal(mask)]
    cache = NpzProposalCache(tmp_path / "cache")

    key, path = store_frame_proposals(
        cache,
        namespace,
        scene_id,
        raw_frame_id,
        image,
        values,
    )
    assert key == proposal_cache_key(
        namespace, f"{scene_id}:{raw_frame_id}", image
    )
    assert path == cache.path_for_key(key)
    assert path.is_file()

    provider = StrictCacheProposalProvider(
        NpzProposalCache(tmp_path / "cache"),
        namespace=namespace,
        missing_policy="error",
    )
    replayed = provider.predict(
        [image], frame_ids=[f"{scene_id}:{raw_frame_id}"]
    )
    assert provider.hits == 1
    assert provider.misses == 0
    assert len(replayed[0]) == 1
    np.testing.assert_array_equal(replayed[0][0].mask, mask)
    np.testing.assert_allclose(replayed[0][0].bbox, values[0].bbox)

    changed = image.copy()
    changed[0, 0, 0] = 1
    with pytest.raises(FileNotFoundError):
        provider.predict(
            [changed], frame_ids=[f"{scene_id}:{raw_frame_id}"]
        )


def test_dry_run_shards_by_scene_and_writes_audit_without_sam3(tmp_path):
    frames_root = tmp_path / "frames"
    scenes = [f"scene{index:04d}_00" for index in range(4)]
    for scene in scenes:
        _write_scene(frames_root, scene)
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    checkpoint = tmp_path / "sam3.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    sam3_root = tmp_path / "sam3-source"
    sam3_root.mkdir()
    output_dir = tmp_path / "cache"
    metadata = tmp_path / "logs" / "shard1.json"

    status = main(
        [
            "--scene-list",
            str(scene_list),
            "--frames-root",
            str(frames_root),
            "--output-dir",
            str(output_dir),
            "--namespace",
            "sam3-scannet-shard-test-v1",
            "--checkpoint",
            str(checkpoint),
            "--sam3-root",
            str(sam3_root),
            "--configured-height",
            "2",
            "--configured-width",
            "3",
            "--num-shards",
            "2",
            "--shard-index",
            "1",
            "--metadata-path",
            str(metadata),
            "--dry-run",
        ]
    )
    assert status == 0
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["shard"]["index"] == 1
    assert payload["shard"]["count"] == 2
    assert [row["scene_id"] for row in payload["frames"]] == [
        scenes[1],
        scenes[3],
    ]
    assert payload["summary"]["status_counts"] == {"dry_run": 2}
    assert list(output_dir.glob("*.npz")) == []


def test_runtime_rgb_staging_is_lossless_and_fail_closed(tmp_path):
    frames_root = tmp_path / "frames"
    scene = "scene0000_00"
    _write_scene(frames_root, scene)
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(f"{scene}\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime_rgb"
    namespace = "sam3-runtime-rgb-test-v1"

    assert main(
        [
            "--scene-list",
            str(scene_list),
            "--frames-root",
            str(frames_root),
            "--output-dir",
            str(tmp_path / "unused-cache"),
            "--runtime-rgb-dir",
            str(runtime_dir),
            "--namespace",
            namespace,
            "--configured-height",
            "2",
            "--configured-width",
            "3",
            "--stage-runtime-rgb",
        ]
    ) == 0

    manifest_path = (
        runtime_dir
        / "manifests"
        / "runtime_rgb_shard_000_of_001.json"
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["complete"] is True
    assert payload["exporter"]["opencv_version"].startswith("4.6.")
    assert payload["frames"][0]["logical_frame_id"] == f"{scene}:0"

    parser_args = type(
        "Args",
        (),
        {
            "runtime_rgb_dir": runtime_dir,
            "shard_index": 0,
            "num_shards": 1,
            "scene_list": scene_list,
            "frames_root": frames_root,
            "namespace": namespace,
            "configured_height": 2,
            "configured_width": 3,
            "gap": 25,
            "proposal_interval": 5,
            "max_frames": None,
        },
    )()
    _, _, records = load_runtime_rgb_manifest(
        args=parser_args,
        scenes=[scene],
        selected_scenes=[(0, scene)],
        full_schedule=[(scene, 0)],
        schedule=[(scene, 0)],
    )
    source = discover_scannet_scene(frames_root, scene)
    staged, orientation = load_staged_runtime_rgb(
        runtime_rgb_dir=runtime_dir,
        record=records[f"{scene}:0"],
        namespace=namespace,
        logical_frame_id=f"{scene}:0",
        source_paths={
            "color": source["color_paths"][0],
            "depth": source["depth_paths"][0],
            "pose": source["pose_paths"][0],
        },
    )
    expected, expected_orientation = load_scannet_rgb(
        source["color_paths"][0],
        source["depth_paths"][0],
        source["poses"][0],
        configured_height=2,
        configured_width=3,
    )
    assert orientation == expected_orientation
    np.testing.assert_array_equal(staged, expected)

    staged_path = Path(payload["frames"][0]["path"])
    changed = np.load(staged_path, allow_pickle=False)
    changed[0, 0, 0] ^= np.uint8(1)
    with staged_path.open("wb") as handle:
        np.save(handle, changed, allow_pickle=False)
    with pytest.raises(ValueError, match="file hash mismatch"):
        load_staged_runtime_rgb(
            runtime_rgb_dir=runtime_dir,
            record=records[f"{scene}:0"],
            namespace=namespace,
            logical_frame_id=f"{scene}:0",
            source_paths={
                "color": source["color_paths"][0],
                "depth": source["depth_paths"][0],
                "pose": source["pose_paths"][0],
            },
        )
