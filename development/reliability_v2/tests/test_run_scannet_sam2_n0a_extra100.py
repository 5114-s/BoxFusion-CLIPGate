from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys
import warnings

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import run_scannet_sam2_n0a_extra100 as runner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _emit_expected_warnings() -> None:
    for line in runner.EXPECTED_WARNING_LINES:
        warnings.warn_explicit(
            runner.EXPECTED_WARNING_MESSAGE,
            UserWarning,
            str(runner.EXPECTED_WARNING_SOURCE_PATH),
            line,
        )


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _reseal_receipt(path: Path, mutate) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="ascii"))
    mutate(value)
    value.pop("content_sha256", None)
    value["content_sha256"] = runner._canonical_json_sha256(value)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )
    return value


def _candidate(rank: int) -> dict[str, object]:
    box = [80 + 140 * rank, 80, 179 + 140 * rank, 179]
    return {
        "rank": rank,
        "raw_index": 7 - 4 * rank,
        "mask_sha256": hashlib.sha256(f"mask-{rank}".encode()).hexdigest(),
        "points_and_voxel_keys_sha256": hashlib.sha256(
            f"points-{rank}".encode()
        ).hexdigest(),
        "tight_box_xyxy": box,
        "world_q02": [-1.0 + rank, -1.0, 0.5],
        "world_q98": [1.0 + rank, 1.0, 1.5],
        "world_center": [float(rank), 0.0, 1.0],
        "world_extent": [2.0, 2.0, 1.0],
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    scenes = ("scene9000_00", "scene9001_00")
    intrinsic = tmp_path / "intrinsic_depth.txt"
    intrinsic.write_text(
        "100 0 320 0\n0 100 240 0\n0 0 1 0\n0 0 0 1\n", encoding="ascii"
    )
    merged_rows = []
    for scene_index, scene_id in enumerate(scenes):
        inputs_dir = tmp_path / "inputs" / scene_id
        inputs_dir.mkdir(parents=True)
        rgb = inputs_dir / "0.jpg"
        depth = inputs_dir / "0.png"
        pose = inputs_dir / "0.txt"
        rgb.write_bytes(b"sealed-rgb-" + scene_id.encode())
        depth.write_bytes(b"sealed-depth-" + scene_id.encode())
        pose.write_text("1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="ascii")
        candidates = [_candidate(0), _candidate(1)]
        diagnostics = [
            {
                "rank": row["rank"],
                "raw_index": row["raw_index"],
                "mask_sha256": row["mask_sha256"],
                "tight_box_xyxy": row["tight_box_xyxy"],
                "selected": True,
                "decision": "selected",
            }
            for row in candidates
        ]
        frame = {
            "frame_ordinal": 0,
            "frame_id": 0,
            "successful": True,
            "inputs": {
                "current_pose_valid": True,
                "f0_pose_forward_filled": False,
                "producer_orientation": 0,
                "producer_rotation_k": 0,
                "producer_depth_shape": [480, 640],
                "producer_image_shape": [480, 640, 3],
                "rgb_path": str(rgb.resolve()),
                "rgb_sha256": _sha256(rgb),
                "depth_path": str(depth.resolve()),
                "depth_sha256": _sha256(depth),
                "pose_path": str(pose.resolve()),
                "pose_sha256": _sha256(pose),
            },
            "funnel": {"candidates": candidates, "masks": diagnostics},
            "runtime": {"complete_ms": 2.0},
        }
        sidecar = tmp_path / "f0" / f"{scene_id}.json"
        _write_json(
            sidecar,
            {
                "schema": runner.EXPECTED_F0_SCENE_SCHEMA,
                "protocol_id": runner.EXPECTED_F0_PROTOCOL,
                "complete": True,
                "scene_id": scene_id,
                "scene_index": scene_index,
                "intrinsic": {
                    "path": str(intrinsic.resolve()),
                    "sha256": _sha256(intrinsic),
                },
                "frames": [frame],
            },
        )
        merged_rows.append(
            {
                "scene_id": scene_id,
                "scene_index": scene_index,
                "sidecar": {
                    "path": str(sidecar.resolve()),
                    "sha256": _sha256(sidecar),
                },
            }
        )
    scene_list = tmp_path / "full200.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    receipt = tmp_path / "F0.json"
    _write_json(
        receipt,
        {
            "schema": runner.EXPECTED_F0_SCHEMA,
            "protocol_id": runner.EXPECTED_F0_PROTOCOL,
            "complete": True,
            "overall_pass": True,
            "coverage": {"scene_count": len(scenes)},
            "scenes": merged_rows,
            "run_signature_sha256": hashlib.sha256(b"fixture").hexdigest(),
        },
    )
    return {
        "scenes": scenes,
        "receipt": receipt,
        "scene_list": scene_list,
        "rows": merged_rows,
        "output": tmp_path / "n0a",
    }


class _FakeProvider:
    def __init__(self, calls: list[np.ndarray]) -> None:
        self.calls = calls

    def predict(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> object:
        assert image_rgb.shape == (480, 640, 3)
        assert image_rgb.dtype == np.uint8
        self.calls.append(boxes_xyxy.copy())
        _emit_expected_warnings()
        count = len(boxes_xyxy)
        masks = np.zeros((count, 480, 640), dtype=bool)
        for index, box in enumerate(boxes_xyxy.astype(int)):
            x0, y0, x1, y1 = box
            masks[index, y0 : y1 + 1, x0 : x1 + 1] = True
        all_ious = np.tile(
            np.asarray([[0.1, 0.9, 0.2]], dtype=np.float32), (count, 1)
        )
        return SimpleNamespace(
            masks=masks,
            selected_hypothesis_indices=np.ones(count, dtype=np.int64),
            predicted_ious=all_ious[:, 1].copy(),
            all_predicted_ious=all_ious,
            timing=SimpleNamespace(
                encoder_ms=1.0,
                decoder_and_host_mask_ms=2.0,
                complete_ms=3.0,
                cuda_synchronized=False,
                peak_allocated_memory_bytes=0,
            ),
        )


def _frame_loader(
    _rgb_path: Path, _depth_path: Path, _pose_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.ones((480, 640), dtype=np.float64),
        np.eye(4, dtype=np.float64),
    )


def _run(data: dict[str, object], calls: list[np.ndarray], **overrides):
    provider = _FakeProvider(calls)

    def provider_factory() -> _FakeProvider:
        return provider

    values = {
        "f0_receipt_path": data["receipt"],
        "full200_scene_list_path": data["scene_list"],
        "output_root": data["output"],
        "shard_index": 0,
        "num_shards": 1,
        "cohort_start": 0,
        "expected_scene_count": 2,
        "expected_keyframes": 2,
        "expected_successful_frames": 2,
        "expected_sources": 4,
        "provider_factory": provider_factory,
        "frame_loader": _frame_loader,
    }
    values.update(overrides)
    return runner.run_n0a(**values)


def test_fake_provider_end_to_end_packed_evidence_create_only_and_resume(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)
    calls: list[np.ndarray] = []
    manifest = _run(data, calls)

    assert manifest["complete"] is True
    assert manifest["totals"]["source_count"] == 4
    assert manifest["totals"]["provider_forward_count"] == 2
    assert manifest["totals"]["authenticated_warning_count"] == 4
    assert manifest["warning_policy"]["policy_id"] == runner.WARNING_POLICY_ID
    assert manifest["native_output_mutation_count"] == 0
    assert manifest["contracts"]["ground_truth_access"] is False
    assert len(calls) == 2
    excluded = manifest["excluded_runtime_reporting"]
    assert excluded["included_in_online_or_warm_distributions"] is False
    assert excluded["cold_model_load_is_combined_with_first_forward"] is True
    assert manifest["unsealed_return_only_runtime"][
        "shard_manifest_json_serialization_write_ms"
    ] >= 0.0
    scene_aggregate = excluded["scene_aggregate_ms"]
    for key in runner.SCENE_EXCLUDED_RUNTIME_KEYS:
        assert scene_aggregate[key] >= 0.0
        assert scene_aggregate[key] == pytest.approx(
            sum(row["excluded_runtime_reporting"][key] for row in manifest["scenes"])
        )
    assert excluded["input_pre_rehash_total_ms"] == pytest.approx(
        excluded["sealed_universe_pre_authentication_ms"]
        + scene_aggregate["input_pre_rehash_ms"]
    )
    assert excluded["input_end_rehash_total_ms"] == pytest.approx(
        excluded["global_input_end_rehash_ms"]
        + scene_aggregate["input_end_rehash_ms"]
    )
    np.testing.assert_array_equal(
        calls[0],
        np.asarray([[80, 80, 179, 179], [220, 80, 319, 179]], dtype=np.float32),
    )

    scene_path = Path(manifest["scenes"][0]["sidecar"]["path"])
    scene = json.loads(scene_path.read_text(encoding="ascii"))
    sources = scene["frames"][0]["sources"]
    assert [row["source_id"] for row in sources] == [
        "scene9000_00/frame_000000/raw_007",
        "scene9000_00/frame_000000/raw_003",
    ]
    assert scene["frames"][0]["runtime"]["sam2_provider_timing"] == {
        "encoder_ms": 1.0,
        "decoder_and_host_mask_ms": 2.0,
        "complete_ms": 3.0,
        "cuda_synchronized": False,
        "peak_allocated_memory_bytes": 0,
    }
    assert scene["frames"][0]["authenticated_warning_count"] == 2
    assert scene["frames"][0]["runtime"]["deterministic_warning_evidence"] == (
        runner._warning_evidence_receipt()
    )
    assert scene["counts"]["authenticated_warning_count"] == 2
    evidence_path = Path(scene["evidence_npz"]["path"])
    with np.load(evidence_path, allow_pickle=False) as evidence:
        assert evidence["mask_packbits"].shape == (2, runner.MASK_PACKED_BYTES)
        np.testing.assert_array_equal(evidence["point_offsets"].shape, (3,))
        np.testing.assert_array_equal(
            evidence["selected_hypothesis_indices"], [1, 1]
        )
        assert evidence["points_world"].shape[1:] == (3,)
        assert int(evidence["point_offsets"][-1]) <= 2 * 2048

    resumed = _run(data, calls)
    assert resumed["resumed_complete"] is True
    assert len(calls) == 2

    create_only = tmp_path / "create-only.json"
    runner._atomic_create_json(create_only, {"a": 1})
    with pytest.raises(runner.N0ARunnerError, match="refusing to overwrite"):
        runner._atomic_create_json(create_only, {"a": 2})


def test_complete_universe_is_authenticated_before_modulo_sharding_and_tamper(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)
    plan = _run(
        data,
        [],
        shard_index=1,
        num_shards=2,
        plan_only=True,
    )
    assert plan["full_universe_authenticated"] is True
    assert plan["universe_census"] == {
        "scene_count": 2,
        "keyframe_count": 2,
        "successful_frame_count": 2,
        "source_count": 4,
        "provider_forward_count": 2,
        "successful_empty_frame_count": 0,
    }
    assert plan["scene_indices"] == [1]
    assert plan["scene_ids"] == ["scene9001_00"]

    first_sidecar = Path(data["rows"][0]["sidecar"]["path"])
    first_sidecar.write_bytes(first_sidecar.read_bytes() + b"\n")
    with pytest.raises(runner.N0ARunnerError, match="rehash differs"):
        _run(data, [], shard_index=1, num_shards=2, plan_only=True)


def test_malformed_provider_is_fail_closed_before_any_scene_output(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)

    class BadProvider:
        def predict(self, _image_rgb, boxes_xyxy):
            _emit_expected_warnings()
            count = len(boxes_xyxy)
            return SimpleNamespace(
                masks=np.zeros((count, 480, 640), dtype=bool),
                selected_hypothesis_indices=np.zeros(count, dtype=np.int64),
                predicted_ious=np.zeros(count, dtype=np.float32),
                all_predicted_ious=np.zeros((count, 3), dtype=np.float32),
                timing=None,
            )

    bad = BadProvider()

    def factory():
        return bad

    with pytest.raises(runner.N0ARunnerError, match="timing contract"):
        runner.run_n0a(
            f0_receipt_path=data["receipt"],
            full200_scene_list_path=data["scene_list"],
            output_root=data["output"],
            shard_index=0,
            num_shards=1,
            cohort_start=0,
            expected_scene_count=2,
            expected_keyframes=2,
            expected_successful_frames=2,
            expected_sources=4,
            provider_factory=factory,
            frame_loader=_frame_loader,
        )
    assert not (Path(data["output"]) / "scenes").exists()


def test_default_loader_converts_native_bgr_then_resizes_to_depth_grid(
    tmp_path: Path,
) -> None:
    cv2 = pytest.importorskip("cv2")
    rgb_path = tmp_path / "native.jpg"
    depth_path = tmp_path / "depth.png"
    pose_path = tmp_path / "pose.txt"
    bgr = np.empty((968, 1296, 3), dtype=np.uint8)
    bgr[:, :, 0] = 10
    bgr[:, :, 1] = 20
    bgr[:, :, 2] = 30
    assert cv2.imwrite(str(rgb_path), bgr)
    assert cv2.imwrite(
        str(depth_path), np.full((480, 640), 1_250, dtype=np.uint16)
    )
    pose_path.write_text(
        "1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n", encoding="ascii"
    )

    rgb, depth_m, pose = runner._default_frame_loader(
        rgb_path, depth_path, pose_path
    )

    assert rgb.shape == (480, 640, 3)
    assert rgb.dtype == np.uint8
    np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
    np.testing.assert_allclose(depth_m, 1.25)
    np.testing.assert_array_equal(pose, np.eye(4, dtype=np.float64))


def _warning_row(
    line: int,
    *,
    message: Warning | None = None,
    category: type[Warning] = UserWarning,
    filename: str | None = None,
) -> warnings.WarningMessage:
    return warnings.WarningMessage(
        message=(
            UserWarning(runner.EXPECTED_WARNING_MESSAGE)
            if message is None
            else message
        ),
        category=category,
        filename=filename or str(runner.EXPECTED_WARNING_SOURCE_PATH),
        lineno=line,
    )


def test_exact_warning_pair_and_compact_evidence() -> None:
    source = runner._warning_policy_source()
    rows = [_warning_row(143), _warning_row(144)]
    evidence = runner._validate_forward_warnings(rows, source=source)

    assert evidence == runner._warning_evidence_receipt()
    encoded = json.dumps(evidence, separators=(",", ":")).encode("ascii")
    assert len(encoded) < 384
    assert runner.EXPECTED_WARNING_MESSAGE.encode("ascii") not in encoded


@pytest.mark.parametrize(
    "rows,error",
    [
        ([], "exactly two"),
        ([_warning_row(143)], "exactly two"),
        ([_warning_row(143), _warning_row(144), _warning_row(144)], "exactly two"),
        ([_warning_row(143), _warning_row(143)], "source line"),
        ([_warning_row(144), _warning_row(143)], "source line"),
        (
            [_warning_row(143, category=RuntimeWarning), _warning_row(144)],
            "category or message type",
        ),
        (
            [
                _warning_row(143, message=RuntimeWarning(runner.EXPECTED_WARNING_MESSAGE)),
                _warning_row(144),
            ],
            "category or message type",
        ),
        (
            [_warning_row(143, message=UserWarning("prefix " + runner.EXPECTED_WARNING_MESSAGE)), _warning_row(144)],
            "message differs",
        ),
        (
            [_warning_row(143, filename="/tmp/position_encoding.py"), _warning_row(144)],
            "source path",
        ),
        (
            [_warning_row(143), _warning_row(144),],
            None,
        ),
    ],
)
def test_warning_policy_rejects_every_non_exact_tuple(rows, error) -> None:
    source = runner._warning_policy_source()
    if error is None:
        assert runner._validate_forward_warnings(rows, source=source)["count"] == 2
    else:
        with pytest.raises(runner.N0ARunnerError, match=error):
            runner._validate_forward_warnings(rows, source=source)


def test_scene_warning_distribution_cannot_hide_zero_plus_four() -> None:
    policy = runner._warning_policy_receipt(runner._warning_policy_source())
    evidence = runner._warning_evidence_receipt()
    receipt = {
        "warning_policy": policy,
        "frames": [
            {
                "provider_invoked": True,
                "authenticated_warning_count": 0,
                "runtime": {"deterministic_warning_evidence": evidence},
            },
            {
                "provider_invoked": True,
                "authenticated_warning_count": 4,
                "runtime": {"deterministic_warning_evidence": evidence},
            },
        ],
        "counts": {
            "provider_forward_count": 2,
            "authenticated_warning_count": 4,
        },
    }
    with pytest.raises(runner.N0ARunnerError, match="per-frame warning count"):
        runner._validate_scene_warning_evidence(receipt, warning_policy=policy)


def test_completed_manifest_rejects_empty_scene_list_and_recomputed_self_hash(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)
    _run(data, [])
    manifest_path = Path(data["output"]) / "shards/shard-000-of-001.json"

    def mutate(value):
        value["scenes"] = []
        value["totals"] = {key: 0 for key in runner.TOTAL_COUNT_KEYS}
        value["expected_shard_census"] = {
            key: 0 for key in runner.SEALED_CENSUS_KEYS
        }
        value["excluded_runtime_reporting"]["scene_aggregate_ms"] = {
            key: 0.0 for key in runner.SCENE_EXCLUDED_RUNTIME_KEYS
        }

    _reseal_receipt(manifest_path, mutate)
    with pytest.raises(runner.N0ARunnerError, match="scene census"):
        _run(data, [])


def test_completed_manifest_rejects_cross_root_scene_references(tmp_path: Path) -> None:
    data = _fixture(tmp_path)
    _run(data, [])
    source_manifest = Path(data["output"]) / "shards/shard-000-of-001.json"
    copied_root = tmp_path / "copied-root"
    copied_manifest = copied_root / "shards/shard-000-of-001.json"
    copied_manifest.parent.mkdir(parents=True)
    shutil.copyfile(source_manifest, copied_manifest)
    copied_data = dict(data)
    copied_data["output"] = copied_root

    with pytest.raises(runner.N0ARunnerError, match="cross-root"):
        _run(copied_data, [])


def test_completed_manifest_rejects_reordered_scenes_and_wrong_totals(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)
    _run(data, [])
    manifest_path = Path(data["output"]) / "shards/shard-000-of-001.json"

    _reseal_receipt(manifest_path, lambda value: value["scenes"].reverse())
    with pytest.raises(runner.N0ARunnerError, match="identity/order"):
        _run(data, [])

    totals_root = tmp_path / "totals"
    totals_root.mkdir()
    _run_data = _fixture(totals_root)
    _run(_run_data, [])
    totals_manifest = Path(_run_data["output"]) / "shards/shard-000-of-001.json"

    def mutate_totals(value):
        value["totals"]["valid_hs_count"] += 1

    _reseal_receipt(totals_manifest, mutate_totals)
    with pytest.raises(runner.N0ARunnerError, match="totals do not recompute"):
        _run(_run_data, [])


def test_completed_manifest_revalidates_nested_frame_warning_distribution(
    tmp_path: Path,
) -> None:
    data = _fixture(tmp_path)
    manifest = _run(data, [])
    manifest_path = Path(data["output"]) / "shards/shard-000-of-001.json"
    scene_path = Path(manifest["scenes"][0]["sidecar"]["path"])

    def mutate_scene(value):
        value["frames"][0]["authenticated_warning_count"] = 0

    _reseal_receipt(scene_path, mutate_scene)

    def mutate_manifest(value):
        value["scenes"][0]["sidecar"]["sha256"] = _sha256(scene_path)

    _reseal_receipt(manifest_path, mutate_manifest)
    with pytest.raises(runner.N0ARunnerError, match="per-frame warning count"):
        _run(data, [])


def test_production_output_root_rejects_invalid_v1_and_nonempty_no_resume(
    tmp_path: Path,
) -> None:
    invalid_alias = runner.PERMANENTLY_INVALID_V1_OUTPUT_ROOT / "nested-v2"
    with pytest.raises(runner.N0ARunnerError, match="permanently invalid"):
        runner._validate_production_output_root(invalid_alias, resume=True)

    nonempty = tmp_path / "nonempty"
    (nonempty / "scenes").mkdir(parents=True)
    (nonempty / "scenes" / "orphan.json").write_text("{}", encoding="ascii")
    with pytest.raises(runner.N0ARunnerError, match="new empty v2"):
        runner._validate_production_output_root(nonempty, resume=False)

    empty = tmp_path / "empty"
    empty.mkdir()
    assert runner._validate_production_output_root(empty, resume=False) == empty.resolve()
