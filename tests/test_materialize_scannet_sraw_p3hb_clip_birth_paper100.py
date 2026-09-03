import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.materialize_scannet_sraw_p3hb_clip_birth_paper100 import (
    APPENDED_CLASS_ID,
    APPENDED_SCORE,
    MANIFEST_NAME,
    EXPECTED_PROTOCOL_SHA256,
    PROTOCOL_ID,
    SCHEMA,
    SHADOW_SCHEMA,
    SRAWBirthMaterializationError,
    _parser,
    materialize_scannet_sraw_p3hb_clip_birth_paper100,
)


SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float32,
)


def _corners(center, extent=1.0):
    extent_xyz = np.broadcast_to(np.asarray(extent, dtype=np.float32), (3,))
    return SIGNS * (extent_xyz / 2.0) + np.asarray(center, dtype=np.float32)


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _birth(track_id=1, center=(0.0, 0.0, 0.0), extent=1.0, **updates):
    confirmation = track_id * 100 + 50
    evidence_frames = [confirmation - 50, confirmation - 25, confirmation]
    evidence_sources = [
        f"scene0000_00/frame_{frame_id:06d}/raw_{track_id:03d}"
        for frame_id in evidence_frames
    ]
    selected_source_id = updates.pop("selected_source_id", evidence_sources[0])
    target_group = updates.pop("target_group", "chair")
    corners_world = updates.pop("corners_world", _corners(center, extent).tolist())
    record = {
        "track_id": track_id,
        "confirmation_frame_id": confirmation,
        "selected_source_id": selected_source_id,
        "target_group": target_group,
        "corners_world": corners_world,
        "evidence_source_ids": evidence_sources,
        "evidence_frame_ids": evidence_frames,
        "geometry": {
            "gate_pass": True,
            "selected_source_id": selected_source_id,
            "corners_world": corners_world,
        },
        "semantic": {"gate_pass": True, "target_group": target_group},
    }
    record.update(updates)
    return record


def _fixture(
    tmp_path: Path,
    births,
    native_center=(20.0, 20.0, 20.0),
    native_extent=1.0,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    native_row = (7, _corners(native_center, native_extent), np.float32(0.4321))
    native_path = baseline / f"{scene}_boxes.pkl"
    with native_path.open("wb") as handle:
        pickle.dump([[native_row]], handle, protocol=4)
    shadow = tmp_path / "shadow.json"
    shadow.write_text(
        json.dumps(
            {
                "schema": SHADOW_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "complete": True,
                "scene_count": 1,
                "scene_order": [scene],
                "accepted_birth_count": len(births),
                "inputs": {"protocol_sha256": EXPECTED_PROTOCOL_SHA256},
                "contracts": {
                    "shadow_only": True,
                    "birth_enabled": False,
                    "native_output_mutation": False,
                    "ground_truth_access": False,
                    "annotation_access": False,
                    "evaluator_access": False,
                    "future_frame_access": False,
                    "training": False,
                    "online_learning": False,
                    "past_only": True,
                },
                "scenes": [
                    {
                        "scene_id": scene,
                        "scene_index": 0,
                        "accepted_births": births,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return {
        "scene": scene,
        "scene_list": scene_list,
        "baseline": baseline,
        "native_row": native_row,
        "native_path": native_path,
        "shadow": shadow,
    }


def _run(inputs, output, plan_only=False):
    return materialize_scannet_sraw_p3hb_clip_birth_paper100(
        scene_list=inputs["scene_list"],
        baseline_root=inputs["baseline"],
        shadow_path=inputs["shadow"],
        output_root=output,
        expected_scene_count=1,
        plan_only=plan_only,
    )


def _mutate_shadow(inputs, mutation):
    payload = json.loads(inputs["shadow"].read_text(encoding="utf-8"))
    mutation(payload)
    inputs["shadow"].write_text(json.dumps(payload), encoding="utf-8")


def test_active_appends_score_one_and_preserves_exact_native_prefix(tmp_path):
    inputs = _fixture(tmp_path, [_birth()])
    output = tmp_path / "active"
    native_sha = _sha256(inputs["native_path"])
    manifest = _run(inputs, output)

    assert manifest["schema"] == SCHEMA
    assert manifest["native_count"] == 1
    assert manifest["shadow_accepted_birth_count"] == 1
    assert manifest["birth_count"] == 1
    assert manifest["strict_online_native_novelty"] is False
    assert manifest["terminal_replay_materialization"] is True
    assert manifest["native_prediction_sha256"][inputs["scene"]] == native_sha
    assert manifest["inputs"]["shadow_sha256"] == _sha256(inputs["shadow"])
    assert (output / MANIFEST_NAME).is_file()

    with (output / f"{inputs['scene']}_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    before = inputs["native_row"]
    after = rows[0]
    assert type(after) is type(before)
    assert type(after[0]) is type(before[0]) and after[0] == before[0]
    assert after[1].dtype == before[1].dtype
    assert after[1].tobytes() == before[1].tobytes()
    assert type(after[2]) is type(before[2])
    assert np.asarray(after[2]).tobytes() == np.asarray(before[2]).tobytes()
    assert rows[1][0] == APPENDED_CLASS_ID
    assert rows[1][2] == APPENDED_SCORE
    assert rows[1][1].dtype == np.float32
    np.testing.assert_allclose(rows[1][1], _corners((0.0, 0.0, 0.0)))
    assert _sha256(inputs["native_path"]) == native_sha


def test_plan_only_selects_without_creating_output(tmp_path):
    inputs = _fixture(tmp_path, [_birth()])
    output = tmp_path / "planned"
    manifest = _run(inputs, output, plan_only=True)
    assert manifest["mode"] == "plan_only"
    assert manifest["birth_count"] == 1
    assert manifest["output_prediction_sha256"] == {}
    assert not output.exists()
    assert (
        manifest["scenes"][inputs["scene"]][
            "native_prefix_row_identity_verified"
        ]
        is False
    )


def test_terminal_native_iou_and_containment_gates(tmp_path):
    overlap = _fixture(
        tmp_path / "iou", [_birth(center=(0.0, 0.0, 0.0))], native_center=(0, 0, 0)
    )
    report = _run(overlap, tmp_path / "iou-out", plan_only=True)
    scene = report["scenes"][overlap["scene"]]
    assert scene["birth_count"] == 0
    assert scene["decision_counts"]["native_overlap"] == 1

    contained = _fixture(
        tmp_path / "contain",
        [_birth(center=(0.0, 0.0, 0.0), extent=0.2)],
        native_center=(0, 0, 0),
        native_extent=2.0,
    )
    report = _run(contained, tmp_path / "contain-out", plan_only=True)
    scene = report["scenes"][contained["scene"]]
    # IoU is below .10; directed containment must still reject it.
    assert scene["terminal_decisions"][0]["max_native_aabb_iou"] < 0.10
    assert scene["decision_counts"]["native_containment"] == 1


@pytest.mark.parametrize(
    "second",
    [
        _birth(2, (0.6, 0.0, 0.0)),
        _birth(2, (0.0, 0.0, 0.0), extent=0.2),
    ],
)
def test_suffix_self_nms_uses_iou_or_either_containment_and_keeps_order(
    tmp_path, second
):
    births = [_birth(1, (0.0, 0.0, 0.0)), second]
    inputs = _fixture(tmp_path, births, native_center=(100, 100, 100))
    manifest = _run(inputs, tmp_path / "out", plan_only=True)
    scene = manifest["scenes"][inputs["scene"]]
    assert [row["track_id"] for row in scene["suffix"]] == [1]
    assert scene["decision_counts"]["self_nms"] == 1


@pytest.mark.parametrize(
    ("contract", "bad_value"),
    [
        ("ground_truth_access", True),
        ("annotation_access", True),
        ("evaluator_access", True),
        ("future_frame_access", True),
        ("training", True),
        ("online_learning", True),
        ("birth_enabled", True),
        ("past_only", False),
    ],
)
def test_rejects_forbidden_or_noncausal_shadow_contract(
    tmp_path, contract, bad_value
):
    inputs = _fixture(tmp_path, [_birth()])
    payload = json.loads(inputs["shadow"].read_text(encoding="utf-8"))
    payload["contracts"][contract] = bad_value
    inputs["shadow"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SRAWBirthMaterializationError, match=contract):
        _run(inputs, tmp_path / "out", plan_only=True)


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("protocol_id", "protocol_id"),
        ("protocol_sha256", "protocol_sha256"),
    ],
)
def test_rejects_unsealed_protocol_identity(tmp_path, field, match):
    inputs = _fixture(tmp_path, [_birth()])

    def mutate(payload):
        if field == "protocol_id":
            payload[field] = "SRAW-P3HB-CLIP-UNSEALED"
        else:
            payload["inputs"][field] = "0" * 64

    _mutate_shadow(inputs, mutate)
    with pytest.raises(SRAWBirthMaterializationError, match=match):
        _run(inputs, tmp_path / "out", plan_only=True)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda birth: birth.__setitem__(
                "evidence_source_ids", birth["evidence_source_ids"][:2]
            ),
            "exactly three",
        ),
        (
            lambda birth: birth.__setitem__(
                "evidence_frame_ids", [100, 100, 150]
            ),
            "strictly increasing",
        ),
        (
            lambda birth: birth.__setitem__(
                "evidence_frame_ids", [100, 125, 149]
            ),
            "third evidence frame",
        ),
        (
            lambda birth: birth["evidence_source_ids"].__setitem__(
                0, "scene0000_00/frame_000099/raw_001"
            ),
            "disagrees with scene/frame",
        ),
    ],
    ids=("three-sources", "strict-frame-order", "third-is-confirmation", "source-frame-pair"),
)
def test_rejects_invalid_three_view_evidence(tmp_path, mutation, match):
    inputs = _fixture(tmp_path, [_birth()])

    def mutate(payload):
        mutation(payload["scenes"][0]["accepted_births"][0])

    _mutate_shadow(inputs, mutate)
    with pytest.raises(SRAWBirthMaterializationError, match=match):
        _run(inputs, tmp_path / "out", plan_only=True)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda birth: birth["geometry"].__setitem__("gate_pass", False),
            r"geometry\.gate_pass",
        ),
        (
            lambda birth: birth["semantic"].__setitem__("gate_pass", False),
            r"semantic\.gate_pass",
        ),
        (
            lambda birth: birth["geometry"].__setitem__(
                "selected_source_id", birth["evidence_source_ids"][1]
            ),
            r"geometry\.selected_source_id",
        ),
        (
            lambda birth: birth["geometry"]["corners_world"][0].__setitem__(
                0, birth["geometry"]["corners_world"][0][0] + 0.01
            ),
            r"geometry\.corners_world disagrees",
        ),
        (
            lambda birth: birth["semantic"].__setitem__(
                "target_group", "table"
            ),
            r"semantic\.target_group",
        ),
        (
            lambda birth: birth.__setitem__(
                "target_group", "not_a_frozen_target_group"
            ),
            "TARGET_GROUP_ALIASES",
        ),
    ],
    ids=(
        "geometry-gate",
        "semantic-gate",
        "geometry-source",
        "geometry-corners",
        "semantic-group",
        "frozen-group-vocabulary",
    ),
)
def test_rejects_unsealed_geometry_or_semantic_decision(
    tmp_path, mutation, match
):
    inputs = _fixture(tmp_path, [_birth()])

    def mutate(payload):
        mutation(payload["scenes"][0]["accepted_births"][0])

    _mutate_shadow(inputs, mutate)
    with pytest.raises(SRAWBirthMaterializationError, match=match):
        _run(inputs, tmp_path / "out", plan_only=True)


def test_rejects_more_than_two_births_or_decreasing_confirmation_order(tmp_path):
    inputs = _fixture(
        tmp_path / "cap",
        [_birth(1), _birth(2), _birth(3)],
    )
    with pytest.raises(SRAWBirthMaterializationError, match="cap=2"):
        _run(inputs, tmp_path / "cap-out", plan_only=True)

    inputs = _fixture(
        tmp_path / "order",
        [_birth(2), _birth(1)],
    )
    with pytest.raises(SRAWBirthMaterializationError, match="confirmation order"):
        _run(inputs, tmp_path / "order-out", plan_only=True)


def test_rejects_schema_future_source_duplicates_and_existing_output(tmp_path):
    inputs = _fixture(tmp_path / "schema", [_birth()])
    payload = json.loads(inputs["shadow"].read_text(encoding="utf-8"))
    payload["schema"] = "boxfusion.wrong.v1"
    inputs["shadow"].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SRAWBirthMaterializationError, match="unsupported"):
        _run(inputs, tmp_path / "schema-out", plan_only=True)

    future = _birth(
        selected_source_id="scene0000_00/frame_999999/raw_001"
    )
    inputs = _fixture(tmp_path / "future", [future])
    with pytest.raises(SRAWBirthMaterializationError, match="first-three"):
        _run(inputs, tmp_path / "future-out", plan_only=True)

    inputs = _fixture(tmp_path / "duplicate", [_birth(), _birth()])
    with pytest.raises(SRAWBirthMaterializationError, match="duplicate accepted track"):
        _run(inputs, tmp_path / "duplicate-out", plan_only=True)

    inputs = _fixture(tmp_path / "overwrite", [_birth()])
    output = tmp_path / "already"
    output.mkdir()
    with pytest.raises(SRAWBirthMaterializationError, match="overwrite"):
        _run(inputs, output)


def test_full100_publishes_exact_prediction_census(tmp_path):
    scenes = [f"scene{index:04d}_00" for index in range(100)]
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes) + "\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    native = (7, _corners((20, 20, 20)), np.float32(1.0))
    for scene in scenes:
        with (baseline / f"{scene}_boxes.pkl").open("wb") as handle:
            pickle.dump([[native]], handle, protocol=4)
    shadow = tmp_path / "shadow.json"
    shadow.write_text(
        json.dumps(
            {
                "schema": SHADOW_SCHEMA,
                "protocol_id": PROTOCOL_ID,
                "complete": True,
                "scene_count": 100,
                "scene_order": scenes,
                "accepted_birth_count": 0,
                "inputs": {"protocol_sha256": EXPECTED_PROTOCOL_SHA256},
                "contracts": {
                    "shadow_only": True,
                    "birth_enabled": False,
                    "native_output_mutation": False,
                    "ground_truth_access": False,
                    "annotation_access": False,
                    "evaluator_access": False,
                    "future_frame_access": False,
                    "training": False,
                    "online_learning": False,
                    "past_only": True,
                },
                "scenes": [
                    {
                        "scene_id": scene,
                        "scene_index": index,
                        "accepted_births": [],
                    }
                    for index, scene in enumerate(scenes)
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "active100"
    manifest = materialize_scannet_sraw_p3hb_clip_birth_paper100(
        scene_list=scene_list,
        baseline_root=baseline,
        shadow_path=shadow,
        output_root=output,
    )
    assert manifest["scene_count"] == 100
    assert manifest["native_count"] == 100
    assert manifest["birth_count"] == 0
    assert len(list(output.glob("scene*_boxes.pkl"))) == 100
    assert {path.name for path in output.iterdir()} == {
        *(f"{scene}_boxes.pkl" for scene in scenes),
        MANIFEST_NAME,
    }


def test_cli_has_plan_only_and_no_gt_evaluator_or_oracle_surface():
    destinations = {action.dest for action in _parser()._actions}
    assert "plan_only" in destinations
    assert "shadow" in destinations
    assert not any(
        token in name
        for name in destinations
        for token in ("gt", "eval", "annot", "oracle")
    )
