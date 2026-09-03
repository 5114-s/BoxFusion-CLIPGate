from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.run_scannet_sraw_p3hb_clip_shadow_paper100 import (
    CONTRACTS,
    F0_MERGE_SCHEMA,
    F0_SCENE_SCHEMA,
    SCHEMA,
    SRAWShadowError,
    _InputLedger,
    _admit_candidates,
    _atomic_create,
    _cache_manifest_records,
    _first_three_sources,
    _geometry_summary,
    _semantic_summary,
    _sha256,
    run_shadow,
)
from tools.seal_scannet_l0_f3_f4_perview_paper100 import F4_SCHEMA
from tools.seal_scannet_l2_source_preserving_paper100 import (
    PROTOCOL_ID as L2_PROTOCOL_ID,
    SCHEMA as L2_SCHEMA,
)


def _cube(center=(0.0, 0.0, 0.0), extent=(1.0, 1.0, 1.0)):
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    signs = np.asarray(
        [
            (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
            (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
        ],
        dtype=np.float64,
    )
    return center[None] + signs * extent[None] * 0.5


def _source(source_id, center=(0, 0, 0), confidence=0.8, extent=(1, 1, 1)):
    corners = _cube(center, extent)
    lower, upper = corners.min(0), corners.max(0)
    return {
        "source_id": source_id,
        "tight_box_xyxy": [1, 1, 10, 10],
        "hypotheses": {
            "HB": {
                "valid": True,
                "world_corners": corners.tolist(),
                "world_center": np.asarray(center, dtype=float).tolist(),
                "local_extent": list(extent),
                "confidence": confidence,
            },
            "H0": {"valid": True, "q02": lower.tolist(), "q98": upper.tolist()},
        },
    }


def _candidate(track_id, frame, center, support=0.8):
    evidence_frames = [frame - 50, frame - 25, frame]
    evidence_sources = [
        f"scene0000_00/frame_{value:06d}/raw_{track_id:03d}"
        for value in evidence_frames
    ]
    return {
        "track_id": track_id,
        "confirmation_frame_id": frame,
        "evidence_source_ids": evidence_sources,
        "evidence_frame_ids": evidence_frames,
        "geometry": {
            "corners_world": _cube(center).tolist(),
            "selected_source_id": f"scene0000_00/frame_{frame:06d}/raw_{track_id:03d}",
            "medoid_mean_hb_aabb_iou": support,
            "selected_hb_confidence": 0.9,
        },
        "semantic": {
            "target_group": "chair",
            "same_target_alias_group_votes": 3,
            "all_vocab_top1_target_votes": 3,
            "median_best_target_cosine": 0.4,
            "median_target_non_target_margin": 0.1,
        },
    }


def test_first_three_are_distinct_causal_source_frames():
    scene = "scene0000_00"
    ids = [
        f"{scene}/frame_000050/raw_000",
        f"{scene}/frame_000000/raw_000",
        f"{scene}/frame_000025/raw_000",
        f"{scene}/frame_000075/raw_000",
    ]
    order = {ids[1]: 0, ids[2]: 1, ids[0]: 2, ids[3]: 3}
    sources = {source_id: {} for source_id in order}
    assert _first_three_sources(
        {"source_ids": ids}, order, sources, scene
    ) == [ids[1], ids[2], ids[0]]


def test_hb_medoid_ties_use_confidence_then_earlier_source():
    ids = [f"s{index}" for index in range(3)]
    sources = {
        ids[0]: _source(ids[0], confidence=0.7),
        ids[1]: _source(ids[1], confidence=0.9),
        ids[2]: _source(ids[2], confidence=0.9),
    }
    result = _geometry_summary(ids, sources)
    assert result["gate_pass"] is True
    assert result["selected_source_id"] == ids[1]
    assert result["selected_source_ordinal"] == 1


def test_geometry_gate_is_all_checks_not_partial():
    ids = [f"s{index}" for index in range(3)]
    sources = {source_id: _source(source_id) for source_id in ids}
    sources[ids[2]]["hypotheses"]["HB"]["confidence"] = 0.54
    sources[ids[0]]["hypotheses"]["HB"]["local_extent"] = [0.2, 1.0, 1.0]
    result = _geometry_summary(ids, sources)
    assert result["gate_pass"] is False
    assert "three_hb_confidences" in result["gate_rejection_reasons"]


@pytest.mark.parametrize(
    ("target_flags", "groups", "expected"),
    [
        ([True, True, False], ["chair", "chair", "table"], True),
        ([True, False, False], ["chair", "chair", "chair"], False),
        ([True, True, True], ["chair", "table", "sofa"], False),
    ],
)
def test_clip_three_crop_gate(target_flags, groups, expected):
    evidence = [
        {
            "all_vocab_top1_is_target": target_flags[index],
            "all_vocab_top1_target_alias_groups": (
                [groups[index]] if target_flags[index] else []
            ),
            "target_best_cosine": 0.25,
            "target_non_target_margin": 0.0,
        }
        for index in range(3)
    ]
    result = _semantic_summary(evidence)
    assert result["gate_pass"] is expected
    if expected:
        assert result["target_group"] == "chair"


def test_novelty_is_confirmation_causal_and_past_birth_only():
    candidates = [
        _candidate(0, 50, (0, 0, 0)),
        _candidate(1, 75, (0, 0, 0)),
    ]
    # The frame-75 CuTR box must not reject the frame-50 candidate.
    cutr = {75: _cube((10, 0, 0))[None]}
    accepted, decisions = _admit_candidates(candidates, cutr)
    assert [row["track_id"] for row in accepted] == [0]
    assert [row["decision"] for row in decisions] == ["accepted", "past_birth_nms"]


def test_birth_cap_is_two_after_causal_nms():
    candidates = [
        _candidate(0, 50, (0, 0, 0)),
        _candidate(1, 50, (3, 0, 0)),
        _candidate(2, 50, (6, 0, 0)),
    ]
    accepted, decisions = _admit_candidates(candidates, {})
    assert len(accepted) == 2
    assert [row["decision"] for row in decisions] == ["accepted", "accepted", "scene_cap"]


def test_cache_manifest_record_census_is_interlocked(tmp_path):
    cache_root = tmp_path / "cache" / "scene0000_00"
    cache_root.mkdir(parents=True)
    cache_path = cache_root / "frame_000000.pt"
    cache_path.write_bytes(b"sealed")
    manifest = {
        "namespace": "scannet-score05-gap25-postfilter-v2",
        "records": [{"frame_id": 0, "count": 2}],
        "record_count": 1,
        "recorded_frame_ids": [0],
        "proposal_count": 2,
    }
    manifest_path = cache_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    frames = [{"inputs": {"cutr_cache_path": str(cache_path)}}]
    schedule = {
        "namespace": "scannet-score05-gap25-postfilter-v2",
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    _, records = _cache_manifest_records(
        frames, "scene0000_00", schedule, _InputLedger()
    )
    assert records[0]["count"] == 2
    manifest["proposal_count"] = 3
    manifest_path.write_text(json.dumps(manifest))
    schedule["manifest_sha256"] = _sha256(manifest_path)
    with pytest.raises(SRAWShadowError, match="census"):
        _cache_manifest_records(
            frames, "scene0000_00", schedule, _InputLedger()
        )


def test_plan_only_validates_seals_and_exposes_exact_contracts(tmp_path, capsys):
    scene = "scene0000_00"
    source_ids = [
        f"{scene}/frame_{frame:06d}/raw_000" for frame in (0, 25, 50)
    ]
    frames = []
    for ordinal, (frame_id, source_id) in enumerate(zip((0, 25, 50), source_ids)):
        source = _source(source_id)
        source.update({"frame_id": frame_id, "frame_ordinal": ordinal})
        frames.append({"frame_id": frame_id, "frame_ordinal": ordinal, "sources": [source]})
    f4_path = tmp_path / "f4.json"
    f4_path.write_text(
        json.dumps(
            {
                "schema": F4_SCHEMA,
                "complete": True,
                "contracts": {"gt_access": False, "prediction_access": False, "evaluator_access": False},
                "frames": frames,
            }
        )
    )
    f0_scene_path = tmp_path / "f0_scene.json"
    f0_scene_path.write_text(json.dumps({"schema": F0_SCENE_SCHEMA, "complete": True, "frames": []}))
    l2_path = tmp_path / "l2.json"
    l2_path.write_text(
        json.dumps(
            {
                "schema": L2_SCHEMA,
                "protocol_id": L2_PROTOCOL_ID,
                "complete": True,
                "overall_pass": True,
                "scene_order": [scene],
                "contracts": {key: False for key in ("ground_truth_access", "annotation_access", "evaluator_access", "training", "online_learning")},
                "scenes": [{"scene_id": scene, "scene_index": 0, "f4": {"path": str(f4_path), "sha256": _sha256(f4_path)}, "f4_source_order": source_ids, "tracks": [{"track_id": 0, "source_ids": source_ids}]}],
            }
        )
    )
    f0_path = tmp_path / "f0.json"
    f0_path.write_text(
        json.dumps(
            {
                "schema": F0_MERGE_SCHEMA,
                "complete": True,
                "coverage": {"scene_order": [scene]},
                "contracts": {key: False for key in ("ground_truth_access", "annotation_access", "evaluator_access", "training", "online_learning")},
                "scenes": [{"scene_id": scene, "sidecar": {"path": str(f0_scene_path), "sha256": _sha256(f0_scene_path)}}],
            }
        )
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen\n")
    output = tmp_path / "must_not_exist.json"
    plan = run_shadow(l2_seal=l2_path, f0_manifest=f0_path, protocol_path=protocol, output_path=output, expected_scene_count=1, plan_only=True)
    assert plan["schema"] == SCHEMA
    assert plan["geometry_pass_track_count"] == 1
    assert plan["contracts"] == CONTRACTS
    assert output.exists() is False
    assert json.loads(capsys.readouterr().out)["clip_crop_count"] == 3


def test_atomic_output_refuses_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    _atomic_create(path, {"schema": SCHEMA})
    with pytest.raises(SRAWShadowError, match="overwrite"):
        _atomic_create(path, {"schema": SCHEMA})
