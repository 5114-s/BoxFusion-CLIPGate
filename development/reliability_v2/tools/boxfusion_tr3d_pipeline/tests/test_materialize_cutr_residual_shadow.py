import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools.materialize_cutr_residual_shadow import materialize


PREFIX = "CuTR-residual-birth-lite shadow JSON | "
R1_PREFIX = "CuTR-residual-cross-view-R1 shadow JSON | "


def cube(center=(0.0, 0.0, 0.0)):
    center = np.asarray(center, dtype=np.float64)
    signs = np.asarray(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float64,
    )
    return center + 0.5 * signs


def setup_case(tmp_path, *, candidate=True, audit_complete=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene = "scene0000_00"
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n")
    native_root = tmp_path / "native"
    log_root = tmp_path / "logs"
    native_root.mkdir()
    log_root.mkdir()
    native = [(0, cube((0, 0, 0)).astype(np.float32), np.float32(0.6))]
    (native_root / f"{scene}_boxes.pkl").write_bytes(pickle.dumps([native]))
    candidates = []
    if candidate:
        candidates = [{
            "track_id": 7,
            "corners": cube((3, 0, 0)).tolist(),
            "appended_score": 0.2,
            "max_native_iou": 0.0,
        }]
    summary = {
        "schema": "boxfusion.cutr_residual_birth_lite_shadow.v1",
        "enabled": True,
        "observer_only": True,
        "active_authorized": False,
        "native_mutation_applied": False,
        "native_export_appended": False,
        "audit_complete": audit_complete,
        "training_free": True,
        "online_learning": False,
        "gt_access": False,
        "clip_access": False,
        "counterfactual_candidate_count": len(candidates),
        "close_result": {
            "observer_only": True,
            "active_authorized": False,
            "native_mutation_applied": False,
            "audit_complete": audit_complete,
            "candidates": candidates,
        },
    }
    (log_root / f"{scene}.log").write_text(PREFIX + json.dumps(summary) + "\n")
    return scene_list, native_root, log_root, native, summary


def run_case(tmp_path, **kwargs):
    scene_list, native_root, log_root, native, summary = setup_case(tmp_path, **kwargs)
    output = tmp_path / "out"
    manifest = tmp_path / "manifest.json"
    result = materialize(
        scene_list=scene_list,
        native_root=native_root,
        log_root=log_root,
        output_root=output,
        manifest_path=manifest,
    )
    return result, output, manifest, native, summary, log_root


def r1_summary_from_s0(summary):
    result = json.loads(json.dumps(summary))
    result.update(
        {
            "schema": "boxfusion.cutr_residual_cross_view_r1_shadow.v1",
            "cutr_descriptor_access": True,
            "descriptor_is_clip": False,
            "closed": True,
            "descriptor_dim": 256,
            "descriptor_cosine": 0.80,
            "translation_gap_m": 0.80,
            "rotation_gap_deg": 30.0,
            "depth_alpha": 0.05,
            "frame_visibility": 0.30,
            "box_visibility": 0.90,
            "min_component_nodes": 3,
            "min_component_edges": 2,
            "max_nodes_per_track": 5,
            "projection_budget_points": 8192,
            "max_receipts": 1024,
            "history_depth_frames_retained": 0,
        }
    )
    candidates = result["close_result"]["candidates"]
    ids = [row["track_id"] for row in candidates]
    result["counterfactual_candidate_track_ids"] = ids
    result["base_counterfactual_candidate_count"] = len(candidates)
    result["close_result"]["admitted_track_ids"] = ids
    result["close_result"]["rejected_track_ids"] = []
    def edge(earlier, later):
        return {
            "earlier_frame_id": earlier,
            "later_frame_id": later,
            "descriptor_cosine": 0.9,
            "translation_m": 0.9,
            "rotation_deg": 0.0,
            "ray_angle_deg": 10.0,
            "frame_visibility": 0.5,
            "depth_consistency": 0.8,
            "box_visibility": 1.0,
            "box_depth_consistency": 0.8,
            "affinity": 0.8,
            "supporting": True,
        }

    result["receipts"] = [
        {
            "track_id": track_id,
            "confirmation_frame_id": 3,
            "component_frame_ids": [1, 2, 3],
            "supporting_edge_count": 2,
            "supporting_edges": [edge(1, 2), edge(2, 3)],
        }
        for track_id in ids
    ]
    result["receipt_count"] = len(result["receipts"])
    return result


def test_materializes_create_only_with_exact_native_prefix(tmp_path):
    result, output, manifest, native, _, _ = run_case(tmp_path)
    rows = pickle.loads((output / "scene0000_00_boxes.pkl").read_bytes())[0]
    assert len(rows) == 2
    assert type(rows[0][0]) is type(native[0][0])
    assert type(rows[0][1]) is type(native[0][1])
    assert type(rows[0][2]) is type(native[0][2])
    assert np.array_equal(rows[0][1], native[0][1])
    assert rows[0][2] == native[0][2]
    assert rows[1][0] == 0 and rows[1][2] == 0.2
    assert result["native_prefix_exact"] is True
    assert json.loads(manifest.read_text())["appended_rows"] == 1


def test_existing_output_or_manifest_fails_create_only(tmp_path):
    result, output, manifest, *_ = run_case(tmp_path)
    assert result["output_rows"] == 2
    scene_list, native_root, log_root, *_ = setup_case(tmp_path / "second")
    with pytest.raises(FileExistsError):
        materialize(
            scene_list=scene_list,
            native_root=native_root,
            log_root=log_root,
            output_root=output,
            manifest_path=tmp_path / "another.json",
        )
    with pytest.raises(FileExistsError):
        materialize(
            scene_list=scene_list,
            native_root=native_root,
            log_root=log_root,
            output_root=tmp_path / "another-out",
            manifest_path=manifest,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("observer_only", False),
        ("active_authorized", True),
        ("native_export_appended", True),
        ("audit_complete", False),
        ("gt_access", True),
    ],
)
def test_rejects_unsafe_summary(tmp_path, field, value):
    scene_list, native_root, log_root, _, summary = setup_case(tmp_path)
    summary[field] = value
    (log_root / "scene0000_00.log").write_text(PREFIX + json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="unsafe"):
        materialize(
            scene_list=scene_list, native_root=native_root, log_root=log_root,
            output_root=tmp_path / "out", manifest_path=tmp_path / "m.json"
        )


def test_rejects_duplicate_summary_line_and_nonfinite_candidate(tmp_path):
    scene_list, native_root, log_root, _, summary = setup_case(tmp_path)
    path = log_root / "scene0000_00.log"
    path.write_text(path.read_text() + path.read_text())
    with pytest.raises(ValueError, match="exactly one"):
        materialize(
            scene_list=scene_list, native_root=native_root, log_root=log_root,
            output_root=tmp_path / "out", manifest_path=tmp_path / "m.json"
        )
    path.write_text(PREFIX + json.dumps(summary).replace("3.5", "NaN") + "\n")
    summary["close_result"]["candidates"][0]["corners"][0][0] = float("nan")
    path.write_text(PREFIX + json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="non-finite"):
        materialize(
            scene_list=scene_list, native_root=native_root, log_root=log_root,
            output_root=tmp_path / "out", manifest_path=tmp_path / "m.json"
        )


def test_rejects_score_or_novelty_contract_violation(tmp_path):
    scene_list, native_root, log_root, _, summary = setup_case(tmp_path)
    summary["close_result"]["candidates"][0]["appended_score"] = 0.7
    (log_root / "scene0000_00.log").write_text(PREFIX + json.dumps(summary) + "\n")
    with pytest.raises(ValueError, match="strictly below"):
        materialize(
            scene_list=scene_list, native_root=native_root, log_root=log_root,
            output_root=tmp_path / "out", manifest_path=tmp_path / "m.json"
        )


def test_empty_candidate_list_is_valid_and_native_only(tmp_path):
    result, output, _, native, *_ = run_case(tmp_path, candidate=False)
    rows = pickle.loads((output / "scene0000_00_boxes.pkl").read_bytes())[0]
    assert result["appended_rows"] == 0
    assert len(rows) == len(native)


def test_r1_variant_reads_only_r1_summary_and_materializes_subset(tmp_path):
    scene_list, native_root, log_root, native, summary = setup_case(tmp_path)
    r1_summary = r1_summary_from_s0(summary)
    (log_root / "scene0000_00.log").write_text(
        PREFIX
        + json.dumps(summary)
        + "\n"
        + R1_PREFIX
        + json.dumps(r1_summary)
        + "\n"
    )
    output = tmp_path / "r1-out"
    result = materialize(
        scene_list=scene_list,
        native_root=native_root,
        log_root=log_root,
        output_root=output,
        manifest_path=tmp_path / "r1-manifest.json",
        variant="r1",
    )
    rows = pickle.loads((output / "scene0000_00_boxes.pkl").read_bytes())[0]
    assert result["schema"] == (
        "boxfusion.cutr_residual_cross_view_r1_materialization.v1"
    )
    assert result["variant"] == "r1"
    assert np.array_equal(rows[0][1], native[0][1])
    assert len(rows) == 2


def test_r1_variant_rejects_clip_descriptor_claim(tmp_path):
    scene_list, native_root, log_root, _, summary = setup_case(tmp_path)
    r1_summary = r1_summary_from_s0(summary)
    r1_summary["descriptor_is_clip"] = True
    (log_root / "scene0000_00.log").write_text(
        PREFIX
        + json.dumps(summary)
        + "\n"
        + R1_PREFIX
        + json.dumps(r1_summary)
        + "\n"
    )
    with pytest.raises(ValueError, match="descriptor_is_clip"):
        materialize(
            scene_list=scene_list,
            native_root=native_root,
            log_root=log_root,
            output_root=tmp_path / "r1-out",
            manifest_path=tmp_path / "r1-manifest.json",
            variant="r1",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("closed", False),
        ("descriptor_dim", 128),
        ("descriptor_cosine", 0.79),
        ("translation_gap_m", 0.7),
        ("rotation_gap_deg", 29.0),
        ("depth_alpha", 0.1),
        ("frame_visibility", 0.2),
        ("box_visibility", 0.8),
        ("min_component_nodes", 2),
        ("min_component_edges", 1),
        ("max_nodes_per_track", 6),
        ("projection_budget_points", 9000),
        ("max_receipts", 999),
        ("history_depth_frames_retained", 1),
    ],
)
def test_r1_variant_rejects_frozen_contract_drift(tmp_path, field, value):
    scene_list, native_root, log_root, _, summary = setup_case(tmp_path)
    r1_summary = r1_summary_from_s0(summary)
    r1_summary[field] = value
    (log_root / "scene0000_00.log").write_text(
        PREFIX + json.dumps(summary) + "\n"
        + R1_PREFIX + json.dumps(r1_summary) + "\n"
    )
    with pytest.raises(ValueError, match="R1 field"):
        materialize(
            scene_list=scene_list,
            native_root=native_root,
            log_root=log_root,
            output_root=tmp_path / "r1-out",
            manifest_path=tmp_path / "r1-manifest.json",
            variant="r1",
        )


@pytest.mark.parametrize(
    "mutation",
    ("top_ids", "admitted_ids", "base_count", "rejected_ids", "candidate_body"),
)
def test_r1_variant_rejects_non_subset_or_inconsistent_ids(tmp_path, mutation):
    scene_list, native_root, log_root, _, summary = setup_case(tmp_path)
    r1_summary = r1_summary_from_s0(summary)
    if mutation == "top_ids":
        r1_summary["counterfactual_candidate_track_ids"] = [999]
    elif mutation == "admitted_ids":
        r1_summary["close_result"]["admitted_track_ids"] = []
    elif mutation == "base_count":
        r1_summary["base_counterfactual_candidate_count"] = 0
    elif mutation == "rejected_ids":
        r1_summary["close_result"]["rejected_track_ids"] = [8]
    else:
        r1_summary["close_result"]["candidates"][0]["appended_score"] = 0.1
    (log_root / "scene0000_00.log").write_text(
        PREFIX + json.dumps(summary) + "\n"
        + R1_PREFIX + json.dumps(r1_summary) + "\n"
    )
    with pytest.raises(ValueError):
        materialize(
            scene_list=scene_list,
            native_root=native_root,
            log_root=log_root,
            output_root=tmp_path / "r1-out",
            manifest_path=tmp_path / "r1-manifest.json",
            variant="r1",
        )


@pytest.mark.parametrize(
    "mutation",
    ("empty_edges", "bad_count", "disconnected", "weak_cosine", "rejected_receipt"),
)
def test_r1_variant_rejects_invalid_or_fake_receipt(tmp_path, mutation):
    scene_list, native_root, log_root, _, summary = setup_case(tmp_path)
    r1_summary = r1_summary_from_s0(summary)
    receipt = r1_summary["receipts"][0]
    if mutation == "empty_edges":
        receipt["supporting_edges"] = []
    elif mutation == "bad_count":
        receipt["supporting_edge_count"] = 999
    elif mutation == "disconnected":
        receipt["component_frame_ids"] = [1, 2, 3, 4]
        receipt["confirmation_frame_id"] = 4
    elif mutation == "weak_cosine":
        receipt["supporting_edges"][0]["descriptor_cosine"] = 0.79
    else:
        r1_summary["close_result"]["admitted_track_ids"] = []
        r1_summary["close_result"]["rejected_track_ids"] = [7]
        r1_summary["counterfactual_candidate_track_ids"] = []
        r1_summary["counterfactual_candidate_count"] = 0
        r1_summary["close_result"]["candidates"] = []
    (log_root / "scene0000_00.log").write_text(
        PREFIX + json.dumps(summary) + "\n"
        + R1_PREFIX + json.dumps(r1_summary) + "\n"
    )
    with pytest.raises(ValueError):
        materialize(
            scene_list=scene_list,
            native_root=native_root,
            log_root=log_root,
            output_root=tmp_path / "r1-out",
            manifest_path=tmp_path / "r1-manifest.json",
            variant="r1",
        )


def test_r1_zero_candidate_is_valid_exact_subset_of_nonempty_s0(tmp_path):
    scene_list, native_root, log_root, native, summary = setup_case(tmp_path)
    r1_summary = r1_summary_from_s0(summary)
    r1_summary["counterfactual_candidate_count"] = 0
    r1_summary["counterfactual_candidate_track_ids"] = []
    r1_summary["close_result"]["candidates"] = []
    r1_summary["close_result"]["admitted_track_ids"] = []
    r1_summary["close_result"]["rejected_track_ids"] = [7]
    r1_summary["receipts"] = []
    r1_summary["receipt_count"] = 0
    (log_root / "scene0000_00.log").write_text(
        PREFIX + json.dumps(summary) + "\n"
        + R1_PREFIX + json.dumps(r1_summary) + "\n"
    )
    output = tmp_path / "r1-zero-out"
    result = materialize(
        scene_list=scene_list,
        native_root=native_root,
        log_root=log_root,
        output_root=output,
        manifest_path=tmp_path / "r1-zero-manifest.json",
        variant="r1",
    )
    rows = pickle.loads((output / "scene0000_00_boxes.pkl").read_bytes())[0]
    assert result["appended_rows"] == 0
    assert len(rows) == len(native)
    assert np.array_equal(rows[0][1], native[0][1])
