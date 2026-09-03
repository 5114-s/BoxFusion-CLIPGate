import json

import numpy as np
import pytest

from boxfusion.fragment_stitch import build_fragment_stitch_candidates
from tools.report_fragment_stitch_ablation import (
    FragmentRule,
    connected_components,
    pairwise_fragment_geometry,
    resolve_consistent_fragment_stitch_config,
    resolve_fragment_stitch_config_provenance,
    select_recommended_clusters,
)


def test_pairwise_fragment_geometry_reports_iou_containment_and_distance():
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            [0.5, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )

    iou, containment, distance = pairwise_fragment_geometry(boxes)

    assert np.isclose(iou[0, 1], 0.125)
    assert np.isclose(containment[0, 1], 1.0)
    assert np.isclose(distance[0, 1], 0.5)
    assert np.allclose(iou, iou.T)
    assert np.allclose(containment, containment.T)


def test_connected_components_ignores_singletons():
    edges = np.asarray(
        [
            [False, True, False, False],
            [True, False, True, False],
            [False, True, False, False],
            [False, False, False, False],
        ]
    )

    assert connected_components(edges) == [[0, 1, 2]]


def test_recommended_or_rule_accepts_either_geometry_branch():
    rule = FragmentRule(
        "recommended_or",
        "or",
        minimum_iou=0.40,
        minimum_containment=0.60,
        maximum_center_distance=0.25,
    )
    iou = np.asarray([[0.50, 0.10]])
    containment = np.asarray([[0.20, 0.70]])
    distance = np.asarray([[1.00, 0.20]])

    assert rule.edges(iou, containment, distance).tolist() == [
        [True, True]
    ]


def _three_fragment_chain():
    # A-B and A-C pass the recommended IoU branch.  B-C passes neither the
    # IoU nor the containment+center branch.
    boxes = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [0.3, 0.0, 0.0, 1.0, 1.0, 1.0],
            [-0.3, 0.0, 0.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    fragments = {
        "graph_component_track_ids": np.asarray([1, 2, 3]),
        "graph_component_states": np.asarray(
            ["active", "discarded", "discarded"]
        ),
        "graph_component_event_frames": np.asarray([0, 10, 20]),
        "graph_component_boxes": boxes,
        "graph_component_view_counts": np.asarray([1, 1, 1]),
        "graph_component_node_counts": np.asarray([1, 1, 1]),
        "graph_component_edge_counts": np.asarray([0, 0, 0]),
        "graph_component_unique_frame_counts": np.asarray([1, 1, 1]),
        "graph_component_confirmed": np.asarray([False, False, False]),
        "graph_component_mean_detector_score": np.asarray(
            [0.90, 0.85, 0.85]
        ),
        "graph_component_labels": np.asarray(
            ["Dining_Table", "dining-table", " dining table "]
        ),
        "graph_component_memory_geometry_points": np.asarray(
            [300, 200, 200]
        ),
        "boxes": np.empty((0, 6), dtype=np.float32),
        "track_ids": np.empty(0, dtype=np.int64),
        "output_is_supplemental": np.empty(0, dtype=bool),
    }
    snapshots = [
        {
            "track_id": int(fragments["graph_component_track_ids"][index]),
            "lifecycle_state": str(
                fragments["graph_component_states"][index]
            ),
            "event_frame": int(
                fragments["graph_component_event_frames"][index]
            ),
            "box": boxes[index],
            "view_count": int(
                fragments["graph_component_view_counts"][index]
            ),
            "node_count": int(
                fragments["graph_component_node_counts"][index]
            ),
            "edge_count": int(
                fragments["graph_component_edge_counts"][index]
            ),
            "memory_geometry_points": int(
                fragments[
                    "graph_component_memory_geometry_points"
                ][index]
            ),
            "mean_detector_score": float(
                fragments[
                    "graph_component_mean_detector_score"
                ][index]
            ),
            "label": str(
                fragments["graph_component_labels"][index]
            ),
            "graph_confirmed": False,
        }
        for index in range(3)
    ]
    return fragments, snapshots


def test_recommended_report_reuses_runtime_clique_without_three_way_closure():
    fragments, snapshots = _three_fragment_chain()
    runtime = build_fragment_stitch_candidates(
        snapshots,
        {
            "enabled": True,
            "minimum_pair_iou": 0.40,
            "minimum_pair_containment": 0.60,
            "maximum_center_distance": 0.25,
            "minimum_max_detector_score": 0.85,
            "minimum_mean_detector_score": 0.70,
            "minimum_event_frame_separation": 5,
            "require_live_member": True,
        },
    )
    report = select_recommended_clusters(
        fragments,
        baseline_world_boxes=np.empty((0, 6), dtype=np.float64),
    )

    assert len(runtime) == len(report) == 1
    assert runtime[0].track_ids == (1, 2)
    assert report[0]["member_track_ids"] == [1, 2]
    assert runtime[0].representative_track_id == 1
    assert report[0]["anchor_track_id"] == 1


def test_recommended_report_inherits_runtime_graph_confirmation_gate():
    fragments, _ = _three_fragment_chain()
    fragments["graph_component_confirmed"][1] = True

    report = select_recommended_clusters(
        fragments,
        baseline_world_boxes=np.empty((0, 6), dtype=np.float64),
    )

    assert report == []


def test_recommended_report_skips_one_invalid_snapshot_like_runtime():
    fragments, _ = _three_fragment_chain()
    fragments["graph_component_boxes"][2] = np.nan

    report = select_recommended_clusters(
        fragments,
        baseline_world_boxes=np.empty((0, 6), dtype=np.float64),
    )

    assert len(report) == 1
    assert report[0]["member_track_ids"] == [1, 2]


def test_recorded_custom_threshold_is_resolved_and_used():
    fragments, _ = _three_fragment_chain()
    custom = {
        "enabled": True,
        "minimum_pair_iou": 0.60,
        "minimum_pair_containment": 0.80,
        "maximum_center_distance": 0.10,
        "minimum_max_detector_score": 0.85,
        "minimum_mean_detector_score": 0.70,
        "minimum_event_frame_separation": 5,
        "require_live_member": True,
    }
    fragments["fragment_stitch_config_json"] = np.asarray(
        json.dumps(custom)
    )

    provenance = resolve_fragment_stitch_config_provenance(
        fragments, minimum_frame_gap=5
    )
    default_report = select_recommended_clusters(
        fragments,
        baseline_world_boxes=np.empty((0, 6), dtype=np.float64),
    )
    custom_report = select_recommended_clusters(
        fragments,
        baseline_world_boxes=np.empty((0, 6), dtype=np.float64),
        fragment_stitch_config=provenance["effective_config"],
    )

    assert provenance["config_source"] == "diagnostic_npz"
    assert provenance["effective_config"]["minimum_pair_iou"] == 0.60
    assert len(default_report) == 1
    assert custom_report == []


def test_legacy_config_fallback_records_effective_defaults():
    fragments, _ = _three_fragment_chain()

    provenance = resolve_fragment_stitch_config_provenance(
        fragments, minimum_frame_gap=5
    )

    assert provenance["config_source"] == "legacy_default"
    assert provenance["effective_config"]["enabled"] is True
    assert (
        provenance["effective_config"][
            "minimum_event_frame_separation"
        ]
        == 5
    )


def test_recorded_config_rejects_cli_frame_gap_conflict():
    fragments, _ = _three_fragment_chain()
    custom = {
        "enabled": True,
        "minimum_event_frame_separation": 7,
    }
    fragments["fragment_stitch_config_json"] = np.asarray(
        json.dumps(custom)
    )

    with pytest.raises(ValueError, match="CLI.*conflicts"):
        resolve_fragment_stitch_config_provenance(
            fragments, minimum_frame_gap=5
        )


def test_cross_scene_effective_config_mismatch_is_rejected():
    first, _ = _three_fragment_chain()
    second, _ = _three_fragment_chain()
    first["fragment_stitch_config_json"] = np.asarray(
        json.dumps(
            {
                "enabled": True,
                "minimum_pair_iou": 0.40,
                "minimum_event_frame_separation": 5,
            }
        )
    )
    second["fragment_stitch_config_json"] = np.asarray(
        json.dumps(
            {
                "enabled": True,
                "minimum_pair_iou": 0.45,
                "minimum_event_frame_separation": 5,
            }
        )
    )

    with pytest.raises(ValueError, match="inconsistent.*across scenes"):
        resolve_consistent_fragment_stitch_config(
            [("scene_a", first), ("scene_b", second)],
            minimum_frame_gap=5,
        )
