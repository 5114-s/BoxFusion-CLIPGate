import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools.materialize_s2_yoloe_direct_shadow import (
    ARRAY_FILENAME,
    AUDIT_FILENAME,
    DEV3_SCENES,
    FROZEN_CONFIG_PATH,
    FROZEN_PREREGISTRATION_SHA256,
    SCHEMA,
    S2ShadowError,
    _QUALITY_FEATURE_NAMES,
    _build_parser,
    materialize_s2_yoloe_direct_shadow,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = REPOSITORY_ROOT / "docs" / "S2_YOLOE_DIRECT_PREREGISTRATION.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _corners(center, extent=(1.0, 1.0, 1.0)):
    center = np.asarray(center, dtype=np.float32)
    extent = np.asarray(extent, dtype=np.float32)
    lower = center - extent / 2.0
    upper = center + extent / 2.0
    return np.asarray(
        [
            [x, y, z]
            for x in (lower[0], upper[0])
            for y in (lower[1], upper[1])
            for z in (lower[2], upper[2])
        ],
        dtype=np.float32,
    )


def _summary(supplemental_count: int) -> dict:
    return {
        "active_supplemental_tracks": supplemental_count,
        "appearance_seconds": 0.0,
        "archived_supplemental_tracks": 0,
        "candidate_archived_total": 0,
        "candidate_discarded_total": 0,
        "candidate_ttl_clock": "provider_call",
        "candidate_updates": supplemental_count,
        "confirmed_supplemental_tracks": supplemental_count,
        "enabled": True,
        "geometry_seconds": 0.1,
        "global_memories": 0,
        "keyframes": 3,
        "lifted": supplemental_count,
        "matched_global": 0,
        "neural_refits_accepted": 0,
        "proposals": supplemental_count,
        "provider_calls": 3,
        "provider_seconds": 0.1,
        "refit_rejections": {},
        "refits_accepted": 0,
        "refits_attempted": 0,
        "supplemental_considered": supplemental_count,
        "supplemental_deduplicated": 0,
        "supplemental_output": supplemental_count,
        "supplemental_rejected_extent": 0,
        "supplemental_rejected_global": 0,
        "supplemental_rejected_projection": 0,
        "supplemental_rejected_score": 0,
    }


def _write_diagnostic(
    root: Path,
    scene: str,
    centers,
    scores,
    *,
    source_indices=None,
    summary_override=None,
):
    root.mkdir(parents=True, exist_ok=True)
    centers = np.asarray(centers, dtype=np.float32).reshape((-1, 3))
    scores = np.asarray(scores, dtype=np.float32)
    count = len(scores)
    if source_indices is None:
        source_indices = -np.ones(count, dtype=np.int64)
    else:
        source_indices = np.asarray(source_indices, dtype=np.int64)
    boxes = np.concatenate((centers, np.ones((count, 3), dtype=np.float32)), axis=1)
    features = np.full((count, len(_QUALITY_FEATURE_NAMES)), 0.5, dtype=np.float32)
    features[:, 0] = scores
    summary = _summary(int(np.sum(source_indices == -1)))
    if summary_override:
        summary.update(summary_override)
    np.savez_compressed(
        root / f"{scene}_tracks.npz",
        scene_id=np.asarray(scene),
        boxes=boxes,
        scores=scores,
        quality_features=features,
        points=np.zeros((count, 512, 3), dtype=np.float32),
        point_mask=np.ones((count, 512), dtype=bool),
        source_indices=source_indices,
        track_ids=np.asarray(
            [index if source_indices[index] >= 0 else -(index + 1) for index in range(count)],
            dtype=np.int64,
        ),
        result_indices=np.arange(count, dtype=np.int64),
        labels=np.asarray(["ignored semantic"] * count),
        quality_feature_names=np.asarray(_QUALITY_FEATURE_NAMES),
        summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
    )


def _fixture(tmp_path: Path):
    candidates = tmp_path / "candidates"
    baseline = tmp_path / "baseline"
    baseline.mkdir(parents=True)

    # Scene 0 exercises native novelty, stable self-NMS and the six-row cap.
    _write_diagnostic(
        candidates,
        DEV3_SCENES[0],
        [
            [10.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.4, 0.0, 0.0],
            [20.0, 0.0, 0.0],
            [30.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [50.0, 0.0, 0.0],
            [60.0, 0.0, 0.0],
            [70.0, 0.0, 0.0],
        ],
        [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55],
    )
    # Scene 1 exercises an empty native prefix.
    _write_diagnostic(
        candidates,
        DEV3_SCENES[1],
        [[1.0, 1.0, 1.0]],
        [0.70],
        # A mask without valid depth may be proposed but not lifted.
        summary_override={"proposals": 2},
    )
    # Scene 2 proves that only source_indices == -1 is materialized.
    _write_diagnostic(
        candidates,
        DEV3_SCENES[2],
        [[100.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        [0.90, 0.80],
        source_indices=[0, -1],
    )

    native_rows = {}
    for index, scene in enumerate(DEV3_SCENES):
        rows = [] if index == 1 else [(7, _corners([0.0, 0.0, 0.0]), 0.5)]
        native_rows[scene] = rows
        with (baseline / f"{scene}_boxes.pkl").open("wb") as handle:
            pickle.dump([rows], handle, protocol=pickle.HIGHEST_PROTOCOL)
    return candidates, baseline, native_rows


def _run(tmp_path: Path):
    candidates, baseline, native_rows = _fixture(tmp_path)
    output = tmp_path / "counterfactual"
    report = materialize_s2_yoloe_direct_shadow(
        candidate_root=candidates,
        baseline_root=baseline,
        preregistration=PREREGISTRATION,
        output_prediction_root=output,
    )
    return candidates, baseline, native_rows, output, report


def test_materializes_fixed_order_filters_and_exact_native_prefix(tmp_path):
    assert _sha256(PREREGISTRATION) == FROZEN_PREREGISTRATION_SHA256
    candidates, baseline, native_rows, output, report = _run(tmp_path)

    assert report["schema"] == SCHEMA
    assert report["scene_order"] == list(DEV3_SCENES)
    assert report["candidate_count"] == 8
    assert report["gt_access"] is False
    assert report["oracle_access"] is False
    assert report["birth"] is False
    assert report["score_mode_for_formal_evaluation"] == "constant_1.0"
    assert all(report["input_hash_identity"].values())

    first = report["scenes"][DEV3_SCENES[0]]
    assert [row["diagnostic_row"] for row in first["accepted_candidates"]] == [
        0,
        3,
        4,
        5,
        6,
        7,
    ]
    assert first["terminal_rejections"] == {
        "native_overlap_rejected_diagnostic_rows": [1],
        "self_nms_rejected_diagnostic_rows": [2],
        "output_cap_rejected_diagnostic_rows": [8],
    }
    assert first["native_prefix_exact"] is True
    assert all(row["formal_evaluation_score"] == 1.0 for row in first["accepted_candidates"])

    third = report["scenes"][DEV3_SCENES[2]]
    assert third["supplemental_rows_read_source_index_minus_one"] == 1
    assert [row["diagnostic_row"] for row in third["accepted_candidates"]] == [1]

    for scene in DEV3_SCENES:
        with (output / f"{scene}_boxes.pkl").open("rb") as handle:
            output_rows = pickle.load(handle)[0]
        expected = native_rows[scene]
        assert len(output_rows) == len(expected) + report["scenes"][scene][
            "accepted_candidate_count"
        ]
        for actual, frozen in zip(output_rows, expected):
            assert type(actual) is type(frozen)
            assert actual[0] == frozen[0] and type(actual[0]) is type(frozen[0])
            np.testing.assert_array_equal(actual[1], frozen[1])
            assert actual[1].dtype == frozen[1].dtype
            assert actual[2] == frozen[2] and type(actual[2]) is type(frozen[2])
        assert _sha256(baseline / f"{scene}_boxes.pkl") == report["scenes"][scene][
            "native_prediction_sha256_before"
        ]
        assert _sha256(candidates / f"{scene}_tracks.npz") == report["scenes"][scene][
            "diagnostic_sha256_before"
        ]

    stored = json.loads((output / AUDIT_FILENAME).read_text(encoding="utf-8"))
    assert stored["npz_sha256"] == _sha256(output / ARRAY_FILENAME)
    with np.load(output / ARRAY_FILENAME, allow_pickle=False) as arrays:
        assert arrays["scene_ids"].tolist() == list(DEV3_SCENES)
        assert arrays["candidate_corners_world"].shape == (8, 8, 3)
        assert np.all(arrays["candidate_formal_evaluation_score"] == 1.0)
        assert np.all(arrays["counterfactual_formal_evaluation_score"] == 1.0)
        assert arrays["candidate_diagnostic_row"].tolist()[-1] == 1


def test_outputs_are_create_only_and_candidate_npz_is_byte_deterministic(tmp_path):
    candidates, baseline, _, output_a, _ = _run(tmp_path / "a")
    output_b = tmp_path / "b" / "counterfactual"
    output_b.parent.mkdir()
    report_b = materialize_s2_yoloe_direct_shadow(
        candidate_root=candidates,
        baseline_root=baseline,
        preregistration=PREREGISTRATION,
        output_prediction_root=output_b,
    )
    assert report_b["candidate_count"] == 8
    assert (output_a / ARRAY_FILENAME).read_bytes() == (output_b / ARRAY_FILENAME).read_bytes()
    with pytest.raises(S2ShadowError, match="refusing to overwrite"):
        materialize_s2_yoloe_direct_shadow(
            candidate_root=candidates,
            baseline_root=baseline,
            preregistration=PREREGISTRATION,
            output_prediction_root=output_a,
        )


def test_fails_closed_on_diagnostic_ranking_or_summary_drift(tmp_path):
    candidates, baseline, _ = _fixture(tmp_path)
    _write_diagnostic(
        candidates,
        DEV3_SCENES[0],
        [[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        [0.50, 0.60],
    )
    with pytest.raises(S2ShadowError, match="ranking changed"):
        materialize_s2_yoloe_direct_shadow(
            candidate_root=candidates,
            baseline_root=baseline,
            preregistration=PREREGISTRATION,
            output_prediction_root=tmp_path / "ranking_output",
        )

    _write_diagnostic(
        candidates,
        DEV3_SCENES[0],
        [[10.0, 0.0, 0.0]],
        [0.60],
        summary_override={"matched_global": 1},
    )
    with pytest.raises(S2ShadowError, match="absorbed into native"):
        materialize_s2_yoloe_direct_shadow(
            candidate_root=candidates,
            baseline_root=baseline,
            preregistration=PREREGISTRATION,
            output_prediction_root=tmp_path / "summary_output",
        )


def test_fails_closed_on_config_or_preregistration_hash_drift(tmp_path):
    candidates, baseline, _ = _fixture(tmp_path)
    changed_config = tmp_path / "changed.yaml"
    changed_config.write_bytes(FROZEN_CONFIG_PATH.read_bytes() + b"\n# changed\n")
    with pytest.raises(S2ShadowError, match="config SHA-256 mismatch"):
        materialize_s2_yoloe_direct_shadow(
            candidate_root=candidates,
            baseline_root=baseline,
            preregistration=PREREGISTRATION,
            output_prediction_root=tmp_path / "config_output",
            config_path=changed_config,
        )

    changed_preregistration = tmp_path / "changed_preregistration.md"
    changed_preregistration.write_text("not the frozen preregistration\n", encoding="utf-8")
    with pytest.raises(S2ShadowError, match="preregistration SHA-256 mismatch"):
        materialize_s2_yoloe_direct_shadow(
            candidate_root=candidates,
            baseline_root=baseline,
            preregistration=changed_preregistration,
            output_prediction_root=tmp_path / "prereg_output",
        )


def test_cli_has_no_ground_truth_or_oracle_input_surface():
    option_strings = {
        option
        for action in _build_parser()._actions
        for option in action.option_strings
    }
    assert not any("ground" in option.lower() for option in option_strings)
    assert not any("oracle" in option.lower() for option in option_strings)
    assert not any(option.lower().startswith("--gt") for option in option_strings)
