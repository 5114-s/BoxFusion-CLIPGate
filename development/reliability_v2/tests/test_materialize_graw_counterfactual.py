from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.materialize_graw_counterfactual import (
    AUDIT_FILENAME,
    GCLEAN_AUDIT_FILENAME,
    GCLEAN_AUDIT_SCHEMA,
    GrawCounterfactualError,
    PUF_AUDIT_FILENAME,
    PUF_AUDIT_SCHEMA,
    PUF_CANDIDATE_SOURCE,
    main,
    materialize_graw_counterfactual,
)


SCENE = "scene0001_00"


def _observer_payload():
    return {
        "schema_version": 1,
        "scene_id": SCENE,
        "mode": "shadow",
        "trace_valid": True,
        "errors": [],
        "frame_count": 3,
        "frames": [
            {
                "frame_id": 0,
                "proposal_ids": [0, 1, 2, 3, 4],
                "proposal_track_ids": [0, 1, 2, 3, 4],
                "native_status": ["unmatched_retained"] * 5,
                "begin_past_track_ids": [],
                "active_track_ids": [0, 1, 2, 3, 4],
                "track_aliases": [],
            },
            {
                "frame_id": 10,
                "proposal_ids": [10, 11, 12, 13, 14],
                "proposal_track_ids": [10, 11, 12, 13, 14],
                "native_status": ["unmatched_retained"] * 5,
                "begin_past_track_ids": [0, 1, 2, 3, 4],
                "active_track_ids": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
                "track_aliases": [],
            },
            {
                "frame_id": 20,
                "proposal_ids": [20],
                "proposal_track_ids": [None],
                "native_status": ["unmatched_dropped"],
                "begin_past_track_ids": [0, 1, 2, 3, 4, 10, 11, 12, 13, 14],
                "active_track_ids": [0, 1, 3, 4, 12, 13],
                "track_aliases": [[10, 0], [14, 13]],
            },
        ],
        "terminal": {
            "snapshot_frame_id": 20,
            "native_row_count": 6,
            "kept_native_indices": [0, 1, 2, 3, 4, 5],
            "output_track_ids": [0, 1, 3, 4, 12, 13],
        },
    }


def _graw_payload():
    associations = [
        {"proposal_id": 10, "native_track_id": 10, "past_track_id": 0},
        {"proposal_id": 11, "native_track_id": 11, "past_track_id": 1},
        {"proposal_id": 12, "native_track_id": 12, "past_track_id": 2},
        {"proposal_id": 13, "native_track_id": 13, "past_track_id": 3},
        {"proposal_id": 14, "native_track_id": 14, "past_track_id": 4},
    ]
    return {
        "schema": "boxfusion.graw_shadow.v1",
        "scene_id": SCENE,
        "trace_valid": True,
        "frame_count": 1,
        "frames": [
            {
                "frame_id": 10,
                "candidate_proposal_ids": [10, 11, 12, 13, 14],
                "candidate_native_track_ids": [10, 11, 12, 13, 14],
                "associations": associations,
            }
        ],
    }


def _gclean_payload():
    payload = copy.deepcopy(_graw_payload())
    payload["schema"] = "boxfusion.gclean_shadow.v1"
    payload["mode"] = "shadow"
    payload["fragment_source"] = "smov_clean"
    for frame in payload["frames"]:
        frame["mode"] = "shadow"
        frame["fragment_source"] = "smov_clean"
    payload["summary"] = {
        "schema": "boxfusion.gclean_shadow.v1",
        "mode": "shadow",
        "fragment_source": "smov_clean",
        "pending": False,
    }
    return payload


def _puf_payload():
    payload = copy.deepcopy(_graw_payload())
    payload.update(
        {
            "schema": "boxfusion.puf_gclean_shadow.v1",
            "mode": "shadow",
            "fragment_source": "smov_clean",
            "candidate_source": "gclean_positive_overlap_top8",
            "birth_enabled": False,
        }
    )
    for frame in payload["frames"]:
        frame.update(
            {
                "schema": "boxfusion.puf_gclean_shadow.v1",
                "mode": "shadow",
                "fragment_source": "smov_clean",
                "candidate_source": "gclean_positive_overlap_top8",
                "birth_enabled": False,
                "fail_open": False,
            }
        )
        for association in frame["associations"]:
            association.update(
                {
                    "beta_track": 0.6,
                    "beta_null": 0.4,
                    "margin": 0.2,
                    "birth_enabled": False,
                }
            )
    payload["summary"] = {
        "schema": "boxfusion.puf_gclean_shadow.v1",
        "mode": "shadow",
        "fragment_source": "smov_clean",
        "candidate_source": "gclean_positive_overlap_top8",
        "birth_enabled": False,
        "pending": False,
        "fail_open": False,
    }
    return payload


def _prediction_rows():
    rows = []
    for index, track_id in enumerate([0, 1, 3, 4, 12, 13]):
        corners = np.full((8, 3), track_id + 0.25, dtype=np.float32)
        rows.append((index % 3, corners, np.float32(0.5 + index / 20.0)))
    return rows


def _write_inputs(tmp_path: Path, observer=None, graw=None):
    observer_root = tmp_path / "observer"
    graw_root = tmp_path / "graw"
    native_root = tmp_path / "native"
    for root in (observer_root, graw_root, native_root):
        root.mkdir()
    (observer_root / f"{SCENE}.observer_tracks.json").write_text(
        json.dumps(_observer_payload() if observer is None else observer),
        encoding="utf-8",
    )
    (graw_root / f"{SCENE}.graw_shadow.json").write_text(
        json.dumps(_graw_payload() if graw is None else graw),
        encoding="utf-8",
    )
    rows = _prediction_rows()
    with (native_root / f"{SCENE}_boxes.pkl").open("wb") as handle:
        pickle.dump([rows], handle, protocol=pickle.HIGHEST_PROTOCOL)
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(SCENE + "\n", encoding="utf-8")
    return observer_root, graw_root, native_root, scene_list, rows


def _run(tmp_path: Path, observer=None, graw=None):
    observer_root, graw_root, native_root, scene_list, rows = _write_inputs(
        tmp_path, observer=observer, graw=graw
    )
    output_root = tmp_path / "output"
    audit = materialize_graw_counterfactual(
        observer_root=observer_root,
        graw_root=graw_root,
        native_prediction_root=native_root,
        output_prediction_root=output_root,
        scene_list=scene_list,
    )
    return output_root, audit, rows


def _write_gclean_inputs(tmp_path: Path, *, gclean=None):
    observer_root, shadow_root, native_root, scene_list, rows = _write_inputs(
        tmp_path
    )
    (shadow_root / f"{SCENE}.graw_shadow.json").unlink()
    (shadow_root / f"{SCENE}.gclean_shadow.json").write_text(
        json.dumps(_gclean_payload() if gclean is None else gclean),
        encoding="utf-8",
    )
    return observer_root, shadow_root, native_root, scene_list, rows


def _write_puf_inputs(tmp_path: Path, *, puf=None):
    observer_root, shadow_root, native_root, scene_list, rows = _write_inputs(
        tmp_path
    )
    (shadow_root / f"{SCENE}.graw_shadow.json").unlink()
    (shadow_root / f"{SCENE}.puf_gclean_shadow.json").write_text(
        json.dumps(_puf_payload() if puf is None else puf),
        encoding="utf-8",
    )
    return observer_root, shadow_root, native_root, scene_list, rows


def test_classifies_all_outcomes_and_deduplicates_create_only_deletion(tmp_path):
    output_root, audit, input_rows = _run(tmp_path)
    scene = audit["scenes"][0]
    assert scene["classification_counts"] == {
        "later-native-same": 1,
        "candidate-dropped": 1,
        "target-dropped": 1,
        "both-survive-distinct": 2,
    }
    assert scene["deleted_native_rows"] == [5]
    assert scene["deleted_terminal_track_ids"] == [13]
    assert audit["totals"]["deleted_row_count"] == 1
    by_candidate = {
        row["candidate_native_track_id"]: row for row in scene["associations"]
    }
    assert by_candidate[10]["classification"] == "later-native-same"
    assert by_candidate[10]["candidate_alias_path"] == [
        {"frame_id": 20, "source": 10, "target": 0}
    ]
    assert by_candidate[11]["classification"] == "candidate-dropped"
    assert by_candidate[12]["classification"] == "target-dropped"
    assert by_candidate[13]["classification"] == "both-survive-distinct"
    assert by_candidate[14]["terminal_candidate_track_id"] == 13

    with (output_root / f"{SCENE}_boxes.pkl").open("rb") as handle:
        output_rows = pickle.load(handle)[0]
    assert len(output_rows) == 5
    for actual, expected in zip(output_rows, input_rows[:5]):
        assert type(actual) is type(expected)
        assert actual[0] == expected[0] and type(actual[0]) is type(expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        assert actual[1].dtype == expected[1].dtype
        assert actual[2] == expected[2] and type(actual[2]) is type(expected[2])
    stored_audit = json.loads((output_root / AUDIT_FILENAME).read_text())
    assert stored_audit["validation"]["passed"] is True
    assert stored_audit["contract"]["retained_row_order"] == "unchanged"
    assert stored_audit["shadow_kind"] == "graw"
    assert stored_audit["fragment_source"] == "raw_depth"


def test_gclean_materializes_with_unambiguous_audit_and_same_safety_contract(
    tmp_path,
):
    observer_root, shadow_root, native_root, scene_list, input_rows = (
        _write_gclean_inputs(tmp_path)
    )
    output_root = tmp_path / "output"
    audit = materialize_graw_counterfactual(
        observer_root=observer_root,
        graw_root=shadow_root,
        native_prediction_root=native_root,
        output_prediction_root=output_root,
        scene_list=scene_list,
        shadow_kind="gclean",
    )
    assert audit["schema"] == GCLEAN_AUDIT_SCHEMA
    assert audit["shadow_kind"] == "gclean"
    assert audit["fragment_source"] == "smov_clean"
    assert audit["inputs"]["shadow_kind"] == "gclean"
    assert audit["inputs"]["fragment_source"] == "smov_clean"
    assert "graw_root" not in audit["inputs"]
    assert audit["scenes"][0]["shadow_kind"] == "gclean"
    assert audit["scenes"][0]["fragment_source"] == "smov_clean"
    assert "graw_shadow_sha256" not in audit["scenes"][0]
    assert audit["totals"]["deleted_row_count"] == 1
    assert (output_root / GCLEAN_AUDIT_FILENAME).is_file()
    assert not (output_root / AUDIT_FILENAME).exists()
    with (output_root / f"{SCENE}_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    assert len(rows) == len(input_rows) - 1


def test_puf_materializes_only_active_safe_associations_with_independent_audit(
    tmp_path,
):
    observer_root, shadow_root, native_root, scene_list, input_rows = (
        _write_puf_inputs(tmp_path)
    )
    output_root = tmp_path / "output"
    audit = materialize_graw_counterfactual(
        observer_root=observer_root,
        graw_root=shadow_root,
        native_prediction_root=native_root,
        output_prediction_root=output_root,
        scene_list=scene_list,
        shadow_kind="puf",
    )
    assert audit["schema"] == PUF_AUDIT_SCHEMA
    assert audit["shadow_kind"] == "puf"
    assert audit["fragment_source"] == "smov_clean"
    assert audit["candidate_source"] == PUF_CANDIDATE_SOURCE
    assert audit["birth_enabled"] is False
    assert audit["inputs"]["candidate_source"] == PUF_CANDIDATE_SOURCE
    assert audit["inputs"]["birth_enabled"] is False
    assert audit["scenes"][0]["candidate_source"] == PUF_CANDIDATE_SOURCE
    assert audit["scenes"][0]["birth_enabled"] is False
    assert audit["totals"]["deleted_row_count"] == 1
    first_association = audit["scenes"][0]["associations"][0]
    assert first_association["beta_track"] == pytest.approx(0.6)
    assert first_association["beta_null"] == pytest.approx(0.4)
    assert first_association["margin"] == pytest.approx(0.2)
    assert first_association["birth_enabled"] is False
    assert (output_root / PUF_AUDIT_FILENAME).is_file()
    assert not (output_root / AUDIT_FILENAME).exists()
    assert not (output_root / GCLEAN_AUDIT_FILENAME).exists()
    with (output_root / f"{SCENE}_boxes.pkl").open("rb") as handle:
        rows = pickle.load(handle)[0]
    assert len(rows) == len(input_rows) - 1


@pytest.mark.parametrize(
    ("location", "field", "value", "message"),
    [
        ("top", "schema", "boxfusion.gclean_shadow.v1", "schema"),
        ("top", "mode", "active", "mode"),
        ("top", "fragment_source", "raw_depth", "fragment_source"),
        ("top", "candidate_source", "gclean_accepted", "candidate_source"),
        ("top", "birth_enabled", True, "birth_enabled"),
        ("summary", "schema", "boxfusion.gclean_shadow.v1", "summary schema"),
        ("summary", "mode", "active", "summary mode"),
        ("summary", "fragment_source", "raw_depth", "fragment_source"),
        ("summary", "candidate_source", "gclean_accepted", "candidate_source"),
        ("summary", "birth_enabled", True, "birth_enabled"),
        ("summary", "pending", True, "pending"),
        ("summary", "fail_open", True, "fail_open"),
        ("frame", "schema", "boxfusion.gclean_shadow.v1", "frame 10 schema"),
        ("frame", "mode", "active", "frame 10 mode"),
        ("frame", "fragment_source", "raw_depth", "fragment_source"),
        ("frame", "candidate_source", "gclean_accepted", "candidate_source"),
        ("frame", "birth_enabled", True, "birth_enabled"),
        ("frame", "fail_open", True, "fail_open"),
    ],
)
def test_puf_contract_mismatch_fails_closed_before_publication(
    tmp_path, location, field, value, message
):
    payload = _puf_payload()
    target = (
        payload
        if location == "top"
        else payload["summary"]
        if location == "summary"
        else payload["frames"][0]
    )
    target[field] = value
    observer_root, shadow_root, native_root, scene_list, _ = _write_puf_inputs(
        tmp_path, puf=payload
    )
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match=message):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=shadow_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
            shadow_kind="puf",
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("beta_track", -0.01, "beta_track"),
        ("beta_track", 1.01, "beta_track"),
        ("beta_track", float("nan"), "beta_track"),
        ("beta_track", float("inf"), "beta_track"),
        ("beta_track", True, "beta_track"),
        ("beta_null", -0.01, "beta_null"),
        ("beta_null", 1.01, "beta_null"),
        ("beta_null", float("nan"), "beta_null"),
        ("margin", 0.0, "margin"),
        ("margin", -0.01, "margin"),
        ("margin", float("inf"), "margin"),
        ("margin", False, "margin"),
        ("birth_enabled", True, "birth_enabled"),
    ],
)
def test_puf_probability_or_birth_violation_fails_closed(
    tmp_path, field, value, message
):
    payload = _puf_payload()
    payload["frames"][0]["associations"][0][field] = value
    observer_root, shadow_root, native_root, scene_list, _ = _write_puf_inputs(
        tmp_path, puf=payload
    )
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match=message):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=shadow_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
            shadow_kind="puf",
        )
    assert not output_root.exists()


@pytest.mark.parametrize(
    ("beta_track", "beta_null"), [(0.4, 0.4), (0.3, 0.4)]
)
def test_puf_requires_track_probability_above_null(
    tmp_path, beta_track, beta_null
):
    payload = _puf_payload()
    association = payload["frames"][0]["associations"][0]
    association["beta_track"] = beta_track
    association["beta_null"] = beta_null
    observer_root, shadow_root, native_root, scene_list, _ = _write_puf_inputs(
        tmp_path, puf=payload
    )
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match="beta_track"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=shadow_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
            shadow_kind="puf",
        )
    assert not output_root.exists()


def test_puf_allows_candidates_without_an_active_safe_association(tmp_path):
    payload = _puf_payload()
    payload["frames"][0]["associations"].pop()
    observer_root, shadow_root, native_root, scene_list, _ = _write_puf_inputs(
        tmp_path, puf=payload
    )
    output_root = tmp_path / "output"
    audit = materialize_graw_counterfactual(
        observer_root=observer_root,
        graw_root=shadow_root,
        native_prediction_root=native_root,
        output_prediction_root=output_root,
        scene_list=scene_list,
        shadow_kind="puf",
    )
    assert audit["totals"]["association_count"] == 4


def test_puf_rejects_same_past_track_conflict_as_not_active_safe(tmp_path):
    payload = _puf_payload()
    payload["frames"][0]["associations"][1]["past_track_id"] = 0
    observer_root, shadow_root, native_root, scene_list, _ = _write_puf_inputs(
        tmp_path, puf=payload
    )
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match="duplicate past_track_id"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=shadow_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
            shadow_kind="puf",
        )
    assert not output_root.exists()


def test_puf_cli_uses_independent_audit_name(tmp_path, capsys):
    observer_root, shadow_root, native_root, scene_list, _ = _write_puf_inputs(
        tmp_path
    )
    output_root = tmp_path / "output"
    assert main(
        [
            "--observer-root",
            str(observer_root),
            "--shadow-root",
            str(shadow_root),
            "--shadow-kind",
            "puf",
            "--native-prediction-root",
            str(native_root),
            "--output-prediction-root",
            str(output_root),
            "--scene-list",
            str(scene_list),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert Path(stdout["audit"]).name == PUF_AUDIT_FILENAME
    assert (output_root / PUF_AUDIT_FILENAME).is_file()


@pytest.mark.parametrize(
    ("location", "field", "value", "message"),
    [
        ("top", "schema", "boxfusion.graw_shadow.v1", "schema"),
        ("top", "mode", "active", "mode"),
        ("top", "fragment_source", "raw_depth", "fragment_source"),
        ("summary", "schema", "boxfusion.graw_shadow.v1", "summary schema"),
        ("summary", "pending", True, "pending"),
        ("frame", "mode", "active", "frame 10 mode"),
        ("frame", "fragment_source", "raw_depth", "fragment_source"),
    ],
)
def test_gclean_format_mismatch_fails_closed_before_publication(
    tmp_path, location, field, value, message
):
    payload = _gclean_payload()
    target = (
        payload
        if location == "top"
        else payload["summary"]
        if location == "summary"
        else payload["frames"][0]
    )
    target[field] = value
    observer_root, shadow_root, native_root, scene_list, _ = (
        _write_gclean_inputs(tmp_path, gclean=payload)
    )
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match=message):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=shadow_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
            shadow_kind="gclean",
        )
    assert not output_root.exists()


def test_gclean_cli_uses_shadow_alias_and_gclean_audit_name(tmp_path, capsys):
    observer_root, shadow_root, native_root, scene_list, _ = (
        _write_gclean_inputs(tmp_path)
    )
    output_root = tmp_path / "output"
    assert main(
        [
            "--observer-root",
            str(observer_root),
            "--shadow-root",
            str(shadow_root),
            "--shadow-kind",
            "gclean",
            "--native-prediction-root",
            str(native_root),
            "--output-prediction-root",
            str(output_root),
            "--scene-list",
            str(scene_list),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert Path(stdout["audit"]).name == GCLEAN_AUDIT_FILENAME
    assert (output_root / GCLEAN_AUDIT_FILENAME).is_file()


def test_gclean_exact_scene_set_rejects_extra_diagnostic(tmp_path):
    observer_root, shadow_root, native_root, scene_list, _ = (
        _write_gclean_inputs(tmp_path)
    )
    extra = _gclean_payload()
    extra["scene_id"] = "scene9999_00"
    (shadow_root / "scene9999_00.gclean_shadow.json").write_text(
        json.dumps(extra), encoding="utf-8"
    )
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match="extra=scene9999_00"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=shadow_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
            shadow_kind="gclean",
        )
    assert not output_root.exists()


def test_unknown_programmatic_shadow_kind_fails_without_output(tmp_path):
    observer_root, graw_root, native_root, scene_list, _ = _write_inputs(tmp_path)
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match="shadow_kind"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=graw_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
            shadow_kind="Graw",
        )
    assert not output_root.exists()


@pytest.mark.parametrize("which", ["observer", "graw"])
def test_invalid_trace_fails_closed_without_publishing_output(tmp_path, which):
    observer, graw = _observer_payload(), _graw_payload()
    (observer if which == "observer" else graw)["trace_valid"] = False
    observer_root, graw_root, native_root, scene_list, _ = _write_inputs(
        tmp_path, observer=observer, graw=graw
    )
    output_root = tmp_path / "output"
    with pytest.raises(GrawCounterfactualError, match="trace_valid"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=graw_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
        )
    assert not output_root.exists()


def test_alias_cycle_fails_closed(tmp_path):
    observer = _observer_payload()
    final = observer["frames"][-1]
    final["active_track_ids"] = [0, 1, 3, 4, 10, 12, 13, 14]
    final["track_aliases"] = [[10, 14], [14, 10]]
    final["native_status"] = ["unmatched_dropped"]
    observer["terminal"] = {
        "snapshot_frame_id": 20,
        "native_row_count": 8,
        "kept_native_indices": list(range(8)),
        "output_track_ids": [0, 1, 3, 4, 10, 12, 13, 14],
    }
    observer_root, graw_root, native_root, scene_list, _ = _write_inputs(
        tmp_path, observer=observer
    )
    with pytest.raises(GrawCounterfactualError, match="cycle"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=graw_root,
            native_prediction_root=native_root,
            output_prediction_root=tmp_path / "output",
            scene_list=scene_list,
        )
    assert not (tmp_path / "output").exists()


def test_ambiguous_terminal_row_identity_and_scene_set_drift_fail_closed(tmp_path):
    observer = _observer_payload()
    observer["terminal"]["output_track_ids"][-1] = 12
    observer_root, graw_root, native_root, scene_list, _ = _write_inputs(
        tmp_path, observer=observer
    )
    with pytest.raises(GrawCounterfactualError, match="duplicate identities"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=graw_root,
            native_prediction_root=native_root,
            output_prediction_root=tmp_path / "output",
            scene_list=scene_list,
        )

    extra = copy.deepcopy(_graw_payload())
    extra["scene_id"] = "scene9999_00"
    (graw_root / "scene9999_00.graw_shadow.json").write_text(
        json.dumps(extra), encoding="utf-8"
    )
    with pytest.raises(GrawCounterfactualError, match="extra=scene9999_00"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=graw_root,
            native_prediction_root=native_root,
            output_prediction_root=tmp_path / "output2",
            scene_list=scene_list,
        )


def test_terminal_output_ids_must_align_with_kept_native_rows(tmp_path):
    observer = _observer_payload()
    observer["terminal"]["output_track_ids"][:2] = [1, 0]
    observer_root, graw_root, native_root, scene_list, _ = _write_inputs(
        tmp_path, observer=observer
    )
    with pytest.raises(GrawCounterfactualError, match="kept native rows"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=graw_root,
            native_prediction_root=native_root,
            output_prediction_root=tmp_path / "output",
            scene_list=scene_list,
        )
    assert not (tmp_path / "output").exists()


def test_cli_writes_audit_and_refuses_existing_output_root(tmp_path, capsys):
    observer_root, graw_root, native_root, scene_list, _ = _write_inputs(tmp_path)
    output_root = tmp_path / "output"
    assert main(
        [
            "--observer-root",
            str(observer_root),
            "--graw-root",
            str(graw_root),
            "--native-prediction-root",
            str(native_root),
            "--output-prediction-root",
            str(output_root),
            "--scene-list",
            str(scene_list),
        ]
    ) == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["validation"]["passed"] is True
    assert stdout["totals"]["deleted_row_count"] == 1
    with pytest.raises(GrawCounterfactualError, match="already exists"):
        materialize_graw_counterfactual(
            observer_root=observer_root,
            graw_root=graw_root,
            native_prediction_root=native_root,
            output_prediction_root=output_root,
            scene_list=scene_list,
        )
