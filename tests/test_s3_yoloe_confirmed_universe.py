import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import tools.export_s3_yoloe_confirmed_universe as exporter
from tools.audit_scannet_boxer_unexplained_oracle import official_constant_evaluate
from tools.audit_scannet_s3_yoloe_confirmed_universe import (
    _geometry_report,
    _point_quantile_minmax,
)


class _Memory:
    def __init__(self, track_id, center):
        self.track_id = track_id
        self._center = np.asarray(center, dtype=np.float32)
        self._points = self._center + np.asarray(
            [[-0.2, -0.2, -0.2], [-0.2, 0.2, 0.2], [0.2, -0.2, 0.2], [0.2, 0.2, -0.2]],
            dtype=np.float32,
        )
        self.observation_count = 3
        self.unique_view_count = 3

    @property
    def aabb(self):
        return self._center.copy(), np.ones(3, dtype=np.float32)

    @property
    def points(self):
        return self._points.copy()


def _track(track_id, center, created):
    return SimpleNamespace(
        track_id=track_id,
        memory=_Memory(track_id, center),
        created_frame=created,
        last_frame=1,
        created_lifecycle_step=created,
        last_lifecycle_step=1,
        hit_count=3,
        view_count=3,
        confirmed=True,
    )


class _Manager:
    def __init__(self, tracks):
        self._tracks = tuple(tracks)
        self.archived_tracks = {tracks[-1].track_id: tracks[-1]}

    def confirmed_tracks(self, include_archived=False):
        assert include_archived is True
        return self._tracks


def _fixture():
    tracks = (_track(0, [2, 0, 0], 0), _track(1, [4, 0, 0], 1))
    metadata = {}
    for track in tracks:
        stats = SimpleNamespace(
            scores=[0.9, 0.8, 0.7],
            box_records=[
                (0.9, 0, np.concatenate((track.memory._center, np.ones(3))).astype(np.float32)),
                (0.8, 1, np.concatenate((track.memory._center, np.ones(3))).astype(np.float32)),
            ],
        )
        metadata[track.track_id] = SimpleNamespace(stats=stats)
    controller = SimpleNamespace(
        track_manager=_Manager(tracks), supplemental_metadata=metadata
    )
    boxes = np.asarray(
        [[0, 0, 0, 1, 1, 1], [2, 0, 0, 1, 1, 1]], dtype=np.float32
    )
    scores = np.asarray([0.5, 0.8], dtype=np.float32)
    stable_ids = np.asarray([9, -1], dtype=np.int64)
    source_indices = np.asarray([0, -1], dtype=np.int64)
    quality = np.zeros((2, 12), dtype=np.float32)
    summary = {
        "confirmed_supplemental_tracks": 2,
        "provider_seconds": 3.0,
        "geometry_seconds": 4.0,
        "supplemental_output": 1,
    }
    result = SimpleNamespace(
        boxes=boxes,
        scores=scores,
        stable_ids=stable_ids,
        source_indices=source_indices,
        quality_features=quality,
        summary=summary,
    )
    points = np.zeros((1, 512, 3), dtype=np.float32)
    mask = np.zeros((1, 512), dtype=bool)
    points[0, :4] = tracks[0].memory.points
    mask[0, :4] = True
    frozen_summary = dict(summary, provider_seconds=30.0, geometry_seconds=40.0)
    frozen = {
        "boxes": boxes[1:].copy(),
        "scores": scores[1:].copy(),
        "track_ids": stable_ids[1:].copy(),
        "result_indices": np.asarray([1], dtype=np.int64),
        "quality_features": quality[1:].copy(),
        "source_indices": source_indices[1:].copy(),
        "points": points,
        "point_mask": mask,
        "summary_json": np.asarray(json.dumps(frozen_summary)),
    }
    return controller, result, frozen


def test_exports_complete_universe_without_semantics_and_keeps_terminal_identity(
    tmp_path, monkeypatch
):
    scene = exporter.DEV3_SCENES[0]
    monkeypatch.setitem(exporter.EXPECTED_CONFIRMED_COUNTS, scene, 2)
    controller, result, frozen = _fixture()
    report = exporter._export_controller_universe(
        scene=scene,
        controller=controller,
        result=result,
        processed_source_frames=[0, 25],
        expected_source_frames=[0, 25],
        frozen_diagnostic=frozen,
        deterministic_bounded_sample=lambda points, count: points[:count],
        output_root=tmp_path,
        provenance={"exporter_source_sha256": "a" * 64},
    )
    assert report["confirmed_track_count"] == 2
    assert report["terminal_output_count"] == 1
    assert report["preterminal_rejected_track_count"] == 1
    assert report["labels_read"] is False
    assert report["labels_exported"] is False
    assert report["terminal_identity_to_frozen_s2"]["point_sample_identity"] is True
    with np.load(tmp_path / f"{scene}{exporter.SCENE_SUFFIX}.npz", allow_pickle=False) as arrays:
        assert arrays["scene_id"].shape == (1,)
        assert arrays["scene_id"].tolist() == [scene]
        assert arrays["track_id"].tolist() == [0, 1]
        assert arrays["archived"].tolist() == [False, True]
        assert arrays["terminal_output"].tolist() == [True, False]
        assert arrays["point_offsets"].tolist() == [0, 4, 8]
        assert not any("label" in name or "clip" in name for name in arrays.files)


def test_export_fails_before_writing_on_stream_or_terminal_drift(tmp_path, monkeypatch):
    scene = exporter.DEV3_SCENES[0]
    monkeypatch.setitem(exporter.EXPECTED_CONFIRMED_COUNTS, scene, 2)
    controller, result, frozen = _fixture()
    with pytest.raises(exporter.S3ExportError, match="exact sealed stream"):
        exporter._export_controller_universe(
            scene=scene,
            controller=controller,
            result=result,
            processed_source_frames=[25, 0],
            expected_source_frames=[0, 25],
            frozen_diagnostic=frozen,
            deterministic_bounded_sample=lambda points, count: points[:count],
            output_root=tmp_path,
            provenance={},
        )
    assert not list(tmp_path.iterdir())
    frozen["scores"][0] += 0.01
    with pytest.raises(exporter.S3ExportError, match="terminal scores differs"):
        exporter._export_controller_universe(
            scene=scene,
            controller=controller,
            result=result,
            processed_source_frames=[0, 25],
            expected_source_frames=[0, 25],
            frozen_diagnostic=frozen,
            deterministic_bounded_sample=lambda points, count: points[:count],
            output_root=tmp_path,
            provenance={},
        )


def test_dev3_seal_is_create_only_and_binds_all_scene_hashes(tmp_path, monkeypatch):
    for scene in exporter.DEV3_SCENES:
        monkeypatch.setitem(exporter.EXPECTED_CONFIRMED_COUNTS, scene, 1)
        npz_path = tmp_path / f"{scene}{exporter.SCENE_SUFFIX}.npz"
        np.savez(npz_path, value=np.asarray([1], dtype=np.int32))
        scene_manifest = {
            "schema": exporter.SCENE_SCHEMA,
            "mode": "shadow",
            "output_inert": True,
            "birth": False,
            "active_authorized": False,
            "gt_access": False,
            "oracle_access": False,
            "labels_read": False,
            "labels_exported": False,
            "scene_id": scene,
            "confirmed_track_count": 1,
            "expected_confirmed_track_count": 1,
            "terminal_output_count": 1,
            "preterminal_rejected_track_count": 0,
            "processed_source_frames_exactly_match_stream_seal": True,
            "npz_file": npz_path.name,
            "npz_sha256": exporter._sha256(npz_path),
            "array_content_sha256": "b" * 64,
            "provenance": {"exporter_source_sha256": "a" * 64},
        }
        (tmp_path / f"{scene}{exporter.SCENE_SUFFIX}.json").write_text(
            json.dumps(scene_manifest), encoding="utf-8"
        )
    seal_path = tmp_path / "seal.json"
    seal = exporter.seal_dev3(scene_root=tmp_path, output_manifest=seal_path)
    assert seal["scene_order"] == list(exporter.DEV3_SCENES)
    assert seal["confirmed_track_count"] == 3
    assert seal["gt_access"] is False
    with pytest.raises(exporter.S3ExportError, match="refusing to overwrite"):
        exporter.seal_dev3(scene_root=tmp_path, output_manifest=seal_path)


def test_geometry_report_quantifies_union_recall_and_ap_ceiling():
    native = [np.asarray([[0.9, 0.0]], dtype=np.float64)]
    candidates = [np.asarray([[0.0, 0.9]], dtype=np.float64)]
    baseline = {
        threshold: official_constant_evaluate(native, [2], threshold)
        for threshold in (0.15, 0.25, 0.50)
    }
    report = _geometry_report(
        scenes=["synthetic"],
        candidate_iou=candidates,
        native_iou=native,
        gt_counts=[2],
        baseline_official=baseline,
    )
    for row in report["per_threshold"].values():
        assert row["native_maximum_matching_count"] == 1
        assert row["native_union_maximum_matching_count"] == 2
        assert row["additional_union_recall_points"] == 50.0
        assert row["necessary_recall_ceiling_can_support_plus_10_ap"] is True


def test_point_quantiles_and_exporter_cli_have_fixed_no_gt_surface():
    points = np.asarray([[0, 0, 0], [2, 4, 6], [4, 8, 12]], dtype=np.float32)
    minmax = _point_quantile_minmax(points, np.asarray([0, 3]), 0.0)
    np.testing.assert_array_equal(minmax, [[0, 0, 0, 4, 8, 12]])
    options = {
        option
        for action in exporter._build_parser()._actions
        for option in action.option_strings
    }
    assert not any("gt" in option.lower() or "oracle" in option.lower() for option in options)
