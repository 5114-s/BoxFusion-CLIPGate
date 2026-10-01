from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools import audit_scannet_s3r0_raw_boxer_receipt_oracle as s3r0


def _corners(center, extent=(1.0, 1.0, 1.0), angle_degrees=0.0):
    center = np.asarray(center, dtype=np.float64)
    extent = np.asarray(extent, dtype=np.float64)
    angle = np.deg2rad(angle_degrees)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return s3r0._SIGNS * (extent / 2.0) @ rotation.T + center


def test_axis_alignment_transforms_obb_corners_before_enclosing_aabb():
    corners = _corners([1.0, 2.0, 3.0], [4.0, 1.0, 0.5], angle_degrees=23.0)
    angle = np.deg2rad(37.0)
    alignment = np.eye(4, dtype=np.float64)
    alignment[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    alignment[:3, 3] = [5.0, -3.0, 1.0]

    aligned_corners = corners @ alignment[:3, :3].T + alignment[:3, 3]
    expected = np.concatenate(
        (aligned_corners.min(axis=0), aligned_corners.max(axis=0))
    )
    actual = s3r0._aligned_enclosing_aabb(corners[None, ...], alignment)[0]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

    world_lower = corners.min(axis=0)
    world_upper = corners.max(axis=0)
    wrong_world_aabb_corners = np.asarray(
        [
            [x, y, z]
            for x in (world_lower[0], world_upper[0])
            for y in (world_lower[1], world_upper[1])
            for z in (world_lower[2], world_upper[2])
        ]
    )
    wrong_aligned = wrong_world_aabb_corners @ alignment[:3, :3].T + alignment[:3, 3]
    wrong = np.concatenate((wrong_aligned.min(axis=0), wrong_aligned.max(axis=0)))
    assert not np.allclose(actual, wrong, rtol=0.0, atol=1e-12)


def test_any_evidence_is_one_row_per_receipt_and_never_exports_an_argmax():
    evidence_iou = np.asarray(
        [
            [0.90, 0.00],
            [0.00, 0.90],
            [0.10, 0.10],
        ],
        dtype=np.float64,
    )
    matrix = s3r0._track_any_evidence_iou(
        evidence_iou, np.asarray([0, 3], dtype=np.int64)
    )
    np.testing.assert_array_equal(matrix, [[0.90, 0.90]])
    assert len(s3r0.strict_maximum_matching(matrix, 0.50)) == 1
    assert len(s3r0.strict_maximum_matching(evidence_iou, 0.50)) == 2

    source = Path(s3r0.__file__).read_text(encoding="utf-8")
    assert "np.argmax(" not in source
    assert "candidate_maximum_matching_pairs" not in source


def test_any_evidence_zero_receipts_has_exact_empty_matrix_shape():
    matrix = s3r0._track_any_evidence_iou(
        np.empty((0, 7), dtype=np.float64), np.asarray([0], dtype=np.int64)
    )
    assert matrix.shape == (0, 7)
    assert matrix.dtype == np.float64


def test_strict_thresholds_maximum_cardinality_and_native_union_additions():
    # A greedy first-row assignment can consume GT0, but augmenting-path MM
    # reassigns it to GT1 so both candidates match.
    reroute = np.asarray([[0.90, 0.80], [0.85, 0.00]], dtype=np.float64)
    assert len(s3r0.strict_maximum_matching(reroute, 0.50)) == 2

    native = np.asarray([[0.90, 0.00, 0.00, 0.00]], dtype=np.float64)
    candidate = np.asarray(
        [
            [0.80, 0.00, 0.00, 0.00],
            [0.00, 0.80, 0.00, 0.00],
            [0.00, 0.00, 0.80, 0.00],
            [0.00, 0.00, 0.00, 0.80],
        ],
        dtype=np.float64,
    )
    report = s3r0._matching_report(
        scenes=("scene0000_00",),
        candidate_iou=(candidate,),
        native_iou=(native,),
    )
    for threshold in ("0.15", "0.25", "0.50"):
        row = report[threshold]
        assert row["candidate_maximum_matching_count"] == 4
        assert row["native_maximum_matching_count"] == 1
        assert row["native_union_maximum_matching_count"] == 4
        assert row["additional_union_matching_over_native"] == 3
        assert row["passes_plus3_continuation_floor"] is True

    boundary = np.diag([0.15, 0.25, 0.50]).astype(np.float64)
    assert len(s3r0.strict_maximum_matching(boundary, 0.15)) == 2
    assert len(s3r0.strict_maximum_matching(boundary, 0.25)) == 1
    assert len(s3r0.strict_maximum_matching(boundary, 0.50)) == 0
    above = np.diag(
        [
            np.nextafter(0.15, np.inf),
            np.nextafter(0.25, np.inf),
            np.nextafter(0.50, np.inf),
        ]
    )
    assert len(s3r0.strict_maximum_matching(above, 0.15)) == 3


def test_matching_report_aggregates_scenes_without_cross_scene_edges():
    reports = s3r0._matching_report(
        scenes=("scene0000_00", "scene0001_00"),
        candidate_iou=(
            np.asarray([[0.9]], dtype=np.float64),
            np.asarray([[0.9]], dtype=np.float64),
        ),
        native_iou=(
            np.empty((0, 1), dtype=np.float64),
            np.empty((0, 1), dtype=np.float64),
        ),
    )
    assert reports["0.50"]["candidate_maximum_matching_count"] == 2
    assert reports["0.50"]["additional_union_matching_over_native"] == 2
    assert set(reports["0.50"]["per_scene"]) == {
        "scene0000_00",
        "scene0001_00",
    }
    with pytest.raises(s3r0.S3R0OracleError, match="incompatible"):
        s3r0._matching_report(
            scenes=("scene0000_00",),
            candidate_iou=(np.empty((1, 2), dtype=np.float64),),
            native_iou=(np.empty((1, 3), dtype=np.float64),),
        )


def test_continuation_gate_requires_both_geometries_at_every_threshold():
    def rows(value):
        return {
            f"{threshold:.2f}": {"additional_union_matching_over_native": value}
            for threshold in s3r0.THRESHOLDS
        }

    passing = s3r0._continuation_gate(rows(3), rows(3))
    assert passing["passes_all_thresholds"] is True

    medoid_fails = rows(3)
    medoid_fails["0.25"]["additional_union_matching_over_native"] = 2
    assert (
        s3r0._continuation_gate(medoid_fails, rows(99))["passes_all_thresholds"]
        is False
    )

    evidence_fails = rows(3)
    evidence_fails["0.50"]["additional_union_matching_over_native"] = 2
    assert (
        s3r0._continuation_gate(rows(99), evidence_fails)["passes_all_thresholds"]
        is False
    )


@pytest.fixture(scope="module")
def frozen_context():
    # This validates only frozen no-GT assets and hashes.  It never opens GT,
    # axis metadata, or deserializes formal-T05 predictions.
    return s3r0._validate_no_gt_frozen()


def test_formal_no_gt_artifact_and_exact_receipt_closure_are_valid(frozen_context):
    assert frozen_context.shadow_manifest["selected_row_count"] == 1571
    assert frozen_context.shadow_manifest["receipt_count"] == 198
    assert frozen_context.shadow_manifest["evidence_count"] == 594
    assert tuple(frozen_context.shadow_arrays["scene_ids"].tolist()) == s3r0.DEV3_SCENES
    assert frozen_context.no_gt_hashes_before["shadow_json"] == (
        s3r0.EXPECTED_SHADOW_JSON_SHA256
    )
    assert frozen_context.no_gt_hashes_before["geometry_helpers"]["sha256"] == (
        s3r0.EXPECTED_GEOMETRY_HELPERS_SHA256
    )


def test_receipt_track_identity_is_scene_qualified(frozen_context):
    arrays = frozen_context.shadow_arrays
    offsets = arrays["scene_receipt_offsets"]
    per_scene = [
        set(
            int(value)
            for value in arrays["receipt_track_id"][
                int(offsets[index]) : int(offsets[index + 1])
            ]
        )
        for index in range(len(s3r0.DEV3_SCENES))
    ]
    assert [len(values) for values in per_scene] == [66, 104, 28]
    assert per_scene[0] & per_scene[1]
    assert per_scene[0] & per_scene[2]
    assert per_scene[1] & per_scene[2]
    # Duplicate numeric track IDs across scenes are expected and valid.  The
    # oracle keeps three independent matrices and never uses track_id globally.


@pytest.mark.parametrize(
    "name,mutate,error",
    [
        (
            "evidence_selected_index",
            lambda arrays: arrays["evidence_selected_index"].__setitem__(0, 1),
            "evidence binding|receipt evidence|one selected observation",
        ),
        (
            "evidence_source_instance_id",
            lambda arrays: arrays["evidence_source_instance_id"].__setitem__(0, 999999),
            "evidence binding",
        ),
        (
            "receipt_medoid_evidence_index",
            lambda arrays: arrays["receipt_medoid_evidence_index"].__setitem__(
                0, (int(arrays["receipt_medoid_evidence_index"][0]) + 1) % 3
            ),
            "receipt medoid",
        ),
        (
            "receipt_corners_world",
            lambda arrays: arrays["receipt_corners_world"].__setitem__(
                (0, 0, 0), 999.0
            ),
            "receipt medoid corners",
        ),
        (
            "receipt_pairwise_aabb_iou",
            lambda arrays: arrays["receipt_pairwise_aabb_iou"].__setitem__(
                (0, 0, 1), 0.0
            ),
            "pairwise AABB IoU",
        ),
        (
            "receipt_scene_index",
            lambda arrays: arrays["receipt_scene_index"].__setitem__(0, 1),
            "receipt scene index|crosses scene",
        ),
        (
            "evidence_offsets",
            lambda arrays: arrays["evidence_offsets"].__setitem__(1, 2),
            "evidence offsets",
        ),
    ],
)
def test_receipt_or_evidence_provenance_tamper_fails_closed(
    frozen_context, name, mutate, error
):
    arrays = {
        key: np.array(value, copy=True)
        for key, value in frozen_context.shadow_arrays.items()
    }
    mutate(arrays)
    with pytest.raises(s3r0.S3R0OracleError, match=error):
        s3r0._validate_s3r_arrays(
            manifest=frozen_context.shadow_manifest,
            arrays=arrays,
            raw_arrays=frozen_context.raw_arrays,
            selections=frozen_context.selections,
        )


def test_no_gt_failure_prevents_oracle_stage(monkeypatch):
    touched = {"gt": False}

    def fail_no_gt():
        raise s3r0.S3R0OracleError("sealed barrier failure")

    def forbidden_gt(*_args, **_kwargs):
        touched["gt"] = True
        raise AssertionError("GT stage must not run")

    monkeypatch.setattr(s3r0, "_validate_no_gt_frozen", fail_no_gt)
    monkeypatch.setattr(s3r0, "load_gt_minmax", forbidden_gt)
    monkeypatch.setattr(s3r0, "load_axis_alignment", forbidden_gt)
    with pytest.raises(s3r0.S3R0OracleError, match="barrier failure"):
        s3r0.audit_scannet_s3r0_raw_boxer_receipt_oracle()
    assert touched["gt"] is False


def test_formal_no_gt_validator_never_touches_gt_or_axis_paths(monkeypatch):
    original = s3r0._regular_file

    def guarded(path, label):
        candidate = Path(path)
        for forbidden_root in (s3r0.GT_ROOT, s3r0.SCAN_ROOT):
            try:
                candidate.relative_to(forbidden_root)
            except ValueError:
                continue
            raise AssertionError(f"no-GT validator touched forbidden path: {candidate}")
        return original(candidate, label)

    monkeypatch.setattr(s3r0, "_regular_file", guarded)
    context = s3r0._validate_no_gt_frozen()
    assert context.shadow_manifest["audit_complete"] is True


def test_dependency_hash_failure_precedes_any_repo_local_import(monkeypatch):
    original_sha256 = s3r0._sha256
    imported = {"called": False}

    def wrong_geometry_hash(path):
        if Path(path) == s3r0.GEOMETRY_HELPERS_SOURCE.resolve():
            return "0" * 64
        return original_sha256(path)

    def forbidden_import(_name):
        imported["called"] = True
        raise AssertionError("dependency import must not run after a hash failure")

    monkeypatch.setattr(s3r0, "_DEPENDENCIES", None)
    monkeypatch.setattr(s3r0, "_sha256", wrong_geometry_hash)
    monkeypatch.setattr(s3r0.importlib, "import_module", forbidden_import)
    with pytest.raises(s3r0.S3R0OracleError, match="bootstrap SHA-256"):
        s3r0._load_dependency_bundle()
    assert imported["called"] is False


def test_import_surface_does_not_eagerly_execute_repo_oracle_dependencies():
    command = (
        "import sys; "
        "import tools.audit_scannet_s3r0_raw_boxer_receipt_oracle; "
        "assert 'tools.audit_scannet_boxer_per_view_topk_ceiling' not in sys.modules; "
        "assert 'tools.audit_scannet_boxer_unexplained_oracle' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=s3r0.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_fresh_process_loads_only_exact_hashed_dependency_modules():
    command = "\n".join(
        (
            "from pathlib import Path",
            "import sys",
            "from tools import audit_scannet_s3r0_raw_boxer_receipt_oracle as oracle",
            "names = ('tools.audit_scannet_boxer_per_view_topk_ceiling', "
            "'tools.audit_scannet_boxer_unexplained_oracle')",
            "assert all(name not in sys.modules for name in names)",
            "bundle = oracle._load_dependency_bundle()",
            "topk, geometry = (sys.modules[name] for name in names)",
            "assert Path(topk.__file__).resolve() == oracle.TOPK_TOOL",
            "assert Path(topk.__spec__.origin).resolve() == oracle.TOPK_TOOL",
            "assert Path(geometry.__file__).resolve() == oracle.GEOMETRY_HELPERS_SOURCE",
            "assert Path(geometry.__spec__.origin).resolve() == "
            "oracle.GEOMETRY_HELPERS_SOURCE",
            "assert bundle.load_sealed_sidecar.__module__ == names[0]",
            "assert bundle.aligned_iou_matrix.__module__ == names[1]",
        )
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=s3r0.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_uncached_bundle_rejects_preloaded_dependency(monkeypatch):
    module_name = "tools.audit_scannet_boxer_per_view_topk_ceiling"
    monkeypatch.setitem(sys.modules, module_name, object())
    monkeypatch.setattr(s3r0, "_DEPENDENCIES", None)
    with pytest.raises(s3r0.S3R0OracleError, match="fresh process"):
        s3r0._load_dependency_bundle()


def test_symlink_inputs_rejected_and_report_writer_is_create_only(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(s3r0.S3R0OracleError, match="must not be a symlink"):
        s3r0._regular_file(link, "synthetic")

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    nested_target = real_parent / "nested.json"
    nested_target.write_text("{}", encoding="utf-8")
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(s3r0.S3R0OracleError, match="must not be a symlink"):
        s3r0._regular_file(parent_link / "nested.json", "synthetic parent")

    output = tmp_path / "report.json"
    payload = {
        "schema": s3r0.SCHEMA,
        "posthoc_dev_diagnostic": True,
        "not_deployable": True,
        "ap_computed": False,
    }
    s3r0._write_json_create_only(output, payload)
    before = output.read_bytes()
    with pytest.raises(s3r0.S3R0OracleError, match="refusing to overwrite"):
        s3r0._write_json_create_only(output, payload)
    assert output.read_bytes() == before


def test_report_writer_rejects_a_protected_root(tmp_path, monkeypatch):
    protected = tmp_path / "sealed-shadow"
    protected.mkdir()
    monkeypatch.setattr(s3r0, "SHADOW_ROOT", protected)
    output = protected / "forbidden.json"
    with pytest.raises(s3r0.S3R0OracleError, match="protected input root"):
        s3r0._write_json_create_only(
            output, {"posthoc_dev_diagnostic": True, "not_deployable": True}
        )
    assert not output.exists()


def test_report_writer_serialization_failure_publishes_no_partial_file(tmp_path):
    output = tmp_path / "must-remain-absent.json"
    with pytest.raises(s3r0.S3R0OracleError, match="could not serialize"):
        s3r0._write_json_create_only(output, {"unsupported": object()})
    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


def test_main_rejects_existing_output_before_running_oracle(tmp_path, monkeypatch):
    output = tmp_path / "already-exists.json"
    output.write_text("do not overwrite", encoding="utf-8")
    called = {"oracle": False}

    def forbidden_oracle():
        called["oracle"] = True
        raise AssertionError("oracle must not run after output preflight failure")

    monkeypatch.setattr(
        s3r0, "audit_scannet_s3r0_raw_boxer_receipt_oracle", forbidden_oracle
    )
    with pytest.raises(s3r0.S3R0OracleError, match="refusing to overwrite"):
        s3r0.main(["--output", str(output)])
    assert called["oracle"] is False
    assert output.read_text(encoding="utf-8") == "do not overwrite"


def test_main_rejects_output_parent_symlink_before_oracle_or_target_access(
    tmp_path, monkeypatch
):
    protected = tmp_path / "synthetic-gt-root"
    protected.mkdir()
    alias = tmp_path / "gt-alias"
    alias.symlink_to(protected, target_is_directory=True)
    called = {"oracle": False}

    def forbidden_oracle():
        called["oracle"] = True
        raise AssertionError("oracle must not run for a symlinked output parent")

    monkeypatch.setattr(s3r0, "GT_ROOT", protected)
    monkeypatch.setattr(
        s3r0, "audit_scannet_s3r0_raw_boxer_receipt_oracle", forbidden_oracle
    )
    with pytest.raises(s3r0.S3R0OracleError, match="output parent must not be"):
        s3r0.main(["--output", str(alias / "must-not-be-probed.json")])
    assert called["oracle"] is False
    assert list(protected.iterdir()) == []


def test_cli_has_only_output_and_no_tuning_or_active_surface():
    options = {
        option
        for action in s3r0._build_parser()._actions
        for option in action.option_strings
    }
    assert options == {"-h", "--help", "--output"}
    assert s3r0.THRESHOLDS == (0.15, 0.25, 0.50)
    assert s3r0.CONTINUATION_MIN_MATCHES == 3
    report_source = Path(s3r0.__file__).read_text(encoding="utf-8")
    assert '"frozen_pre_gt_k8_membership_revalidated": True' in report_source
    assert '"posthoc_gt_informed_candidate_selection_applied": False' in (report_source)
    source = report_source
    forbidden = (
        "--threshold",
        "--top-k",
        "--gt-root",
        "--scan-root",
        "--baseline-root",
        "compute_ap",
        "evaluate_predictions",
    )
    assert not any(token in source for token in forbidden)
