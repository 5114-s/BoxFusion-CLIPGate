from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_dynamic_run_coverage import (
    DynamicCoverageError,
    main,
    read_manifest_scenes,
    validate_dynamic_run_coverage,
)


SCENES = ("scene0001_00", "scene0002_00")


def _manifest(path: Path, scenes=SCENES) -> Path:
    path.write_text(
        json.dumps({scene: {"T_frame": 25} for scene in scenes}),
        encoding="utf-8",
    )
    return path


def _prediction_root(path: Path, scenes=SCENES) -> Path:
    path.mkdir()
    for scene in scenes:
        (path / f"{scene}_boxes.pkl").write_bytes(b"not-empty")
        # A normal output sidecar is not another evaluator artifact.
        (path / f"{scene}_boxes.pkl.dual_state.json").write_text(
            "{}", encoding="utf-8"
        )
    return path


def _event_root(path: Path, scenes=SCENES) -> Path:
    path.mkdir()
    for scene in scenes:
        (path / f"{scene}_events.jsonl").write_text(
            '{"frame_id": 0}\n'
            + json.dumps(
                {
                    "schema": "boxfusion.causal_dynamic_branch.v1",
                    "type": "summary",
                    "scene_id": scene,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return path


def test_paired_score_roots_require_exact_nonempty_scene_coverage(tmp_path):
    manifest = _manifest(tmp_path / "manifest.json")
    persistent = _prediction_root(tmp_path / "persistent")
    current = _prediction_root(tmp_path / "current")

    report = validate_dynamic_run_coverage(
        manifest,
        persistent_root=persistent,
        current_root=current,
    )

    assert report["passed"] is True
    assert report["manifest"]["scene_count"] == 2
    assert [root["kind"] for root in report["artifact_roots"]] == [
        "persistent",
        "current",
    ]
    assert all(root["scene_count"] == 2 for root in report["artifact_roots"])
    assert all(
        len(record["sha256"]) == 64
        for root in report["artifact_roots"]
        for record in root["files"].values()
    )


@pytest.mark.parametrize("mutation", ("missing", "extra", "empty"))
def test_score_root_rejects_missing_extra_and_empty_files(tmp_path, mutation):
    manifest = _manifest(tmp_path / "manifest.json")
    persistent = _prediction_root(tmp_path / "persistent")
    current = _prediction_root(tmp_path / "current")
    if mutation == "missing":
        (current / "scene0002_00_boxes.pkl").unlink()
        message = "missing=scene0002_00"
    elif mutation == "extra":
        (persistent / "scene9999_00_boxes.pkl").write_bytes(b"extra")
        message = "extra=scene9999_00"
    else:
        (current / "scene0001_00_boxes.pkl").write_bytes(b"")
        message = "invalid_or_empty=scene0001_00_boxes.pkl"

    with pytest.raises(DynamicCoverageError, match=message):
        validate_dynamic_run_coverage(
            manifest,
            persistent_root=persistent,
            current_root=current,
        )


def test_manifest_duplicate_scene_ids_fail_closed(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"scenes":[{"scene_id":"scene0001_00"},'
        '{"scene_id":"scene0001_00"}]}',
        encoding="utf-8",
    )
    events = _event_root(tmp_path / "events", ("scene0001_00",))

    with pytest.raises(DynamicCoverageError, match="duplicate scene IDs"):
        validate_dynamic_run_coverage(manifest, event_roots=(events,))


def test_duplicate_manifest_json_keys_are_rejected_before_overwrite(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        '{"scene0001_00":{},"scene0001_00":{}}', encoding="utf-8"
    )

    with pytest.raises(DynamicCoverageError, match="duplicate JSON keys"):
        read_manifest_scenes(manifest)


def test_event_root_supports_names_and_rejects_duplicate_scene_artifacts(tmp_path):
    manifest = _manifest(tmp_path / "manifest.json")
    events = _event_root(tmp_path / "events")
    report = validate_dynamic_run_coverage(manifest, event_roots=(events,))
    assert report["artifact_roots"][0]["kind"] == "events[0]"

    (events / "scene0001_00.jsonl").write_text(
        '{"frame_id": 1}\n', encoding="utf-8"
    )
    with pytest.raises(DynamicCoverageError, match="duplicate=scene0001_00"):
        validate_dynamic_run_coverage(manifest, event_roots=(events,))


def test_event_root_rejects_truncated_or_wrong_scene_summary(tmp_path):
    manifest = _manifest(tmp_path / "manifest.json")
    events = _event_root(tmp_path / "events")
    (events / "scene0001_00_events.jsonl").write_text(
        '{"schema":"boxfusion.causal_dynamic_branch.v1","type":"frame",'
        '"scene_id":"scene0001_00"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DynamicCoverageError, match="incomplete_event=scene0001_00"):
        validate_dynamic_run_coverage(manifest, event_roots=(events,))

    (events / "scene0001_00_events.jsonl").write_text(
        '{"schema":"boxfusion.causal_dynamic_branch.v1","type":"summary",'
        '"scene_id":"scene9999_00"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DynamicCoverageError, match="incomplete_event=scene0001_00"):
        validate_dynamic_run_coverage(manifest, event_roots=(events,))


def test_roots_must_be_paired_distinct_and_cli_fails_closed(tmp_path, capsys):
    manifest = _manifest(tmp_path / "manifest.json")
    persistent = _prediction_root(tmp_path / "persistent")

    with pytest.raises(DynamicCoverageError, match="supplied together"):
        validate_dynamic_run_coverage(manifest, persistent_root=persistent)
    with pytest.raises(DynamicCoverageError, match="distinct"):
        validate_dynamic_run_coverage(
            manifest,
            persistent_root=persistent,
            current_root=persistent,
        )

    rc = main(
        [
            "--manifest",
            str(manifest),
            "--persistent-root",
            str(persistent),
        ]
    )
    assert rc == 2
    error = json.loads(capsys.readouterr().err)
    assert error["passed"] is False


def test_cli_success_emits_machine_readable_receipt(tmp_path, capsys):
    manifest = _manifest(tmp_path / "manifest.json")
    events = _event_root(tmp_path / "events")

    assert main(
        ["--manifest", str(manifest), "--event-root", str(events)]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "boxfusion.dynamic_run_coverage.v1"
    assert report["passed"] is True
