from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import numpy as np
import pytest

import boxfusion.s3r_h10_provider_core as core
from boxfusion.s3r_h10_provider_core import (
    EXPECTED_RAW_COUNTS,
    EXPECTED_SCENE_ORDER,
    FrameToken,
    FrameTransaction,
    FrameTransactionError,
    HOLDOUT_LIST_SHA256,
    MAX_RAW_ROWS_PER_FRAME,
    SCHEDULE_SCHEMA,
    ScheduleValidationError,
    parse_exact_schedule_bundle,
)

_HASH = "1" * 64
_POSE_HASH = "2" * 64


def _bundle_dict():
    scenes = []
    for scene_id, raw_count in zip(EXPECTED_SCENE_ORDER, EXPECTED_RAW_COUNTS):
        raw_ids = list(range(0, raw_count * 25, 25))
        excluded_ids = [2325] if scene_id == "scene0412_00" else []
        valid_ids = [frame_id for frame_id in raw_ids if frame_id not in excluded_ids]
        scenes.append(
            {
                "scene_id": scene_id,
                "source_schedule_manifest_relpath": f"{scene_id}/manifest.json",
                "source_schedule_manifest_sha256": core.EXPECTED_SOURCE_MANIFEST_SHA256[
                    scene_id
                ],
                "formal_t05_relpath": (
                    f"results/scannet_topk_fusion_score05/{scene_id}_boxes.pkl"
                ),
                "formal_t05_sha256": core.EXPECTED_FORMAL_T05_SHA256[scene_id],
                "intrinsic_color_relpath": "frames/intrinsic/intrinsic_color.txt",
                "intrinsic_color_sha256": _HASH,
                "raw_frame_ids": raw_ids,
                "valid_frame_ids": valid_ids,
                "excluded_frames": (
                    [
                        {
                            "frame_id": 2325,
                            "reason": "nonfinite_pose",
                            "pose_relpath": "frames/pose/2325.txt",
                            "pose_sha256": core.EXCLUDED_POSE_SHA256,
                        }
                    ]
                    if excluded_ids
                    else []
                ),
                "frames": [
                    {
                        "frame_id": frame_id,
                        "color_relpath": f"frames/color/{frame_id}.jpg",
                        "color_sha256": _HASH,
                        "depth_relpath": f"frames/depth/{frame_id}.png",
                        "depth_sha256": _HASH,
                        "pose_relpath": f"frames/pose/{frame_id}.txt",
                        "pose_sha256": _POSE_HASH,
                    }
                    for frame_id in valid_ids
                ],
            }
        )
    return {
        "schema": SCHEDULE_SCHEMA,
        "scene_order": list(EXPECTED_SCENE_ORDER),
        "raw_frame_count": 770,
        "valid_frame_count": 769,
        "holdout_list_sha256": HOLDOUT_LIST_SHA256,
        "provider": {
            "annotation_path": None,
            "track": False,
            "directory_enumeration": False,
            "prefetch": False,
            "persist_before_advance": True,
        },
        "scenes": scenes,
    }


@pytest.fixture(scope="module")
def parsed_bundle():
    return parse_exact_schedule_bundle(_bundle_dict())


def _empty_rows():
    return {
        "center": np.empty((0, 3), dtype=np.float64),
        "extent": np.empty((0, 3), dtype=np.float64),
        "quaternion": np.empty((0, 4), dtype=np.float64),
        "score": np.empty((0,), dtype=np.float64),
        "source_row": np.empty((0,), dtype=np.int64),
    }


def _one_row():
    return {
        "center": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        "extent": np.asarray([[0.5, 0.6, 0.7]], dtype=np.float64),
        "quaternion": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
        "score": np.asarray([0.75], dtype=np.float64),
        "source_row": np.asarray([0], dtype=np.int64),
    }


def test_parse_exact_self_contained_bundle_and_file_identity(tmp_path):
    mapping = _bundle_dict()
    parsed = parse_exact_schedule_bundle(mapping)
    assert parsed.scene_order == EXPECTED_SCENE_ORDER
    assert parsed.raw_frame_count == 770
    assert parsed.valid_frame_count == 769
    assert len(parsed.ordered_frames) == 769
    assert len(parsed.scenes[1].raw_frame_ids) == 94
    assert len(parsed.scenes[1].valid_frame_ids) == 93
    assert parsed.scenes[1].excluded_frames[0].frame_id == 2325
    assert 2325 not in parsed.scenes[1].valid_frame_ids

    path = tmp_path / "schedule.json"
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(payload)
    from_file = parse_exact_schedule_bundle(path)
    assert from_file.sha256 == core.sha256(payload).hexdigest()
    assert from_file.ordered_frames[-1][1].frame_id == 1900


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda value: value.update({"labels": []}), "keys differ"),
        (lambda value: value["provider"].update({"track": True}), "safety"),
        (lambda value: value["provider"].pop("prefetch"), "keys differ"),
        (lambda value: value["scene_order"].reverse(), "scene_order"),
        (lambda value: value.update({"raw_frame_count": 769}), "770 raw"),
        (
            lambda value: value["scenes"][0].update(
                {"source_schedule_manifest_sha256": "A" * 64}
            ),
            "lowercase",
        ),
        (
            lambda value: value["scenes"][0].update(
                {"intrinsic_color_relpath": "../future.txt"}
            ),
            "dot components",
        ),
        (
            lambda value: value["scenes"][0]["raw_frame_ids"].__setitem__(1, 0),
            "strictly increasing",
        ),
        (
            lambda value: value["scenes"][1]["excluded_frames"][0].update(
                {"reason": "other"}
            ),
            "nonfinite_pose",
        ),
        (
            lambda value: value["scenes"][1]["frames"].append(
                deepcopy(value["scenes"][1]["frames"][-1])
            ),
            "exactly match",
        ),
    ],
)
def test_parser_fails_closed_on_schema_order_duplicates_and_unsafe_data(
    mutation, match
):
    value = _bundle_dict()
    mutation(value)
    with pytest.raises(ScheduleValidationError, match=match):
        parse_exact_schedule_bundle(value)


def test_parser_rejects_duplicate_json_keys_and_symlink(tmp_path):
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":1,"schema":2}', encoding="utf-8")
    with pytest.raises(ScheduleValidationError, match="duplicate JSON key"):
        parse_exact_schedule_bundle(duplicate)

    real = tmp_path / "real.json"
    real.write_text(json.dumps(_bundle_dict()), encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(real)
    with pytest.raises(ScheduleValidationError, match="non-symlink"):
        parse_exact_schedule_bundle(link)


def test_empty_frame_is_persisted_before_advance_and_fsync_order(
    tmp_path, parsed_bundle, monkeypatch
):
    events = []
    original_fsync = core._fsync

    with FrameTransaction(tmp_path / "run", parsed_bundle) as transaction:
        token = transaction.begin("scene0304_00", 0)

        def recording_fsync(descriptor, role):
            events.append(role)
            original_fsync(descriptor, role)

        monkeypatch.setattr(core, "_fsync", recording_fsync)
        commit = transaction.commit(token, **_empty_rows(), runtime_seconds=0.125)
        assert events == ["frame-file", "frame-directory", "journal"]
        assert commit.row_count == 0
        assert transaction.completed_frame_count == 1
        next_token = transaction.begin("scene0304_00", 25)
        assert next_token.frame_id == 25

    frame_path = tmp_path / "run" / commit.relative_path
    with np.load(frame_path, allow_pickle=False) as payload:
        assert frozenset(payload.files) == core._NPZ_KEYS
        assert payload["center"].shape == (0, 3)
        assert payload["center"].dtype == np.float64
        assert payload["source_row"].dtype == np.int64
        assert payload["input_sha256"].shape == (4, 32)
        assert payload["input_sha256"].dtype == np.uint8
        assert payload["runtime_seconds"].tolist() == [0.125]
    lines = (tmp_path / "run" / "frames.journal.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["row_count"] == 0


def test_pending_off_order_duplicate_and_forged_token_fail_closed(
    tmp_path, parsed_bundle
):
    transaction = FrameTransaction(tmp_path / "pending", parsed_bundle)
    transaction.begin("scene0304_00", 0)
    with pytest.raises(FrameTransactionError, match="previous frame"):
        transaction.begin("scene0304_00", 25)
    assert transaction.poisoned
    transaction.close()

    transaction = FrameTransaction(tmp_path / "off-order", parsed_bundle)
    with pytest.raises(FrameTransactionError, match="off-order"):
        transaction.begin("scene0304_00", 25)
    assert transaction.poisoned
    transaction.close()

    transaction = FrameTransaction(tmp_path / "forged", parsed_bundle)
    token = transaction.begin("scene0304_00", 0)
    forged = FrameToken(token.scene_id, token.frame_id, object())
    with pytest.raises(FrameTransactionError, match="exact pending"):
        transaction.commit(forged, **_empty_rows(), runtime_seconds=0.0)
    assert transaction.poisoned
    transaction.close()


@pytest.mark.parametrize(
    "edit,match",
    [
        (
            lambda rows: rows.update(
                {"center": np.asarray([[np.nan, 0.0, 0.0]], dtype=np.float64)}
            ),
            "non-finite",
        ),
        (
            lambda rows: rows.update(
                {
                    "center": np.zeros((2, 3)),
                    "extent": np.ones((2, 3)),
                    "quaternion": np.ones((2, 4)),
                    "score": np.ones(2),
                    "source_row": np.asarray([0, 0], dtype=np.int64),
                }
            ),
            "0..N-1",
        ),
        (
            lambda rows: rows.update(
                {
                    "center": np.zeros((MAX_RAW_ROWS_PER_FRAME + 1, 3)),
                    "extent": np.ones((MAX_RAW_ROWS_PER_FRAME + 1, 3)),
                    "quaternion": np.ones((MAX_RAW_ROWS_PER_FRAME + 1, 4)),
                    "score": np.ones(MAX_RAW_ROWS_PER_FRAME + 1),
                    "source_row": np.arange(MAX_RAW_ROWS_PER_FRAME + 1),
                }
            ),
            "cap exceeded",
        ),
        (
            lambda rows: rows.update(
                {"quaternion": np.asarray([[1e-7, 0.0, 0.0, 0.0]])}
            ),
            "squared norm",
        ),
    ],
)
def test_nonfinite_duplicate_and_cap_poison_transaction(
    tmp_path, parsed_bundle, edit, match
):
    transaction = FrameTransaction(tmp_path / f"bad-{match[:3]}", parsed_bundle)
    token = transaction.begin("scene0304_00", 0)
    rows = _one_row()
    edit(rows)
    with pytest.raises(FrameTransactionError, match=match):
        transaction.commit(token, **rows, runtime_seconds=0.1)
    assert transaction.poisoned
    with pytest.raises(FrameTransactionError, match="poisoned"):
        transaction.begin("scene0304_00", 25)
    transaction.close()


def test_existing_root_and_symlink_root_are_rejected(tmp_path, parsed_bundle):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_bytes(b"mine")
    with pytest.raises(FrameTransactionError, match="create-only"):
        FrameTransaction(existing, parsed_bundle)
    assert sentinel.read_bytes() == b"mine"

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(FrameTransactionError, match="create-only|symlink"):
        FrameTransaction(link, parsed_bundle)


def test_created_root_name_swap_between_stat_and_open_fails_closed(
    tmp_path, parsed_bundle, monkeypatch
):
    output = tmp_path / "swapped-root"
    real_open = os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == output.name and dir_fd is not None and not swapped:
            os.rename(
                output.name,
                "original-root",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.mkdir(output.name, mode=0o700, dir_fd=dir_fd)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(core.os, "open", swapping_open)
    with pytest.raises(FrameTransactionError, match="identity changed before open"):
        FrameTransaction(output, parsed_bundle)
    assert swapped is True


def test_existing_frame_symlink_and_hardlink_race_are_preserved(
    tmp_path, parsed_bundle, monkeypatch
):
    transaction = FrameTransaction(tmp_path / "symlink-frame", parsed_bundle)
    token = transaction.begin("scene0304_00", 0)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    final = tmp_path / "symlink-frame" / "frames" / "scene0304_00.000000.npz"
    final.symlink_to(outside)
    with pytest.raises(FrameTransactionError, match="already exists"):
        transaction.commit(token, **_one_row(), runtime_seconds=0.1)
    assert final.is_symlink()
    assert outside.read_bytes() == b"outside"
    transaction.close()

    transaction = FrameTransaction(tmp_path / "race", parsed_bundle)
    token = transaction.begin("scene0304_00", 0)
    original_link = os.link

    def competing_link(src, dst, **kwargs):
        descriptor = os.open(
            dst,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=kwargs["dst_dir_fd"],
        )
        os.write(descriptor, b"competitor")
        os.close(descriptor)
        raise FileExistsError(dst)

    monkeypatch.setattr(os, "link", competing_link)
    with pytest.raises(FileExistsError):
        transaction.commit(token, **_one_row(), runtime_seconds=0.1)
    monkeypatch.setattr(os, "link", original_link)
    raced = tmp_path / "race" / "frames" / "scene0304_00.000000.npz"
    assert raced.read_bytes() == b"competitor"
    assert transaction.poisoned
    transaction.close()


def test_crash_after_frame_publish_before_journal_fsync_blocks_advance(
    tmp_path, parsed_bundle, monkeypatch
):
    transaction = FrameTransaction(tmp_path / "crash", parsed_bundle)
    token = transaction.begin("scene0304_00", 0)
    original_fsync = core._fsync

    def crash_journal(descriptor, role):
        if role == "journal":
            raise OSError("simulated journal crash")
        original_fsync(descriptor, role)

    monkeypatch.setattr(core, "_fsync", crash_journal)
    with pytest.raises(OSError, match="simulated"):
        transaction.commit(token, **_one_row(), runtime_seconds=0.1)
    assert transaction.poisoned
    assert transaction.completed_frame_count == 0
    assert (tmp_path / "crash" / "frames" / "scene0304_00.000000.npz").is_file()
    with pytest.raises(FrameTransactionError, match="poisoned"):
        transaction.begin("scene0304_00", 25)
    with pytest.raises(FrameTransactionError, match="poisoned"):
        transaction.seal(run_provenance_sha256=_HASH)
    transaction.close()


def test_seal_is_impossible_until_every_frame_and_contains_no_science_fields(
    tmp_path, parsed_bundle, monkeypatch
):
    transaction = FrameTransaction(tmp_path / "seal", parsed_bundle)
    with pytest.raises(FrameTransactionError, match="cannot seal 0/769"):
        transaction.seal(run_provenance_sha256=_HASH)
    monkeypatch.setattr(core, "_fsync", lambda descriptor, role: None)
    for scene, frame in parsed_bundle.ordered_frames:
        token = transaction.begin(scene.scene_id, frame.frame_id)
        transaction.commit(token, **_empty_rows(), runtime_seconds=0.0)
    provenance_payload = b'{"audit_complete":true}\n'
    provenance_hash = transaction.publish_run_provenance(provenance_payload)
    seal = transaction.seal(run_provenance_sha256=provenance_hash)
    assert seal["completed_frame_count"] == 769
    assert transaction.sealed
    assert set(seal) == {
        "schema",
        "schedule_sha256",
        "run_provenance_sha256",
        "completed_frame_count",
        "journal_sha256",
        "frame_record_sha256",
        "total_runtime_seconds",
        "runtime_seconds_semantics",
    }
    forbidden = {"labels", "gt", "clip", "native", "ap"}
    assert forbidden.isdisjoint(key.lower() for key in seal)
    persisted = json.loads((tmp_path / "seal" / "FINAL_SEAL.json").read_text())
    assert persisted == seal
    assert (tmp_path / "seal" / "RUN_PROVENANCE.json").read_bytes() == (
        provenance_payload
    )
    with pytest.raises(FrameTransactionError, match="already sealed"):
        transaction.begin("scene0025_00", 1900)
    transaction.close()


def test_output_root_name_swap_before_provenance_fails_closed(
    tmp_path, parsed_bundle, monkeypatch
):
    output = tmp_path / "late-swap"
    transaction = FrameTransaction(output, parsed_bundle)
    monkeypatch.setattr(core, "_fsync", lambda descriptor, role: None)
    for scene, frame in parsed_bundle.ordered_frames:
        token = transaction.begin(scene.scene_id, frame.frame_id)
        transaction.commit(token, **_empty_rows(), runtime_seconds=0.0)
    original = tmp_path / "late-swap-original"
    output.rename(original)
    output.mkdir()
    with pytest.raises(FrameTransactionError, match="namespace identity changed"):
        transaction.publish_run_provenance(b"{}\n")
    assert transaction.poisoned
    assert not (original / "RUN_PROVENANCE.json").exists()
    assert not (output / "RUN_PROVENANCE.json").exists()
    transaction.close()


def test_output_parent_name_swap_before_begin_fails_closed(
    tmp_path, parsed_bundle
):
    parent = tmp_path / "logs"
    parent.mkdir()
    output = parent / "run"
    transaction = FrameTransaction(output, parsed_bundle)

    original_parent = tmp_path / "logs-original"
    parent.rename(original_parent)
    parent.mkdir()

    with pytest.raises(FrameTransactionError, match="namespace identity changed"):
        transaction.begin("scene0304_00", 0)
    assert transaction.poisoned
    assert not output.exists()
    assert (original_parent / "run").is_dir()
    transaction.close()
