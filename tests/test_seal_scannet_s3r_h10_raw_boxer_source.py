from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pytest

from boxfusion.s3r_h10_provider_core import (
    JOURNAL_SCHEMA,
    PRECOMMIT_RUNTIME_SEMANTICS,
    SEAL_SCHEMA,
    parse_exact_schedule_bundle,
)
from tools import seal_scannet_s3r_h10_raw_boxer_source as sealer

SCHEDULE = Path("docs/data/S3R_H10_EXACT_SCHEDULE_V2.json").resolve()
_CONTRACT_HASH = sealer.EXPECTED_PROVIDER_CONTRACT_SHA256
_RUNNER_HASH = sealer.EXPECTED_PROVIDER_RUNNER_SHA256
_CORE_HASH = sealer.EXPECTED_PROVIDER_CORE_SHA256


def _replace_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.replace")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _json_bytes(value) -> bytes:
    return sealer._canonical_json_bytes(value)


def _frame_values(scene, frame, schedule_index, count):
    source_row = np.arange(count, dtype=np.int64)
    center = np.stack(
        (
            source_row.astype(np.float64),
            np.full(count, float(schedule_index), dtype=np.float64),
            np.ones(count, dtype=np.float64),
        ),
        axis=1,
    )
    extent = np.ones((count, 3), dtype=np.float64)
    quaternion = np.zeros((count, 4), dtype=np.float64)
    quaternion[:, 0] = 1.0
    if schedule_index == 0:
        score = np.asarray(
            [0.1, 0.9, 0.9, 0.2, 0.8, 0.7, 0.6, 0.5, 0.4, 0.95],
            dtype=np.float64,
        )
    else:
        score = np.linspace(0.3, 0.2, count, dtype=np.float64)
    hashes = np.stack(
        [
            np.frombuffer(bytes.fromhex(value), dtype=np.uint8)
            for value in (
                scene.intrinsic_color_sha256,
                frame.color_sha256,
                frame.depth_sha256,
                frame.pose_sha256,
            )
        ]
    )
    return {
        "center": center,
        "extent": extent,
        "quaternion": quaternion,
        "score": score,
        "source_row": source_row,
        "input_sha256": hashes,
        "runtime_seconds": np.asarray([0.01], dtype=np.float64),
    }


def _npz_bytes(arrays) -> bytes:
    output = io.BytesIO()
    np.savez(output, **arrays)
    return output.getvalue()


def _provider_contract():
    return {
        "model_process_count": 1,
        "owl_instance_count": 1,
        "boxernet_instance_count": 1,
        "taxonomy": "lvisplus",
        "prompt_count": 1220,
        "threshold_2d": 0.25,
        "nms_iou_2d": 0.5,
        "threshold_3d": 0.5,
        "score_rule": "mean(owl_2d_score,boxer_3d_score)_after_3d_threshold",
        "image_hw": [960, 960],
        "precision": "bfloat16",
        "seed": 0,
        "temporal_state": False,
        "prefetch": False,
        "frame_directory_enumeration": False,
        "coordinate_convention": (
            "absolute_scannet_world=center_boxer_recentered+"
            "translation_of_first_valid_exact_schedule_pose;"
            "extent_unchanged;Hamilton_wxyz_quaternion_l2_normalized"
        ),
    }


def _asset_ledger():
    records = {}

    def add(name, digest, expected):
        records[name] = {
            "path": f"/frozen/{name.replace(':', '_')}",
            "sha256_before": digest,
            "expected_sha256": expected,
            "sha256_after": digest,
        }

    add("schedule", sealer.EXPECTED_SCHEDULE_SHA256, sealer.EXPECTED_SCHEDULE_SHA256)
    add(
        "holdout_list",
        sealer.EXPECTED_HOLDOUT_LIST_SHA256,
        sealer.EXPECTED_HOLDOUT_LIST_SHA256,
    )
    add("provider_contract", _CONTRACT_HASH, _CONTRACT_HASH)
    for name, digest in sealer._EXPECTED_MODEL_HASHES.items():
        add(name, digest, digest)
    add("runner_source", _RUNNER_HASH, None)
    add("provider_core_source", _CORE_HASH, None)
    for relative, digest in sealer._EXPECTED_EXTERNAL_CODE_HASHES.items():
        add(f"boxer_code:{relative}", digest, digest)
    return records


def _build_provider(root: Path):
    bundle = parse_exact_schedule_bundle(SCHEDULE)
    root.mkdir()
    frames_directory = root / sealer.FRAMES_DIRECTORY_NAME
    frames_directory.mkdir()
    journal_lines = [
        sealer._canonical_json_line(
            {
                "schema": JOURNAL_SCHEMA,
                "schedule_sha256": bundle.sha256,
                "expected_frame_count": bundle.valid_frame_count,
            }
        )
    ]
    runtime_rows = []
    row_counts = []
    for schedule_index, (scene, frame) in enumerate(bundle.ordered_frames):
        count = 10 if schedule_index == 0 else 2 if schedule_index == 1 else 0
        arrays = _frame_values(scene, frame, schedule_index, count)
        payload = _npz_bytes(arrays)
        name = sealer._expected_frame_name(scene, frame)
        (frames_directory / name).write_bytes(payload)
        row = {
            "scene_id": scene.scene_id,
            "frame_id": frame.frame_id,
            "relative_path": f"frames/{name}",
            "row_count": count,
            "file_sha256": hashlib.sha256(payload).hexdigest(),
            "input_sha256": {
                "intrinsic_color": scene.intrinsic_color_sha256,
                "color": frame.color_sha256,
                "depth": frame.depth_sha256,
                "pose": frame.pose_sha256,
            },
            "runtime_seconds": 0.01,
            "runtime_seconds_semantics": PRECOMMIT_RUNTIME_SEMANTICS,
        }
        journal_lines.append(sealer._canonical_json_line(row))
        runtime_rows.append(
            {
                "scene_id": scene.scene_id,
                "frame_id": frame.frame_id,
                "row_count": count,
                "precommit_compute_seconds": 0.01,
                "end_to_end_seconds": 0.02,
            }
        )
        row_counts.append(count)
    journal_payload = b"".join(journal_lines)
    (root / sealer.JOURNAL_NAME).write_bytes(journal_payload)

    input_count, input_hash = sealer._exact_input_ledger_sha256(bundle)
    precommit = [0.01] * bundle.valid_frame_count
    end_to_end = [0.02] * bundle.valid_frame_count
    provenance = {
        "schema": sealer.PROVIDER_RUN_SCHEMA,
        "audit_complete": True,
        "shadow_only": True,
        "birth_enabled": False,
        "ap_evaluated": False,
        "gt_used": False,
        "target_dataset_training_used": False,
        "schedule": {
            "schema": bundle.schema,
            "sha256": bundle.sha256,
            "scene_order": list(bundle.scene_order),
            "valid_frame_count": bundle.valid_frame_count,
            "raw_frame_count": bundle.raw_frame_count,
            "excluded_frame_count": bundle.raw_frame_count - bundle.valid_frame_count,
        },
        "provider_contract": _provider_contract(),
        "model_runtime": {
            "owl_instance_count": 1,
            "boxernet_instance_count": 1,
            "prompt_count": 1220,
            "boxer_image_hw": 960,
            "owl_use_bfloat16": True,
        },
        "environment": {},
        "frozen_assets": _asset_ledger(),
        # The source sealer deliberately treats this provider-bound field as opaque.
        "formal_t05": {},
        "frame_inputs": {
            "before_each_frame_read_verified": True,
            "after_complete_stream_verified": True,
            "frame_inputs_before_read_and_after_stream_verified": True,
            "verified_file_count": input_count,
            "expected_file_count": input_count,
            "exact_input_ledger_sha256": input_hash,
        },
        "runtime": {
            "cold_start_model_load_and_warmup_seconds": 0.5,
            "cold_first_frame": dict(runtime_rows[0]),
            "cold_first_frame_end_to_end_seconds": 0.02,
            "cold_start_total_seconds": 0.52,
            "precommit_compute_definition": (
                "current-frame verified reads + synchronous datum construction + "
                "OWL + Boxer + CUDA synchronize; excludes persistence"
            ),
            "end_to_end_definition": (
                "precommit compute + frame NPZ fsync + frame-directory fsync + "
                "journal fsync"
            ),
            "precommit_compute_summary": sealer._percentile_summary(
                precommit, expected_count=bundle.valid_frame_count
            ),
            "all_frame_end_to_end_summary": sealer._percentile_summary(
                end_to_end, expected_count=bundle.valid_frame_count
            ),
            "warm_frame_end_to_end_summary": sealer._percentile_summary(
                end_to_end[1:], expected_count=bundle.valid_frame_count - 1
            ),
            "warm_frame_count": bundle.valid_frame_count - 1,
            "deadline_uses": (
                "warm_frame_end_to_end_summary_after_global_first_committed_frame"
            ),
            "process_peak_rss_bytes": 123456,
            "integrated_realtime_qualified": False,
            "frames": runtime_rows,
        },
        "output": {
            "committed_frame_count": bundle.valid_frame_count,
            "raw_row_count": sum(row_counts),
            "empty_frame_count": sum(count == 0 for count in row_counts),
            "native_prediction_mutation": False,
            "tracked_csv_created": False,
        },
    }
    provenance_payload = _json_bytes(provenance)
    (root / sealer.PROVENANCE_NAME).write_bytes(provenance_payload)
    record_hash = hashlib.sha256(b"".join(journal_lines[1:])).hexdigest()
    final_seal = {
        "schema": SEAL_SCHEMA,
        "schedule_sha256": bundle.sha256,
        "run_provenance_sha256": hashlib.sha256(provenance_payload).hexdigest(),
        "completed_frame_count": bundle.valid_frame_count,
        "journal_sha256": hashlib.sha256(journal_payload).hexdigest(),
        "frame_record_sha256": record_hash,
        "total_runtime_seconds": sum([0.01] * bundle.valid_frame_count),
        "runtime_seconds_semantics": PRECOMMIT_RUNTIME_SEMANTICS,
    }
    (root / sealer.PROVIDER_SEAL_NAME).write_bytes(_json_bytes(final_seal))
    return bundle


@pytest.fixture(scope="module")
def provider_template(tmp_path_factory):
    root = tmp_path_factory.mktemp("h10-provider") / "provider"
    _build_provider(root)
    return root


def _copy_provider(template: Path, tmp_path: Path) -> Path:
    target = tmp_path / "provider"
    shutil.copytree(template, target, copy_function=os.link)
    return target


def _rewrite_provenance(provider: Path, mutate) -> None:
    path = provider / sealer.PROVENANCE_NAME
    value = json.loads(path.read_text())
    mutate(value)
    payload = _json_bytes(value)
    _replace_bytes(path, payload)
    seal_path = provider / sealer.PROVIDER_SEAL_NAME
    seal = json.loads(seal_path.read_text())
    seal["run_provenance_sha256"] = hashlib.sha256(payload).hexdigest()
    _replace_bytes(seal_path, _json_bytes(seal))


def _rewrite_first_frame(provider: Path, mutate) -> None:
    bundle = parse_exact_schedule_bundle(SCHEDULE)
    scene, frame = bundle.ordered_frames[0]
    frame_path = (
        provider
        / sealer.FRAMES_DIRECTORY_NAME
        / sealer._expected_frame_name(scene, frame)
    )
    with np.load(frame_path, allow_pickle=False) as source:
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    mutate(arrays)
    payload = _npz_bytes(arrays)
    _replace_bytes(frame_path, payload)

    journal_path = provider / sealer.JOURNAL_NAME
    values = [json.loads(line) for line in journal_path.read_text().splitlines()]
    values[1]["file_sha256"] = hashlib.sha256(payload).hexdigest()
    lines = [sealer._canonical_json_line(value) for value in values]
    journal_payload = b"".join(lines)
    _replace_bytes(journal_path, journal_payload)
    seal_path = provider / sealer.PROVIDER_SEAL_NAME
    seal = json.loads(seal_path.read_text())
    seal["journal_sha256"] = hashlib.sha256(journal_payload).hexdigest()
    seal["frame_record_sha256"] = hashlib.sha256(b"".join(lines[1:])).hexdigest()
    _replace_bytes(seal_path, _json_bytes(seal))


def test_seals_complete_numeric_source_and_exact_score_only_k8(
    provider_template, tmp_path
):
    output = tmp_path / "sealed"
    manifest = sealer.seal_raw_source(
        provider_root=provider_template,
        schedule_path=SCHEDULE,
        output_root=output,
    )
    assert manifest["raw_row_count"] == 12
    assert manifest["exact_frame_count"] == 769
    assert manifest["empty_frame_count"] == 767
    assert len(manifest["frame_row_ledger"]) == 769
    assert manifest["frame_row_ledger"][:3] == [[0, 0, 10], [0, 25, 2], [0, 50, 0]]
    assert manifest["tracking_enabled"] is False
    assert manifest["association_applied"] is False
    assert manifest["input_identity"]["byte_identical"] is True
    assert manifest["k8"]["membership_count"] == 10
    assert [row[2] for row in manifest["k8"]["membership_identities"][:8]] == [
        9,
        1,
        2,
        4,
        5,
        6,
        7,
        8,
    ]
    assert manifest["k8"]["membership_identities"][8:] == [
        [0, 25, 0, 2048, 10],
        [0, 25, 1, 2049, 11],
    ]
    with np.load(output / sealer.OUTPUT_NPZ_NAME, allow_pickle=False) as source:
        assert set(source.files) == sealer._SOURCE_ARRAYS
        arrays = {name: np.array(source[name], copy=True) for name in source.files}
    assert arrays["scene_ids"].tolist() == list(
        parse_exact_schedule_bundle(SCHEDULE).scene_order
    )
    assert arrays["per_view_source_row"].dtype == np.int64
    assert arrays["per_view_source_score"].dtype == np.float64
    assert arrays["per_view_center_world"].shape == (12, 3)
    assert sealer._array_content_sha256(arrays) == manifest["array_content_sha256"]
    assert (
        hashlib.sha256((output / sealer.OUTPUT_NPZ_NAME).read_bytes()).hexdigest()
        == manifest["npz_sha256"]
    )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda arrays: arrays["source_row"].__setitem__(1, 0),
            "source rows",
        ),
        (
            lambda arrays: arrays["center"].__setitem__((0, 0), np.nan),
            "non-finite",
        ),
        (
            lambda arrays: arrays["extent"].__setitem__((0, 0), 0.0),
            "extent",
        ),
        (
            lambda arrays: arrays["center"].__setitem__((0, 0), 10_001.0),
            "center exceeds",
        ),
        (
            lambda arrays: arrays["extent"].__setitem__((0, 0), 101.0),
            "extent exceeds",
        ),
        (
            lambda arrays: arrays["quaternion"].__setitem__(0, [1e-8, 0, 0, 0]),
            "quaternion norm",
        ),
        (
            lambda arrays: arrays["quaternion"].__setitem__(0, [1.001, 0, 0, 0]),
            "quaternion norm",
        ),
        (
            lambda arrays: arrays["input_sha256"].__setitem__((0, 0), 255),
            "input hashes",
        ),
    ],
)
def test_numeric_frame_corruption_fails_closed(
    provider_template, tmp_path, mutate, match
):
    provider = _copy_provider(provider_template, tmp_path)
    _rewrite_first_frame(provider, mutate)
    with pytest.raises(sealer.RawSourceSealError, match=match):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "sealed",
        )


def test_artifact_hash_missing_empty_frame_and_tracked_extra_fail_closed(
    provider_template, tmp_path
):
    provider = _copy_provider(provider_template, tmp_path)
    first = sorted((provider / sealer.FRAMES_DIRECTORY_NAME).iterdir())[0]
    _replace_bytes(first, first.read_bytes() + b"tamper")
    with pytest.raises(sealer.RawSourceSealError, match="hash differs"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "bad-hash",
        )

    provider = _copy_provider(provider_template, tmp_path / "missing-copy")
    files = sorted((provider / sealer.FRAMES_DIRECTORY_NAME).iterdir())
    files[-1].unlink()
    with pytest.raises(sealer.RawSourceSealError, match="file set differs"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "missing",
        )

    provider = _copy_provider(provider_template, tmp_path / "tracked-copy")
    (provider / "boxer_3dbbs_tracked.csv").write_bytes(b"")
    with pytest.raises(sealer.RawSourceSealError, match="tracked, temporary, or extra"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "tracked",
        )


def test_provenance_contract_assets_and_seal_binding_fail_closed(
    provider_template, tmp_path
):
    provider = _copy_provider(provider_template, tmp_path / "contract-copy")
    _rewrite_provenance(
        provider,
        lambda value: value["provider_contract"].update({"temporal_state": True}),
    )
    with pytest.raises(sealer.RawSourceSealError, match="temporal_state"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "contract",
        )

    provider = _copy_provider(provider_template, tmp_path / "asset-copy")
    _rewrite_provenance(
        provider,
        lambda value: value["frozen_assets"]["boxer_checkpoint"].update(
            {"sha256_after": "0" * 64}
        ),
    )
    with pytest.raises(sealer.RawSourceSealError, match="changed during inference"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "asset",
        )

    provider = _copy_provider(provider_template, tmp_path / "seal-copy")
    seal_path = provider / sealer.PROVIDER_SEAL_NAME
    seal = json.loads(seal_path.read_text())
    seal["run_provenance_sha256"] = "0" * 64
    _replace_bytes(seal_path, _json_bytes(seal))
    with pytest.raises(sealer.RawSourceSealError, match="provenance hash differs"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "seal",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("birth_enabled", True),
        ("ap_evaluated", True),
        ("gt_used", True),
        ("target_dataset_training_used", True),
    ],
)
def test_forbidden_provider_controls_fail_closed(
    provider_template, tmp_path, field, value
):
    provider = _copy_provider(provider_template, tmp_path)
    _rewrite_provenance(provider, lambda receipt: receipt.__setitem__(field, value))
    with pytest.raises(sealer.RawSourceSealError, match="shadow-only controls"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "forbidden-control",
        )


@pytest.mark.parametrize(
    "asset_name", ["provider_contract", "runner_source", "provider_core_source"]
)
def test_approved_contract_and_provider_code_hashes_are_pinned(
    provider_template, tmp_path, asset_name
):
    provider = _copy_provider(provider_template, tmp_path)

    def replace_with_self_consistent_unapproved_hash(receipt):
        record = receipt["frozen_assets"][asset_name]
        record["sha256_before"] = "1" * 64
        record["expected_sha256"] = "1" * 64
        record["sha256_after"] = "1" * 64

    _rewrite_provenance(provider, replace_with_self_consistent_unapproved_hash)
    with pytest.raises(sealer.RawSourceSealError, match="frozen hash differs"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "unapproved-provider",
        )


def test_formal_t05_receipt_is_opaque_and_only_provenance_hash_is_bound(
    provider_template, tmp_path
):
    provider = _copy_provider(provider_template, tmp_path)
    _rewrite_provenance(
        provider,
        lambda receipt: receipt.__setitem__(
            "formal_t05", {"opaque_provider_owned_receipt": [None, 17, "unparsed"]}
        ),
    )
    manifest = sealer.seal_raw_source(
        provider_root=provider,
        schedule_path=SCHEDULE,
        output_root=tmp_path / "opaque-t05",
    )
    assert (
        manifest["provider_bindings"]["run_provenance_sha256"]
        == hashlib.sha256((provider / sealer.PROVENANCE_NAME).read_bytes()).hexdigest()
    )


def test_provenance_precommit_runtime_must_match_journal_and_npz(
    provider_template, tmp_path
):
    provider = _copy_provider(provider_template, tmp_path)

    def change_provenance_runtime_consistently(receipt):
        runtime = receipt["runtime"]
        runtime["frames"][0]["precommit_compute_seconds"] = 0.011
        runtime["cold_first_frame"] = dict(runtime["frames"][0])
        values = [row["precommit_compute_seconds"] for row in runtime["frames"]]
        runtime["precommit_compute_summary"] = sealer._percentile_summary(
            values, expected_count=len(values)
        )

    _rewrite_provenance(provider, change_provenance_runtime_consistently)
    with pytest.raises(sealer.RawSourceSealError, match="precommit runtimes differ"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "runtime-ledger-mismatch",
        )


def test_journal_cap_off_order_and_runtime_mismatch_fail_closed(
    provider_template, tmp_path
):
    provider = _copy_provider(provider_template, tmp_path / "cap-copy")
    journal_path = provider / sealer.JOURNAL_NAME
    values = [json.loads(line) for line in journal_path.read_text().splitlines()]
    values[1]["row_count"] = sealer.MAX_RAW_ROWS_PER_FRAME + 1
    lines = [sealer._canonical_json_line(value) for value in values]
    payload = b"".join(lines)
    _replace_bytes(journal_path, payload)
    seal_path = provider / sealer.PROVIDER_SEAL_NAME
    seal = json.loads(seal_path.read_text())
    seal["journal_sha256"] = hashlib.sha256(payload).hexdigest()
    seal["frame_record_sha256"] = hashlib.sha256(b"".join(lines[1:])).hexdigest()
    _replace_bytes(seal_path, _json_bytes(seal))
    with pytest.raises(sealer.RawSourceSealError, match="cap exceeded"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "cap",
        )

    provider = _copy_provider(provider_template, tmp_path / "order-copy")
    journal_path = provider / sealer.JOURNAL_NAME
    values = [json.loads(line) for line in journal_path.read_text().splitlines()]
    values[1]["frame_id"] = 25
    lines = [sealer._canonical_json_line(value) for value in values]
    payload = b"".join(lines)
    _replace_bytes(journal_path, payload)
    seal_path = provider / sealer.PROVIDER_SEAL_NAME
    seal = json.loads(seal_path.read_text())
    seal["journal_sha256"] = hashlib.sha256(payload).hexdigest()
    seal["frame_record_sha256"] = hashlib.sha256(b"".join(lines[1:])).hexdigest()
    _replace_bytes(seal_path, _json_bytes(seal))
    with pytest.raises(sealer.RawSourceSealError, match="frame order"):
        sealer.seal_raw_source(
            provider_root=provider,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "order",
        )


def test_before_after_snapshot_change_is_rejected(
    provider_template, tmp_path, monkeypatch
):
    real_snapshot = sealer._rehash_snapshot
    calls = 0

    def changing_snapshot(*args, **kwargs):
        nonlocal calls
        value = real_snapshot(*args, **kwargs)
        calls += 1
        if calls == 2:
            value = dict(value)
            value["schedule"] = "0" * 64
        return value

    monkeypatch.setattr(sealer, "_rehash_snapshot", changing_snapshot)
    with pytest.raises(sealer.RawSourceSealError, match="changed while sealing"):
        sealer.seal_raw_source(
            provider_root=provider_template,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "changed",
        )


def test_atomic_pair_is_create_only_and_race_preserves_competitor(
    provider_template, tmp_path, monkeypatch
):
    source = tmp_path / "source"
    manifest = sealer.seal_raw_source(
        provider_root=provider_template,
        schedule_path=SCHEDULE,
        output_root=source,
    )
    with np.load(source / sealer.OUTPUT_NPZ_NAME, allow_pickle=False) as npz:
        arrays = {name: np.array(npz[name], copy=True) for name in npz.files}

    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "sentinel"
    sentinel.write_bytes(b"mine")
    with pytest.raises(sealer.RawSourceSealError, match="refusing to overwrite"):
        sealer._publish_create_only(
            output_root=existing, arrays=arrays, manifest=manifest
        )
    assert sentinel.read_bytes() == b"mine"

    destination = tmp_path / "race"
    real_rename = sealer._rename_noreplace

    def competing_rename(source_fd, source_name, destination_fd, destination_name):
        destination.mkdir()
        (destination / "competitor").write_bytes(b"theirs")
        real_rename(source_fd, source_name, destination_fd, destination_name)

    monkeypatch.setattr(sealer, "_rename_noreplace", competing_rename)
    with pytest.raises(sealer.RawSourceSealError, match="refusing to overwrite"):
        sealer._publish_create_only(
            output_root=destination, arrays=arrays, manifest=manifest
        )
    assert (destination / "competitor").read_bytes() == b"theirs"
    assert not list(tmp_path.glob(".race.stage.*"))


def test_output_parent_name_swap_is_detected_without_publishing_to_replacement(
    provider_template, tmp_path, monkeypatch
):
    source = tmp_path / "source-for-parent-race"
    manifest = sealer.seal_raw_source(
        provider_root=provider_template,
        schedule_path=SCHEDULE,
        output_root=source,
    )
    with np.load(source / sealer.OUTPUT_NPZ_NAME, allow_pickle=False) as npz:
        arrays = {name: np.array(npz[name], copy=True) for name in npz.files}

    parent = tmp_path / "publish-parent"
    parent.mkdir()
    moved_parent = tmp_path / "moved-publish-parent"
    real_rename = sealer._rename_noreplace

    def swapping_rename(source_fd, source_name, destination_fd, destination_name):
        parent.rename(moved_parent)
        parent.mkdir()
        real_rename(source_fd, source_name, destination_fd, destination_name)

    monkeypatch.setattr(sealer, "_rename_noreplace", swapping_rename)
    with pytest.raises(sealer.RawSourceSealError, match="parent identity changed"):
        sealer._publish_create_only(
            output_root=parent / "sealed", arrays=arrays, manifest=manifest
        )
    assert not (parent / "sealed").exists()
    assert (moved_parent / "sealed" / sealer.OUTPUT_NPZ_NAME).is_file()


def test_staging_inode_swap_and_extra_entry_fail_closed(
    provider_template, tmp_path, monkeypatch
):
    source = tmp_path / "source-for-staging-races"
    manifest = sealer.seal_raw_source(
        provider_root=provider_template,
        schedule_path=SCHEDULE,
        output_root=source,
    )
    with np.load(source / sealer.OUTPUT_NPZ_NAME, allow_pickle=False) as npz:
        arrays = {name: np.array(npz[name], copy=True) for name in npz.files}

    real_open = sealer.os.open
    swapped = False

    def swapping_open(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if (
            not swapped
            and isinstance(path, str)
            and path.startswith(".inode-swap.stage.")
            and flags & getattr(os, "O_DIRECTORY", 0)
            and dir_fd is not None
        ):
            os.rename(
                path,
                f"{path}.moved",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            os.mkdir(path, mode=0o700, dir_fd=dir_fd)
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(sealer.os, "open", swapping_open)
    with pytest.raises(sealer.RawSourceSealError, match="staging identity changed"):
        sealer._publish_create_only(
            output_root=tmp_path / "inode-swap", arrays=arrays, manifest=manifest
        )
    assert not (tmp_path / "inode-swap").exists()
    monkeypatch.setattr(sealer.os, "open", real_open)

    real_write = sealer._write_exclusive_fsync_at
    writes = 0

    def injecting_write(directory_fd, name, payload):
        nonlocal writes
        real_write(directory_fd, name, payload)
        writes += 1
        if writes == 2:
            descriptor = os.open(
                "unexpected-entry",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=directory_fd,
            )
            os.close(descriptor)

    monkeypatch.setattr(sealer, "_write_exclusive_fsync_at", injecting_write)
    with pytest.raises(sealer.RawSourceSealError, match="entry set differs"):
        sealer._publish_create_only(
            output_root=tmp_path / "extra-entry", arrays=arrays, manifest=manifest
        )
    assert not (tmp_path / "extra-entry").exists()


def test_content_hash_is_independent_of_zip_metadata():
    arrays = {
        "x": np.asarray([[1.0, 2.0]], dtype=np.float64),
        "y": np.asarray([3], dtype=np.int64),
    }
    deterministic = sealer._deterministic_npz_bytes(arrays)
    alternate = io.BytesIO()
    with zipfile.ZipFile(alternate, "w") as archive:
        for name, array in arrays.items():
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", (2026, 8, 24, 12, 0, 0))
            archive.writestr(info, payload.getvalue())
    assert (
        hashlib.sha256(deterministic).digest()
        != hashlib.sha256(alternate.getvalue()).digest()
    )
    with np.load(io.BytesIO(alternate.getvalue()), allow_pickle=False) as source:
        loaded = {name: np.array(source[name], copy=True) for name in source.files}
    assert sealer._array_content_sha256(loaded) == sealer._array_content_sha256(arrays)


def test_symlink_provider_and_output_parent_fail_closed(provider_template, tmp_path):
    provider_link = tmp_path / "provider-link"
    provider_link.symlink_to(provider_template, target_is_directory=True)
    with pytest.raises(sealer.RawSourceSealError, match="symlink path component"):
        sealer.seal_raw_source(
            provider_root=provider_link,
            schedule_path=SCHEDULE,
            output_root=tmp_path / "unused",
        )

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(real_parent, target_is_directory=True)
    arrays = {
        "scene_ids": np.asarray(["scene0000_00"], dtype="<U12"),
        "per_view_scene_index": np.empty(0, dtype=np.int16),
        "per_view_frame_id": np.empty(0, dtype=np.int64),
        "per_view_source_row": np.empty(0, dtype=np.int64),
        "per_view_source_instance_id": np.empty(0, dtype=np.int64),
        "per_view_source_score": np.empty(0, dtype=np.float64),
        "per_view_center_world": np.empty((0, 3), dtype=np.float64),
        "per_view_extent_xyz": np.empty((0, 3), dtype=np.float64),
        "per_view_quaternion_wxyz": np.empty((0, 4), dtype=np.float64),
    }
    payload = sealer._deterministic_npz_bytes(arrays)
    manifest = {
        "npz_sha256": hashlib.sha256(payload).hexdigest(),
        "array_content_sha256": sealer._array_content_sha256(arrays),
    }
    with pytest.raises(sealer.RawSourceSealError, match="symlink path component"):
        sealer._publish_create_only(
            output_root=parent_link / "sealed", arrays=arrays, manifest=manifest
        )


def test_sealer_source_has_no_dataset_record_or_evaluator_interface():
    source = Path(sealer.__file__).read_text(encoding="utf-8")
    forbidden = (
        "bbox.npy",
        "axisAlignment",
        "full_annotations.json",
        "import pickle",
        "pickle.load",
        "evaluation.data_util",
        "--annotation",
    )
    assert all(token not in source for token in forbidden)
