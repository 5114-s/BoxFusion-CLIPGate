from __future__ import annotations

import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from boxfusion.tr3d_r2_geometry import yaw_obb_corners_world
from tools import audit_openbox_smov_r2_shadow as audit_r2


def _receipt(
    index: int,
    stable_id: int,
    native: np.ndarray,
    candidate: np.ndarray | None,
) -> dict[str, object]:
    replacing = candidate is not None
    return {
        "native_index": index,
        "stable_id": stable_id,
        "reason": "loo_improved" if replacing else "no_track_memory",
        "hypothesis": "native_yaw_quantile+base" if replacing else None,
        "view_frame_ids": [0, 25, 50] if replacing else [],
        "native_corners": native.tolist(),
        "candidate_corners": None if candidate is None else candidate.tolist(),
        "native_projection_iou": 0.60 if replacing else None,
        "candidate_projection_iou": 0.70 if replacing else None,
        "native_support": 0.60 if replacing else None,
        "candidate_support": 0.70 if replacing else None,
        "native_free_space": 0.20 if replacing else None,
        "candidate_free_space": 0.10 if replacing else None,
        "center_shift_m": 0.10 if replacing else None,
        "volume_ratio": 1.0 if replacing else None,
        "would_replace": replacing,
        "face_extension_signs": [0, 0] if replacing else None,
        "face_extension_delta_m": [0.0, 0.0] if replacing else None,
        "face_strong_mask": [False] * 4 if replacing else None,
        "face_weak_mask": [False] * 4 if replacing else None,
    }


def _fixture(tmp_path: Path) -> dict[str, object]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    scene = "scene0000_00"
    roots = {
        name: tmp_path / name
        for name in ("predictions", "diagnostics", "logs", "anchors", "controls")
    }
    for root in roots.values():
        root.mkdir()
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text(scene + "\n", encoding="utf-8")

    native = np.stack(
        (
            yaw_obb_corners_world(
                np.asarray([0.0, 0.0, 3.0, 2.0, 1.0, 2.0, 0.0])
            ),
            yaw_obb_corners_world(
                np.asarray([4.0, 0.0, 3.0, 1.0, 1.0, 1.0, 0.0])
            ),
        )
    ).astype(np.float32)
    candidate = yaw_obb_corners_world(
        np.asarray([0.1, 0.0, 3.0, 2.0, 1.0, 2.0, 0.0])
    ).astype(np.float32)
    scores = np.asarray([0.9, 0.8], dtype=np.float32)
    stable_ids = np.asarray([7, 8], dtype=np.int64)
    mask = np.asarray([True, False], dtype=np.bool_)
    counterfactual = native.copy()
    counterfactual[0] = candidate
    receipts = [
        _receipt(0, 7, native[0], candidate),
        _receipt(1, 8, native[1], None),
    ]

    prediction_path = roots["predictions"] / f"{scene}_boxes.pkl"
    payload = [[
        (0, native[index].copy(), float(scores[index]))
        for index in range(len(native))
    ]]
    with prediction_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    (roots["anchors"] / prediction_path.name).write_bytes(prediction_path.read_bytes())

    sidecar_path = roots["diagnostics"] / f"{scene}{audit_r2.SIDECAR_SUFFIX}"
    np.savez_compressed(
        sidecar_path,
        schema=np.asarray(audit_r2.SUMMARY_SCHEMA),
        native_corners=native,
        native_scores=scores,
        stable_ids=stable_ids,
        counterfactual_corners=counterfactual,
        would_replace_mask=mask,
        receipts_json=np.frombuffer(
            json.dumps(
                receipts,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
    )

    summary = {
        "schema": audit_r2.SUMMARY_SCHEMA,
        "enabled": True,
        "observer_only": True,
        "closed": True,
        **{key: False for key in audit_r2._FALSE_ATTESTATIONS},
        "scene_id": scene,
        "effective_config": {
            **audit_r2.EXPECTED_CONFIG,
            "diagnostics": {"root": str(roots["diagnostics"])},
        },
        "keyframes": 3,
        "proposals": 2,
        "proposal_cap_drops": 0,
        "valid_fragments": 2,
        "invalid_fragments": 0,
        "accepted_views": 2,
        "same_frame_duplicates": 0,
        "track_capacity_drops": 0,
        "retired_tracks": 0,
        "prepare_failures": 0,
        "would_replace": 1,
        "active_tracks_at_close": 2,
        "core_timing": {"mean_ms": 2.0, "p95_ms": 3.0, "max_ms": 4.0},
        "wrapper_timing": {"mean_ms": 3.0, "p95_ms": 4.0, "max_ms": 5.0},
        "receipts": receipts,
        "terminal": {
            "native_count": 2,
            "counterfactual_count": 1,
            "would_replace_native_indices": [0],
            "would_replace_stable_ids": [7],
            "native_export_mutated": False,
            "counterfactual_geometry_applied": False,
        },
    }
    log_path = roots["logs"] / f"{scene}.log"
    log_path.write_text(
        "Average FPS: 30.00\n"
        + audit_r2.SUMMARY_PREFIX
        + json.dumps(summary, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    (roots["controls"] / f"{scene}.log").write_text(
        "Average FPS: 31.00\n", encoding="utf-8"
    )
    return {
        "scene": scene,
        "scene_list": scene_list,
        "roots": roots,
        "prediction_path": prediction_path,
        "sidecar_path": sidecar_path,
        "log_path": log_path,
        "summary": summary,
        "native": native,
        "scores": scores,
        "stable_ids": stable_ids,
        "counterfactual": counterfactual,
        "mask": mask,
        "receipts": receipts,
    }


def _args(
    data: dict[str, object],
    report: Path,
    *,
    paired: bool = True,
    contract: str | None = None,
):
    roots = data["roots"]
    argv = [
        "--scene-list",
        str(data["scene_list"]),
        "--prediction-root",
        str(roots["predictions"]),
        "--diagnostics-root",
        str(roots["diagnostics"]),
        "--log-root",
        str(roots["logs"]),
        "--anchor-root",
        str(roots["anchors"]),
        "--report",
        str(report),
    ]
    if paired:
        argv.extend(("--control-log-root", str(roots["controls"])))
    if contract is not None:
        argv.extend(("--contract", contract))
    return audit_r2.parser().parse_args(argv)


def _rewrite_log(data: dict[str, object]) -> None:
    data["log_path"].write_text(
        "Average FPS: 30.00\n"
        + audit_r2.SUMMARY_PREFIX
        + json.dumps(data["summary"], sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )


def _rewrite_sidecar(data: dict[str, object]) -> None:
    np.savez_compressed(
        data["sidecar_path"],
        schema=np.asarray(data["summary"]["schema"]),
        native_corners=data["native"],
        native_scores=data["scores"],
        stable_ids=data["stable_ids"],
        counterfactual_corners=data["counterfactual"],
        would_replace_mask=data["mask"],
        receipts_json=np.frombuffer(
            json.dumps(
                data["receipts"],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8"),
            dtype=np.uint8,
        ),
    )


def test_audit_binds_native_prediction_sidecar_receipts_and_paired_latency(tmp_path):
    data = _fixture(tmp_path)
    report_path = tmp_path / "report.json"
    report = audit_r2.audit(_args(data, report_path))

    assert report["passed"] is True
    assert report["scene_count"] == 1
    assert report["native_rows"] == 2
    assert report["would_replace"] == 1
    assert report["paired_realtime_checked"] is True
    assert report["contract"] == "visibility-v2"
    assert report["minimum_paired_fps_ratio"] == pytest.approx(30.0 / 31.0)
    assert report_path.stat().st_mode & 0o222 == 0
    with pytest.raises(FileExistsError, match="refusing existing"):
        audit_r2.audit(_args(data, report_path))


def test_audit_rejects_native_prediction_mutation(tmp_path):
    data = _fixture(tmp_path)
    with data["prediction_path"].open("rb") as handle:
        payload = pickle.load(handle)
    payload[0][0][1][0, 0] += np.float32(0.25)
    with data["prediction_path"].open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with pytest.raises(audit_r2.AuditError, match="sidecar-identical"):
        audit_r2.audit(_args(data, tmp_path / "report.json", paired=False))


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda summary: summary.update(active_authorized=True), "unsafe attestation"),
        (
            lambda summary: summary["wrapper_timing"].update(
                p95_ms=11.0, max_ms=20.0
            ),
            "p95 budget exceeded",
        ),
        (
            lambda summary: summary["effective_config"].update(max_views_per_track=6),
            "effective_config.max_views_per_track drifted",
        ),
    ],
)
def test_audit_rejects_unsafe_summary_or_budget_drift(tmp_path, mutation, match):
    data = _fixture(tmp_path)
    mutation(data["summary"])
    _rewrite_log(data)
    with pytest.raises(audit_r2.AuditError, match=match):
        audit_r2.audit(_args(data, tmp_path / "report.json", paired=False))


def test_audit_rejects_counterfactual_on_an_unselected_row(tmp_path):
    data = _fixture(tmp_path)
    data["counterfactual"][1, :, 0] += np.float32(0.1)
    _rewrite_sidecar(data)
    with pytest.raises(audit_r2.AuditError, match="unselected geometry changed"):
        audit_r2.audit(_args(data, tmp_path / "report.json", paired=False))


def test_audit_rejects_receipt_without_strict_loo_dominance(tmp_path):
    data = _fixture(tmp_path)
    data["receipts"][0]["candidate_projection_iou"] = 0.50
    data["summary"]["receipts"] = data["receipts"]
    _rewrite_sidecar(data)
    _rewrite_log(data)
    with pytest.raises(audit_r2.AuditError, match="LOO dominance mismatch"):
        audit_r2.audit(_args(data, tmp_path / "report.json", paired=False))


def _downgrade_fixture_to_v1(data: dict[str, object]) -> None:
    face_fields = {
        "face_extension_signs",
        "face_extension_delta_m",
        "face_strong_mask",
        "face_weak_mask",
    }
    for receipt in data["receipts"]:
        for field in face_fields:
            receipt.pop(field)
        if receipt["hypothesis"] is not None:
            receipt["hypothesis"] = str(receipt["hypothesis"]).split("+", 1)[0]
    summary = data["summary"]
    summary["schema"] = audit_r2.SUMMARY_SCHEMA_V1
    for field in set(audit_r2.EXPECTED_CONFIG) - set(audit_r2.EXPECTED_CONFIG_V1):
        summary["effective_config"].pop(field)
    summary["receipts"] = data["receipts"]
    _rewrite_sidecar(data)
    _rewrite_log(data)


def test_audit_contracts_keep_v1_and_visibility_v2_strictly_separate(tmp_path):
    old = _fixture(tmp_path / "old")
    _downgrade_fixture_to_v1(old)
    report = audit_r2.audit(
        _args(
            old,
            tmp_path / "v1-report.json",
            paired=False,
            contract="r2-v1",
        )
    )
    assert report["contract"] == "r2-v1"
    assert report["schema"] == audit_r2.REPORT_SCHEMA_V1

    with pytest.raises(audit_r2.AuditError, match="wrong summary schema"):
        audit_r2.audit(
            _args(
                old,
                tmp_path / "wrong-v2.json",
                paired=False,
                contract="visibility-v2",
            )
        )

    current = _fixture(tmp_path / "current")
    with pytest.raises(audit_r2.AuditError, match="wrong summary schema"):
        audit_r2.audit(
            _args(
                current,
                tmp_path / "wrong-v1.json",
                paired=False,
                contract="r2-v1",
            )
        )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda receipt: receipt.update(
                face_strong_mask=[True, False, False, False],
                face_weak_mask=[False, False, False, False],
            ),
            "strong face",
        ),
        (
            lambda receipt: receipt.update(
                hypothesis="native_yaw_quantile+face_x",
                face_extension_signs=[1, 0],
                face_extension_delta_m=[0.31, 0.0],
                face_strong_mask=[True, False, False, False],
                face_weak_mask=[True, False, False, False],
            ),
            "out of bounds",
        ),
        (
            lambda receipt: receipt.update(
                hypothesis="native_yaw_quantile+face_x",
                face_extension_signs=[1, 0],
                face_extension_delta_m=[0.30, 0.0],
                face_strong_mask=[True, False, False, False],
                face_weak_mask=[True, True, False, False],
            ),
            "visible-anchor/unseen-face",
        ),
    ],
)
def test_visibility_v2_audit_rejects_invalid_face_metadata(
    tmp_path, mutation, match
):
    data = _fixture(tmp_path)
    mutation(data["receipts"][0])
    data["summary"]["receipts"] = data["receipts"]
    _rewrite_sidecar(data)
    _rewrite_log(data)
    with pytest.raises(audit_r2.AuditError, match=match):
        audit_r2.audit(_args(data, tmp_path / "report.json", paired=False))
