from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from tools import materialize_openbox_smov_r2_counterfactual as tool


_SIGNS = np.asarray(
    [
        [-1, -1, -1],
        [-1, -1, 1],
        [-1, 1, -1],
        [-1, 1, 1],
        [1, -1, -1],
        [1, -1, 1],
        [1, 1, -1],
        [1, 1, 1],
    ],
    dtype=np.float32,
)


def _corners(center: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(center, dtype=np.float32) + _SIGNS


def _score(value: float) -> float:
    return float(np.float32(value))


def _write_prediction(
    path: Path,
    corners: np.ndarray,
    scores: tuple[float, ...],
    labels: tuple[int, ...],
    *,
    order: str = "C",
) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [[
        (
            labels[index],
            np.array(corners[index], dtype=np.float32, order=order, copy=True),
            scores[index],
        )
        for index in range(len(labels))
    ]]
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    path.write_bytes(encoded)
    return encoded


def _receipt(
    *,
    index: int,
    stable_id: int,
    native: np.ndarray,
    candidate: np.ndarray | None,
    replacing: bool,
    face: bool = False,
) -> dict[str, object]:
    if candidate is None:
        hypothesis = None
        native_iou = candidate_iou = None
        native_support = candidate_support = None
        native_free = candidate_free = None
        shift = volume = None
        signs = deltas = strong = weak = None
        frames: list[int] = []
    else:
        hypothesis = (
            "native_yaw_quantile+face_x"
            if face
            else "native_yaw_quantile+base"
        )
        if replacing:
            native_iou, candidate_iou = 0.4, 0.6
            native_support, candidate_support = 0.5, 0.7
            native_free, candidate_free = 0.3, 0.2
        else:
            native_iou, candidate_iou = 0.6, 0.5
            native_support, candidate_support = 0.7, 0.5
            native_free, candidate_free = 0.2, 0.3
        shift, volume = 0.1, 1.1
        frames = [0, 25, 50]
        if face:
            signs = [1, 0]
            deltas = [0.30, 0.0]
            strong = [True, False, False, False]
            weak = [True, False, False, False]
        else:
            signs = [0, 0]
            deltas = [0.0, 0.0]
            strong = [False, False, False, False]
            weak = [False, False, False, False]
    return {
        "native_index": index,
        "stable_id": stable_id,
        "reason": (
            "loo_improved"
            if replacing
            else "loo_not_improved"
            if candidate is not None
            else "no_track_memory"
        ),
        "hypothesis": hypothesis,
        "view_frame_ids": frames,
        "native_corners": native.tolist(),
        "candidate_corners": None if candidate is None else candidate.tolist(),
        "native_projection_iou": native_iou,
        "candidate_projection_iou": candidate_iou,
        "native_support": native_support,
        "candidate_support": candidate_support,
        "native_free_space": native_free,
        "candidate_free_space": candidate_free,
        "center_shift_m": shift,
        "volume_ratio": volume,
        "would_replace": replacing,
        "face_extension_signs": signs,
        "face_extension_delta_m": deltas,
        "face_strong_mask": strong,
        "face_weak_mask": weak,
    }


def _write_sidecar(
    path: Path,
    native: np.ndarray,
    scores: tuple[float, ...],
    *,
    mask: np.ndarray,
    candidates: tuple[np.ndarray | None, ...],
    face_rows: tuple[int, ...] = (),
    schema: str = tool.SIDECAR_SCHEMA,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(native)
    stable_ids = np.arange(100, 100 + count, dtype=np.int64)
    counterfactual = native.copy()
    receipts = []
    for index in range(count):
        candidate = candidates[index]
        if mask[index]:
            assert candidate is not None
            counterfactual[index] = candidate
        receipts.append(
            _receipt(
                index=index,
                stable_id=int(stable_ids[index]),
                native=native[index],
                candidate=candidate,
                replacing=bool(mask[index]),
                face=index in face_rows,
            )
        )
    encoded = json.dumps(
        receipts, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    np.savez_compressed(
        path,
        schema=np.asarray(schema),
        native_corners=np.asarray(native, dtype=np.float32),
        native_scores=np.asarray(scores, dtype=np.float32),
        stable_ids=stable_ids,
        counterfactual_corners=np.asarray(counterfactual, dtype=np.float32),
        would_replace_mask=np.asarray(mask, dtype=np.bool_),
        receipts_json=np.frombuffer(encoded, dtype=np.uint8),
    )


def _args(tmp_path: Path, scenes: tuple[str, ...]) -> argparse.Namespace:
    scene_list = tmp_path / "scenes.txt"
    scene_list.write_text("\n".join(scenes), encoding="utf-8")
    prediction_root = tmp_path / "native"
    diagnostics_root = tmp_path / "sidecars"
    prediction_root.mkdir()
    diagnostics_root.mkdir()
    return argparse.Namespace(
        scene_list=scene_list,
        prediction_root=prediction_root,
        diagnostics_root=diagnostics_root,
        output_root=tmp_path / "counterfactual",
        manifest=tmp_path / "counterfactual_manifest.json",
    )


def _load_rows(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
        assert handle.read() == b""
    return payload[0]


def test_materializes_only_masked_geometry_and_preserves_row_identity(
    tmp_path: Path,
) -> None:
    scene = "scene0000_00"
    args = _args(tmp_path, (scene,))
    native = np.stack(
        (_corners((0, 0, 2)), _corners((4, 0, 2)), _corners((8, 0, 2)))
    )
    scores = (_score(0.91), _score(0.73), _score(0.21))
    labels = (4, 2, 9)
    native_path = args.prediction_root / f"{scene}{tool.PREDICTION_SUFFIX}"
    native_bytes = _write_prediction(native_path, native, scores, labels)
    # Row 0 intentionally has a non-null, rejected face candidate.  A
    # materializer that keys on candidate presence instead of the mask will
    # corrupt this row.
    rejected = _corners((20, 0, 2))
    accepted = _corners((5, 0, 2))
    mask = np.asarray([False, True, False])
    sidecar_path = args.diagnostics_root / f"{scene}{tool.SIDECAR_SUFFIX}"
    _write_sidecar(
        sidecar_path,
        native,
        scores,
        mask=mask,
        candidates=(rejected, accepted, None),
        face_rows=(0,),
    )
    sidecar_bytes = sidecar_path.read_bytes()

    manifest = tool.materialize(args)

    rows = _load_rows(args.output_root / f"{scene}{tool.PREDICTION_SUFFIX}")
    assert {path.name for path in args.output_root.iterdir()} == {
        f"{scene}{tool.PREDICTION_SUFFIX}"
    }
    assert [row[0] for row in rows] == list(labels)
    assert [row[2] for row in rows] == list(scores)
    assert len(rows) == len(native)
    np.testing.assert_array_equal(rows[0][1], native[0])
    np.testing.assert_array_equal(rows[1][1], accepted)
    np.testing.assert_array_equal(rows[2][1], native[2])
    assert native_path.read_bytes() == native_bytes
    assert sidecar_path.read_bytes() == sidecar_bytes

    assert manifest["schema"] == tool.SCHEMA
    assert manifest["contract"] == "visibility-v2"
    assert manifest["selection_policy"] == "would_replace_mask_only"
    assert manifest["replaced_rows"] == 1
    assert manifest["labels_unchanged"] is True
    assert manifest["scores_unchanged"] is True
    assert manifest["row_order_unchanged"] is True
    scene_row = manifest["scenes"][0]
    assert scene_row["replaced_native_indices"] == [1]
    assert scene_row["replaced_stable_ids"] == [101]
    assert scene_row["replaced_hypotheses"] == [
        "native_yaw_quantile+base"
    ]
    assert scene_row["score_bits_sha256_before"] == scene_row[
        "score_bits_sha256_after"
    ]
    assert json.loads(args.manifest.read_text(encoding="utf-8")) == manifest

    output_before = (
        args.output_root / f"{scene}{tool.PREDICTION_SUFFIX}"
    ).read_bytes()
    with pytest.raises(FileExistsError, match="existing output root"):
        tool.materialize(args)
    assert (
        args.output_root / f"{scene}{tool.PREDICTION_SUFFIX}"
    ).read_bytes() == output_before


def test_no_replacement_scene_is_byte_identical_to_native(tmp_path: Path) -> None:
    scene = "scene0001_00"
    args = _args(tmp_path, (scene,))
    native = np.stack((_corners((0, 0, 2)),))
    scores = (_score(0.5),)
    source = _write_prediction(
        args.prediction_root / f"{scene}{tool.PREDICTION_SUFFIX}",
        native,
        scores,
        (7,),
        order="F",
    )
    _write_sidecar(
        args.diagnostics_root / f"{scene}{tool.SIDECAR_SUFFIX}",
        native,
        scores,
        mask=np.asarray([False]),
        candidates=(None,),
    )

    manifest = tool.materialize(args)

    output = (
        args.output_root / f"{scene}{tool.PREDICTION_SUFFIX}"
    ).read_bytes()
    assert output == source
    assert manifest["scenes"][0]["no_replacement_byte_identity"] is True
    assert manifest["replaced_rows"] == 0


@pytest.mark.parametrize(
    "failure",
    ("schema", "native", "score", "unselected", "receipt", "candidate"),
)
def test_preflight_rejects_inconsistent_sidecar_without_creating_outputs(
    tmp_path: Path, failure: str
) -> None:
    scene = "scene0002_00"
    args = _args(tmp_path, (scene,))
    native = np.stack((_corners((0, 0, 2)),))
    scores = (_score(0.8),)
    _write_prediction(
        args.prediction_root / f"{scene}{tool.PREDICTION_SUFFIX}",
        native,
        scores,
        (0,),
    )
    candidate = _corners((1, 0, 2))
    path = args.diagnostics_root / f"{scene}{tool.SIDECAR_SUFFIX}"
    _write_sidecar(
        path,
        native,
        scores,
        mask=np.asarray([failure in {"receipt", "candidate"}]),
        candidates=(candidate if failure in {"receipt", "candidate"} else None,),
        schema=("boxfusion.openbox_smov_r2_shadow.v1" if failure == "schema" else tool.SIDECAR_SCHEMA),
    )
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.array(archive[name], copy=True) for name in archive.files}
    if failure == "native":
        data["native_corners"][0, 0, 0] += np.float32(0.25)
    elif failure == "score":
        data["native_scores"][0] += np.float32(0.25)
    elif failure == "unselected":
        data["counterfactual_corners"][0, 0, 0] += np.float32(0.25)
    elif failure == "receipt":
        receipts = json.loads(data["receipts_json"].tobytes().decode("utf-8"))
        receipts[0]["would_replace"] = False
        data["receipts_json"] = np.frombuffer(
            json.dumps(receipts, sort_keys=True, separators=(",", ":")).encode(),
            dtype=np.uint8,
        )
    elif failure == "candidate":
        data["counterfactual_corners"][0, 0, 0] += np.float32(0.25)
    np.savez_compressed(path, **data)

    with pytest.raises(tool.MaterializationError):
        tool.materialize(args)
    assert not args.output_root.exists()
    assert not args.manifest.exists()


def test_all_scenes_preflight_before_any_output_is_created(tmp_path: Path) -> None:
    scenes = ("scene0003_00", "scene0004_00")
    args = _args(tmp_path, scenes)
    native = np.stack((_corners((0, 0, 2)),))
    scores = (_score(0.6),)
    for index, scene in enumerate(scenes):
        _write_prediction(
            args.prediction_root / f"{scene}{tool.PREDICTION_SUFFIX}",
            native,
            scores,
            (index,),
        )
        _write_sidecar(
            args.diagnostics_root / f"{scene}{tool.SIDECAR_SUFFIX}",
            native,
            scores,
            mask=np.asarray([False]),
            candidates=(None,),
            schema=(
                tool.SIDECAR_SCHEMA
                if index == 0
                else "boxfusion.openbox_smov_r2_shadow.v1"
            ),
        )

    with pytest.raises(tool.MaterializationError, match="visibility-v2"):
        tool.materialize(args)
    assert not args.output_root.exists()
    assert not args.manifest.exists()


def test_scene_list_and_artifact_sets_are_complete_and_unique(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path, ("scene0005_00",))
    args.scene_list.write_text("scene0005_00\nscene0005_00\n", encoding="utf-8")
    with pytest.raises(tool.MaterializationError, match="unique"):
        tool.materialize(args)
    assert not args.output_root.exists()

    args.scene_list.write_text("scene0005_00\n", encoding="utf-8")
    native = np.stack((_corners((0, 0, 2)),))
    scores = (_score(0.4),)
    _write_prediction(
        args.prediction_root / f"scene0005_00{tool.PREDICTION_SUFFIX}",
        native,
        scores,
        (0,),
    )
    _write_prediction(
        args.prediction_root / f"scene9999_00{tool.PREDICTION_SUFFIX}",
        native,
        scores,
        (0,),
    )
    _write_sidecar(
        args.diagnostics_root / f"scene0005_00{tool.SIDECAR_SUFFIX}",
        native,
        scores,
        mask=np.asarray([False]),
        candidates=(None,),
    )
    with pytest.raises(tool.MaterializationError, match="artifact set differs"):
        tool.materialize(args)
    assert not args.output_root.exists()


def test_prediction_with_trailing_bytes_is_rejected(tmp_path: Path) -> None:
    scene = "scene0006_00"
    args = _args(tmp_path, (scene,))
    native = np.stack((_corners((0, 0, 2)),))
    scores = (_score(0.4),)
    path = args.prediction_root / f"{scene}{tool.PREDICTION_SUFFIX}"
    _write_prediction(path, native, scores, (0,))
    path.write_bytes(path.read_bytes() + b"trailing")
    _write_sidecar(
        args.diagnostics_root / f"{scene}{tool.SIDECAR_SUFFIX}",
        native,
        scores,
        mask=np.asarray([False]),
        candidates=(None,),
    )

    with pytest.raises(tool.MaterializationError, match="trailing bytes"):
        tool.materialize(args)
    assert not args.output_root.exists()


def test_existing_manifest_is_never_overwritten(tmp_path: Path) -> None:
    args = _args(tmp_path, ("scene0007_00",))
    marker = b"existing-manifest\n"
    args.manifest.write_bytes(marker)

    with pytest.raises(FileExistsError, match="existing manifest"):
        tool.materialize(args)

    assert args.manifest.read_bytes() == marker
    assert not args.output_root.exists()
