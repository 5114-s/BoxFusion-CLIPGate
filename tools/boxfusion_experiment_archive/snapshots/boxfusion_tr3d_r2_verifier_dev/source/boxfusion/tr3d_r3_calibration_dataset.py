"""Immutable train-only dataset for the R3 veto calibrator."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from .tr3d_r3_calibrator import CLASS_NAMES, FEATURE_NAMES


DATASET_SCHEMA = "boxfusion.tr3d_r3_veto_dataset.v1"
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
GAIN_CLASS = 0
SAFE_NEUTRAL_CLASS = 1
HARM_CLASS = 2


def _pairwise_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)), dtype=np.float64)
    lower = np.maximum(left[:, None, :3], right[None, :, :3])
    upper = np.minimum(left[:, None, 3:], right[None, :, 3:])
    intersection = np.prod(np.maximum(upper - lower, 0.0), axis=2)
    left_volume = np.prod(left[:, 3:] - left[:, :3], axis=1)
    right_volume = np.prod(right[:, 3:] - right[:, :3], axis=1)
    union = left_volume[:, None] + right_volume[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def greedy_tp_count(
    boxes: object, scores: object, ground_truth: object, threshold: float
) -> int:
    """Match one scene exactly like the frozen ScanNet evaluator."""

    predictions = np.asarray(boxes, dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if predictions.ndim != 2 or predictions.shape[1:] != (6,):
        raise ValueError("boxes must be [N,6]")
    if values.shape != (len(predictions),) or not np.isfinite(values).all():
        raise ValueError("scores must be finite [N]")
    if gt.ndim != 2 or gt.shape[1:] != (6,):
        raise ValueError("ground_truth must be [M,6]")
    if not (0 < float(threshold) <= 1):
        raise ValueError("threshold must be in (0,1]")
    if not len(predictions) or not len(gt):
        return 0
    overlaps = _pairwise_iou(predictions, gt)
    # Stable prediction row is the secondary key after score descending.
    order = np.lexsort((np.arange(len(values), dtype=np.int64), -values))
    used = np.zeros(len(gt), dtype=np.bool_)
    matched = 0
    for prediction in order:
        target = int(np.argmax(overlaps[int(prediction)]))
        if overlaps[int(prediction), target] > threshold and not used[target]:
            used[target] = True
            matched += 1
    return matched


def label_single_replacement(
    anchor_boxes: object,
    anchor_scores: object,
    ground_truth: object,
    *,
    anchor_index: int,
    candidate_box: object,
) -> tuple[int, np.ndarray]:
    """Label one primary replacement from exact scene-level TP changes."""

    anchors = np.asarray(anchor_boxes, dtype=np.float64)
    scores = np.asarray(anchor_scores, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    candidate = np.asarray(candidate_box, dtype=np.float64)
    if candidate.shape != (6,) or not np.isfinite(candidate).all():
        raise ValueError("candidate_box must be finite [6]")
    if anchor_index < 0 or anchor_index >= len(anchors):
        raise ValueError("anchor_index is out of range")
    replacement = anchors.copy()
    replacement[anchor_index] = candidate
    deltas = np.asarray(
        [
            greedy_tp_count(replacement, scores, gt, threshold)
            - greedy_tp_count(anchors, scores, gt, threshold)
            for threshold in IOU_THRESHOLDS
        ],
        dtype=np.int8,
    )
    if np.any(deltas < 0):
        label = HARM_CLASS
    elif int(deltas[2]) > 0 and int(deltas[0]) >= 0 and int(deltas[1]) >= 0:
        label = GAIN_CLASS
    else:
        label = SAFE_NEUTRAL_CLASS
    return label, deltas


def label_joint_replacements(
    anchor_boxes: object,
    anchor_scores: object,
    ground_truth: object,
    *,
    anchor_indices: object,
    candidate_boxes: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Label primary candidates by leave-one-out contribution to raw R3.

    Independent G0-to-candidate labels are not composition-safe: two
    individually neutral replacements can compete for the same GT when both
    are active.  The veto model starts from the *joint raw-primary output*, so
    each label measures what changes when that candidate alone is restored to
    its G0 geometry.  Whole-output OOF AP remains the authoritative gate.
    """

    anchors = np.asarray(anchor_boxes, dtype=np.float64)
    scores = np.asarray(anchor_scores, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    indices = np.asarray(anchor_indices, dtype=np.int64)
    candidates = np.asarray(candidate_boxes, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[1:] != (6,):
        raise ValueError("anchor_boxes must be [A,6]")
    if scores.shape != (len(anchors),) or not np.isfinite(scores).all():
        raise ValueError("anchor_scores must be finite [A]")
    if gt.ndim != 2 or gt.shape[1:] != (6,):
        raise ValueError("ground_truth must be [G,6]")
    if indices.ndim != 1 or candidates.shape != (len(indices), 6):
        raise ValueError("joint candidates must be anchor_indices [N], boxes [N,6]")
    if len(indices) and (
        np.any(indices < 0)
        or np.any(indices >= len(anchors))
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError("joint primary candidates must map to unique valid anchors")
    if not np.isfinite(candidates).all() or (
        len(candidates) and np.any(candidates[:, 3:] <= candidates[:, :3])
    ):
        raise ValueError("joint candidate boxes are invalid")
    raw = anchors.copy()
    raw[indices] = candidates
    raw_counts = np.asarray(
        [greedy_tp_count(raw, scores, gt, threshold) for threshold in IOU_THRESHOLDS],
        dtype=np.int64,
    )
    labels = np.empty(len(indices), dtype=np.int8)
    deltas = np.empty((len(indices), len(IOU_THRESHOLDS)), dtype=np.int8)
    for row, anchor_index in enumerate(indices.tolist()):
        without = raw.copy()
        without[anchor_index] = anchors[anchor_index]
        contribution = raw_counts - np.asarray(
            [
                greedy_tp_count(without, scores, gt, threshold)
                for threshold in IOU_THRESHOLDS
            ],
            dtype=np.int64,
        )
        if np.any(contribution < 0):
            label = HARM_CLASS
        elif int(contribution[2]) > 0 and int(contribution[0]) >= 0 and int(contribution[1]) >= 0:
            label = GAIN_CLASS
        else:
            label = SAFE_NEUTRAL_CLASS
        labels[row] = label
        deltas[row] = contribution.astype(np.int8)
    return labels, deltas


@dataclass(frozen=True)
class R3CalibrationDataset:
    scene_ids: np.ndarray
    anchor_offsets: np.ndarray
    anchor_boxes: np.ndarray
    anchor_scores: np.ndarray
    gt_offsets: np.ndarray
    gt_boxes: np.ndarray
    sample_scene_index: np.ndarray
    sample_anchor_index: np.ndarray
    proposal_ids: np.ndarray
    candidate_boxes: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    tp_deltas: np.ndarray
    ap_deltas: np.ndarray
    provenance: Mapping[str, Any]

    @property
    def scene_count(self) -> int:
        return int(len(self.scene_ids))

    @property
    def sample_count(self) -> int:
        return int(len(self.labels))

    def validate(self) -> "R3CalibrationDataset":
        scenes = np.asarray(self.scene_ids)
        if scenes.ndim != 1 or scenes.dtype.kind != "U" or not len(scenes):
            raise ValueError("scene_ids must be a non-empty unicode vector")
        if len(set(scenes.tolist())) != len(scenes):
            raise ValueError("scene_ids must be unique")
        anchor_offsets = np.asarray(self.anchor_offsets)
        gt_offsets = np.asarray(self.gt_offsets)
        anchors = np.asarray(self.anchor_boxes)
        scores = np.asarray(self.anchor_scores)
        gt = np.asarray(self.gt_boxes)
        for name, value, dtype in (
            ("anchor_offsets", anchor_offsets, np.dtype(np.int64)),
            ("gt_offsets", gt_offsets, np.dtype(np.int64)),
            ("anchor_boxes", anchors, np.dtype(np.float64)),
            ("anchor_scores", scores, np.dtype(np.float64)),
            ("gt_boxes", gt, np.dtype(np.float64)),
        ):
            if value.dtype != dtype:
                raise ValueError(f"{name} must have exact dtype {dtype}")
        if anchor_offsets.shape != (len(scenes) + 1,) or gt_offsets.shape != (
            len(scenes) + 1,
        ):
            raise ValueError("ragged offsets disagree with scene count")
        for name, offsets, count in (
            ("anchor_offsets", anchor_offsets, len(anchors)),
            ("gt_offsets", gt_offsets, len(gt)),
        ):
            if offsets[0] != 0 or offsets[-1] != count or np.any(np.diff(offsets) < 0):
                raise ValueError(f"{name} is not a valid ragged offset vector")
        if anchors.ndim != 2 or anchors.shape[1:] != (6,):
            raise ValueError("anchor_boxes must be [A,6]")
        if scores.shape != (len(anchors),):
            raise ValueError("anchor_scores must be [A]")
        if gt.ndim != 2 or gt.shape[1:] != (6,):
            raise ValueError("gt_boxes must be [G,6]")
        if not np.isfinite(anchors).all() or not np.isfinite(scores).all() or not np.isfinite(gt).all():
            raise ValueError("anchor/GT arrays must be finite")
        if len(np.unique(scores)) != len(scores):
            raise ValueError(
                "global anchor scores must be unique so formal evaluator tie order is irrelevant"
            )
        if np.any(anchors[:, 3:] <= anchors[:, :3]) or (
            len(gt) and np.any(gt[:, 3:] <= gt[:, :3])
        ):
            raise ValueError("anchor/GT boxes must have positive extent")

        count = self.sample_count
        sample_scene = np.asarray(self.sample_scene_index)
        sample_anchor = np.asarray(self.sample_anchor_index)
        proposal_ids = np.asarray(self.proposal_ids)
        candidates = np.asarray(self.candidate_boxes)
        features = np.asarray(self.features)
        labels = np.asarray(self.labels)
        deltas = np.asarray(self.tp_deltas)
        ap_deltas = np.asarray(self.ap_deltas)
        for name, value, dtype in (
            ("sample_scene_index", sample_scene, np.dtype(np.int64)),
            ("sample_anchor_index", sample_anchor, np.dtype(np.int64)),
            ("proposal_ids", proposal_ids, np.dtype(np.int64)),
            ("candidate_boxes", candidates, np.dtype(np.float64)),
            ("features", features, np.dtype(np.float64)),
            ("labels", labels, np.dtype(np.int8)),
            ("tp_deltas", deltas, np.dtype(np.int8)),
            ("ap_deltas", ap_deltas, np.dtype(np.float64)),
        ):
            if value.dtype != dtype:
                raise ValueError(f"{name} must have exact dtype {dtype}")
        for name, value in (
            ("sample_scene_index", sample_scene),
            ("sample_anchor_index", sample_anchor),
            ("proposal_ids", proposal_ids),
            ("labels", labels),
        ):
            if value.shape != (count,):
                raise ValueError(f"{name} must be [sample_count]")
        if candidates.shape != (count, 6):
            raise ValueError("candidate_boxes must be [N,6]")
        if features.shape != (count, len(FEATURE_NAMES)):
            raise ValueError("features must be [N,6]")
        if deltas.shape != (count, len(IOU_THRESHOLDS)):
            raise ValueError("tp_deltas must be [N,3]")
        if ap_deltas.shape != (count, len(IOU_THRESHOLDS)):
            raise ValueError("ap_deltas must be [N,3]")
        if (
            not np.isfinite(candidates).all()
            or not np.isfinite(features).all()
            or not np.isfinite(ap_deltas).all()
            or np.any(candidates[:, 3:] <= candidates[:, :3])
        ):
            raise ValueError("candidate/features arrays are invalid")
        if count and (
            np.any(sample_scene < 0)
            or np.any(sample_scene >= len(scenes))
            or np.any((labels < 0) | (labels >= len(CLASS_NAMES)))
            or np.any((deltas < -1) | (deltas > 1))
        ):
            raise ValueError("sample class/index/delta values are invalid")
        epsilon = 1e-12
        expected_labels = np.full(count, SAFE_NEUTRAL_CLASS, dtype=np.int8)
        harm = np.any(ap_deltas < -epsilon, axis=1) | np.any(deltas < 0, axis=1)
        gain = (
            ~harm
            & (ap_deltas[:, 2] > epsilon)
            & np.all(ap_deltas[:, :2] >= -epsilon, axis=1)
            & np.all(deltas >= 0, axis=1)
        )
        expected_labels[harm] = HARM_CLASS
        expected_labels[gain] = GAIN_CLASS
        if not np.array_equal(labels, expected_labels):
            raise ValueError("labels disagree with rank-aware AP/TP contributions")
        observed: set[tuple[int, int]] = set()
        for row, (scene_index, anchor_index) in enumerate(
            zip(sample_scene.tolist(), sample_anchor.tolist())
        ):
            local_count = int(anchor_offsets[scene_index + 1] - anchor_offsets[scene_index])
            if anchor_index < 0 or anchor_index >= local_count:
                raise ValueError(f"sample {row} anchor index is out of range")
            key = (scene_index, anchor_index)
            if key in observed:
                raise ValueError("dataset has more than one primary sample per anchor")
            observed.add(key)
        encoded = json.dumps(dict(self.provenance), sort_keys=True, allow_nan=False)
        normalized_provenance = json.loads(encoded)
        if not isinstance(normalized_provenance, dict):
            raise ValueError("provenance must be a JSON mapping")
        if normalized_provenance.get("global_anchor_scores_unique") is not True:
            raise ValueError("provenance must attest unique global anchor scores")
        return self

    def as_npz_payload(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "schema": np.asarray(DATASET_SCHEMA),
            "feature_names_json": np.asarray(json.dumps(list(FEATURE_NAMES))),
            "class_names_json": np.asarray(json.dumps(list(CLASS_NAMES))),
            "iou_thresholds": np.asarray(IOU_THRESHOLDS, dtype=np.float64),
            "scene_ids": np.asarray(self.scene_ids),
            "anchor_offsets": np.asarray(self.anchor_offsets, dtype=np.int64),
            "anchor_boxes": np.asarray(self.anchor_boxes, dtype=np.float64),
            "anchor_scores": np.asarray(self.anchor_scores, dtype=np.float64),
            "gt_offsets": np.asarray(self.gt_offsets, dtype=np.int64),
            "gt_boxes": np.asarray(self.gt_boxes, dtype=np.float64),
            "sample_scene_index": np.asarray(self.sample_scene_index, dtype=np.int64),
            "sample_anchor_index": np.asarray(self.sample_anchor_index, dtype=np.int64),
            "proposal_ids": np.asarray(self.proposal_ids, dtype=np.int64),
            "candidate_boxes": np.asarray(self.candidate_boxes, dtype=np.float64),
            "features": np.asarray(self.features, dtype=np.float64),
            "labels": np.asarray(self.labels, dtype=np.int8),
            "tp_deltas": np.asarray(self.tp_deltas, dtype=np.int8),
            "ap_deltas": np.asarray(self.ap_deltas, dtype=np.float64),
            "provenance_json": np.asarray(
                json.dumps(dict(self.provenance), sort_keys=True, allow_nan=False)
            ),
        }


def _text(values: Mapping[str, np.ndarray], name: str) -> str:
    value = np.asarray(values[name])
    if value.shape != () or value.dtype.hasobject:
        raise ValueError(f"{name} must be a non-object scalar")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    if not isinstance(result, str):
        raise ValueError(f"{name} must be text")
    return result


def load_dataset(path: str | Path) -> R3CalibrationDataset:
    with np.load(Path(path), allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in archive.files}
    expected = {
        "schema", "feature_names_json", "class_names_json", "iou_thresholds",
        "scene_ids", "anchor_offsets", "anchor_boxes", "anchor_scores",
        "gt_offsets", "gt_boxes", "sample_scene_index", "sample_anchor_index",
        "proposal_ids", "candidate_boxes", "features", "labels", "tp_deltas",
        "ap_deltas",
        "provenance_json",
    }
    if set(values) != expected or _text(values, "schema") != DATASET_SCHEMA:
        raise ValueError("unsupported or incomplete R3 calibration dataset")
    if json.loads(_text(values, "feature_names_json")) != list(FEATURE_NAMES):
        raise ValueError("dataset feature order changed")
    if json.loads(_text(values, "class_names_json")) != list(CLASS_NAMES):
        raise ValueError("dataset class order changed")
    if not np.array_equal(values["iou_thresholds"], np.asarray(IOU_THRESHOLDS)):
        raise ValueError("dataset IoU thresholds changed")
    dataset = R3CalibrationDataset(
        scene_ids=values["scene_ids"],
        anchor_offsets=values["anchor_offsets"],
        anchor_boxes=values["anchor_boxes"],
        anchor_scores=values["anchor_scores"],
        gt_offsets=values["gt_offsets"],
        gt_boxes=values["gt_boxes"],
        sample_scene_index=values["sample_scene_index"],
        sample_anchor_index=values["sample_anchor_index"],
        proposal_ids=values["proposal_ids"],
        candidate_boxes=values["candidate_boxes"],
        features=values["features"],
        labels=values["labels"],
        tp_deltas=values["tp_deltas"],
        ap_deltas=values["ap_deltas"],
        provenance=json.loads(_text(values, "provenance_json")),
    ).validate()
    recomputed_labels, recomputed_tp, recomputed_ap = (
        label_dataset_global_leave_one_out(dataset)
    )
    if not np.array_equal(dataset.labels, recomputed_labels):
        raise ValueError("stored labels disagree with geometry/GT recomputation")
    if not np.array_equal(dataset.tp_deltas, recomputed_tp):
        raise ValueError("stored TP deltas disagree with geometry/GT recomputation")
    if not np.allclose(dataset.ap_deltas, recomputed_ap, rtol=0.0, atol=1e-12):
        raise ValueError("stored AP deltas disagree with geometry/GT recomputation")
    return dataset


def write_dataset(path: str | Path, dataset: R3CalibrationDataset) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    np.savez_compressed(buffer, **dataset.as_npz_payload())
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp",
            dir=target.parent, delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(buffer.getvalue())
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_name, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"immutable R3 calibration dataset exists: {target}") from error
    finally:
        if temporary_name is not None:
            try:
                os.unlink(temporary_name)
            except FileNotFoundError:
                pass
    return target


def prediction_rows(
    dataset: R3CalibrationDataset,
    accepted: object | None,
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Reconstruct baseline/raw/veto geometry without changing score/order."""

    values = dataset.validate()
    if accepted is None:
        mask = np.zeros(values.sample_count, dtype=np.bool_)
    else:
        mask = np.asarray(accepted, dtype=np.bool_)
        if mask.shape != (values.sample_count,):
            raise ValueError("accepted must be [sample_count]")
    result = []
    for scene_index, scene_id in enumerate(values.scene_ids.tolist()):
        a0, a1 = values.anchor_offsets[scene_index : scene_index + 2]
        g0, g1 = values.gt_offsets[scene_index : scene_index + 2]
        boxes = np.array(values.anchor_boxes[a0:a1], copy=True)
        sample_rows = np.flatnonzero(
            (values.sample_scene_index == scene_index) & mask
        )
        if len(sample_rows):
            boxes[values.sample_anchor_index[sample_rows]] = values.candidate_boxes[
                sample_rows
            ]
        result.append(
            (
                scene_id,
                boxes,
                np.array(values.anchor_scores[a0:a1], copy=True),
                np.array(values.gt_boxes[g0:g1], copy=True),
            )
        )
    return result


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changing = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(
        np.sum((mrec[changing + 1] - mrec[changing]) * mpre[changing + 1])
    )


def global_scored_metrics(
    rows: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
    threshold: float,
) -> tuple[int, float]:
    """Return exact global TP count and VOC AP for unique-score rows."""

    records: list[tuple[float, int, str, int, np.ndarray]] = []
    gt_by_scene: dict[str, np.ndarray] = {}
    total_gt = 0
    scores_seen: list[float] = []
    for scene_order, (scene_id, boxes, scores, gt) in enumerate(rows):
        predictions = np.asarray(boxes, dtype=np.float64)
        values = np.asarray(scores, dtype=np.float64)
        targets = np.asarray(gt, dtype=np.float64)
        if predictions.ndim != 2 or predictions.shape[1:] != (6,):
            raise ValueError("global metric boxes must be [N,6]")
        if values.shape != (len(predictions),) or not np.isfinite(values).all():
            raise ValueError("global metric scores must be finite [N]")
        if targets.ndim != 2 or targets.shape[1:] != (6,):
            raise ValueError("global metric GT must be [M,6]")
        if scene_id in gt_by_scene:
            raise ValueError("global metric scene ids must be unique")
        gt_by_scene[scene_id] = targets
        total_gt += len(targets)
        scores_seen.extend(values.tolist())
        records.extend(
            (float(score), scene_order, scene_id, row, predictions[row])
            for row, score in enumerate(values)
        )
    if len(scores_seen) != len(set(scores_seen)):
        raise ValueError("global metric requires unique scores to avoid tie ambiguity")
    records.sort(key=lambda item: (-item[0], item[1], item[3]))
    used = {
        scene_id: np.zeros(len(gt), dtype=np.bool_)
        for scene_id, gt in gt_by_scene.items()
    }
    tp = np.zeros(len(records), dtype=np.float64)
    fp = np.ones(len(records), dtype=np.float64)
    for index, (_, _, scene_id, _, box) in enumerate(records):
        gt = gt_by_scene[scene_id]
        if not len(gt):
            continue
        overlaps = _pairwise_iou(np.asarray(box)[None], gt)[0]
        target = int(np.argmax(overlaps))
        if overlaps[target] > threshold and not used[scene_id][target]:
            used[scene_id][target] = True
            tp[index] = 1.0
            fp[index] = 0.0
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall = cumulative_tp / float(total_gt + 1e-6)
    precision = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
    )
    return int(tp.sum()), _voc_ap(recall, precision) if len(records) else 0.0


def label_dataset_global_leave_one_out(
    dataset: R3CalibrationDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recompute exact rank-aware contribution of every raw-primary row.

    The reference is the joint raw-primary configuration over the complete
    train100 score ordering.  For each sample, restore just that anchor to G0
    and measure raw-minus-without TP and AP at all three thresholds.
    """

    values = dataset.validate()
    raw_mask = np.ones(values.sample_count, dtype=np.bool_)
    raw_rows = prediction_rows(values, raw_mask)
    raw_tp = np.empty(len(IOU_THRESHOLDS), dtype=np.int64)
    raw_ap = np.empty(len(IOU_THRESHOLDS), dtype=np.float64)
    for column, threshold in enumerate(IOU_THRESHOLDS):
        raw_tp[column], raw_ap[column] = global_scored_metrics(raw_rows, threshold)
    tp_deltas = np.empty(
        (values.sample_count, len(IOU_THRESHOLDS)), dtype=np.int8
    )
    ap_deltas = np.empty(
        (values.sample_count, len(IOU_THRESHOLDS)), dtype=np.float64
    )
    for sample in range(values.sample_count):
        scene_index = int(values.sample_scene_index[sample])
        anchor_index = int(values.sample_anchor_index[sample])
        without_rows = list(raw_rows)
        scene_id, boxes, scores, gt = raw_rows[scene_index]
        restored = np.array(boxes, copy=True)
        anchor_offset = int(values.anchor_offsets[scene_index])
        restored[anchor_index] = values.anchor_boxes[anchor_offset + anchor_index]
        without_rows[scene_index] = (scene_id, restored, scores, gt)
        for column, threshold in enumerate(IOU_THRESHOLDS):
            without_tp, without_ap = global_scored_metrics(without_rows, threshold)
            tp_deltas[sample, column] = np.int8(raw_tp[column] - without_tp)
            ap_deltas[sample, column] = raw_ap[column] - without_ap
    epsilon = 1e-12
    labels = np.full(values.sample_count, SAFE_NEUTRAL_CLASS, dtype=np.int8)
    harm = np.any(ap_deltas < -epsilon, axis=1) | np.any(tp_deltas < 0, axis=1)
    gain = (
        ~harm
        & (ap_deltas[:, 2] > epsilon)
        & np.all(ap_deltas[:, :2] >= -epsilon, axis=1)
        & np.all(tp_deltas >= 0, axis=1)
    )
    labels[harm] = HARM_CLASS
    labels[gain] = GAIN_CLASS
    return labels, tp_deltas, ap_deltas
