#!/usr/bin/env python3
"""Paired ScanNet100 bootstrap for B05 versus Reliable-TopK T05.

This is a read-only evaluator for the BoxFusion class-agnostic ScanNet path.
It deliberately mirrors the score=1.0 evaluator's observable behavior:

* scenes and prediction rows retain their official input order;
* all prediction scores are replaced in memory by exactly 1.0;
* pooled detections are ordered with ``np.argsort(-confidence)`` using
  NumPy's default (quicksort) tie behavior;
* matching is greedy, per scene (or per bootstrap scene copy), and accepts
  only IoU strictly greater than 0.15/0.25/0.50;
* AP is continuous VOC AP with the evaluator's ``npos + 1e-6`` recall term.

The same scene draw is used for both arms in every replicate.  Prediction and
ground-truth inputs are never modified.  Only ``--out`` may be written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


THRESHOLDS = np.asarray([0.15, 0.25, 0.50], dtype=np.float64)
ARMS = ("B05", "T05")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")
    return path


def axis_alignment(path: Path) -> np.ndarray:
    regular_file(path, "axis-alignment metadata")
    for line in path.read_text().splitlines():
        if line.strip().startswith("axisAlignment"):
            try:
                values = [float(value) for value in line.split("=", 1)[1].split()]
            except (IndexError, ValueError) as error:
                raise ValueError(f"malformed axisAlignment in {path}") from error
            if len(values) != 16:
                raise ValueError(f"axisAlignment in {path} has {len(values)} values")
            matrix = np.asarray(values, dtype=np.float64).reshape(4, 4)
            if not np.isfinite(matrix).all():
                raise ValueError(f"non-finite axisAlignment in {path}")
            return matrix
    raise ValueError(f"axisAlignment missing in {path}")


def load_gt_minmax(path: Path) -> np.ndarray:
    regular_file(path, "ScanNet GT")
    gt = np.load(path, allow_pickle=False).astype(np.float64)
    if gt.ndim != 2 or gt.shape[1] < 6 or not np.isfinite(gt[:, :6]).all():
        raise ValueError(f"invalid ScanNet GT array: {path}")
    if np.any(gt[:, 3:6] < 0.0):
        raise ValueError(f"negative ScanNet GT extent: {path}")
    return np.concatenate(
        (gt[:, :3] - gt[:, 3:6] / 2.0, gt[:, :3] + gt[:, 3:6] / 2.0),
        axis=1,
    )


def load_prediction_minmax(path: Path, alignment: np.ndarray) -> np.ndarray:
    regular_file(path, "prediction")
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], list):
        raise ValueError(f"unexpected prediction container: {path}")
    rows = data[0]
    if not rows:
        return np.empty((0, 6), dtype=np.float64)
    try:
        corners = np.asarray([row[1] for row in rows], dtype=np.float64)
        disk_scores = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    except (IndexError, TypeError, ValueError) as error:
        raise ValueError(f"invalid prediction rows: {path}") from error
    if corners.shape != (len(rows), 8, 3) or not np.isfinite(corners).all():
        raise ValueError(f"invalid prediction corners: {path}")
    if disk_scores.shape != (len(rows),) or not np.isfinite(disk_scores).all():
        raise ValueError(f"invalid prediction scores: {path}")
    # The on-disk scores are intentionally audited but never used below.
    aligned = corners @ alignment[:3, :3].T + alignment[:3, 3]
    return np.concatenate((aligned.min(axis=1), aligned.max(axis=1)), axis=1)


def aligned_iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float64)
    lower = np.maximum(pred[:, None, :3], gt[None, :, :3])
    upper = np.minimum(pred[:, None, 3:], gt[None, :, 3:])
    intersection = np.prod(np.maximum(upper - lower, 0.0), axis=2)
    pred_volume = np.prod(np.maximum(pred[:, 3:] - pred[:, :3], 0.0), axis=1)
    gt_volume = np.prod(np.maximum(gt[:, 3:] - gt[:, :3], 0.0), axis=1)
    union = pred_volume[:, None] + gt_volume[None, :] - intersection
    return intersection / np.maximum(union, np.finfo(np.float64).eps)


def load_real_data(
    scenes: Sequence[str],
    roots: Mapping[str, Path],
    gt_root: Path,
    scan_root: Path,
) -> Tuple[Dict[str, List[np.ndarray]], List[int], Dict[str, List[str]]]:
    ious: Dict[str, List[np.ndarray]] = {arm: [] for arm in ARMS}
    gt_counts: List[int] = []
    prediction_hashes: Dict[str, List[str]] = {arm: [] for arm in ARMS}
    for scene in scenes:
        gt = load_gt_minmax(gt_root / f"{scene}_bbox.npy")
        alignment = axis_alignment(scan_root / scene / f"{scene}.txt")
        gt_counts.append(len(gt))
        for arm in ARMS:
            prediction_path = roots[arm] / f"{scene}_boxes.pkl"
            pred = load_prediction_minmax(prediction_path, alignment)
            ious[arm].append(aligned_iou_matrix(pred, gt))
            prediction_hashes[arm].append(sha256(prediction_path))
    return ious, gt_counts, prediction_hashes


def audit_prediction_set(scenes: Sequence[str], arm: str, root: Path) -> None:
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    observed = {
        path.name for path in root.glob("scene*_boxes.pkl")
        if path.is_file() or path.is_symlink()
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ValueError(
            f"{arm} prediction set mismatch: missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )
    for name in sorted(expected):
        regular_file(root / name, f"{arm} prediction")


def voc_ap(tp: np.ndarray, fp: np.ndarray, npos: int) -> float:
    tp_cumulative = np.cumsum(tp, dtype=np.float64)
    fp_cumulative = np.cumsum(fp, dtype=np.float64)
    recall = tp_cumulative / float(npos + 1e-6)
    precision = tp_cumulative / np.maximum(
        tp_cumulative + fp_cumulative, np.finfo(np.float64).eps
    )
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    for index in range(padded_precision.size - 1, 0, -1):
        padded_precision[index - 1] = max(
            padded_precision[index - 1], padded_precision[index]
        )
    changes = np.where(padded_recall[1:] != padded_recall[:-1])[0]
    return float(
        np.sum(
            (padded_recall[changes + 1] - padded_recall[changes])
            * padded_precision[changes + 1]
        )
    )


def evaluate_sample(
    iou_by_scene: Sequence[np.ndarray],
    gt_counts: Sequence[int],
    sample: np.ndarray,
) -> np.ndarray:
    """Evaluate constant-score AP, treating duplicate sampled scenes independently."""
    lengths = np.fromiter(
        (len(iou_by_scene[int(scene)]) for scene in sample), dtype=np.int64
    )
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    total_predictions = int(offsets[-1])
    confidence = np.ones(total_predictions, dtype=np.float64)
    # Do not add ``kind=``: this call shape intentionally matches eval_det.py.
    order = np.argsort(-confidence)
    copy_indices = np.searchsorted(offsets[1:], order, side="right")
    local_indices = order - offsets[copy_indices]
    npos = int(sum(gt_counts[int(scene)] for scene in sample))
    output = np.empty(len(THRESHOLDS), dtype=np.float64)
    for threshold_index, threshold in enumerate(THRESHOLDS):
        matched = [
            np.zeros(gt_counts[int(scene)], dtype=bool) for scene in sample
        ]
        tp = np.zeros(total_predictions, dtype=np.float64)
        for detection_index, (copy_index, prediction_index) in enumerate(
            zip(copy_indices, local_indices)
        ):
            overlaps = iou_by_scene[int(sample[copy_index])][prediction_index]
            if overlaps.size:
                gt_index = int(np.argmax(overlaps))
                if (
                    overlaps[gt_index] > threshold
                    and not matched[copy_index][gt_index]
                ):
                    tp[detection_index] = 1.0
                    matched[copy_index][gt_index] = True
        output[threshold_index] = voc_ap(tp, 1.0 - tp, npos)
    return output


def summary(observed: float, values: np.ndarray) -> dict:
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "observed_delta": float(observed),
        "observed_delta_ap_points": float(100.0 * observed),
        "bootstrap_mean": float(np.mean(values)),
        "bootstrap_mean_ap_points": float(100.0 * np.mean(values)),
        "ci95_percentile": [float(lower), float(upper)],
        "ci95_percentile_ap_points": [float(100.0 * lower), float(100.0 * upper)],
        "positive_rate": float(np.mean(values > 0.0)),
        "nonpositive_rate": float(np.mean(values <= 0.0)),
        "p_one_sided_bootstrap": float(
            (np.count_nonzero(values <= 0.0) + 1) / (len(values) + 1)
        ),
    }


def run_bootstrap(
    scenes: Sequence[str],
    ious: Mapping[str, Sequence[np.ndarray]],
    gt_counts: Sequence[int],
    replicates: int,
    seed: int,
    progress_every: int,
) -> dict:
    scene_count = len(scenes)
    original_sample = np.arange(scene_count, dtype=np.int64)
    observed = {
        arm: evaluate_sample(ious[arm], gt_counts, original_sample) for arm in ARMS
    }
    rng = np.random.default_rng(seed)
    draw_dtype = np.int16 if scene_count <= np.iinfo(np.int16).max else np.int32
    draws = rng.integers(
        0, scene_count, size=(replicates, scene_count), dtype=draw_dtype
    )
    differences = np.empty((replicates, len(THRESHOLDS)), dtype=np.float64)
    for replicate_index, sample in enumerate(draws):
        b05 = evaluate_sample(ious["B05"], gt_counts, sample)
        t05 = evaluate_sample(ious["T05"], gt_counts, sample)
        differences[replicate_index] = t05 - b05
        if progress_every and (replicate_index + 1) % progress_every == 0:
            print(f"bootstrap {replicate_index + 1}/{replicates}", flush=True)
    observed_difference = observed["T05"] - observed["B05"]
    return {
        "schema": "boxfusion.scannet_b05_t05_paired_bootstrap.v1",
        "method": {
            "scene_count": scene_count,
            "replicates": replicates,
            "seed": seed,
            "score": 1.0,
            "iou_thresholds": THRESHOLDS.tolist(),
            "matching": "greedy per independent scene copy; strict IoU > threshold",
            "tie_sort": "np.argsort(-confidence) default quicksort, matching eval_det.py",
            "ap": "continuous VOC AP pooled over sampled scene copies",
            "bootstrap": "paired nonparametric scene bootstrap; shared B05/T05 draws",
            "ci": "two-sided percentile 95% CI",
        },
        "prediction_counts": {
            arm: int(sum(len(matrix) for matrix in ious[arm])) for arm in ARMS
        },
        "gt_count": int(sum(gt_counts)),
        "original_ap": {
            arm: {
                f"AP{int(round(100 * threshold)):02d}": float(
                    observed[arm][threshold_index]
                )
                for threshold_index, threshold in enumerate(THRESHOLDS)
            }
            for arm in ARMS
        },
        "contrast_T05_minus_B05": {
            f"AP{int(round(100 * threshold)):02d}": summary(
                observed_difference[threshold_index],
                differences[:, threshold_index],
            )
            for threshold_index, threshold in enumerate(THRESHOLDS)
        },
    }


def synthetic_data(identical: bool = False):
    scenes = [f"synthetic_{index:02d}" for index in range(10)]
    gt_counts = [2] * len(scenes)
    b05: List[np.ndarray] = []
    t05: List[np.ndarray] = []
    for index in range(len(scenes)):
        base = np.asarray(
            [
                [0.80, 0.00],
                [0.00, 0.30 if index % 2 == 0 else 0.10],
                [0.00, 0.00],
            ],
            dtype=np.float64,
        )
        active = base.copy()
        if index % 2:
            active = np.concatenate((active, [[0.00, 0.70]]), axis=0)
        b05.append(base)
        t05.append(base.copy() if identical else active)
    return scenes, {"B05": b05, "T05": t05}, gt_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("/data/ZhaoX/BoxFusion"))
    parser.add_argument(
        "--scan-root", type=Path, default=Path("/extra/ZhaoX/scannet_data/scans")
    )
    parser.add_argument("--b05", "--baseline", dest="b05", type=Path)
    parser.add_argument("--t05", "--treatment", dest="t05", type=Path)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-identical", action="store_true")
    parser.add_argument(
        "--allow-identical-roots",
        action="store_true",
        help="audit-only escape hatch; forbidden for a formal B05/T05 comparison",
    )
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be nonnegative")
    if not (args.smoke or args.smoke_identical):
        if args.b05 is None or args.t05 is None:
            parser.error("real mode requires --b05 and --t05")
        if (
            args.b05.resolve() == args.t05.resolve()
            and not args.allow_identical_roots
        ):
            parser.error("B05 and T05 roots resolve to the same directory")
    return args


def main() -> None:
    args = parse_args()
    roots: Optional[Dict[str, Path]] = None
    prediction_hashes: Optional[Dict[str, List[str]]] = None
    if args.smoke or args.smoke_identical:
        scenes, ious, gt_counts = synthetic_data(identical=args.smoke_identical)
        mode = "synthetic_identical" if args.smoke_identical else "synthetic_positive"
        scene_list_path = None
    else:
        roots = {"B05": args.b05, "T05": args.t05}
        for arm, root in roots.items():
            if root.is_symlink() or not root.is_dir():
                raise ValueError(f"{arm} root must be a regular directory: {root}")
        scene_list_path = regular_file(
            args.repo / "evaluation/data_util/meta_data/scannetv2_val.txt",
            "official scene list",
        )
        scenes = [
            line.strip()
            for line in scene_list_path.read_text().splitlines()
            if line.strip()
        ]
        if len(scenes) != 100 or len(set(scenes)) != 100:
            raise ValueError("formal experiment requires exactly 100 unique scenes")
        for arm, root in roots.items():
            audit_prediction_set(scenes, arm, root)
        gt_root = args.repo / "evaluation/data_util/scannet_train_detection_data"
        ious, gt_counts, prediction_hashes = load_real_data(
            scenes, roots, gt_root, args.scan_root
        )
        mode = "real_scannet100"

    result = run_bootstrap(
        scenes,
        ious,
        gt_counts,
        args.replicates,
        args.seed,
        args.progress_every,
    )
    result["mode"] = mode
    if roots is not None and prediction_hashes is not None and scene_list_path is not None:
        result["prediction_roots"] = {
            arm: str(root.resolve()) for arm, root in roots.items()
        }
        result["scene_list"] = {
            "path": str(scene_list_path.resolve()),
            "sha256": sha256(scene_list_path),
            "ordered_scenes": scenes,
        }
        result["prediction_set_digest"] = {
            arm: hashlib.sha256("".join(prediction_hashes[arm]).encode()).hexdigest()
            for arm in ARMS
        }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.out is not None:
        if args.out.exists() and args.out.is_symlink():
            raise ValueError(f"refusing symlink output: {args.out}")
        if not args.out.parent.is_dir():
            raise ValueError(f"output parent does not exist: {args.out.parent}")
        args.out.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
