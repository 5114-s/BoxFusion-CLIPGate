"""GT-assisted diagnostics of *raw observation* memberships, never map snapshots.

IoU assignment is a diagnostic proxy for source identity, not pixel-level GT
provenance. In particular, unmatched and multiply-overlapping observations are
not silently counted as clean. This module has no model/runtime dependencies.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


def aabb_iou(corners, gt):
    boxes = np.asarray(corners, dtype=np.float64).reshape(-1, 8, 3)
    targets = np.asarray(gt, dtype=np.float64).reshape(-1, 8, 3)
    for value in (boxes, targets):
        if not np.isfinite(value).all() or (np.ptp(value, axis=1) <= 0).any():
            raise ValueError("Nonfinite or degenerate corners")
    if not len(boxes) or not len(targets):
        return np.zeros((len(boxes), len(targets)))
    lo, hi = boxes.min(1), boxes.max(1)
    gl, gh = targets.min(1), targets.max(1)
    inter = np.maximum(0, np.minimum(hi[:, None], gh) - np.maximum(lo[:, None], gl)).prod(2)
    union = (hi - lo).prod(1)[:, None] + (gh - gl).prod(1) - inter
    return inter / union


def assign_observations(matrix, threshold, *, policy="unique"):
    """-1 unknown; -2 ambiguous; >=0 assigned GT. Comparison is strict >."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError("Expected finite observation-by-GT IoU matrix")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("IoU threshold must be in [0,1]")
    if policy not in ("unique", "best"):
        raise ValueError("policy must be unique or best")
    assignment = np.full(len(matrix), -1, dtype=np.int64)
    if matrix.shape[1] == 0:
        return assignment
    hits = matrix > threshold
    counts = hits.sum(1)
    best = matrix.argmax(1)
    assignment[counts > 0] = best[counts > 0]
    if policy == "unique":
        assignment[counts > 1] = -2
    return assignment


def assess_membership(ids, assignment, frame_ids):
    """Assess one actual PFO input or retained membership without conflating them.

    Repeated observation IDs are deduplicated; support counts distinct frames.
    A view never confirms itself merely by appearing in multiple map snapshots.
    """
    original = list(ids)
    if any(isinstance(value, (bool, np.bool_)) or int(value) != value for value in original):
        raise ValueError("Observation IDs must be integers")
    ids = sorted(set(int(value) for value in original))
    assignment = np.asarray(assignment)
    frame_ids = np.asarray(frame_ids)
    for value in (assignment, frame_ids):
        if value.ndim != 1 or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError("Identity/frame arrays must be finite one-dimensional integer arrays")
        if np.any(value != np.floor(value)):
            raise ValueError("Identity/frame arrays must contain integers")
    if np.any(assignment < -2) or np.any(frame_ids < 0):
        raise ValueError("Invalid identity code or negative frame ID")
    assignment = assignment.astype(np.int64)
    frame_ids = frame_ids.astype(np.int64)
    if len(assignment) != len(frame_ids):
        raise ValueError("Frame and identity arrays must align")
    if any(index < 0 or index >= len(assignment) for index in ids):
        raise ValueError("Membership references an unknown observation ID")
    values = assignment[ids]
    groups = defaultdict(list)
    for index in ids:
        if assignment[index] >= 0:
            groups[int(assignment[index])].append(index)
    unknown = int(np.count_nonzero(values == -1))
    ambiguous = int(np.count_nonzero(values == -2))
    if not ids:
        status = "empty"
    elif len(groups) >= 2:
        status = "mixed_known"
    elif len(groups) == 1:
        status = "clean_known" if not (unknown or ambiguous) else "one_known_with_unresolved"
    elif unknown == len(ids):
        status = "all_unknown"
    else:
        status = "unresolved_without_unique_identity"
    supports = {str(key): {
        "observation_ids": group,
        "frames": sorted(set(int(frame_ids[index]) for index in group)),
        "distinct_frames": len(set(int(frame_ids[index]) for index in group)),
    } for key, group in sorted(groups.items())}
    frame_groups = defaultdict(set)
    for key, group in groups.items():
        for index in group:
            frame_groups[int(frame_ids[index])].add(key)
    counts = sorted((len(group) for group in groups.values()), reverse=True)
    return {
        "status": status, "observation_ids": ids,
        "observations": len(ids), "duplicate_memberships": len(original) - len(ids),
        "distinct_frames": len(set(int(frame_ids[index]) for index in ids)),
        "unknown_observations": unknown, "ambiguous_observations": ambiguous,
        "known_groups": supports,
        "known_identities": len(groups),
        "minority_known_fraction": (float(sum(counts[1:]) / sum(counts)) if counts else None),
        "same_frame_known_coexistence_frames": sorted(
            key for key, labels in frame_groups.items() if len(labels) >= 2),
        "groups_with_2_frames": sum(item["distinct_frames"] >= 2 for item in supports.values()),
        "groups_with_3_frames": sum(item["distinct_frames"] >= 3 for item in supports.values()),
    }


def summarize_memberships(memberships, assignment, frame_ids):
    details = [assess_membership(ids, assignment, frame_ids) for ids in memberships]
    statuses = Counter(item["status"] for item in details)
    fully_identified = sum(item["unknown_observations"] == 0 and item["ambiguous_observations"] == 0
                           and item["observations"] > 0 for item in details)
    return {
        "groups": len(details), "status_counts": dict(sorted(statuses.items())),
        "fully_identified_groups": fully_identified,
        "mixed_known_groups": statuses.get("mixed_known", 0),
        "mixed_with_two_2frame_groups": sum(
            item["groups_with_2_frames"] >= 2 for item in details),
        "mixed_with_two_3frame_groups": sum(
            item["groups_with_3_frames"] >= 2 for item in details),
        "details": details,
    }


def class_agnostic_ap(predictions, ground_truth, threshold):
    """Same score sort/greedy matching/VOC integration as the CA1M anchor.

    Inputs are ordered scene dictionaries. No additional NMS or output filters
    are introduced. This function is cross-checked against the anchor evaluator
    by the experiment driver before AP results may be reported.
    """
    scene_ids, scores, rows = [], [], []
    matrices, matched = {}, {}
    for scene, prediction in predictions.items():
        corners, confidence = prediction
        confidence = np.asarray(confidence, dtype=np.float64)
        if len(corners) != len(confidence) or not np.isfinite(confidence).all():
            raise ValueError("Malformed prediction scores")
        matrices[scene] = aabb_iou(corners, ground_truth.get(scene, []))
        matched[scene] = np.zeros(len(ground_truth.get(scene, [])), dtype=bool)
        scene_ids.extend([scene] * len(confidence))
        rows.extend(range(len(confidence)))
        scores.extend(confidence.tolist())
    order = np.argsort(-np.asarray(scores))
    tp, fp = np.zeros(len(order)), np.zeros(len(order))
    for rank, index in enumerate(order):
        scene, row = scene_ids[index], rows[index]
        values = matrices[scene][row]
        if len(values) and values.max() > threshold:
            target = int(values.argmax())
            if not matched[scene][target]:
                matched[scene][target] = True
                tp[rank] = 1
            else:
                fp[rank] = 1
        else:
            fp[rank] = 1
    tp, fp = tp.cumsum(), fp.cumsum()
    gt_count = sum(len(value) for value in ground_truth.values())
    recall = tp / (gt_count + 1e-6)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    mr = np.r_[0, recall, 1]
    mp = np.r_[0, precision, 0]
    mp = np.maximum.accumulate(mp[::-1])[::-1]
    changes = np.flatnonzero(mr[1:] != mr[:-1])
    ap = float(np.sum((mr[changes + 1] - mr[changes]) * mp[changes + 1]))
    return {"ap": 100 * ap, "tp": int(tp[-1]) if len(tp) else 0,
            "fp": int(fp[-1]) if len(fp) else 0, "gt": gt_count,
            "predictions": len(order)}
