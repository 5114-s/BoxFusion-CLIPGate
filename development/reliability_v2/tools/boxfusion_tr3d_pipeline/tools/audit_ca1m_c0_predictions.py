#!/usr/bin/env python3
"""Fail closed unless CA-1M C0 has exactly one valid prediction per scene."""

from __future__ import annotations

import argparse
import json
import pickle
import tempfile
from pathlib import Path

import numpy as np


def scene_ids(path: Path) -> tuple[str, ...]:
    rows = tuple(row.strip() for row in path.read_text().splitlines() if row.strip())
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list is empty or contains duplicates")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    score_group = parser.add_mutually_exclusive_group(required=True)
    score_group.add_argument("--expected-score", type=float)
    score_group.add_argument("--require-real-score", action="store_true")
    args = parser.parse_args()

    scenes = scene_ids(args.scene_list.resolve())
    expected_files = {f"{scene}_boxes.pkl" for scene in scenes}
    actual_files = {path.name for path in args.prediction_root.glob("*_boxes.pkl")}
    if actual_files != expected_files:
        raise ValueError(
            f"prediction exact-set mismatch: missing={sorted(expected_files-actual_files)}, "
            f"unexpected={sorted(actual_files-expected_files)}"
        )
    rows = []
    all_scores = []
    for scene in scenes:
        path = args.prediction_root / f"{scene}_boxes.pkl"
        with path.open("rb") as handle:
            prediction = pickle.load(handle)
        if not isinstance(prediction, list) or len(prediction) != 1:
            raise ValueError(f"{scene}: prediction must have one batch")
        count = 0
        for item in prediction[0]:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError(f"{scene}: invalid prediction tuple")
            label, corners, score = item
            corners = np.asarray(corners)
            if int(label) != 0 or corners.shape != (8, 3) or not np.isfinite(corners).all():
                raise ValueError(f"{scene}: invalid class-agnostic box")
            score = float(score)
            if not np.isfinite(score) or score < 0.0 or score > 1.0:
                raise ValueError(f"{scene}: invalid score={score}")
            if args.expected_score is not None and not np.isclose(
                score, args.expected_score, rtol=0, atol=1e-8
            ):
                raise ValueError(f"{scene}: expected score={args.expected_score}, got {score}")
            all_scores.append(score)
            count += 1
        rows.append({"scene_id": scene, "predictions": count})
    score_array = np.asarray(all_scores, dtype=np.float64)
    if args.require_real_score:
        if score_array.size == 0:
            raise ValueError("real-score audit requires at least one prediction")
        if np.allclose(score_array, 1.0, rtol=0, atol=1e-8):
            raise ValueError("all scores are 1.0; the author export bug is still active")
        if float(np.std(score_array)) <= 1e-8:
            raise ValueError("scores are constant; detector confidence was not preserved")
    result = {
        "schema": "boxfusion.ca1m_c0_prediction_audit.v1",
        "ok": True,
        "scenes": len(rows),
        "boxes": sum(row["predictions"] for row in rows),
        "score_mode": "real_detector_score" if args.require_real_score else "constant",
        "expected_score": args.expected_score,
        "score_min": float(score_array.min()) if score_array.size else None,
        "score_max": float(score_array.max()) if score_array.size else None,
        "score_std": float(score_array.std()) if score_array.size else None,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=args.output.parent, delete=False) as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("ok", "scenes", "boxes", "score_mode", "score_min", "score_max", "score_std")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
