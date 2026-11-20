#!/usr/bin/env python3
"""Materialize matched PLR, CALR, and MVSR simple controls on MultiScan-50."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "development" / "final_controls"))
from boxfusion.m1_anchor_online import deterministic_tail_score  # noqa: E402
from boxfusion.online_candidate_map import _aabb_iou_matrix, _size_price  # noqa: E402


def read(path: Path) -> list[tuple[int, np.ndarray, float]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = payload[0] if isinstance(payload, (list, tuple)) and len(payload) == 1 else payload
    return [(int(label), np.asarray(box, dtype=np.float64).reshape(8, 3), float(score)) for label, box, score in rows]


def write(path: Path, rows: list[tuple[int, np.ndarray, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump([[(int(a), np.array(b, copy=True), float(c)) for a, b, c in rows]], handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def same(left, right) -> bool:
    return len(left) == len(right) and all(
        a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2] == b[2]
        for a, b in zip(left, right)
    )


def evidence(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def proposal_rows(scene: str, values: dict[str, np.ndarray], native_boxes: np.ndarray, quota: int):
    lengths = np.asarray(values["proposal_lengths"], dtype=np.int64)
    frames = np.repeat(np.asarray(values["frame_ids"], dtype=np.int64), lengths)
    ids = np.asarray(values["proposal_ids"], dtype=np.int64)
    boxes = np.asarray(values["proposal_corners"], dtype=np.float64)
    scores = np.asarray(values["proposal_scores"], dtype=np.float64)
    if not (len(frames) == len(ids) == len(boxes) == len(scores)):
        raise RuntimeError(f"{scene}: proposal cache is misaligned")
    if len(native_boxes) and len(boxes):
        covered = _aabb_iou_matrix(boxes, native_boxes).max(axis=1) >= 0.25
    else:
        covered = np.zeros(len(boxes), dtype=bool)
    order = sorted(
        np.flatnonzero(~covered).tolist(),
        key=lambda i: (-float(scores[i]), int(frames[i]), int(ids[i]), int(i)),
    )
    selected: list[int] = []
    for index in order:
        if selected and float(_aabb_iou_matrix(boxes[index:index + 1], boxes[selected]).max()) >= 0.50:
            continue
        selected.append(index)
        if len(selected) == quota:
            break
    if len(selected) != quota:
        raise RuntimeError(f"{scene}: only {len(selected)} direct proposals for quota {quota}")
    return [(0, boxes[i], _size_price(boxes[i])) for i in selected]


def anchor_memory_rows(scene: str, values: dict[str, np.ndarray], quota: int):
    lengths = np.asarray(values["anchor_lengths"], dtype=np.int64)
    frames = np.repeat(np.asarray(values["frame_ids"], dtype=np.int64), lengths)
    ids = np.asarray(values["anchor_ids"], dtype=np.int64)
    boxes = np.asarray(values["anchor_corners"], dtype=np.float64)
    scores = np.asarray(values["anchor_scores"], dtype=np.float64)
    if not (len(frames) == len(ids) == len(boxes) == len(scores)):
        raise RuntimeError(f"{scene}: anchor cache is misaligned")
    memory = {}
    for index, (frame, anchor, box, score) in enumerate(zip(frames, ids, boxes, scores)):
        key = tuple(np.floor(box.mean(axis=0) / 0.30).astype(np.int64).tolist())
        rank = (-float(score), int(frame), int(anchor), int(index))
        current = memory.get(key)
        if current is None or rank < current[0]:
            memory[key] = (rank, box.copy(), float(score), int(frame), int(anchor))
    selected = sorted(memory.values(), key=lambda row: row[0])[:quota]
    if len(selected) != quota:
        raise RuntimeError(f"{scene}: only {len(selected)} score-memory voxels for quota {quota}")
    return [
        (0, box, deterministic_tail_score(raw, scene, frame, anchor))
        for _, box, raw, frame, anchor in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--mvsr-patch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.prepared_root / "manifest.json").read_text())
    scenes = [row["alias"] for row in manifest["scans"]]
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise RuntimeError("expected 50 unique MultiScan aliases")
    components = args.experiment_root / "components"
    evidence_root = args.experiment_root / "evidence"
    arms = ("full_reference", "plr_direct", "calr_score_memory", "mvsr_max", "mvsr_latest")
    totals = {arm: 0 for arm in arms}
    per_scene = {}
    for scene in scenes:
        filename = f"{scene}_boxes.pkl"
        patch_components = args.mvsr_patch_root / "components"
        patched_latest = patch_components / "native_factor_no_visibility_latest" / filename
        mvsr_components = patch_components if patched_latest.is_file() else components
        reference_native_max = read(components / "native_factor_no_visibility" / filename)
        native_max = read(
            mvsr_components / (
                "native_factor_no_visibility_max_ge1"
                if mvsr_components == patch_components
                else "native_factor_no_visibility"
            ) / filename
        )
        native_latest = read(mvsr_components / "native_factor_no_visibility_latest" / filename)
        native_raw = read(mvsr_components / "native_factor_no_visibility_native" / filename)
        plr = read(components / "births_plr_v1" / filename)
        calr = read(components / "births_calr_v1" / filename)
        if not (len(native_max) == len(native_latest) == len(native_raw)) or any(
            not np.array_equal(a[1], b[1]) or not np.array_equal(a[1], c[1])
            for a, b, c in zip(native_max, native_latest, native_raw)
        ):
            raise RuntimeError(f"{scene}: invalid native controls")
        native_boxes = np.asarray([row[1] for row in reference_native_max], dtype=np.float64).reshape(-1, 8, 3)
        values = evidence(evidence_root / f"{scene}.npz")
        direct = proposal_rows(scene, values, native_boxes, len(plr))
        score_memory = anchor_memory_rows(scene, values, len(calr))
        outputs = {
            "full_reference": reference_native_max + plr + calr,
            "plr_direct": reference_native_max + direct + calr,
            "calr_score_memory": reference_native_max + plr + score_memory,
            "mvsr_max": native_max + plr + calr,
            "mvsr_latest": native_latest + plr + calr,
        }
        frozen_full = read(args.experiment_root / "factorial_no_visibility" / "full" / filename)
        if not same(outputs["full_reference"], frozen_full):
            raise RuntimeError(f"{scene}: composed full arm does not reproduce frozen full")
        for arm, rows in outputs.items():
            write(args.output / arm / filename, rows)
            totals[arm] += len(rows)
        per_scene[scene] = {
            "native": len(reference_native_max), "plr_budget": len(plr), "calr_budget": len(calr),
            "full_predictions": len(outputs["full_reference"]),
            "mvsr_patch_used": bool(mvsr_components == patch_components),
        }
    report = {
        "schema": "boxfusion.multiscan50.key_simple_controls.v1",
        "scene_count": len(scenes),
        "arms": totals,
        "comparisons": {
            "PLR": "single-view direct acceptance, scene-terminal score Top-K, matched per-scene birth count vs three-keyframe causal PLR",
            "CALR": "terminal count-matched score-only voxel memory vs three-keyframe voxel consensus",
            "MVSR": "latest-view support vs historical maximum support",
        },
        "shared": "final no-visibility three-factor MVSR; fixed native geometry and frozen evidence cache",
        "per_scene": per_scene,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"totals": totals}, indent=2))


if __name__ == "__main__":
    main()
