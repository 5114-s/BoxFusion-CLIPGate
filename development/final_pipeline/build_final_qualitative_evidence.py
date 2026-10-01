#!/usr/bin/env python3
"""Bind selected qualitative cases to rerun traces and cached frame evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def aabb_iou(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64).reshape(-1, 8, 3)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 8, 3)
    if not len(left) or not len(right):
        return np.zeros((len(left), len(right)))
    l0, l1 = left.min(1), left.max(1)
    r0, r1 = right.min(1), right.max(1)
    lo, hi = np.maximum(l0[:, None], r0[None]), np.minimum(l1[:, None], r1[None])
    inter = np.prod(np.maximum(hi - lo, 0), axis=2)
    lv = np.prod(np.maximum(l1 - l0, 0), axis=1)[:, None]
    rv = np.prod(np.maximum(r1 - r0, 0), axis=1)[None]
    return inter / np.maximum(lv + rv - inter, 1e-12)


def closest_trace(rows: list[dict], target: np.ndarray) -> dict:
    if not rows:
        raise RuntimeError("requested qualitative source has no terminal trace")
    boxes = np.asarray([row["box"] for row in rows], dtype=np.float64)
    values = aabb_iou(boxes, target[None])[:, 0]
    index = int(np.argmax(values))
    if values[index] < .98:
        raise RuntimeError(f"terminal trace does not match selected box: IoU={values[index]:.6f}")
    return rows[index]


def slices(values: np.lib.npyio.NpzFile, prefix: str) -> dict[int, slice]:
    result, start = {}, 0
    for frame, length in zip(values["frame_ids"], values[f"{prefix}_lengths"]):
        result[int(frame)] = slice(start, start + int(length)); start += int(length)
    return result


def choose_frame_box(values, prefix: str, frame: int, target: np.ndarray, lookup) -> dict:
    if frame not in lookup:
        raise RuntimeError(f"frame {frame} absent from {prefix} evidence cache")
    region = lookup[frame]
    boxes = np.asarray(values[f"{prefix}_corners"][region], dtype=np.float64)
    if not len(boxes):
        raise RuntimeError(f"frame {frame} has no {prefix} candidates")
    scores = np.asarray(values[f"{prefix}_scores"][region], dtype=np.float64)
    overlaps = aabb_iou(boxes, target[None])[:, 0]
    index = int(np.argmax(overlaps))
    row = {
        "frame_id": int(frame), "candidate_box": boxes[index].tolist(),
        "candidate_score": float(scores[index]), "candidate_target_iou": float(overlaps[index]),
    }
    if prefix == "proposal":
        row["candidate_xyxy"] = np.asarray(values["proposal_boxes_2d"][region][index], dtype=float).tolist()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    parity = json.loads(args.parity.read_text())
    if not parity.get("all_selected_cases_passed"):
        raise RuntimeError("selected-case source/trace parity did not pass")
    args.output.mkdir(parents=True, exist_ok=True)
    output_cases = {}
    for panel, case in selection["cases"].items():
        scene = case["scene"]
        diagnostics_path = args.diagnostics / f"{scene}.json"
        diagnostic = json.loads(diagnostics_path.read_text())
        state = diagnostic["state"]
        if not (diagnostic["strictly_causal"] and diagnostic["online_incremental"]):
            raise RuntimeError(f"non-causal trace: {scene}")
        if state.get("selected_calr_v1") is not True or state["m1p"]["use_children"] is not False:
            raise RuntimeError(f"wrong final configuration in trace: {scene}")
        trace = state.get("qualitative_trace")
        if not trace:
            raise RuntimeError(f"missing qualitative trace: {scene}")
        target = np.asarray(case["box"], dtype=np.float64)
        source = case["source"]
        if source == "plr":
            terminal = closest_trace(trace["plr"], target)
            frame_ids = list(map(int, terminal["evidence_frame_ids"]))
            support = float(terminal["support_frames"])
        elif source == "calr":
            terminal = closest_trace(trace["calr"], target)
            frame_ids = list(map(int, terminal["support_frame_ids"]))
            support = float(terminal["support_frames"])
        else:
            terminal = closest_trace(trace["native"], target)
            matched = [slot for slot in terminal["slots"] if slot["matched"]]
            matched.sort(key=lambda row: (-float(row["strength"]), int(row["frame_id"])))
            frame_ids = [int(row["frame_id"]) for row in matched]
            support = float(terminal["support"])
        if source in ("plr", "calr") and len(set(frame_ids)) < 3:
            raise RuntimeError(f"{panel} lacks three distinct supporting keyframes")
        with np.load(args.evidence / f"{scene}.npz", allow_pickle=False) as values:
            prefix = "anchor" if source == "calr" else "proposal"
            lookup = slices(values, prefix)
            frames = [choose_frame_box(values, prefix, frame, target, lookup)
                      for frame in frame_ids[:3]]
        for row in frames:
            row["rgb_path"] = str(args.raw_root / scene / "color" / f"{row['frame_id']}.jpg")
            row["source"] = source
        export = {
            "schema": "boxfusion.recar3d.qualitative_evidence.v1",
            "scene": scene, "panel": panel, "source": source,
            "future_frames_used": False, "selected_case_parity": True,
            "terminal_trace": terminal, "support": support, "frames": frames,
            "diagnostics_path": str(diagnostics_path.resolve()),
            "diagnostics_sha256": digest(diagnostics_path),
            "evidence_cache_sha256": digest(args.evidence / f"{scene}.npz"),
        }
        path = args.output / f"{panel}_evidence.json"
        path.write_text(json.dumps(export, indent=2) + "\n")
        output_cases[panel] = case | {
            "evidence_export": str(path.resolve()),
            "evidence_sha256": digest(path), "evidence_verified": True,
            "max_geometric_support": support if source == "native" else None,
        }
    selection["cases"] = output_cases
    selection["provisional"] = False
    selection["terminal_parity"] = parity
    args.selection.write_text(json.dumps(selection, indent=2) + "\n")
    manifest = {
        "schema": "boxfusion.recar3d.qualitative_sources.v1",
        "selection": str(args.selection.resolve()),
        "parity": str(args.parity.resolve()),
        "selected_scenes": sorted({row["scene"] for row in output_cases.values()}),
        "source_manifest_sha256": selection["source_manifest_sha256"],
    }
    (args.output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    paths = [args.selection, args.parity] + sorted(args.output.glob("*_evidence.json"))
    (args.output / "source_hashes.sha256").write_text(
        "".join(f"{digest(path)}  {path.resolve()}\n" for path in paths)
    )


if __name__ == "__main__":
    main()
