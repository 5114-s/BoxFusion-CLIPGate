#!/usr/bin/env python3
"""Create an isolated C3 engineering replay from frozen C2 evidence.

The route and score policy are intentionally fixed:

* select ``source_rank <= 5 AND mask2_depth``;
* retain every frozen R3 anchor row first and byte-equivalent after loading;
* append class-agnostic candidates ordered by the frozen C1 track score;
* assign every candidate a positive score below every anchor score.

No ground truth, CLIP model, or teacher label is available to this command.
The output is an engineering active replay, not formal activation authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import (  # noqa: E402
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from boxfusion.tr3d_c2_maskrgbd_observer import GATE_NAMES  # noqa: E402
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.run_tr3d_c2_maskrgbd_observer import (  # noqa: E402
    REPORT_SCHEMA as C2_EXPORT_SCHEMA,
    _code_hash as c2_code_hash,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


SCHEMA = "boxfusion.tr3d_c3_shadow_active_manifest.v1"
ROUTE = "source_rank<=5 AND mask2_depth"
SCORE_POLICY = "global_c1_track_rank_below_all_frozen_anchors_v1"


def _code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_prediction(path: Path) -> list[list[tuple[int, np.ndarray, float]]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - local immutable artifact
    if type(payload) is not list or len(payload) != 1 or type(payload[0]) is not list:
        raise ValueError(f"{path}: prediction must be canonical list[list[tuple]]")
    for index, row in enumerate(payload[0]):
        if type(row) is not tuple or len(row) != 3 or type(row[0]) is not int:
            raise ValueError(f"{path}: malformed prediction row {index}")
        geometry = np.asarray(row[1])
        score = float(row[2])
        if (
            row[0] != 0
            or type(row[1]) is not np.ndarray
            or geometry.dtype != np.float32
            or geometry.shape != (8, 3)
            or not geometry.flags.c_contiguous
            or not np.isfinite(geometry).all()
            or type(row[2]) is not float
            or not math.isfinite(score)
        ):
            raise ValueError(f"{path}: non-canonical prediction row {index}")
    return payload


def _assign_candidate_scores(
    entries: Sequence[tuple[int, int, float]], anchor_floor: float
) -> dict[tuple[int, int], float]:
    """Map global C1 rank to distinct positive float32 scores below anchors."""

    if not math.isfinite(anchor_floor) or anchor_floor <= 0.0:
        raise ValueError("the frozen anchor score floor must be positive and finite")
    ordered = sorted(entries, key=lambda item: (-item[2], item[0], item[1]))
    if any(not math.isfinite(item[2]) for item in ordered):
        raise ValueError("C1 track scores must be finite")
    count = len(ordered)
    if count == 0:
        return {}
    cap = np.float32(anchor_floor * 0.5)
    if not (0.0 < float(cap) < anchor_floor):
        raise ValueError("could not construct a score band below frozen anchors")
    output: dict[tuple[int, int], float] = {}
    previous = float("inf")
    for rank, (scene_index, local_index, _) in enumerate(ordered):
        value = np.float32(float(cap) * (count - rank) / (count + 1.0))
        score = float(value)
        if not (0.0 < score < anchor_floor and score < previous):
            raise ValueError("candidate score quantization changed the frozen rank")
        output[(scene_index, local_index)] = score
        previous = score
    return output


def _append_payload(
    source: list[list[tuple[int, np.ndarray, float]]],
    corners: np.ndarray,
    scores: Sequence[float],
) -> list[list[tuple[int, np.ndarray, float]]]:
    geometry = np.asarray(corners)
    if (
        geometry.dtype != np.float32
        or geometry.ndim != 3
        or geometry.shape[1:] != (8, 3)
        or not np.isfinite(geometry).all()
        or len(geometry) != len(scores)
    ):
        raise ValueError("candidate corners/scores are malformed")
    rows = list(source[0])
    for index, raw_score in enumerate(scores):
        score = float(raw_score)
        if not math.isfinite(score) or score <= 0.0:
            raise ValueError("candidate scores must be positive and finite")
        rows.append(
            (
                0,
                np.array(geometry[index], dtype=np.float32, order="C", copy=True),
                score,
            )
        )
    return [rows]


def _write_pickle_create_only(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite prediction: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return sha256_file(path)


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite manifest: {path}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    scene_list = args.scene_list.resolve()
    c2_report_path = args.c2_export_report.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing existing output root: {output_root}")
    if args.manifest.resolve().exists():
        raise FileExistsError(f"refusing existing manifest: {args.manifest.resolve()}")
    scenes = read_scene_list(scene_list)
    if len(scenes) != 100:
        raise ValueError("formal C3 engineering replay requires exactly 100 scenes")
    c2_report = json.loads(c2_report_path.read_text(encoding="utf-8"))
    if c2_report.get("schema") != C2_EXPORT_SCHEMA:
        raise ValueError("unsupported C2 export report")
    if (
        not c2_report.get("observer_only")
        or c2_report.get("mutation_enabled")
        or int(c2_report.get("applied_count", -1)) != 0
        or c2_report.get("ground_truth_access")
        or c2_report.get("clip_access")
        or c2_report.get("teacher_labels_used_for_gate")
    ):
        raise ValueError("C2 export violates its frozen observer contract")
    if c2_report.get("code_sha256") != c2_code_hash():
        raise ValueError("C2 implementation differs from its export")
    if int(c2_report.get("scene_count", -1)) != len(scenes):
        raise ValueError("C2 scene count mismatch")
    c2_rows = {str(row["scene_id"]): row for row in c2_report["scenes"]}
    if set(c2_rows) != set(scenes) or len(c2_rows) != len(scenes):
        raise ValueError("C2 scene identity mismatch")

    anchor_root = args.active_prediction_root.resolve()
    anchor_before = _tree_snapshot(anchor_root, scenes)
    if anchor_before["tree_sha256"] != c2_report["frozen_active_before"]["tree_sha256"]:
        raise ValueError("frozen R3 anchor differs from C2 lineage")

    prepared: list[dict[str, Any]] = []
    entries: list[tuple[int, int, float]] = []
    anchor_floor = float("inf")
    gate_index = GATE_NAMES.index("mask2_depth")
    for scene_index, scene_id in enumerate(scenes):
        c2_path = sidecar_path(args.c2_cache_root.resolve(), scene_id, args.prefix_id)
        c2_sha = sha256_file(c2_path)
        if c2_sha != str(c2_rows[scene_id]["sidecar_sha256"]):
            raise ValueError(f"{scene_id}: C2 sidecar hash mismatch")
        c2 = load_sidecar(c2_path)
        parent_path = tr3d_residual_cache_path(
            args.parent_cache_root.resolve(), scene_id, args.prefix_id
        )
        parent_sha = sha256_file(parent_path)
        if parent_sha != c2.parent_cache_sha256:
            raise ValueError(f"{scene_id}: parent/C2 lineage mismatch")
        with np.load(parent_path, allow_pickle=False) as raw:
            checkpoint_sha = str(np.asarray(raw["checkpoint_sha256"]).item())
            config_sha = str(np.asarray(raw["config_sha256"]).item())
        parent = load_tr3d_residual_cache(
            parent_path,
            expected_scene_id=scene_id,
            expected_prefix_id=args.prefix_id,
            expected_checkpoint_sha256=checkpoint_sha,
            expected_config_sha256=config_sha,
        )
        if not np.array_equal(parent.proposal_ids[c2.parent_rows], c2.proposal_ids):
            raise ValueError(f"{scene_id}: C2/parent proposal identity mismatch")
        anchor_path = anchor_root / f"{scene_id}_boxes.pkl"
        anchor_sha = sha256_file(anchor_path)
        if anchor_sha != c2.anchor_prediction_sha256:
            raise ValueError(f"{scene_id}: anchor/C2 lineage mismatch")
        payload = _load_prediction(anchor_path)
        if payload[0]:
            anchor_floor = min(anchor_floor, min(float(row[2]) for row in payload[0]))
        mask = (c2.source_ranks <= 5) & c2.observation.gate_mask[:, gate_index]
        parent_rows = np.asarray(c2.parent_rows[mask], dtype=np.int64)
        proposal_ids = np.asarray(c2.proposal_ids[mask], dtype=np.int64)
        c1_scores = np.asarray(c2.c1_track_scores[mask], dtype=np.float64)
        candidate_corners = np.ascontiguousarray(
            parent.corners_world[parent_rows], dtype=np.float32
        )
        if len(candidate_corners) and (
            not np.isfinite(candidate_corners).all()
            or np.any(np.ptp(candidate_corners, axis=1) <= 0.0)
        ):
            raise ValueError(f"{scene_id}: invalid candidate geometry")
        for local_index, score in enumerate(c1_scores.tolist()):
            entries.append((scene_index, local_index, float(score)))
        prepared.append(
            {
                "scene_id": scene_id,
                "payload": payload,
                "anchor_sha256": anchor_sha,
                "c2_path": c2_path,
                "c2_sha256": c2_sha,
                "parent_path": parent_path,
                "parent_sha256": parent_sha,
                "parent_rows": parent_rows,
                "proposal_ids": proposal_ids,
                "candidate_corners": candidate_corners,
                "c1_scores": c1_scores,
            }
        )
    if not math.isfinite(anchor_floor):
        raise ValueError("the frozen R3 tree contains no anchor scores")
    score_map = _assign_candidate_scores(entries, anchor_floor)

    scene_reports: list[dict[str, Any]] = []
    for scene_index, row in enumerate(prepared):
        candidate_scores = [
            score_map[(scene_index, local_index)]
            for local_index in range(len(row["candidate_corners"]))
        ]
        output = _append_payload(
            row["payload"], row["candidate_corners"], candidate_scores
        )
        target = output_root / f"{row['scene_id']}_boxes.pkl"
        output_sha = _write_pickle_create_only(target, output)
        scene_reports.append(
            {
                "scene_id": row["scene_id"],
                "source_prediction_sha256": row["anchor_sha256"],
                "anchor_prediction_sha256": row["anchor_sha256"],
                "c2_sidecar": str(row["c2_path"]),
                "c2_sidecar_sha256": row["c2_sha256"],
                "parent_cache": str(row["parent_path"]),
                "parent_cache_sha256": row["parent_sha256"],
                "anchor_count": len(row["payload"][0]),
                "candidate_count": len(row["candidate_corners"]),
                "anchor_rows": len(row["payload"][0]),
                "candidate_rows": len(row["candidate_corners"]),
                "output_count": len(output[0]),
                "candidate_parent_rows": row["parent_rows"].tolist(),
                "candidate_proposal_ids": row["proposal_ids"].tolist(),
                "candidate_c1_track_scores": row["c1_scores"].tolist(),
                "candidate_output_scores": candidate_scores,
                "output_prediction_sha256": output_sha,
            }
        )

    anchor_after = _tree_snapshot(anchor_root, scenes)
    if anchor_before != anchor_after:
        raise RuntimeError("frozen R3 anchor tree changed during C3 materialization")
    output_tree = _tree_snapshot(output_root, scenes)
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "shadow_only": True,
        "engineering_active_replay": True,
        "formal_active_authorized": False,
        "formal_activation_authorized": False,
        "ground_truth_access": False,
        "counterfactual_report_access": False,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "teacher_labels_used": False,
        "class_agnostic": True,
        "candidate_label": 0,
        "route": ROUTE,
        "score_policy": SCORE_POLICY,
        "anchor_rows_first_and_unchanged": True,
        "candidate_scores_below_every_anchor": True,
        "scene_list": str(scene_list),
        "scene_list_sha256": sha256_file(scene_list),
        "scene_count": len(scenes),
        "c2_export_report": str(c2_report_path),
        "c2_export_report_sha256": sha256_file(c2_report_path),
        "c2_cache_root": str(args.c2_cache_root.resolve()),
        "parent_cache_root": str(args.parent_cache_root.resolve()),
        "anchor_prediction_root": str(anchor_root),
        "active_prediction_root": str(anchor_root),
        "output_prediction_root": str(output_root),
        "output_root": str(output_root),
        "prefix_id": args.prefix_id,
        "materializer_code_sha256": _code_sha256(),
        "anchor_score_floor": anchor_floor,
        "candidate_count": len(entries),
        "anchor_count": sum(int(row["anchor_count"]) for row in scene_reports),
        "output_count": sum(int(row["output_count"]) for row in scene_reports),
        "anchor_tree_before": anchor_before,
        "anchor_tree_after": anchor_after,
        "output_tree": output_tree,
        "input_prediction_tree_sha256": anchor_before["tree_sha256"],
        "output_prediction_tree_sha256": output_tree["tree_sha256"],
        "materialization_wall_s": time.perf_counter() - started,
        "scenes": scene_reports,
    }
    _write_json_create_only(args.manifest.resolve(), manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--c2-export-report", type=Path, required=True)
    value.add_argument("--c2-cache-root", type=Path, required=True)
    value.add_argument("--parent-cache-root", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--manifest", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    manifest = materialize(parser().parse_args(argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
