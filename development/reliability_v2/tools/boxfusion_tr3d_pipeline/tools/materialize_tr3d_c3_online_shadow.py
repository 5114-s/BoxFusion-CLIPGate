#!/usr/bin/env python3
"""Append online-confirmed C3 candidates to a frozen terminal-R3 tree.

This is a GT-free, append-only shadow materializer.  It consumes immutable
``tr3d_c3_online_identity`` diagnostics, selects only candidates whose frozen
route rank is at most five and whose *online* YOLOE Mask-RGBD decision is true,
and appends them below every terminal-R3 anchor score.  It never authorizes a
live mutation; the resulting tree exists solely for an official AP shadow
evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file  # noqa: E402
from boxfusion.tr3d_c3_online_identity import (  # noqa: E402
    ROUTE as IDENTITY_ROUTE,
    SCHEMA as IDENTITY_SCHEMA,
    prediction_state_sha256,
)
from boxfusion.tr3d_residual_cache import (  # noqa: E402
    load_tr3d_residual_cache,
    tr3d_residual_cache_path,
)
from tools.materialize_tr3d_c3_active import (  # noqa: E402
    _append_payload,
    _assign_candidate_scores,
    _load_prediction,
    _write_json_create_only,
    _write_pickle_create_only,
)
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


SCHEMA = "boxfusion.tr3d_c3_online_shadow_manifest.v1"
ROUTE = "source_rank<=5 AND online_yoloe_mask2_depth"
SCORE_POLICY = "global_frozen_c1_rank_below_all_terminal_r3_anchors_v1"
COMPLETION_SCHEMA = "boxfusion.tr3d_terminal_completion.v2"


def _code_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _load_parent(path: Path, scene_id: str, prefix_id: str):
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as raw:
        checkpoint_sha = str(np.asarray(raw["checkpoint_sha256"]).item())
        config_sha = str(np.asarray(raw["config_sha256"]).item())
    return load_tr3d_residual_cache(
        path,
        expected_scene_id=scene_id,
        expected_prefix_id=prefix_id,
        expected_checkpoint_sha256=checkpoint_sha,
        expected_config_sha256=config_sha,
    )


def _load_identity_diagnostic(path: Path, scene_id: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o222:
        raise ValueError(f"{path}: diagnostic must be immutable regular file")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_false = (
        "mutation_enabled",
        "ground_truth_access",
        "clip_access",
        "teacher_labels_used_for_gate",
        "online_sam3_forward",
        "online_dino_forward",
    )
    if (
        payload.get("schema") != IDENTITY_SCHEMA
        or not payload.get("complete")
        or not payload.get("observer_only")
        or not payload.get("enabled")
        or int(payload.get("applied_count", -1)) != 0
        or any(bool(payload.get(name)) for name in required_false)
        or payload.get("scene_id") != scene_id
        or payload.get("route") != IDENTITY_ROUTE
        or payload.get("gate_name") != "mask2_depth"
        or int(payload.get("source_rank_max", -1)) != 5
        or float(payload.get("identity_coverage", -1.0)) != 1.0
        or int(payload.get("missing_identity_count", -1)) != 0
        or int(payload.get("out_of_universe_selected_count", -1)) != 0
        or payload.get("candidate_generation_is_live")
        or payload.get("online_confirmation_provider")
        != "runtime_yoloe_mask_real_depth"
    ):
        raise ValueError(f"{path}: online identity contract failed")
    candidates = payload.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != int(payload.get("candidate_count", -1))
        or len(candidates) != int(payload.get("exact_identity_joined_count", -1))
    ):
        raise ValueError(f"{path}: malformed candidate universe")
    return payload


def _load_completion_marker(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink() or path.stat().st_mode & 0o222:
        raise ValueError(f"{path}: completion marker must be immutable regular file")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line or "=" not in raw_line:
            raise ValueError(f"{path}:{line_number}: malformed completion marker")
        key, value = raw_line.split("=", 1)
        if not key or not value or key in values:
            raise ValueError(f"{path}:{line_number}: invalid/duplicate marker key")
        values[key] = value
    required = {
        "schema",
        "scene_fingerprint",
        "active_prediction_sha256",
        "same_run_baseline_sha256",
        "r3_diagnostic_sha256",
        "boxer_diagnostic_sha256",
        "c3_online_diagnostic_sha256",
    }
    if set(values) != required or values.get("schema") != COMPLETION_SCHEMA:
        raise ValueError(f"{path}: unsupported completion marker contract")
    for key in required - {"schema"}:
        value = values[key]
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{path}: malformed {key}")
    return values


def _prediction_state(payload: list[list[tuple[int, np.ndarray, float]]]) -> str:
    rows = payload[0]
    corners = (
        np.stack([np.asarray(row[1], dtype=np.float32) for row in rows])
        if rows
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    scores = np.asarray([float(row[2]) for row in rows], dtype=np.float32)
    return prediction_state_sha256(corners, scores)


def _selected_candidates(
    diagnostic: dict[str, Any], parent: Any, *, scene_id: str, prefix_id: str,
    parent_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent_rows: list[int] = []
    proposal_ids: list[int] = []
    c1_scores: list[float] = []
    seen_identities: set[str] = set()
    for candidate in diagnostic["candidates"]:
        source_rank = int(candidate.get("source_rank", -1))
        parent_row = int(candidate.get("parent_row", -1))
        proposal_id = int(candidate.get("proposal_id", -1))
        identity = str(candidate.get("identity_key", ""))
        expected_identity = (
            f"{scene_id}:{prefix_id}:{parent_sha256}:{proposal_id}"
        )
        if (
            source_rank < 1
            or source_rank > 5
            or parent_row < 0
            or parent_row >= len(parent.proposal_ids)
            or int(parent.proposal_ids[parent_row]) != proposal_id
            or identity != expected_identity
            or identity in seen_identities
        ):
            raise ValueError(f"{scene_id}: candidate identity/parent mismatch")
        seen_identities.add(identity)
        if not bool(candidate.get("online_yoloe_mask2_depth")):
            continue
        score = float(candidate.get("c1_depth_dino_track_score", float("nan")))
        if not math.isfinite(score):
            raise ValueError(f"{scene_id}: non-finite frozen C1 score")
        parent_rows.append(parent_row)
        proposal_ids.append(proposal_id)
        c1_scores.append(score)
    if len(parent_rows) != int(diagnostic.get("online_selected_count", -1)):
        raise ValueError(f"{scene_id}: online selected count mismatch")
    rows = np.asarray(parent_rows, dtype=np.int64)
    corners = np.ascontiguousarray(parent.corners_world[rows], dtype=np.float32)
    if len(corners) and (
        not np.isfinite(corners).all()
        or np.any(np.ptp(corners, axis=1) <= 0.0)
    ):
        raise ValueError(f"{scene_id}: invalid online candidate geometry")
    return (
        rows,
        np.asarray(proposal_ids, dtype=np.int64),
        np.asarray(c1_scores, dtype=np.float64),
        corners,
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    scene_list = args.scene_list.resolve()
    scenes = read_scene_list(scene_list)
    if len(scenes) not in (1, 10, 100):
        raise ValueError("online C3 shadow requires smoke1, fixed10, or full100")
    output_root = args.output_root.resolve()
    manifest_path = args.manifest.resolve()
    if output_root.exists() or manifest_path.exists():
        raise FileExistsError("refusing an existing online C3 shadow namespace")

    anchor_root = args.anchor_prediction_root.resolve()
    diagnostics_root = args.identity_diagnostics_root.resolve()
    parent_root = args.parent_cache_root.resolve()
    anchor_before = _tree_snapshot(anchor_root, scenes)
    prepared: list[dict[str, Any]] = []
    global_entries: list[tuple[int, int, float]] = []
    anchor_floor = float("inf")

    for scene_index, scene_id in enumerate(scenes):
        diagnostic_path = diagnostics_root / f"{scene_id}_c3_online_identity.json"
        diagnostic = _load_identity_diagnostic(diagnostic_path, scene_id)
        diagnostic_sha = sha256_file(diagnostic_path)
        parent_path = tr3d_residual_cache_path(parent_root, scene_id, args.prefix_id)
        parent_sha = sha256_file(parent_path)
        if (
            Path(str(diagnostic.get("parent_cache", ""))).resolve()
            != parent_path.resolve()
            or str(diagnostic.get("parent_cache_sha256")) != parent_sha
        ):
            raise ValueError(f"{scene_id}: diagnostic/parent lineage mismatch")
        parent = _load_parent(parent_path, scene_id, args.prefix_id)
        parent_rows, proposal_ids, c1_scores, corners = _selected_candidates(
            diagnostic,
            parent,
            scene_id=scene_id,
            prefix_id=args.prefix_id,
            parent_sha256=parent_sha,
        )
        anchor_path = anchor_root / f"{scene_id}_boxes.pkl"
        anchor_sha = sha256_file(anchor_path)
        anchor_payload = _load_prediction(anchor_path)
        marker_path = anchor_root / f"{scene_id}.run_fingerprint"
        marker = _load_completion_marker(marker_path)
        if (
            marker["active_prediction_sha256"] != anchor_sha
            or marker["c3_online_diagnostic_sha256"] != diagnostic_sha
        ):
            raise ValueError(f"{scene_id}: completion marker lineage mismatch")
        state_sha = _prediction_state(anchor_payload)
        if (
            diagnostic.get("prediction_state_before_sha256") != state_sha
            or diagnostic.get("prediction_state_after_sha256") != state_sha
            or int(diagnostic.get("prediction_count", -1)) != len(anchor_payload[0])
        ):
            raise ValueError(f"{scene_id}: diagnostic/terminal prediction mismatch")
        if anchor_payload[0]:
            anchor_floor = min(
                anchor_floor, min(float(row[2]) for row in anchor_payload[0])
            )
        for local_index, score in enumerate(c1_scores.tolist()):
            global_entries.append((scene_index, local_index, float(score)))
        prepared.append(
            {
                "scene_id": scene_id,
                "diagnostic_path": diagnostic_path,
                "diagnostic_sha256": diagnostic_sha,
                "parent_path": parent_path,
                "parent_sha256": parent_sha,
                "anchor_path": anchor_path,
                "anchor_sha256": anchor_sha,
                "anchor_payload": anchor_payload,
                "anchor_state_sha256": state_sha,
                "marker_path": marker_path,
                "marker_sha256": sha256_file(marker_path),
                "scene_fingerprint": marker["scene_fingerprint"],
                "parent_rows": parent_rows,
                "proposal_ids": proposal_ids,
                "c1_scores": c1_scores,
                "corners": corners,
            }
        )
    if not math.isfinite(anchor_floor) or anchor_floor <= 0.0:
        raise ValueError("terminal-R3 anchor score floor must be positive")
    score_map = _assign_candidate_scores(global_entries, anchor_floor)

    scene_reports: list[dict[str, Any]] = []
    for scene_index, row in enumerate(prepared):
        scores = [
            score_map[(scene_index, local_index)]
            for local_index in range(len(row["corners"]))
        ]
        output = _append_payload(row["anchor_payload"], row["corners"], scores)
        target = output_root / f"{row['scene_id']}_boxes.pkl"
        output_sha = _write_pickle_create_only(target, output)
        scene_reports.append(
            {
                "scene_id": row["scene_id"],
                "identity_diagnostic": str(row["diagnostic_path"]),
                "identity_diagnostic_sha256": row["diagnostic_sha256"],
                "parent_cache": str(row["parent_path"]),
                "parent_cache_sha256": row["parent_sha256"],
                "anchor_prediction": str(row["anchor_path"]),
                "anchor_prediction_sha256": row["anchor_sha256"],
                "anchor_prediction_state_sha256": row["anchor_state_sha256"],
                "completion_marker": str(row["marker_path"]),
                "completion_marker_sha256": row["marker_sha256"],
                "scene_fingerprint": row["scene_fingerprint"],
                "output_prediction_sha256": output_sha,
                "anchor_count": len(row["anchor_payload"][0]),
                "candidate_count": len(row["corners"]),
                "output_count": len(output[0]),
                "candidate_parent_rows": row["parent_rows"].tolist(),
                "candidate_proposal_ids": row["proposal_ids"].tolist(),
                "candidate_c1_track_scores": row["c1_scores"].tolist(),
                "candidate_output_scores": scores,
            }
        )

    anchor_after = _tree_snapshot(anchor_root, scenes)
    if anchor_before != anchor_after:
        raise RuntimeError("terminal-R3 anchor tree changed during shadow replay")
    output_tree = _tree_snapshot(output_root, scenes)
    manifest = {
        "schema": SCHEMA,
        "complete": True,
        "shadow_only": True,
        "observer_evidence_replay": True,
        "formal_active_authorized": False,
        "live_mutation_authorized": False,
        "ground_truth_access": False,
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
        "identity_diagnostics_root": str(diagnostics_root),
        "parent_cache_root": str(parent_root),
        "anchor_prediction_root": str(anchor_root),
        "output_prediction_root": str(output_root),
        "prefix_id": args.prefix_id,
        "materializer_code_sha256": _code_sha256(),
        "anchor_score_floor": anchor_floor,
        "anchor_count": sum(row["anchor_count"] for row in scene_reports),
        "candidate_count": sum(row["candidate_count"] for row in scene_reports),
        "output_count": sum(row["output_count"] for row in scene_reports),
        "anchor_tree_before": anchor_before,
        "anchor_tree_after": anchor_after,
        "output_tree": output_tree,
        "materialization_wall_s": time.perf_counter() - started,
        "scenes": scene_reports,
    }
    _write_json_create_only(manifest_path, manifest)
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--identity-diagnostics-root", type=Path, required=True)
    value.add_argument("--parent-cache-root", type=Path, required=True)
    value.add_argument("--anchor-prediction-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--manifest", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    report = materialize(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
