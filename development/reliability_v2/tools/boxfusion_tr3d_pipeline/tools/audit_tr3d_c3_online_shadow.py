#!/usr/bin/env python3
"""GT-free identity and lineage audit for an online C3 shadow tree."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file  # noqa: E402
from boxfusion.tr3d_residual_cache import tr3d_residual_cache_path  # noqa: E402
from tools.audit_tr3d_c3_active import _anchor_row_equal, _geometry_equal  # noqa: E402
from tools.materialize_tr3d_c3_active import (  # noqa: E402
    _load_prediction,
    _write_json_create_only,
)
from tools.materialize_tr3d_c3_online_shadow import (  # noqa: E402
    ROUTE,
    SCHEMA as MANIFEST_SCHEMA,
    SCORE_POLICY,
    _code_sha256,
    _load_completion_marker,
    _load_identity_diagnostic,
    _load_parent,
    _prediction_state,
    _selected_candidates,
)
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


SCHEMA = "boxfusion.tr3d_c3_online_shadow_audit.v1"


def _manifest_path(payload: dict[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest {key} must be a non-empty path")
    return Path(value).resolve()


def _exact_output_set(root: Path, scenes: Sequence[str]) -> None:
    expected = {f"{scene_id}_boxes.pkl" for scene_id in scenes}
    found = {path.name for path in root.iterdir()}
    if found != expected:
        raise ValueError(
            f"shadow output set mismatch: missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}"
        )
    invalid = [
        path.name
        for path in root.iterdir()
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222
    ]
    if invalid:
        raise ValueError(f"shadow outputs are not immutable regular files: {invalid}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    if (
        not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path.stat().st_mode & 0o222
    ):
        raise ValueError("manifest must be immutable regular file")
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or not manifest.get("complete")
        or not manifest.get("shadow_only")
        or manifest.get("formal_active_authorized")
        or manifest.get("live_mutation_authorized")
        or manifest.get("ground_truth_access")
        or manifest.get("clip_access")
        or manifest.get("teacher_labels_used")
        or manifest.get("route") != ROUTE
        or manifest.get("score_policy") != SCORE_POLICY
        or manifest.get("materializer_code_sha256") != _code_sha256()
    ):
        raise ValueError("online C3 shadow manifest contract failed")

    paths = {
        "scene_list": args.scene_list.resolve(),
        "identity_diagnostics_root": args.identity_diagnostics_root.resolve(),
        "parent_cache_root": args.parent_cache_root.resolve(),
        "anchor_prediction_root": args.anchor_prediction_root.resolve(),
        "output_prediction_root": args.output_root.resolve(),
    }
    for key, expected in paths.items():
        if _manifest_path(manifest, key) != expected:
            raise ValueError(f"manifest {key} path mismatch")
    if manifest.get("prefix_id") != args.prefix_id:
        raise ValueError("manifest prefix mismatch")
    scenes = read_scene_list(paths["scene_list"])
    if len(scenes) not in (1, 10, 100):
        raise ValueError("online shadow audit requires smoke1/fixed10/full100")
    if (
        int(manifest.get("scene_count", -1)) != len(scenes)
        or manifest.get("scene_list_sha256") != sha256_file(paths["scene_list"])
    ):
        raise ValueError("manifest scene-list identity mismatch")

    anchor_before = _tree_snapshot(paths["anchor_prediction_root"], scenes)
    output_before = _tree_snapshot(paths["output_prediction_root"], scenes)
    if manifest.get("anchor_tree_before") != anchor_before:
        raise ValueError("manifest anchor_tree_before mismatch")
    if manifest.get("anchor_tree_after") != anchor_before:
        raise ValueError("manifest anchor_tree_after mismatch")
    if manifest.get("output_tree") != output_before:
        raise ValueError("manifest output tree mismatch")
    _exact_output_set(paths["output_prediction_root"], scenes)

    manifest_rows = manifest.get("scenes")
    if (
        not isinstance(manifest_rows, list)
        or [str(row.get("scene_id")) for row in manifest_rows] != scenes
    ):
        raise ValueError("manifest ordered scenes mismatch")

    anchor_floor = float("inf")
    exact_anchor_rows = 0
    exact_candidate_rows = 0
    anchor_count = 0
    candidate_count = 0
    output_count = 0
    global_c1_order: list[tuple[float, int, int]] = []
    global_score_order: list[tuple[float, int, int]] = []
    all_candidate_scores: list[float] = []
    per_scene: list[dict[str, Any]] = []

    for scene_index, (scene_id, manifest_row) in enumerate(zip(scenes, manifest_rows)):
        diagnostic_path = (
            paths["identity_diagnostics_root"]
            / f"{scene_id}_c3_online_identity.json"
        )
        diagnostic = _load_identity_diagnostic(diagnostic_path, scene_id)
        diagnostic_sha = sha256_file(diagnostic_path)
        parent_path = tr3d_residual_cache_path(
            paths["parent_cache_root"], scene_id, args.prefix_id
        )
        parent_sha = sha256_file(parent_path)
        if (
            Path(str(diagnostic.get("parent_cache", ""))).resolve()
            != parent_path.resolve()
            or diagnostic.get("parent_cache_sha256") != parent_sha
        ):
            raise ValueError(f"{scene_id}: parent lineage mismatch")
        parent = _load_parent(parent_path, scene_id, args.prefix_id)
        parent_rows, proposal_ids, c1_scores, corners = _selected_candidates(
            diagnostic,
            parent,
            scene_id=scene_id,
            prefix_id=args.prefix_id,
            parent_sha256=parent_sha,
        )

        anchor_path = paths["anchor_prediction_root"] / f"{scene_id}_boxes.pkl"
        marker_path = paths["anchor_prediction_root"] / f"{scene_id}.run_fingerprint"
        output_path = paths["output_prediction_root"] / f"{scene_id}_boxes.pkl"
        anchor_sha = sha256_file(anchor_path)
        output_sha = sha256_file(output_path)
        marker_sha = sha256_file(marker_path)
        marker = _load_completion_marker(marker_path)
        if (
            marker["active_prediction_sha256"] != anchor_sha
            or marker["c3_online_diagnostic_sha256"] != diagnostic_sha
        ):
            raise ValueError(f"{scene_id}: completion marker mismatch")
        anchor = _load_prediction(anchor_path)
        output = _load_prediction(output_path)
        anchor_state = _prediction_state(anchor)
        if (
            diagnostic.get("prediction_state_before_sha256") != anchor_state
            or diagnostic.get("prediction_state_after_sha256") != anchor_state
            or int(diagnostic.get("prediction_count", -1)) != len(anchor[0])
        ):
            raise ValueError(f"{scene_id}: prediction state binding mismatch")

        expected_scalars = {
            "identity_diagnostic_sha256": diagnostic_sha,
            "parent_cache_sha256": parent_sha,
            "anchor_prediction_sha256": anchor_sha,
            "anchor_prediction_state_sha256": anchor_state,
            "completion_marker_sha256": marker_sha,
            "scene_fingerprint": marker["scene_fingerprint"],
            "output_prediction_sha256": output_sha,
        }
        for key, expected in expected_scalars.items():
            if manifest_row.get(key) != expected:
                raise ValueError(f"{scene_id}: manifest {key} mismatch")
        expected_paths = {
            "identity_diagnostic": diagnostic_path,
            "parent_cache": parent_path,
            "anchor_prediction": anchor_path,
            "completion_marker": marker_path,
        }
        for key, expected in expected_paths.items():
            if Path(str(manifest_row.get(key, ""))).resolve() != expected.resolve():
                raise ValueError(f"{scene_id}: manifest {key} path mismatch")
        arrays = {
            "candidate_parent_rows": parent_rows,
            "candidate_proposal_ids": proposal_ids,
            "candidate_c1_track_scores": c1_scores,
        }
        for key, expected in arrays.items():
            if not np.array_equal(np.asarray(manifest_row.get(key, [])), expected):
                raise ValueError(f"{scene_id}: manifest {key} mismatch")

        candidate_scores = np.asarray(
            manifest_row.get("candidate_output_scores", []), dtype=np.float64
        )
        if (
            candidate_scores.shape != (len(corners),)
            or not np.isfinite(candidate_scores).all()
        ):
            raise ValueError(f"{scene_id}: invalid candidate score vector")
        if len(output[0]) != len(anchor[0]) + len(corners):
            raise ValueError(f"{scene_id}: output row count mismatch")
        for row_index, anchor_row in enumerate(anchor[0]):
            if not _anchor_row_equal(anchor_row, output[0][row_index]):
                raise ValueError(f"{scene_id}: anchor row {row_index} changed")
            exact_anchor_rows += 1
        for local_index, corners_expected in enumerate(corners):
            row = output[0][len(anchor[0]) + local_index]
            score_expected = float(candidate_scores[local_index])
            if (
                type(row) is not tuple
                or len(row) != 3
                or type(row[0]) is not int
                or row[0] != 0
                or not _geometry_equal(row[1], corners_expected)
                or type(row[2]) is not float
                or row[2] != score_expected
            ):
                raise ValueError(f"{scene_id}: candidate row {local_index} changed")
            exact_candidate_rows += 1
            global_c1_order.append((-float(c1_scores[local_index]), scene_index, local_index))
            global_score_order.append((-score_expected, scene_index, local_index))
            all_candidate_scores.append(score_expected)
        if anchor[0]:
            anchor_floor = min(anchor_floor, min(float(row[2]) for row in anchor[0]))
        anchor_count += len(anchor[0])
        candidate_count += len(corners)
        output_count += len(output[0])
        per_scene.append(
            {
                "scene_id": scene_id,
                "anchor_rows": len(anchor[0]),
                "candidate_rows": len(corners),
                "output_rows": len(output[0]),
                "exact_anchor_rows": len(anchor[0]),
                "exact_candidate_rows": len(corners),
            }
        )

    if not math.isfinite(anchor_floor) or anchor_floor <= 0.0:
        raise ValueError("invalid global anchor score floor")
    if float(manifest.get("anchor_score_floor", float("nan"))) != anchor_floor:
        raise ValueError("manifest anchor score floor mismatch")
    if all_candidate_scores and (
        min(all_candidate_scores) <= 0.0
        or max(all_candidate_scores) >= anchor_floor
        or len(set(all_candidate_scores)) != len(all_candidate_scores)
    ):
        raise ValueError("candidate scores violate low-score/uniqueness contract")
    global_c1_order.sort()
    global_score_order.sort()
    if [row[1:] for row in global_c1_order] != [row[1:] for row in global_score_order]:
        raise ValueError("candidate score order differs from frozen C1 order")
    expected_counts = {
        "anchor_count": anchor_count,
        "candidate_count": candidate_count,
        "output_count": output_count,
    }
    for key, expected in expected_counts.items():
        if int(manifest.get(key, -1)) != expected:
            raise ValueError(f"manifest {key} mismatch")

    anchor_after = _tree_snapshot(paths["anchor_prediction_root"], scenes)
    output_after = _tree_snapshot(paths["output_prediction_root"], scenes)
    if anchor_after != anchor_before or output_after != output_before:
        raise RuntimeError("protected prediction tree changed during audit")
    report = {
        "schema": SCHEMA,
        "complete": True,
        "ok": True,
        "shadow_only": True,
        "formal_active_authorized": False,
        "ground_truth_access": False,
        "clip_access": False,
        "route": ROUTE,
        "score_policy": SCORE_POLICY,
        "scene_count": len(scenes),
        "scene_list": str(paths["scene_list"]),
        "scene_list_sha256": sha256_file(paths["scene_list"]),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "anchor_count": anchor_count,
        "candidate_count": candidate_count,
        "output_count": output_count,
        "exact_anchor_rows": exact_anchor_rows,
        "exact_candidate_rows": exact_candidate_rows,
        "candidate_score_order_exact": True,
        "candidate_scores_positive_below_anchor_floor": True,
        "anchor_score_floor": anchor_floor,
        "anchor_tree_unchanged": True,
        "output_tree_unchanged_during_audit": True,
        "scenes": per_scene,
    }
    _write_json_create_only(args.report.resolve(), report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--identity-diagnostics-root", type=Path, required=True)
    value.add_argument("--parent-cache-root", type=Path, required=True)
    value.add_argument("--anchor-prediction-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--report", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    report = audit(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
