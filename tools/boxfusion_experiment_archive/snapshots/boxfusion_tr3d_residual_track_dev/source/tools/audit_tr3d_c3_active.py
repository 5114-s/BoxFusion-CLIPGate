#!/usr/bin/env python3
"""GT-free byte/lineage audit for a C3 append-only shadow materialization."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import pickle
import struct
import sys
from typing import Any, Mapping, Sequence

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
from tools.audit_tr3d_c2_maskrgbd_observer import (  # noqa: E402
    _write_json_create_only,
)
from tools.run_tr3d_c1_track_observer import _tree_snapshot  # noqa: E402
from tools.run_tr3d_c2_maskrgbd_observer import (  # noqa: E402
    REPORT_SCHEMA as C2_EXPORT_SCHEMA,
    _code_hash as c2_code_hash,
)
from tools.tr3d_data import read_scene_list  # noqa: E402


REPORT_SCHEMA = "boxfusion.tr3d_c3_active_identity_audit.v1"
MANIFEST_SCHEMA = "boxfusion.tr3d_c3_shadow_active_manifest.v1"
ROUTE_NAME = "top5_mask2_depth"
PICKLE_PROTOCOL = 5


def _load_prediction(path: Path) -> list[list[tuple[Any, np.ndarray, Any]]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301 - pinned local artifact
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"{path}: expected a one-scene outer list")
    if not isinstance(payload[0], list):
        raise ValueError(f"{path}: prediction rows must be a list")
    for index, row in enumerate(payload[0]):
        if not isinstance(row, tuple) or len(row) != 3:
            raise ValueError(f"{path}: row {index} must be a 3-tuple")
        label, corners, score = row
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"{path}: row {index} label must be integer")
        geometry = np.asarray(corners)
        if (
            geometry.shape != (8, 3)
            or geometry.dtype.hasobject
            or not np.isfinite(geometry).all()
        ):
            raise ValueError(f"{path}: row {index} geometry must be finite [8,3]")
        if isinstance(score, bool) or not isinstance(score, (float, np.floating)):
            raise ValueError(f"{path}: row {index} score must be floating point")
        if not math.isfinite(float(score)):
            raise ValueError(f"{path}: row {index} score must be finite")
    return payload


def _scalar_bytes(value: Any) -> bytes:
    return pickle.dumps(value, protocol=PICKLE_PROTOCOL)


def _score_bytes(value: Any) -> bytes:
    if isinstance(value, np.floating):
        return np.asarray(value).tobytes()
    return struct.pack("!d", float(value))


def _geometry_equal(left: object, right: object) -> bool:
    lhs, rhs = np.asarray(left), np.asarray(right)
    return bool(
        type(left) is type(right)
        and lhs.dtype == rhs.dtype
        and lhs.shape == rhs.shape
        and lhs.strides == rhs.strides
        and lhs.flags.c_contiguous == rhs.flags.c_contiguous
        and lhs.flags.f_contiguous == rhs.flags.f_contiguous
        and lhs.tobytes(order="A") == rhs.tobytes(order="A")
    )


def _anchor_row_equal(before: tuple[Any, Any, Any], after: tuple[Any, Any, Any]) -> bool:
    return bool(
        type(before) is type(after)
        and type(before[0]) is type(after[0])
        and _scalar_bytes(before[0]) == _scalar_bytes(after[0])
        and _geometry_equal(before[1], after[1])
        and type(before[2]) is type(after[2])
        and _score_bytes(before[2]) == _score_bytes(after[2])
    )


def _manifest_path(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"manifest {name} must be a non-empty path")
    return Path(value).resolve()


def _validate_manifest_contract(
    manifest: Mapping[str, Any], *, paths: Mapping[str, Path], prefix_id: str
) -> None:
    if manifest.get("schema") != MANIFEST_SCHEMA or not manifest.get("complete"):
        raise ValueError("unsupported or incomplete C3 materialization manifest")
    if (
        not manifest.get("shadow_only")
        or manifest.get("formal_active_authorized")
        or manifest.get("ground_truth_access")
        or manifest.get("counterfactual_report_access")
        or manifest.get("clip_access")
    ):
        raise ValueError("C3 manifest violates the GT-free shadow contract")
    if manifest.get("prefix_id") != prefix_id:
        raise ValueError("C3 manifest prefix mismatch")
    for name, expected in paths.items():
        if _manifest_path(manifest.get(name), name) != expected:
            raise ValueError(f"C3 manifest {name} path mismatch")


def _load_c2_export(path: Path, scenes: Sequence[str]) -> dict[str, Any]:
    export = json.loads(path.read_text(encoding="utf-8"))
    if export.get("schema") != C2_EXPORT_SCHEMA:
        raise ValueError("unsupported C2 export report")
    if (
        not export.get("observer_only")
        or export.get("mutation_enabled")
        or int(export.get("applied_count", -1)) != 0
        or export.get("ground_truth_access")
        or export.get("clip_access")
        or export.get("teacher_labels_used_for_gate")
    ):
        raise ValueError("C2 export violates its observer-only contract")
    if export.get("code_sha256") != c2_code_hash():
        raise ValueError("current C2 code differs from immutable export")
    ordered = [str(row.get("scene_id")) for row in export.get("scenes", [])]
    if ordered != list(scenes) or int(export.get("scene_count", -1)) != len(scenes):
        raise ValueError("C2 export ordered scene set mismatch")
    if len(set(ordered)) != len(ordered):
        raise ValueError("C2 export contains duplicate scenes")
    return export


def _load_parent(path: Path, scene_id: str, prefix_id: str):
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


def _expected_rows(
    scene_rows: Sequence[dict[str, Any]],
) -> tuple[list[list[dict[str, Any]]], float, bool]:
    anchor_scores = [
        float(row[2])
        for scene in scene_rows
        for row in scene["source"][0]
    ]
    anchor_floor = min(anchor_scores) if anchor_scores else 0.0
    c1_order = [
        (
            -float(candidate["c1_track_score"]),
            scene_index,
            local_index,
        )
        for scene_index, scene in enumerate(scene_rows)
        for local_index, candidate in enumerate(scene["candidates"])
    ]
    score_order = [
        (
            -float(candidate["score"]),
            scene_index,
            local_index,
        )
        for scene_index, scene in enumerate(scene_rows)
        for local_index, candidate in enumerate(scene["candidates"])
    ]
    c1_order.sort()
    score_order.sort()
    strict_scores = all(
        -score_order[index][0] > -score_order[index + 1][0]
        for index in range(len(score_order) - 1)
    )
    rank_exact = strict_scores and (
        [row[1:] for row in c1_order] == [row[1:] for row in score_order]
    )
    # Physical pickle rows deliberately retain C2 source-row order.  The
    # unique output score, not cross-scene row concatenation, defines the
    # global evaluator order.
    return [list(scene["candidates"]) for scene in scene_rows], anchor_floor, rank_exact


def _validate_output_tree(root: Path, scenes: Sequence[str]) -> None:
    expected = {f"{scene}_boxes.pkl" for scene in scenes}
    found = {path.name for path in root.iterdir()}
    if found != expected:
        raise ValueError(
            f"output prediction set mismatch: missing={sorted(expected-found)}, "
            f"extra={sorted(found-expected)}"
        )
    invalid = [
        path.name
        for path in root.iterdir()
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222
    ]
    if invalid:
        raise ValueError(f"output predictions are not immutable regular files: {invalid}")


def _audit_payload(
    source: list[list[tuple[Any, np.ndarray, Any]]],
    output: list[list[tuple[Any, np.ndarray, Any]]],
    expected_candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    anchor_count = len(source[0])
    candidate_count = len(expected_candidates)
    issues: list[str] = []
    if len(output[0]) != anchor_count + candidate_count:
        issues.append(
            f"prediction count differs ({len(output[0])} != "
            f"{anchor_count + candidate_count})"
        )
    exact_anchors = 0
    for index in range(min(anchor_count, len(output[0]))):
        if _anchor_row_equal(source[0][index], output[0][index]):
            exact_anchors += 1
        else:
            issues.append(f"anchor row {index} differs in type/dtype/bytes")
    exact_candidates = 0
    available = max(0, len(output[0]) - anchor_count)
    for offset in range(min(candidate_count, available)):
        row = output[0][anchor_count + offset]
        expected = expected_candidates[offset]
        if type(row) is not tuple or len(row) != 3:
            issues.append(f"candidate row {offset} is not a 3-tuple")
            continue
        if type(row[0]) is not int or row[0] != 0:
            issues.append(f"candidate row {offset} label is not Python int zero")
        if not _geometry_equal(row[1], expected["corners"]):
            issues.append(f"candidate row {offset} geometry bytes differ")
        if type(row[2]) is not float or _score_bytes(row[2]) != _score_bytes(
            expected["score"]
        ):
            issues.append(f"candidate row {offset} score bytes differ")
        if not any(value.startswith(f"candidate row {offset} ") for value in issues):
            exact_candidates += 1
    return {
        "ok": not issues,
        "issues": issues,
        "anchor_rows": anchor_count,
        "candidate_rows": candidate_count,
        "exact_anchor_rows": exact_anchors,
        "exact_candidate_rows": exact_candidates,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("manifest must be a regular non-symlink file")
    if manifest_path.stat().st_mode & 0o222:
        raise ValueError("manifest must be immutable")
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {
        "scene_list": args.scene_list.resolve(),
        "c2_export_report": args.c2_export_report.resolve(),
        "c2_cache_root": args.c2_cache_root.resolve(),
        "parent_cache_root": args.parent_cache_root.resolve(),
        "active_prediction_root": args.active_prediction_root.resolve(),
        "output_root": args.output_root.resolve(),
    }
    report_path = args.report.expanduser().absolute()
    for name in (
        "c2_cache_root",
        "parent_cache_root",
        "active_prediction_root",
        "output_root",
    ):
        protected_root = paths[name]
        if report_path == protected_root or protected_root in report_path.parents:
            raise ValueError(f"report must be outside protected {name}")
    _validate_manifest_contract(manifest, paths=paths, prefix_id=args.prefix_id)
    scenes = read_scene_list(paths["scene_list"])
    if len(scenes) not in (10, 100):
        raise ValueError("C3 identity audit requires fixed10 or full100")
    if manifest.get("scene_list_sha256") != sha256_file(paths["scene_list"]):
        raise ValueError("manifest scene-list hash mismatch")
    if manifest.get("c2_export_report_sha256") != sha256_file(
        paths["c2_export_report"]
    ):
        raise ValueError("manifest C2 export hash mismatch")
    export = _load_c2_export(paths["c2_export_report"], scenes)
    export_rows = {str(row["scene_id"]): row for row in export["scenes"]}
    input_before = _tree_snapshot(paths["active_prediction_root"], scenes)
    output_before = _tree_snapshot(paths["output_root"], scenes)
    if manifest.get("input_prediction_tree_sha256") != input_before["tree_sha256"]:
        raise ValueError("manifest input prediction tree hash mismatch")
    if manifest.get("output_prediction_tree_sha256") != output_before["tree_sha256"]:
        raise ValueError("manifest output prediction tree hash mismatch")
    _validate_output_tree(paths["output_root"], scenes)
    if manifest.get("anchor_tree_before") != input_before:
        raise ValueError("manifest anchor_tree_before mismatch")
    if manifest.get("anchor_tree_after") != input_before:
        raise ValueError("manifest anchor_tree_after mismatch")
    if manifest.get("output_tree") != output_before:
        raise ValueError("manifest output_tree mismatch")
    manifest_rows = manifest.get("scenes", [])
    if [str(row.get("scene_id")) for row in manifest_rows] != scenes:
        raise ValueError("manifest ordered per-scene rows mismatch")
    manifest_by_scene = {str(row["scene_id"]): row for row in manifest_rows}

    gate_index = GATE_NAMES.index("mask2_depth")
    scene_rows: list[dict[str, Any]] = []
    protected: list[tuple[Path, str]] = [
        (manifest_path, manifest_sha),
        (paths["scene_list"], sha256_file(paths["scene_list"])),
        (paths["c2_export_report"], sha256_file(paths["c2_export_report"])),
    ]
    for scene_order, scene_id in enumerate(scenes):
        source_path = paths["active_prediction_root"] / f"{scene_id}_boxes.pkl"
        output_path = paths["output_root"] / f"{scene_id}_boxes.pkl"
        c2_path = sidecar_path(paths["c2_cache_root"], scene_id, args.prefix_id)
        parent_path = tr3d_residual_cache_path(
            paths["parent_cache_root"], scene_id, args.prefix_id
        )
        c2_sha, parent_sha = sha256_file(c2_path), sha256_file(parent_path)
        if c2_sha != export_rows[scene_id]["sidecar_sha256"]:
            raise ValueError(f"{scene_id}: C2 sidecar hash mismatch")
        c2 = load_sidecar(c2_path)
        if parent_sha != c2.parent_cache_sha256:
            raise ValueError(f"{scene_id}: parent cache hash mismatch")
        parent = _load_parent(parent_path, scene_id, args.prefix_id)
        if not np.array_equal(parent.proposal_ids[c2.parent_rows], c2.proposal_ids):
            raise ValueError(f"{scene_id}: C2/parent proposal identity mismatch")
        source_sha, output_sha = sha256_file(source_path), sha256_file(output_path)
        if source_sha != c2.anchor_prediction_sha256:
            raise ValueError(f"{scene_id}: C2 anchor prediction hash mismatch")
        row = manifest_by_scene[scene_id]
        expected_hashes = {
            "source_prediction_sha256": source_sha,
            "c2_sidecar_sha256": c2_sha,
            "parent_cache_sha256": parent_sha,
            "output_prediction_sha256": output_sha,
        }
        for name, expected in expected_hashes.items():
            if row.get(name) != expected:
                raise ValueError(f"{scene_id}: manifest {name} mismatch")
        route = (c2.source_ranks <= 5) & c2.observation.gate_mask[:, gate_index]
        selected = np.flatnonzero(route)
        expected_parent_rows = np.asarray(c2.parent_rows[selected], dtype=np.int64)
        expected_proposal_ids = np.asarray(c2.proposal_ids[selected], dtype=np.int64)
        expected_c1_scores = np.asarray(c2.c1_track_scores[selected], dtype=np.float64)
        manifest_parent_rows = np.asarray(row.get("candidate_parent_rows", []))
        manifest_proposal_ids = np.asarray(row.get("candidate_proposal_ids", []))
        manifest_c1_scores = np.asarray(row.get("candidate_c1_track_scores", []))
        manifest_output_scores = np.asarray(row.get("candidate_output_scores", []))
        if not np.array_equal(manifest_parent_rows, expected_parent_rows):
            raise ValueError(f"{scene_id}: manifest candidate parent rows mismatch")
        if not np.array_equal(manifest_proposal_ids, expected_proposal_ids):
            raise ValueError(f"{scene_id}: manifest candidate proposal ids mismatch")
        if not np.array_equal(manifest_c1_scores, expected_c1_scores):
            raise ValueError(f"{scene_id}: manifest candidate C1 scores mismatch")
        if (
            manifest_output_scores.shape != (len(selected),)
            or not np.isfinite(manifest_output_scores).all()
        ):
            raise ValueError(f"{scene_id}: malformed manifest candidate output scores")
        candidates = [
            {
                "scene_order": scene_order,
                "c2_row": int(c2_row),
                "proposal_id": int(c2.proposal_ids[c2_row]),
                "c1_track_score": float(c2.c1_track_scores[c2_row]),
                "score": float(manifest_output_scores[local_index]),
                "corners": parent.corners_world[int(c2.parent_rows[c2_row])],
            }
            for local_index, c2_row in enumerate(selected)
        ]
        scene_rows.append(
            {
                "scene_id": scene_id,
                "source": _load_prediction(source_path),
                "output": _load_prediction(output_path),
                "candidates": candidates,
            }
        )
        protected.extend(
            ((source_path, source_sha), (output_path, output_sha),
             (c2_path, c2_sha), (parent_path, parent_sha))
        )

    expected_by_scene, anchor_floor, expected_global_order = _expected_rows(scene_rows)
    if float(manifest.get("anchor_score_floor", float("nan"))) != anchor_floor:
        raise ValueError("manifest anchor score floor mismatch")
    scene_reports: list[dict[str, Any]] = []
    candidate_order: list[tuple[float, int, int]] = []
    for scene_index, scene in enumerate(scene_rows):
        expected = expected_by_scene[scene_index]
        row_report = _audit_payload(scene["source"], scene["output"], expected)
        manifest_row = manifest_by_scene[scene["scene_id"]]
        if int(manifest_row.get("anchor_rows", -1)) != row_report["anchor_rows"]:
            row_report["issues"].append("manifest anchor_rows mismatch")
        if int(manifest_row.get("candidate_rows", -1)) != row_report["candidate_rows"]:
            row_report["issues"].append("manifest candidate_rows mismatch")
        row_report["ok"] = not row_report["issues"]
        row_report["scene_id"] = scene["scene_id"]
        scene_reports.append(row_report)
        anchor_count = len(scene["source"][0])
        for local_index, row in enumerate(expected):
            output_index = anchor_count + local_index
            if output_index >= len(scene["output"][0]):
                continue
            output_row = scene["output"][0][output_index]
            candidate_order.append(
                (float(output_row[2]), scene_index, local_index)
            )
    expected_candidate_count = sum(len(rows) for rows in expected_by_scene)
    global_order_ok = (
        expected_global_order and len(candidate_order) == expected_candidate_count
    )
    all_below_anchor = all(
        0.0 < score < anchor_floor for score, _, _ in candidate_order
    )
    if int(manifest.get("candidate_count", -1)) != len(candidate_order):
        issues = ["manifest total candidate_count mismatch"]
    else:
        issues = []
    if int(manifest.get("anchor_count", -1)) != sum(
        len(scene["source"][0]) for scene in scene_rows
    ):
        issues.append("manifest total anchor_count mismatch")
    if int(manifest.get("output_count", -1)) != sum(
        len(scene["output"][0]) for scene in scene_rows
    ):
        issues.append("manifest total output_count mismatch")
    issues.extend(
        f"{row['scene_id']}: {issue}"
        for row in scene_reports
        for issue in row["issues"]
    )
    if not global_order_ok:
        issues.append("global candidate score order differs from C1 rank")
    if not all_below_anchor:
        issues.append("one or more candidate scores do not rank below every anchor")

    input_after = _tree_snapshot(paths["active_prediction_root"], scenes)
    output_after = _tree_snapshot(paths["output_root"], scenes)
    if input_before != input_after:
        issues.append("input prediction tree changed during audit")
    if output_before != output_after:
        issues.append("output prediction tree changed during audit")
    for path, expected in protected:
        if sha256_file(path) != expected:
            issues.append(f"artifact changed during audit: {path}")
    return {
        "schema": REPORT_SCHEMA,
        "ok": not issues,
        "observer_only_identity_audit": True,
        "ground_truth_access": False,
        "clip_access": False,
        "shadow_only": True,
        "formal_active_authorized": False,
        "route": ROUTE_NAME,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "input_prediction_tree_before": input_before,
        "input_prediction_tree_after": input_after,
        "output_prediction_tree_before": output_before,
        "output_prediction_tree_after": output_after,
        "anchor_floor": anchor_floor,
        "candidate_count": len(candidate_order),
        "global_candidate_c1_order_exact": global_order_ok,
        "all_candidate_scores_below_all_anchors": all_below_anchor,
        "scene_reports": scene_reports,
        "issues": issues,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--c2-export-report", type=Path, required=True)
    value.add_argument("--c2-cache-root", type=Path, required=True)
    value.add_argument("--parent-cache-root", type=Path, required=True)
    value.add_argument("--active-prediction-root", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--prefix-id", default="p100")
    value.add_argument("--report", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = audit(args)
    _write_json_create_only(args.report.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
