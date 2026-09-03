#!/usr/bin/env python3
"""GT-free identity audit for incremental novelty append-only outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.tr3d_c2_maskrgbd_cache import sha256_file  # noqa: E402
from boxfusion.tr3d_incremental_gate import IncrementalNoveltyPolicy  # noqa: E402
from tools.materialize_tr3d_c3_active import (  # noqa: E402
    _load_prediction, _write_json_create_only,
)
from tools.materialize_tr3d_incremental_novelty_active import SCHEMA  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


AUDIT_SCHEMA = "boxfusion.tr3d_incremental_novelty_active_audit.v1"


def audit(args: argparse.Namespace) -> dict:
    scenes = read_scene_list(args.scene_list.resolve())
    policy = IncrementalNoveltyPolicy.load(args.policy)
    manifest = json.loads(args.manifest.resolve().read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != SCHEMA
        or not manifest.get("complete")
        or not manifest.get("activation_authorized")
        or manifest.get("ground_truth_access")
        or manifest.get("policy_checkpoint_sha256") != policy.sha256
        or int(manifest.get("scene_count", -1)) != len(scenes)
    ):
        raise ValueError("incremental active manifest contract failed")
    by_scene = {row["scene_id"]: row for row in manifest.get("scenes", [])}
    issues = []
    total_anchor = total_applied = 0
    reports = []
    for scene in scenes:
        row = by_scene.get(scene)
        if row is None:
            issues.append(f"{scene}: missing manifest row")
            continue
        anchor_path = args.anchor_root.resolve() / f"{scene}_boxes.pkl"
        active_path = args.active_root.resolve() / f"{scene}_boxes.pkl"
        anchors = _load_prediction(anchor_path)[0]
        active = _load_prediction(active_path)[0]
        applied = len(active) - len(anchors)
        exact = sum(
            left[0] == right[0]
            and left[2] == right[2]
            and left[1].dtype == right[1].dtype
            and left[1].tobytes(order="C") == right[1].tobytes(order="C")
            for left, right in zip(anchors, active[: len(anchors)])
        )
        floor = min((item[2] for item in anchors), default=float("inf"))
        supplemental = [item[2] for item in active[len(anchors):]]
        if (
            applied < 0 or exact != len(anchors)
            or applied != int(row.get("applied_count", -1))
            or len(active) != int(row.get("output_count", -1))
            or sha256_file(anchor_path) != row.get("anchor_prediction_sha256")
            or sha256_file(active_path) != row.get("output_prediction_sha256")
            or any(score >= floor or score <= 0.0 for score in supplemental)
            or supplemental != sorted(supplemental, reverse=True)
        ):
            issues.append(f"{scene}: append-only identity/score contract failed")
        total_anchor += len(anchors); total_applied += max(applied, 0)
        reports.append({"scene_id": scene, "anchor_count": len(anchors),
                        "applied_count": applied, "output_count": len(active),
                        "exact_anchor_rows": exact})
    result = {
        "schema": AUDIT_SCHEMA, "complete": True, "ok": not issues,
        "ground_truth_access": False, "append_only": True,
        "policy_checkpoint_sha256": policy.sha256,
        "manifest_sha256": sha256_file(args.manifest.resolve()),
        "scene_count": len(scenes), "anchor_count": total_anchor,
        "applied_count": total_applied, "output_count": total_anchor + total_applied,
        "issues": issues, "scenes": reports,
    }
    _write_json_create_only(args.output.resolve(), result)
    return result


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--policy", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--anchor-root", type=Path, required=True)
    value.add_argument("--active-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    result = audit(parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["ok"] else 1)
