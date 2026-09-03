#!/usr/bin/env python3
"""Seal the exact100 GT-free Stage-O overlay collection and OOF provenance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import regular_file  # noqa: E402
from boxfusion.ca1m_tr3d_overlay_binding_v4 import (  # noqa: E402
    validate_overlay_authorization,
)
from boxfusion.ca1m_tr3d_terminal_gate_v4 import load_oof_row_scores  # noqa: E402
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    load_overlay_cache,
    sha256_file,
)
from tools.preflight_ca1m_tr3d_terminal_train100_v4 import (  # noqa: E402
    validate_config,
)


SCHEMA = "boxfusion.ca1m_tr3d_terminal_overlay_collection.v2"


def _sealed(path: Path, name: str) -> Path:
    source = regular_file(path, name)
    if source.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {source}")
    return source


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = _sealed(path, name)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return source, value


def _record(binding: Mapping[str, Any], name: str) -> Path:
    source = _sealed(Path(str(binding.get("path", ""))), name)
    if binding.get("sha256") != sha256_file(source):
        raise ValueError(f"{name} SHA256 differs")
    return source


def _publish(path: Path, value: Mapping[str, Any]) -> Path:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing overlay collection: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def build(config_path: Path) -> dict[str, Any]:
    config_source, cfg = _json(config_path, "Stage-O O2 config")
    report = validate_config(config_source)
    if (
        report.get("overlay_stage_runtime_authorized") is not True
        or (report.get("overlay_inventory") or {}).get("complete") is not True
        or (report.get("overlay_inventory") or {}).get("count") != 100
    ):
        raise ValueError("Stage-O exact100 preflight is incomplete")
    authorization = validate_overlay_authorization(config_source, cfg)
    proposal_rows = authorization.pop("proposal_rows")
    scene_path = regular_file(Path(str(cfg["scene_contract"]["path"])), "scene list")
    scenes = tuple(row.strip() for row in scene_path.read_text().splitlines() if row.strip())
    if (
        len(scenes) != 100
        or len(set(scenes)) != 100
        or sha256_file(scene_path) != cfg["scene_contract"]["sha256"]
    ):
        raise ValueError("overlay collection requires exact100 scenes")

    stage_o = cfg["stage_o_binding"]
    upstream = {
        name: dict(stage_o[name])
        for name in (
            "proposal_collection", "final_base_manifest",
            "native_b6_v2_collection_manifest",
            "native_b6_v2_deployment_checkpoint",
            "native_b6_v2_deployment_checkpoint_manifest",
            "native_b6_v2_oof_row_scores",
            "native_b6_v2_oof_row_scores_manifest",
        )
    }
    final_path, final = _json(
        _record(upstream["final_base_manifest"], "final-base manifest"),
        "final-base manifest",
    )
    b6_path, b6 = _json(
        _record(upstream["native_b6_v2_collection_manifest"], "B6 collection"),
        "B6 collection",
    )
    checkpoint = _record(
        upstream["native_b6_v2_deployment_checkpoint"], "B6 deploy checkpoint"
    )
    checkpoint_manifest_path, checkpoint_manifest = _json(
        _record(
            upstream["native_b6_v2_deployment_checkpoint_manifest"],
            "B6 checkpoint manifest",
        ),
        "B6 checkpoint manifest",
    )
    oof, oof_manifest = load_oof_row_scores(
        upstream["native_b6_v2_oof_row_scores"],
        upstream["native_b6_v2_oof_row_scores_manifest"],
        checkpoint=checkpoint,
        checkpoint_manifest=checkpoint_manifest_path,
    )
    if (
        len(oof["scene_ids"]) != 6682
        or len(set(oof["scene_ids"].astype(str).tolist())) != 100
        or oof_manifest.get("each_row_model_excludes_scene") is not True
        or oof_manifest.get("scene_group_oof") is not True
    ):
        raise ValueError("B6 all-fold OOF exact100 provenance differs")
    checkpoint_oof = checkpoint_manifest.get("all_fold_oof_row_scores") or {}
    if (
        checkpoint_oof.get("sha256")
        != upstream["native_b6_v2_oof_row_scores"]["sha256"]
        or checkpoint_oof.get("manifest_sha256")
        != upstream["native_b6_v2_oof_row_scores_manifest"]["sha256"]
        or checkpoint_oof.get("checkpoint_manifest_binds_sidecar") is not True
        or checkpoint_oof.get("sidecar_manifest_binds_checkpoint") is not True
    ):
        raise ValueError("B6 checkpoint/OOF sidecar bidirectional binding differs")

    final_rows = final.get("per_scene") or {}
    b6_rows = {str(row.get("scene_id")): row for row in b6.get("scenes", ())}
    if set(final_rows) != set(scenes) or set(b6_rows) != set(scenes):
        raise ValueError("final-base/B6 collection scene sets differ")
    overlay_root = Path(str(cfg["overlay_stage"]["output_root"])).resolve()
    anchor_root = Path(str(cfg["overlay_stage"]["final_anchor_root"])).resolve()
    diagnostic_root = Path(
        str(cfg["overlay_stage"]["native_b6_v2_diagnostics_root"])
    ).resolve()
    completion_root = Path(
        str(cfg["overlay_stage"]["native_b6_v2_completion_root"])
    ).resolve()
    rows: list[dict[str, Any]] = []
    totals = {"anchors": 0, "candidates": 0, "near_candidates": 0,
              "represented_anchors": 0}
    for scene in scenes:
        proposal_row = proposal_rows[scene]
        overlay_path = _sealed(
            overlay_root / f"{scene}_ca1m_tr3d_overlay_v4.npz",
            f"overlay {scene}",
        )
        loaded = load_overlay_cache(
            overlay_path,
            expected_scene=scene,
            expected_proposal_sha256=str(proposal_row["sha256"]),
        )
        summary = loaded["summary"]
        anchor = _sealed(anchor_root / f"{scene}_boxes.pkl", f"anchor {scene}")
        diagnostic = _sealed(
            diagnostic_root / f"{scene}_ca1m_native_b6.npz", f"B6 diagnostic {scene}"
        )
        completion_path, completion = _json(
            completion_root / f"{scene}.json", f"B6 completion {scene}"
        )
        completion_artifacts = completion.get("artifacts") or {}
        if (
            summary.candidate_count != proposal_row.get("candidate_count")
            or summary.proposal_cache_sha256 != proposal_row.get("sha256")
            or summary.final_anchor_sha256 != sha256_file(anchor)
            or summary.final_anchor_sha256
            != (final_rows[scene] or {}).get("active_prediction_sha256")
            or summary.final_anchor_manifest_sha256 != sha256_file(final_path)
            or summary.native_b6_diagnostic_sha256 != sha256_file(diagnostic)
            or summary.native_b6_collection_manifest_sha256 != sha256_file(b6_path)
            or summary.native_b6_checkpoint_sha256 != sha256_file(checkpoint)
            or summary.native_b6_checkpoint_manifest_sha256
            != sha256_file(checkpoint_manifest_path)
            or b6_rows[scene].get("observer_completion_sha256")
            != sha256_file(completion_path)
            or b6_rows[scene].get("final_base_prediction_sha256")
            != sha256_file(anchor)
            or (completion_artifacts.get("native_b6_diagnostic") or {}).get("sha256")
            != sha256_file(diagnostic)
            or (completion_artifacts.get("final_base_anchor") or {}).get("sha256")
            != sha256_file(anchor)
        ):
            raise ValueError(f"{scene}: overlay/upstream cross-binding differs")
        counts = {
            "anchor_count": summary.anchor_count,
            "candidate_count": summary.candidate_count,
            "near_candidate_count": summary.near_candidate_count,
            "represented_anchor_count": summary.represented_anchor_count,
        }
        totals["anchors"] += summary.anchor_count
        totals["candidates"] += summary.candidate_count
        totals["near_candidates"] += summary.near_candidate_count
        totals["represented_anchors"] += summary.represented_anchor_count
        rows.append(
            {
                "scene_id": scene,
                "path": str(overlay_path),
                "sha256": loaded["sha256"],
                "proposal_sha256": summary.proposal_cache_sha256,
                "final_anchor_sha256": summary.final_anchor_sha256,
                "native_b6_diagnostic_sha256": summary.native_b6_diagnostic_sha256,
                "active_anchor_scores_sha256": summary.active_anchor_scores_sha256,
                **counts,
            }
        )
    return {
        "schema": SCHEMA,
        "complete": True,
        "create_only": True,
        "stage": "O",
        "cpu_only": True,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "gpu_started_by_manifest_sealer": False,
        "scene_count": 100,
        "scene_list": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
        "stage_o_config": {"path": str(config_source), "sha256": sha256_file(config_source)},
        "stage_o_authorization": {
            "path": authorization["authorization_path"],
            "sha256": authorization["authorization_sha256"],
        },
        "upstream": upstream,
        "score_roles": {
            "overlay_anchor_scores": "deployment_overlay_only",
            "deployment_scores_allowed_for_stacked_gate_training": False,
            "stacked_gate_training_score_source": "all_fold_oof_row_scores_v2",
            "oof_scores_consumed_by_overlay": False,
        },
        "oof_provenance": {
            "row_count": 6682,
            "scene_count": 100,
            "scene_group_oof": True,
            "each_row_model_excludes_scene": True,
            "score_array_for_stacked_training": "deployment_blend_oof_scores",
            "checkpoint_manifest_binds_sidecar": True,
            "sidecar_manifest_binds_checkpoint": True,
        },
        "totals": {
            **totals,
            "near_candidate_fraction": (
                totals["near_candidates"] / totals["candidates"]
                if totals["candidates"] else 0.0
            ),
        },
        "scenes": rows,
        "code": {
            "sealer": {"path": str(Path(__file__).resolve()),
                       "sha256": sha256_file(Path(__file__).resolve())},
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config", type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v4_o2.json",
    )
    value.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/ca1m_tr3d_terminal_ca_native_train100_v4/"
        "overlay_collection_manifest_v2.json",
    )
    value.add_argument("--seal", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    value = build(args.config)
    if args.seal:
        target = _publish(args.output, value)
        result = {"complete": True, "output": str(target),
                  "sha256": sha256_file(target), "totals": value["totals"]}
    else:
        result = {"ready": True, "scene_count": value["scene_count"],
                  "totals": value["totals"], "gpu_started": False,
                  "ground_truth_access": False}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
