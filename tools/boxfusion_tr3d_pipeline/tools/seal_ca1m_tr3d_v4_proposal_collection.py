#!/usr/bin/env python3
"""Seal an exact100, GT-free manifest after terminal-v4 Stage P completes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import regular_directory, regular_file  # noqa: E402
from boxfusion.ca1m_tr3d_terminal_v4 import load_proposal_cache, sha256_file  # noqa: E402
from tools.preflight_ca1m_tr3d_terminal_train100_v4 import validate_config  # noqa: E402


SCHEMA = "boxfusion.ca1m_tr3d_proposal_collection.v4"


def _create_only(path: Path, value: dict[str, Any]) -> Path:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite proposal manifest: {target}") from error
        target.chmod(0o444)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    source = regular_file(path, name)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return source, value


def build(config_path: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    report = validate_config(config_path)
    source, cfg = _json(config_path, "Stage-P runtime config")
    scene_path = regular_file(Path(cfg["scene_contract"]["path"]), "train100 scene list")
    scenes = tuple(row.strip() for row in scene_path.read_text().splitlines() if row.strip())
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise ValueError("proposal collection requires exact100 scenes")
    root = regular_directory(Path(cfg["proposal_stage"]["output_root"]), "proposal root")
    missing = [
        scene
        for scene in scenes
        if not (root / f"{scene}_ca1m_tr3d_proposals_v4.npz").is_file()
    ]
    status = {
        "schema": SCHEMA,
        "ready": not missing,
        "scene_count": len(scenes),
        "valid_count": len(scenes) - len(missing),
        "missing_count": len(missing),
        "missing_scenes": missing,
        "ground_truth_access": False,
        "gpu_started": False,
    }
    if missing:
        return None, status
    parity_path, parity = _json(
        Path(cfg["distribution_parity"]["receipt"]), "exact100 point parity"
    )
    parity_scenes = parity.get("scenes") or {}
    rows: list[dict[str, Any]] = []
    total_candidates = 0
    total_points = 0
    for scene in scenes:
        path = root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
        loaded = load_proposal_cache(
            path,
            expected_scene=scene,
            expected_binding_sha256=cfg["ca_native_tr3d_binding"]["sha256"],
        )
        summary = loaded["summary"]
        if (
            summary.adapter_mode != "genuine"
            or summary.config_sha256 != cfg["ca_native_tr3d_inference"]["sha256"]
            or summary.checkpoint_sha256
            != cfg["ca_native_tr3d_binding"]["checkpoint_sha256"]
            or summary.source_points_sha256
            != (parity_scenes.get(scene) or {}).get("world_point_array_sha256")
        ):
            raise ValueError(f"{scene}: proposal provenance differs from Stage-P binding")
        code_manifest = json.loads(str(loaded["code_manifest_json"].item()))
        code_files = code_manifest.get("files") or {}
        if (
            code_files.get("ca_point_inference_config")
            != cfg["ca_native_tr3d_inference"]["sha256"]
            or code_files.get("checkpoint_binding")
            != cfg["ca_native_tr3d_binding"]["sha256"]
        ):
            raise ValueError(f"{scene}: proposal code manifest lacks CA inference binding")
        rows.append(
            {
                "scene_id": scene,
                "path": str(path.resolve()),
                "sha256": loaded["sha256"],
                "source_points_sha256": summary.source_points_sha256,
                "frame_lineage_sha256": summary.frame_lineage_sha256,
                "point_count": summary.point_count,
                "candidate_count": summary.candidate_count,
                "code_manifest_sha256": summary.code_manifest_sha256,
            }
        )
        total_candidates += summary.candidate_count
        total_points += summary.point_count
    authorization = regular_file(
        Path(cfg["proposal_stage"]["authorization_receipt"]),
        "Stage-P authorization receipt",
    )
    value = {
        "schema": SCHEMA,
        "complete": True,
        "create_only": True,
        "stage": "P",
        "scene_count": 100,
        "scene_list": {
            "path": str(scene_path),
            "sha256": sha256_file(scene_path),
        },
        "runtime_config": {"path": str(source), "sha256": sha256_file(source)},
        "stage_p_authorization": {
            "path": str(authorization),
            "sha256": sha256_file(authorization),
        },
        "checkpoint_binding": dict(cfg["ca_native_tr3d_binding"]),
        "point_inference_config": dict(cfg["ca_native_tr3d_inference"]),
        "distribution_parity": {
            "path": str(parity_path),
            "sha256": sha256_file(parity_path),
        },
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "anchor_access": False,
        "b6_access": False,
        "gpu_started_by_manifest_sealer": False,
        "totals": {"points": total_points, "candidates": total_candidates},
        "scenes": rows,
        "preflight_proposal_inventory": report["proposal_inventory"],
    }
    return value, status


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v4_p3.json",
    )
    value.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "reports/ca1m_tr3d_terminal_ca_native_train100_v4/"
        "proposal_collection_manifest_v3.json",
    )
    value.add_argument("--seal", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    value, status = build(args.config)
    if value is None:
        print(json.dumps(status, indent=2, sort_keys=True))
        return 2 if args.seal else 0
    if args.seal:
        target = _create_only(args.output, value)
        status.update({"sealed": True, "output": str(target), "sha256": sha256_file(target)})
    print(json.dumps(status, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
