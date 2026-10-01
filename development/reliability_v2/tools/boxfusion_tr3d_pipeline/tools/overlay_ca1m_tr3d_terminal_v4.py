#!/usr/bin/env python3
"""Create the later CPU-only CA-1M terminal-v4 association overlay.

The checked-in config intentionally has pending final-base and native-B6-v2
bindings, so this entry point currently fails before opening any artifact.
Once those bindings are sealed, it combines immutable stage-P candidates with
the final-base anchor and the newly CA-trained B6 v2 scores.  It never invokes
TR3D, an evaluator, or a ground-truth loader.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_score import (  # noqa: E402
    CHECKPOINT_MANIFEST_SCHEMA,
    load_ca1m_native_b6_scorer,
    load_native_observer_diagnostic,
)
from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    regular_directory,
    regular_file,
)
from boxfusion.ca1m_tr3d_terminal import associate_terminal_candidates  # noqa: E402
from boxfusion.ca1m_tr3d_overlay_binding_v4 import (  # noqa: E402
    validate_overlay_authorization,
)
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    OverlaySummary,
    load_overlay_cache,
    load_proposal_cache,
    overlay_payload,
    sha256_array,
    sha256_file,
    write_npz_create_only,
)
from tools.preflight_ca1m_tr3d_terminal_train100_v4 import (  # noqa: E402
    validate_config,
)


FINAL_ANCHOR_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
B6_COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
B6_COMPLETION_SCHEMA = "boxfusion.ca1m_native_b6_final_base_scene_completion.v2"
SCENE = re.compile(r"^[0-9]{8}$")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--collection-config",
        type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v4_p5.json",
    )
    value.add_argument("--scene", action="append", default=[])
    return value


def _immutable_file(path: Path, name: str) -> Path:
    source = regular_file(path, name)
    if source.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {source}")
    return source


def _json(
    path: Path, name: str, *, immutable: bool = False
) -> tuple[Path, dict[str, Any]]:
    source = _immutable_file(path, name) if immutable else regular_file(path, name)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return source, value


def _prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    source = _immutable_file(path, "final-base anchor prediction")
    with source.open("rb") as handle:
        value = pickle.load(handle)  # noqa: S301 - sealed local artifact
        if handle.read(1):
            raise ValueError(f"anchor prediction has trailing bytes: {source}")
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
        raise ValueError(f"anchor prediction has invalid batch shape: {source}")
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(value[0]):
        if not isinstance(row, tuple) or len(row) != 3 or type(row[0]) is not int or row[0] != 0:
            raise ValueError(f"invalid anchor row {index}: {source}")
        corner = np.asarray(row[1])
        score = float(row[2])
        if (
            corner.dtype != np.dtype(np.float32)
            or corner.shape != (8, 3)
            or not np.isfinite(corner).all()
            or not np.isfinite(score)
            or not 0.0 <= score <= 1.0
        ):
            raise ValueError(f"malformed anchor row {index}: {source}")
        corners.append(np.array(corner, dtype=np.float32, order="C", copy=True))
        scores.append(score)
    return (
        np.stack(corners) if corners else np.empty((0, 8, 3), dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
    )


def _scenes(
    path: Path, selected: list[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    source = regular_file(path, "train100 scene list")
    scenes = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if len(scenes) != 100 or len(set(scenes)) != 100 or any(SCENE.fullmatch(x) is None for x in scenes):
        raise ValueError("overlay requires exact CA train100 scene list")
    if selected:
        requested = tuple(str(value) for value in selected)
        if len(requested) != len(set(requested)) or any(value not in scenes for value in requested):
            raise ValueError("overlay --scene values differ from train100")
        wanted = set(requested)
        return scenes, tuple(scene for scene in scenes if scene in wanted)
    return scenes, scenes


def _final_anchor_manifest(path: Path, scenes: tuple[str, ...]) -> tuple[Path, dict[str, Any]]:
    source, value = _json(path, "final-base train100 manifest", immutable=True)
    required = {
        "schema": FINAL_ANCHOR_SCHEMA,
        "ok": True,
        "dataset": "CA1M",
        "split": "train100",
        "scene_count": 100,
        "ground_truth_access": False,
        "evaluation_invoked": False,
        "training_invoked": False,
        "scannet_learned_b6_or_gate_reused": False,
        "clip_appearance_gate_active": True,
        "reliable_view_top_k": 3,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(f"final-base manifest field {name} differs")
    rows = value.get("per_scene")
    if not isinstance(rows, dict) or set(rows) != set(scenes):
        raise ValueError("final-base manifest does not cover exact train100")
    same = value.get("same_run") or {}
    if any(same.get(name) != 100 for name in (
        "byte_identity_scenes", "semantic_identity_scenes", "hard_link_identity_scenes"
    )):
        raise ValueError("final-base same-run identity coverage differs")
    return source, value


def _b6_collection(
    path: Path,
    scenes: tuple[str, ...],
    final_manifest: Path,
    anchor_root: Path,
    scene_list_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    source, value = _json(path, "native-B6 v2 collection manifest", immutable=True)
    required = {
        "schema": B6_COLLECTION_SCHEMA,
        "complete": True,
        "train_only": True,
        "scene_count": 100,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(f"native-B6 v2 collection field {name} differs")
    modules = value.get("source_modules") or {}
    if modules != {
        "selective_boxer_g0": True,
        "clip_appearance_gate": True,
        "reliable_view_top_k": 3,
        "b6_evidence_top_k": 5,
    }:
        raise ValueError("native-B6 v2 source-module contract differs")
    source_final = value.get("source_final_base_collection") or {}
    if (
        Path(str(source_final.get("path", ""))).resolve() != final_manifest
        or source_final.get("sha256") != sha256_file(final_manifest)
        or source_final.get("schema") != FINAL_ANCHOR_SCHEMA
        or Path(str(value.get("source_final_base_root", ""))).resolve() != anchor_root
        or value.get("scene_ids_sha256") != scene_list_sha256
    ):
        raise ValueError("native-B6 v2 collection is not bound to final-base train100")
    rows = value.get("scenes")
    if (
        not isinstance(rows, list)
        or len(rows) != 100
        or len({str(row.get("scene_id")) for row in rows}) != 100
        or {str(row.get("scene_id")) for row in rows} != set(scenes)
    ):
        raise ValueError("native-B6 v2 collection does not cover exact train100")
    return source, value


def _b6_checkpoint_manifest(path: Path, final_manifest: Path) -> tuple[Path, dict[str, Any]]:
    source, value = _json(path, "native-B6 v2 checkpoint manifest", immutable=True)
    if (
        value.get("schema") != CHECKPOINT_MANIFEST_SCHEMA
        or value.get("complete") is not True
        or value.get("train_only") is not True
        or value.get("activation_authorized") is not True
        or value.get("validation_ground_truth_access") is not False
        or value.get("validation_prediction_access") is not False
        or value.get("official_validation_comparable") is not False
    ):
        raise ValueError("native-B6 v2 checkpoint is not train-only activation-authorized")
    dataset = value.get("dataset") or {}
    source_final = dataset.get("source_final_base_collection") or {}
    modules = dataset.get("source_modules") or {}
    if (
        dataset.get("source_collection_schema") != B6_COLLECTION_SCHEMA
        or dataset.get("old_native_b6_diagnostics_reused") is not False
        or dataset.get("old_native_b6_checkpoint_reused") is not False
        or source_final.get("schema") != FINAL_ANCHOR_SCHEMA
        or Path(str(source_final.get("path", ""))).resolve() != final_manifest
        or source_final.get("sha256") != sha256_file(final_manifest)
        or modules.get("clip_appearance_gate") is not True
        or modules.get("reliable_view_top_k") != 3
    ):
        raise ValueError("native-B6 checkpoint lacks final-base-v2 provenance")
    return source, value


def _completion(
    *,
    path: Path,
    scene: str,
    diagnostic: Path,
    anchor: Path,
    final_manifest: Path,
    collection_row: dict[str, Any],
) -> None:
    source, value = _json(
        path, f"native-B6 v2 completion {scene}", immutable=True
    )
    required = {
        "schema": B6_COMPLETION_SCHEMA,
        "phase": "sealed_final_base_offline_native_b6_observer",
        "scene_id": scene,
        "complete": True,
        "train_only": True,
        "evaluation_invoked": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "output_mutation_authorized": False,
        "old_native_b6_diagnostics_reused": False,
        "old_native_b6_checkpoint_reused": False,
        "offline_direct_observer": True,
        "geometry_authority": "sealed_final_base_prediction",
        "cross_run_boxfusion_replay_invoked": False,
        "cross_run_exact_identity_required": False,
        "rgb_pixels_accessed": False,
        "stable_id_policy": "sealed_prediction_row_index",
    }
    for name, expected in required.items():
        if value.get(name) != expected:
            raise ValueError(f"{scene}: native-B6 completion field {name} differs")
    source_final = value.get("source_final_base_manifest") or {}
    if (
        Path(str(source_final.get("path", ""))).resolve() != final_manifest
        or source_final.get("sha256") != sha256_file(final_manifest)
        or source_final.get("schema") != FINAL_ANCHOR_SCHEMA
    ):
        raise ValueError(f"{scene}: B6 completion final-base binding differs")
    artifacts = value.get("artifacts") or {}
    for name, expected_path in (
        ("native_b6_diagnostic", diagnostic), ("final_base_anchor", anchor)
    ):
        record = artifacts.get(name) or {}
        if (
            Path(str(record.get("path", ""))).resolve() != expected_path
            or record.get("sha256") != sha256_file(expected_path)
        ):
            raise ValueError(f"{scene}: completion artifact {name} differs")
    if (
        collection_row.get("scene_id") != scene
        or collection_row.get("observer_completion_sha256") != sha256_file(source)
        or collection_row.get("final_base_prediction_sha256") != sha256_file(anchor)
    ):
        raise ValueError(f"{scene}: B6 collection/completion reverse binding differs")
    if value.get("source_modules") != {
        "selective_boxer_g0": True,
        "clip_appearance_gate": True,
        "reliable_view_top_k": 3,
        "b6_evidence_top_k": 5,
    }:
        raise ValueError(f"{scene}: B6 completion source modules differ")


def run(args: argparse.Namespace) -> dict[str, Any]:
    config_path, cfg = _json(args.collection_config, "terminal-v4 config")
    overlay_cfg = cfg.get("overlay_stage") or {}
    # Fail before opening a proposal, anchor, B6 file, or model checkpoint.
    if overlay_cfg.get("run_authorized") is not True:
        raise PermissionError("terminal-v4 CPU overlay is not authorized")
    stage_o_authorization = validate_overlay_authorization(config_path, cfg)
    proposal_rows = stage_o_authorization.pop("proposal_rows")
    preflight = validate_config(config_path)
    if preflight["overlay_stage_runtime_authorized"] is not True:
        raise PermissionError("terminal-v4 overlay preflight is not authorized")
    contract_scenes, run_scenes = _scenes(
        Path(cfg["scene_contract"]["path"]), list(args.scene)
    )
    proposal_root = regular_directory(Path(overlay_cfg["proposal_cache_root"]), "v4 proposal root")
    anchor_root = regular_directory(Path(overlay_cfg["final_anchor_root"]), "final-base anchor root")
    diagnostic_root = regular_directory(
        Path(overlay_cfg["native_b6_v2_diagnostics_root"]), "native-B6 v2 diagnostic root"
    )
    completion_root = regular_directory(
        Path(overlay_cfg["native_b6_v2_completion_root"]), "native-B6 v2 completion root"
    )
    final_manifest, final_value = _final_anchor_manifest(
        Path(overlay_cfg["final_anchor_manifest"]), contract_scenes
    )
    if sha256_file(final_manifest) != overlay_cfg["final_anchor_manifest_sha256"]:
        raise ValueError("final-base manifest SHA256 differs from v4 binding")
    collection_manifest, collection_value = _b6_collection(
        Path(overlay_cfg["native_b6_v2_collection_manifest"]),
        contract_scenes,
        final_manifest,
        anchor_root,
        str(cfg["scene_contract"]["sha256"]),
    )
    collection_rows = {
        str(row["scene_id"]): dict(row) for row in collection_value["scenes"]
    }
    if sha256_file(collection_manifest) != overlay_cfg["native_b6_v2_collection_manifest_sha256"]:
        raise ValueError("native-B6 v2 collection SHA256 differs from v4 binding")
    checkpoint = _immutable_file(
        Path(overlay_cfg["native_b6_v2_checkpoint"]), "native-B6 v2 checkpoint"
    )
    checkpoint_manifest, _ = _b6_checkpoint_manifest(
        Path(overlay_cfg["native_b6_v2_checkpoint_manifest"]), final_manifest
    )
    if (
        sha256_file(checkpoint) != overlay_cfg["native_b6_v2_checkpoint_sha256"]
        or sha256_file(checkpoint_manifest)
        != overlay_cfg["native_b6_v2_checkpoint_manifest_sha256"]
    ):
        raise ValueError("native-B6 v2 checkpoint binding SHA256 differs")
    if (
        checkpoint.name != "ca1m_native_b6_final_base_iou_mlp_v2.npz"
        or checkpoint_manifest.name
        != "ca1m_native_b6_final_base_iou_mlp_v2.manifest.json"
        or "ca1m_native_b6_final_base_train100_v2" not in str(diagnostic_root)
    ):
        raise ValueError("old/ScanNet B6 artifact path is forbidden")
    scorer = load_ca1m_native_b6_scorer(
        checkpoint, checkpoint_manifest, require_activation_authorized=True
    )
    output_root = Path(overlay_cfg["output_root"])
    if output_root.is_symlink():
        raise ValueError("overlay output root must not be a symlink")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    for scene in run_scenes:
        proposal_row = proposal_rows[scene]
        proposal_path = _immutable_file(
            Path(str(proposal_row["path"])), "sealed-manifest v4 proposal cache"
        )
        if (
            proposal_path
            != proposal_root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
            or proposal_row.get("sha256") != sha256_file(proposal_path)
        ):
            raise ValueError(f"{scene}: proposal differs from sealed P manifest")
        proposal = load_proposal_cache(
            proposal_path,
            expected_scene=scene,
            expected_binding_sha256=cfg["ca_native_tr3d_binding"]["sha256"],
        )
        anchor = _immutable_file(
            anchor_root / f"{scene}_boxes.pkl", "final-base anchor"
        )
        final_row = final_value["per_scene"][scene]
        if final_row.get("active_prediction_sha256") != sha256_file(anchor):
            raise ValueError(f"{scene}: final-base manifest/prediction SHA256 differs")
        diagnostic = _immutable_file(
            diagnostic_root / f"{scene}_ca1m_native_b6.npz",
            "native-B6 v2 diagnostic",
        )
        _completion(
            path=completion_root / f"{scene}.json",
            scene=scene,
            diagnostic=diagnostic,
            anchor=anchor,
            final_manifest=final_manifest,
            collection_row=collection_rows[scene],
        )
        corners, detector_scores = _prediction(anchor)
        evidence = load_native_observer_diagnostic(
            diagnostic, scene_id=scene, corners=corners, scores=detector_scores
        )
        deployment_scores = np.asarray(
            scorer.predict(evidence["features"], detector_scores).scores,
            dtype=np.float32,
        )
        association = associate_terminal_candidates(
            anchor_corners=corners,
            anchor_scores=deployment_scores,
            candidate_corners=proposal["candidate_corners_world"],
            candidate_scores=proposal["candidate_scores"],
            near_iou=float(cfg["protocol"]["near_iou"]),
        )
        summary = OverlaySummary(
            scene_id=scene,
            anchor_count=len(corners),
            candidate_count=len(proposal["candidate_scores"]),
            near_candidate_count=int(association.near_mask.sum()),
            represented_anchor_count=len(association.represented_anchor_indices),
            proposal_cache_sha256=proposal["sha256"],
            final_anchor_sha256=sha256_file(anchor),
            final_anchor_manifest_sha256=sha256_file(final_manifest),
            native_b6_diagnostic_sha256=sha256_file(diagnostic),
            native_b6_collection_manifest_sha256=sha256_file(collection_manifest),
            native_b6_checkpoint_sha256=scorer.checkpoint_sha256,
            native_b6_checkpoint_manifest_sha256=scorer.manifest_sha256,
            active_anchor_scores_sha256=sha256_array(deployment_scores),
        )
        target = output_root / f"{scene}_ca1m_tr3d_overlay_v4.npz"
        if target.exists() or target.is_symlink():
            loaded = load_overlay_cache(
                target,
                expected_scene=scene,
                expected_proposal_sha256=sha256_file(proposal_path),
            )
            if loaded["summary"].as_dict() != summary.as_dict():
                raise ValueError(f"{scene}: resumed overlay upstream SHA summary differs")
            expected_arrays = {
                "anchor_corners": corners,
                "active_anchor_scores": deployment_scores,
                "candidate_corners_world": proposal["candidate_corners_world"],
                "candidate_scores": proposal["candidate_scores"],
                "best_anchor_indices": association.best_anchor_indices,
                "best_anchor_iou": association.best_anchor_iou,
                "best_anchor_center_distance_m":
                    association.best_anchor_center_distance_m,
                "near_mask": association.near_mask,
                "represented_anchor_indices": association.represented_anchor_indices,
            }
            for name, expected in expected_arrays.items():
                if not np.array_equal(loaded[name], expected):
                    raise ValueError(f"{scene}: resumed overlay field {name} differs")
            reports[scene] = {
                **summary.as_dict(),
                "anchor_score_role": "deployment_overlay_only",
                "stacked_gate_training_authorized": False,
                "resumed": True,
            }
            continue
        write_npz_create_only(
            target,
            overlay_payload(
                summary=summary,
                anchor_corners=corners,
                anchor_scores=deployment_scores,
                proposal=proposal,
                association=association,
            ),
        )
        reports[scene] = {
            **summary.as_dict(),
            "anchor_score_role": "deployment_overlay_only",
            "stacked_gate_training_authorized": False,
            "resumed": False,
        }
    return {
        "schema": "boxfusion.ca1m_tr3d_terminal_overlay_run.v4",
        "complete": True,
        "stage": "O",
        "cpu_only": True,
        "ground_truth_access": False,
        "stage_o_authorization": stage_o_authorization,
        "anchor_score_source": "deployable_ca1m_native_b6_v2_checkpoint",
        "anchor_score_role": "deployment_overlay_only",
        "oof_usage": "provenance_binding_only",
        "oof_scores_consumed": False,
        "deployment_scores_allowed_for_stacked_gate_training": False,
        "stacked_gate_training_score_source": "all_fold_oof_row_scores_v2",
        "scene_count": len(run_scenes),
        "resumed_count": sum(bool(row["resumed"]) for row in reports.values()),
        "scenes": reports,
    }


def main() -> int:
    print(json.dumps(run(parser().parse_args()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
