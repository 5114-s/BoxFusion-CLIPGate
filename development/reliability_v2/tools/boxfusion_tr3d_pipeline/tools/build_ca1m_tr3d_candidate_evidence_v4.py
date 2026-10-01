#!/usr/bin/env python3
"""Collect and seal GT-free native evidence for CA-1M terminal-v4 candidates.

The collector consumes only stage-P proposals, stage-O lineage, and processed
train RGB-D calibration.  It has no GT, validation, checkpoint, or prediction
writer argument.  ``--seal`` revalidates the exact train100 set and publishes a
create-only manifest consumed by the terminal-benefit training binding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_native_b6_observer import (  # noqa: E402
    CA1MNativeB6Config,
    CA1MNativeB6Observer,
    FEATURE_NAMES as NATIVE_FEATURE_NAMES,
    SCHEMA as NATIVE_EVIDENCE_SCHEMA,
)
from boxfusion.ca1m_tr3d_terminal_gate_v4 import (  # noqa: E402
    CANDIDATE_EVIDENCE_MANIFEST_SCHEMA,
    write_binding_create_only,
)
from boxfusion.ca1m_tr3d_terminal_v4 import (  # noqa: E402
    OVERLAY_SCHEMA,
    PROPOSAL_SCHEMA,
    load_overlay_cache,
    load_proposal_cache,
    sha256_file,
)


SCENE_RE = re.compile(r"^[0-9]{8}$")
SUFFIX = "_ca1m_tr3d_candidate_evidence_v4.npz"
OVERLAY_COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_terminal_overlay_collection.v2"


def _regular(path: Path, name: str, *, sealed: bool = False) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    if sealed and result.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {result}")
    return result


def _directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_dir() or result.is_symlink():
        raise FileNotFoundError(f"missing directory {name}: {result}")
    return result


def _scenes(path: Path) -> tuple[Path, tuple[str, ...]]:
    source = _regular(path, "train100 scene list", sealed=True)
    rows = tuple(line.strip() for line in source.read_text().splitlines() if line.strip())
    if (
        len(rows) != 100
        or len(set(rows)) != 100
        or any(SCENE_RE.fullmatch(row) is None for row in rows)
    ):
        raise ValueError("candidate-evidence v4 requires exact 100 unique CA train scenes")
    return source, rows


def _numeric_pngs(path: Path) -> dict[int, Path]:
    directory = _directory(path, "depth directory")
    result: dict[int, Path] = {}
    for item in directory.iterdir():
        if item.is_symlink() or not item.is_file() or item.suffix.lower() != ".png":
            continue
        try:
            frame = int(item.stem)
        except ValueError:
            continue
        if frame in result:
            raise ValueError(f"duplicate numeric depth frame {frame}: {directory}")
        result[frame] = item.resolve()
    if not result:
        raise ValueError(f"no numeric depth PNGs in {directory}")
    return result


def _scene_metadata(scene_root: Path) -> tuple[dict[int, Path], np.ndarray, np.ndarray]:
    depth = _numeric_pngs(scene_root / "depth")
    poses = np.asarray(
        np.load(_regular(scene_root / "all_poses.npy", "poses"), allow_pickle=False),
        dtype=np.float64,
    )
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"invalid pose array: {scene_root}")
    per_frame = scene_root / "K_depth_per_frame.npy"
    if per_frame.exists() or per_frame.is_symlink():
        intrinsics = np.asarray(
            np.load(_regular(per_frame, "per-frame intrinsics"), allow_pickle=False),
            dtype=np.float64,
        )
    else:
        intrinsic = np.asarray(
            np.loadtxt(_regular(scene_root / "K_depth.txt", "depth intrinsics")),
            dtype=np.float64,
        ).reshape(3, 3)
        intrinsics = np.broadcast_to(intrinsic, (len(poses), 3, 3)).copy()
    if intrinsics.shape != (len(poses), 3, 3) or not np.isfinite(intrinsics).all():
        raise ValueError(f"intrinsics/pose count mismatch: {scene_root}")
    return depth, poses, intrinsics


def _scalar(archive: Any, name: str) -> Any:
    if name not in archive.files:
        raise ValueError(f"candidate evidence lacks {name}")
    value = np.asarray(archive[name])
    if value.shape != ():
        raise ValueError(f"candidate evidence {name} must be scalar")
    return value.item()


def validate_evidence(
    path: Path,
    *,
    scene: str,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    source = _regular(path, f"candidate evidence {scene}", sealed=True)
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "schema", "complete", "observer_only", "mutation_enabled",
            "applied_count", "ground_truth_access", "clip_access", "scene_id",
            "result_indices", "stable_ids", "corners", "scores", "used_frame_ids",
            "feature_names", "features", "valid_evidence", "summary_json",
        }
        if not required.issubset(set(archive.files)):
            raise ValueError(f"candidate evidence key set is incomplete: {scene}")
        fixed = {
            "schema": NATIVE_EVIDENCE_SCHEMA,
            "complete": True,
            "observer_only": True,
            "mutation_enabled": False,
            "applied_count": 0,
            "ground_truth_access": False,
            "clip_access": False,
            "scene_id": scene,
        }
        for name, expected in fixed.items():
            if _scalar(archive, name) != expected:
                raise ValueError(f"candidate evidence scalar {name} differs: {scene}")
        corners = np.asarray(proposal["candidate_corners_world"])
        scores = np.asarray(proposal["candidate_scores"])
        frames = np.asarray(proposal["used_frame_ids"])
        count = len(corners)
        names = tuple(str(value) for value in np.asarray(archive["feature_names"]).tolist())
        features = np.asarray(archive["features"])
        valid = np.asarray(archive["valid_evidence"])
        if (
            not np.array_equal(archive["result_indices"], np.arange(count, dtype=np.int64))
            or not np.array_equal(archive["stable_ids"], np.arange(count, dtype=np.int64))
            or not np.array_equal(archive["corners"], corners)
            or not np.array_equal(archive["scores"], scores)
            or not np.array_equal(archive["used_frame_ids"], frames)
            or names != NATIVE_FEATURE_NAMES
            or features.dtype != np.dtype(np.float32)
            or features.shape != (count, len(NATIVE_FEATURE_NAMES))
            or valid.dtype != np.dtype(np.bool_)
            or valid.shape != (count,)
            or not np.isfinite(features).all()
            or np.any((features < 0.0) | (features > 1.0))
            or not np.array_equal(features[:, 0], scores)
        ):
            raise ValueError(f"candidate evidence row/feature identity differs: {scene}")
        summary = json.loads(str(_scalar(archive, "summary_json")))
        if (
            summary.get("ground_truth_access") is not False
            or summary.get("mutation_enabled") is not False
            or summary.get("mapping_rows") != count
            or summary.get("prediction_rows") != count
        ):
            raise ValueError(f"candidate evidence summary differs: {scene}")
        valid_count = int(np.count_nonzero(valid))
    return {
        "path": str(source),
        "sha256": sha256_file(source),
        "candidate_rows": count,
        "valid_evidence_rows": valid_count,
    }


def collect_scene(
    *,
    scene: str,
    data_root: Path,
    proposal_root: Path,
    overlay_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    proposal_path = proposal_root / f"{scene}_ca1m_tr3d_proposals_v4.npz"
    overlay_path = overlay_root / f"{scene}_ca1m_tr3d_overlay_v4.npz"
    proposal = load_proposal_cache(proposal_path, expected_scene=scene)
    overlay = load_overlay_cache(
        overlay_path,
        expected_scene=scene,
        expected_proposal_sha256=sha256_file(proposal_path),
    )
    if not np.array_equal(
        overlay["candidate_corners_world"], proposal["candidate_corners_world"]
    ):
        raise ValueError(f"{scene}: P/O candidate geometry differs")
    target = output_root / f"{scene}{SUFFIX}"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing candidate evidence: {target}")
    scene_root = _directory(data_root / scene, f"processed train scene {scene}")
    depth, poses, intrinsics = _scene_metadata(scene_root)
    observer = CA1MNativeB6Observer(
        CA1MNativeB6Config(
            enabled=True,
            diagnostics_root=str(output_root),
            top_k=5,
            pixel_stride=4,
            margin=0.05,
            min_depth=0.10,
            max_depth=8.0,
            near_clip=1e-3,
            max_cached_keyframes=256,
        )
    )
    for frame in np.asarray(proposal["used_frame_ids"], dtype=np.int64).tolist():
        if frame >= len(poses) or frame not in depth:
            raise ValueError(f"{scene}: proposal lineage frame {frame} is unavailable")
        depth_m = np.asarray(Image.open(depth[frame]), dtype=np.float32) / 1000.0
        observer.record_keyframe(
            scene_id=scene,
            frame_id=frame,
            source_frame_id=str(frame),
            depth_meters=depth_m,
            intrinsics=intrinsics[frame],
            camera_to_world=poses[frame],
        )
    # The generic observer writes ``*_ca1m_native_b6.npz``.  Use a private
    # temporary diagnostics root and atomically publish the v4 filename.
    summary = observer.finalize(
        scene_id=scene,
        corners=proposal["candidate_corners_world"],
        scores=proposal["candidate_scores"],
        stable_ids=np.arange(len(proposal["candidate_scores"]), dtype=np.int64),
    )
    produced = Path(summary.diagnostic_path).resolve()
    if produced == target.resolve():
        raise RuntimeError("candidate evidence temporary and target paths alias")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.hardlink_to(produced)
        target.chmod(0o444)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        produced.unlink(missing_ok=True)
    return validate_evidence(target, scene=scene, proposal=proposal)


def seal(
    *,
    scene_list: Path,
    proposal_root: Path,
    overlay_root: Path,
    overlay_collection_manifest: Path,
    evidence_root: Path,
    output: Path,
) -> dict[str, Any]:
    scene_path, scenes = _scenes(scene_list)
    proposals = _directory(proposal_root, "proposal root")
    overlays = _directory(overlay_root, "overlay root")
    evidence = _directory(evidence_root, "candidate evidence root")
    overlay_manifest_path = _regular(
        overlay_collection_manifest, "overlay collection manifest", sealed=True
    )
    try:
        overlay_manifest = json.loads(overlay_manifest_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError("overlay collection manifest is not JSON") from error
    if (
        not isinstance(overlay_manifest, dict)
        or overlay_manifest.get("schema") != OVERLAY_COLLECTION_SCHEMA
        or overlay_manifest.get("complete") is not True
        or overlay_manifest.get("stage") != "O"
        or overlay_manifest.get("cpu_only") is not True
        or overlay_manifest.get("ground_truth_access") is not False
        or overlay_manifest.get("validation_ground_truth_access") is not False
        or overlay_manifest.get("scene_count") != 100
        or (overlay_manifest.get("scene_list") or {}).get("sha256")
        != sha256_file(scene_path)
        or (overlay_manifest.get("score_roles") or {}).get(
            "deployment_scores_allowed_for_stacked_gate_training"
        ) is not False
        or (overlay_manifest.get("score_roles") or {}).get(
            "stacked_gate_training_score_source"
        ) != "all_fold_oof_row_scores_v2"
    ):
        raise ValueError("overlay collection manifest safety contract differs")
    overlay_manifest_rows = {
        str(row.get("scene_id")): row for row in overlay_manifest.get("scenes", ())
        if isinstance(row, dict)
    }
    if set(overlay_manifest_rows) != set(scenes) or len(overlay_manifest_rows) != 100:
        raise ValueError("overlay collection manifest is not exact train100")
    expected = {f"{scene}{SUFFIX}" for scene in scenes}
    actual = {
        item.name for item in evidence.iterdir()
        if item.is_file() and not item.is_symlink()
    }
    symlinks = [item.name for item in evidence.iterdir() if item.is_symlink()]
    if actual != expected or symlinks:
        raise ValueError("candidate evidence v4 root is not exact train100")
    rows: dict[str, Any] = {}
    total_candidates = 0
    total_valid = 0
    lineage: set[tuple[str, str, str, str]] = set()
    for scene in scenes:
        proposal_path = proposals / f"{scene}_ca1m_tr3d_proposals_v4.npz"
        overlay_path = overlays / f"{scene}_ca1m_tr3d_overlay_v4.npz"
        proposal = load_proposal_cache(proposal_path, expected_scene=scene)
        overlay = load_overlay_cache(
            overlay_path,
            expected_scene=scene,
            expected_proposal_sha256=sha256_file(proposal_path),
        )
        sealed_overlay = overlay_manifest_rows[scene]
        if (
            Path(str(sealed_overlay.get("path", ""))).resolve() != overlay_path.resolve()
            or sealed_overlay.get("sha256") != sha256_file(overlay_path)
            or sealed_overlay.get("proposal_sha256") != sha256_file(proposal_path)
            or sealed_overlay.get("candidate_count")
            != int(overlay["summary"].candidate_count)
            or sealed_overlay.get("anchor_count") != int(overlay["summary"].anchor_count)
            or sealed_overlay.get("near_candidate_count")
            != int(overlay["summary"].near_candidate_count)
        ):
            raise ValueError(f"{scene}: overlay differs from sealed O collection")
        report = validate_evidence(
            evidence / f"{scene}{SUFFIX}", scene=scene, proposal=proposal
        )
        summary = overlay["summary"]
        lineage.add((
            summary.final_anchor_manifest_sha256,
            summary.native_b6_collection_manifest_sha256,
            summary.native_b6_checkpoint_sha256,
            summary.native_b6_checkpoint_manifest_sha256,
        ))
        report.update({
            "proposal_sha256": sha256_file(proposal_path),
            "overlay_sha256": sha256_file(overlay_path),
            "final_anchor_manifest_sha256": summary.final_anchor_manifest_sha256,
            "native_b6_collection_manifest_sha256": summary.native_b6_collection_manifest_sha256,
            "native_b6_checkpoint_sha256": summary.native_b6_checkpoint_sha256,
            "native_b6_checkpoint_manifest_sha256": summary.native_b6_checkpoint_manifest_sha256,
        })
        rows[scene] = report
        total_candidates += int(report["candidate_rows"])
        total_valid += int(report["valid_evidence_rows"])
    if len(lineage) != 1:
        raise ValueError("candidate evidence v4 overlays do not share one final-base/B6-v2 lineage")
    payload = {
        "schema": CANDIDATE_EVIDENCE_MANIFEST_SCHEMA,
        "complete": True,
        "train_only": True,
        "scene_count": 100,
        "candidate_rows": total_candidates,
        "valid_evidence_rows": total_valid,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "mutation_enabled": False,
        "source_proposal_schema": PROPOSAL_SCHEMA,
        "source_overlay_schema": OVERLAY_SCHEMA,
        "native_b6_evidence_top_k": 5,
        "old_candidate_evidence_reused": False,
        "scene_list": {"path": str(scene_path), "sha256": sha256_file(scene_path)},
        "source_roots": {
            "proposal": str(proposals), "overlay": str(overlays), "evidence": str(evidence)
        },
        "overlay_collection_manifest": {
            "path": str(overlay_manifest_path),
            "sha256": sha256_file(overlay_manifest_path),
            "schema": OVERLAY_COLLECTION_SCHEMA,
        },
        "proposal_collection_manifest": dict(
            (overlay_manifest.get("upstream") or {}).get("proposal_collection") or {}
        ),
        "overlay_totals": dict(overlay_manifest.get("totals") or {}),
        "shared_final_base_b6_v2_lineage": list(next(iter(lineage))),
        "scenes": rows,
    }
    write_binding_create_only(output, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--collect", action="store_true")
    mode.add_argument("--seal", action="store_true")
    value.add_argument("--scene-list", type=Path, required=True)
    value.add_argument("--scene", action="append", default=[])
    value.add_argument("--data-root", type=Path)
    value.add_argument("--proposal-root", type=Path, required=True)
    value.add_argument("--overlay-root", type=Path, required=True)
    value.add_argument("--overlay-collection-manifest", type=Path)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path)
    return value


def main() -> int:
    args = parser().parse_args()
    _, scenes = _scenes(args.scene_list)
    if args.collect:
        if (
            args.data_root is None
            or args.manifest is not None
            or args.overlay_collection_manifest is not None
        ):
            raise ValueError(
                "--collect requires --data-root and forbids manifest arguments"
            )
        selected = tuple(args.scene) if args.scene else scenes
        if len(set(selected)) != len(selected) or set(selected) - set(scenes):
            raise ValueError("selected candidate-evidence scenes differ from train100")
        output = args.output_root.resolve()
        if output.is_symlink():
            raise ValueError("candidate evidence output root must not be a symlink")
        output.mkdir(parents=True, exist_ok=True)
        reports = {
            scene: collect_scene(
                scene=scene,
                data_root=_directory(args.data_root, "processed train root"),
                proposal_root=_directory(args.proposal_root, "proposal root"),
                overlay_root=_directory(args.overlay_root, "overlay root"),
                output_root=output,
            )
            for scene in selected
        }
        print(json.dumps({"complete": True, "ground_truth_access": False, "scenes": reports}, sort_keys=True))
        return 0
    if (
        args.data_root is not None
        or args.scene
        or args.manifest is None
        or args.overlay_collection_manifest is None
    ):
        raise ValueError(
            "--seal requires --manifest/--overlay-collection-manifest and forbids --data-root/--scene"
        )
    report = seal(
        scene_list=args.scene_list,
        proposal_root=args.proposal_root,
        overlay_root=args.overlay_root,
        overlay_collection_manifest=args.overlay_collection_manifest,
        evidence_root=args.output_root,
        output=args.manifest,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
