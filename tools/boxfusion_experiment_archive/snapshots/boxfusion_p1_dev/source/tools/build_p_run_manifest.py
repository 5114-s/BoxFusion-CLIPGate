#!/usr/bin/env python3
"""Create or verify an immutable P0/P1 run manifest before resume."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
from typing import Mapping, Sequence


SCHEMA = "boxfusion.p_ablation.run_manifest.v1"
_STAGE_PROFILE = {
    "P0": "p0_frozen_b6",
    "P1": "p1_residual_proposal_observer",
}
_SCENE_PATTERN = re.compile(r"^scene[0-9]{4}_[0-9]{2}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml"}
_CODE_DIRECTORIES = ("boxfusion", "config", "scripts", "tools", "evaluation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_ids(path: Path) -> list[str]:
    rows = [
        row.strip()
        for row in path.read_text(encoding="utf-8").splitlines()
        if row.strip()
    ]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError("scene list must be non-empty and unique")
    invalid = [row for row in rows if _SCENE_PATTERN.fullmatch(row) is None]
    if invalid:
        raise ValueError(f"invalid ScanNet scene id: {invalid[0]!r}")
    return rows


def _code_files(root: Path) -> tuple[Path, ...]:
    files: list[Path] = [root / "demo.py"]
    for directory_name in _CODE_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        files.extend(
            path
            for path in directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in _CODE_SUFFIXES
                and "__pycache__" not in path.parts
            )
        )
    unique = {path.resolve(): path for path in files if path.is_file()}
    return tuple(
        unique[key]
        for key in sorted(unique, key=lambda row: row.as_posix())
    )


def _tree_sha256(root: Path, paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _resolved_inside(root: Path, path: Path, role: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{role} must remain inside isolated checkout") from error
    return resolved


def _finite(
    value: float, role: str, *, lower: float, upper: float | None = None
) -> float:
    result = float(value)
    if not math.isfinite(result) or result < lower:
        raise ValueError(f"{role} must be finite and >= {lower}")
    if upper is not None and result > upper:
        raise ValueError(f"{role} must be <= {upper}")
    return result


def _torch_version() -> str:
    try:
        import torch
    except Exception as error:  # pragma: no cover - runner preflight catches it
        raise RuntimeError("P manifest requires the runtime PyTorch") from error
    return str(torch.__version__)


def _p1_provenance(
    checkpoint: Path,
    *,
    expected_b6_sha256: str,
    forbidden_scene_list: Path,
) -> dict:
    import torch

    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
    except TypeError:  # pragma: no cover - old PyTorch compatibility
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("P1 checkpoint must contain a mapping")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("P1 checkpoint lacks train-only provenance")
    scenes = provenance.get("train_scene_ids")
    b6_sha = provenance.get("b6_checkpoint_sha256")
    train_list_sha = provenance.get("train_scene_list_sha256")
    forbidden_list_sha = provenance.get("forbidden_scene_list_sha256")
    valid_scenes = (
        isinstance(scenes, Sequence)
        and not isinstance(scenes, (str, bytes))
        and bool(scenes)
        and len(set(scenes)) == len(scenes)
        and all(
            isinstance(scene, str)
            and _SCENE_PATTERN.fullmatch(scene) is not None
            for scene in scenes
        )
    )
    valid_hashes = all(
        isinstance(value, str)
        and _SHA256_PATTERN.fullmatch(value.lower()) is not None
        for value in (b6_sha, train_list_sha, forbidden_list_sha)
    )
    if (
        not valid_scenes
        or not valid_hashes
        or provenance.get("forbidden_overlap") != []
    ):
        raise ValueError("P1 checkpoint train-only provenance is invalid")
    if b6_sha.lower() != expected_b6_sha256.lower():
        raise ValueError("P1 checkpoint does not match frozen B6 checkpoint")
    actual_forbidden_sha = _sha256(forbidden_scene_list)
    if forbidden_list_sha.lower() != actual_forbidden_sha:
        raise ValueError(
            "P1 checkpoint was not trained with the canonical validation "
            "split forbidden"
        )
    return {
        "train_scene_count": len(scenes),
        "train_scene_list_sha256": train_list_sha.lower(),
        "forbidden_scene_list": str(forbidden_scene_list.resolve()),
        "forbidden_scene_list_sha256": forbidden_list_sha.lower(),
        "b6_checkpoint_sha256": b6_sha.lower(),
    }


def build_manifest(args: argparse.Namespace) -> dict:
    expected_profile = _STAGE_PROFILE[args.stage]
    if args.profile != expected_profile:
        raise ValueError(
            f"{args.stage} requires canonical profile {expected_profile}"
        )
    if args.stage == "P0" and args.p1_checkpoint is not None:
        raise ValueError("P0 must not bind a P1 checkpoint")
    if args.stage == "P1" and args.p1_checkpoint is None:
        raise ValueError("P1 requires a train-only checkpoint")
    if args.stage == "P1" and args.forbidden_scene_list is None:
        raise ValueError("P1 requires a canonical forbidden scene list")
    required = {
        "scene_list": args.scene_list,
        "config": args.config,
        "b6_checkpoint": args.b6_checkpoint,
        "yoloe_checkpoint": args.yoloe_checkpoint,
        "cutr_checkpoint": args.cutr_checkpoint,
        "clip_checkpoint": args.clip_checkpoint,
        "class_features": args.class_features,
        "class_list": args.class_list,
        "pst_texture": args.pst_texture,
    }
    if args.stage == "P1":
        required["p1_checkpoint"] = args.p1_checkpoint
        required["forbidden_scene_list"] = args.forbidden_scene_list
    for role, path in required.items():
        if path is None or not path.is_file():
            raise FileNotFoundError(f"missing {role}: {path}")
    root = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    code_paths = _code_files(root)
    if not code_paths:
        raise ValueError("isolated checkout contains no runnable code")
    output_roots = {
        "prediction_root": _resolved_inside(
            root, args.prediction_root, "prediction_root"
        ),
        "log_root": _resolved_inside(root, args.log_root, "log_root"),
        "diagnostics_root": _resolved_inside(
            root, args.diagnostics_root, "diagnostics_root"
        ),
        "evaluation_root": _resolved_inside(
            root, args.evaluation_root, "evaluation_root"
        ),
    }
    if len(set(output_roots.values())) != len(output_roots):
        raise ValueError("P output roots must be pairwise distinct")
    b6_blend = _finite(
        args.b6_detector_blend,
        "b6_detector_blend",
        lower=0.0,
        upper=1.0,
    )
    minimum_extent = _finite(
        args.minimum_extent, "minimum_extent", lower=0.0
    )
    if int(args.proposal_interval) < 1:
        raise ValueError("proposal_interval must be positive")
    if args.quality_mode != "iou_mlp":
        raise ValueError("P0/P1 freeze B6 in iou_mlp quality mode")
    if args.proposal_provider != "yoloe":
        raise ValueError("P0/P1 require the frozen YOLOE provider")
    if args.candidate_ttl_clock != "provider_call":
        raise ValueError("P0/P1 require provider_call TTL")
    if int(args.candidate_track_ttl) < 0:
        raise ValueError("candidate_track_ttl must be non-negative")
    if int(args.archive_confirmed_tracks) not in (0, 1):
        raise ValueError("archive_confirmed_tracks must be 0 or 1")
    if int(args.inference_seed) < 0 or int(args.evaluation_seed) < 0:
        raise ValueError("P seeds must be non-negative")
    post_minimum_extent = None
    if args.post_minimum_extent not in (None, ""):
        post_minimum_extent = _finite(
            args.post_minimum_extent,
            "post_minimum_extent",
            lower=0.0,
        )
    for role, directory in {
        "live_root": args.live_root,
        "frames_root": args.frames_root,
        "ground_truth_root": args.ground_truth_root,
        "scans_root": args.scans_root,
    }.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"missing {role}: {directory}")
    scenes = _scene_ids(args.scene_list)
    b6_checkpoint_sha256 = _sha256(args.b6_checkpoint)
    p1_training_provenance = (
        _p1_provenance(
            args.p1_checkpoint,
            expected_b6_sha256=b6_checkpoint_sha256,
            forbidden_scene_list=args.forbidden_scene_list,
        )
        if args.stage == "P1"
        else None
    )
    return {
        "schema": SCHEMA,
        "stage": args.stage,
        "profile": args.profile,
        "scene_count": len(scenes),
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": _sha256(args.scene_list),
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "b6_checkpoint": str(args.b6_checkpoint.resolve()),
        "b6_checkpoint_sha256": b6_checkpoint_sha256,
        "yoloe_checkpoint": str(args.yoloe_checkpoint.resolve()),
        "yoloe_checkpoint_sha256": _sha256(args.yoloe_checkpoint),
        "p1_checkpoint": (
            str(args.p1_checkpoint.resolve())
            if args.p1_checkpoint is not None
            else None
        ),
        "p1_checkpoint_sha256": (
            _sha256(args.p1_checkpoint)
            if args.p1_checkpoint is not None
            else None
        ),
        "p1_training_provenance": p1_training_provenance,
        "parameters": {
            "b6_detector_blend": b6_blend,
            "minimum_extent": minimum_extent,
            "post_minimum_extent": post_minimum_extent,
            "proposal_interval": int(args.proposal_interval),
            "quality_mode": args.quality_mode,
            "proposal_provider": args.proposal_provider,
            "candidate_ttl_clock": args.candidate_ttl_clock,
            "candidate_track_ttl": int(args.candidate_track_ttl),
            "archive_confirmed_tracks": bool(args.archive_confirmed_tracks),
            "inference_seed": int(args.inference_seed),
            "evaluation_seed": int(args.evaluation_seed),
        },
        "assets": {
            role: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for role, path in {
                "cutr_checkpoint": args.cutr_checkpoint,
                "clip_checkpoint": args.clip_checkpoint,
                "class_features": args.class_features,
                "class_list": args.class_list,
                "pst_texture": args.pst_texture,
            }.items()
        },
        "live_root": str(args.live_root.resolve()),
        "frames_root": str(args.frames_root.resolve()),
        "ground_truth_root": str(args.ground_truth_root.resolve()),
        "scans_root": str(args.scans_root.resolve()),
        **{key: str(value) for key, value in output_roots.items()},
        "runtime": {
            "python": platform.python_version(),
            "executable": str(Path(args.python_executable).resolve()),
            "torch": _torch_version(),
        },
        "code_tree_file_count": len(code_paths),
        "code_tree_sha256": _tree_sha256(root, code_paths),
        "code_sha256": {
            path.relative_to(root).as_posix(): _sha256(path)
            for path in code_paths
            if path
            in {
                root / "boxfusion" / "residual_proposal.py",
                root / "boxfusion" / "online_refinement.py",
                root / "boxfusion" / "p_ablation.py",
                root / "demo.py",
                root / "scripts" / "run_scannet_online_refinement.sh",
                root / "scripts" / "run_scannet_p_ablation.sh",
                root / "tools" / "build_p_run_manifest.py",
            }
        },
    }


def _has_artifacts(path: Path, patterns: Sequence[str]) -> bool:
    return path.is_dir() and any(
        any(path.glob(pattern)) for pattern in patterns
    )


def write_or_verify(
    payload: dict,
    *,
    manifest_path: Path,
    prediction_root: Path,
    diagnostics_root: Path,
) -> str:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                "existing P run manifest disagrees with current inputs; "
                "choose a new BOXFUSION_P_RUN_TAG"
            )
        return "verified"
    if _has_artifacts(prediction_root, ("scene*_boxes.pkl",)) or _has_artifacts(
        diagnostics_root, ("scene*_tracks.npz",)
    ):
        raise ValueError(
            "P artifacts exist without a manifest; choose a new run tag"
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, manifest_path)
    return "created"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=("P0", "P1"))
    parser.add_argument("--profile", required=True)
    parser.add_argument("--scene-list", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--b6-checkpoint", required=True, type=Path)
    parser.add_argument("--yoloe-checkpoint", required=True, type=Path)
    parser.add_argument("--cutr-checkpoint", required=True, type=Path)
    parser.add_argument("--clip-checkpoint", required=True, type=Path)
    parser.add_argument("--class-features", required=True, type=Path)
    parser.add_argument("--class-list", required=True, type=Path)
    parser.add_argument("--pst-texture", required=True, type=Path)
    parser.add_argument("--p1-checkpoint", type=Path)
    parser.add_argument("--forbidden-scene-list", type=Path)
    parser.add_argument("--b6-detector-blend", required=True, type=float)
    parser.add_argument("--minimum-extent", required=True, type=float)
    parser.add_argument("--post-minimum-extent")
    parser.add_argument("--proposal-interval", required=True, type=int)
    parser.add_argument("--quality-mode", required=True)
    parser.add_argument("--proposal-provider", required=True)
    parser.add_argument("--candidate-ttl-clock", required=True)
    parser.add_argument("--candidate-track-ttl", required=True, type=int)
    parser.add_argument("--archive-confirmed-tracks", required=True, type=int)
    parser.add_argument("--inference-seed", required=True, type=int)
    parser.add_argument("--evaluation-seed", required=True, type=int)
    parser.add_argument("--live-root", required=True, type=Path)
    parser.add_argument("--frames-root", required=True, type=Path)
    parser.add_argument("--ground-truth-root", required=True, type=Path)
    parser.add_argument("--scans-root", required=True, type=Path)
    parser.add_argument("--python-executable", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--diagnostics-root", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = build_manifest(args)
    status = write_or_verify(
        payload,
        manifest_path=args.manifest,
        prediction_root=args.prediction_root,
        diagnostics_root=args.diagnostics_root,
    )
    print(
        json.dumps(
            {
                "status": status,
                "manifest": str(args.manifest.resolve()),
                "stage": args.stage,
                "scene_count": payload["scene_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
