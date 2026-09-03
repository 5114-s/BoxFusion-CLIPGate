#!/usr/bin/env python3
"""Build or verify an immutable YiDu ablation run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boxfusion.yidu_ablation import (  # noqa: E402
    YIDU_STAGE_TO_PROFILE,
    resolve_yidu_stage,
)


HASH_CHUNK = 1024 * 1024
CODE_SUFFIXES = {".py", ".sh", ".yaml", ".yml", ".json"}
CODE_DIRS = ("boxfusion", "config", "scripts", "tools")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(HASH_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def code_files(root: Path) -> Iterable[Path]:
    for directory_name in CODE_DIRS:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and path.suffix.lower() in CODE_SUFFIXES
                and "__pycache__" not in path.parts
            ):
                yield path


def code_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in code_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def optional_hash(path_value: Optional[str]) -> Optional[str]:
    if not path_value:
        return None
    return sha256_file(Path(path_value).resolve())


def cache_metadata_hash(cache: Path) -> Optional[str]:
    if not cache.is_dir():
        return None
    candidates = sorted(
        path
        for path in cache.rglob("*")
        if path.is_file()
        and (
            path.name in {"metadata.json", "manifest.json"}
            or any(part.startswith("manifest") for part in path.parts)
        )
    )
    if not candidates:
        return None
    digest = hashlib.sha256()
    for path in candidates:
        relative = path.relative_to(cache).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build(args: argparse.Namespace) -> Dict[str, object]:
    stage = resolve_yidu_stage(args.stage)
    if args.profile != YIDU_STAGE_TO_PROFILE[stage]:
        raise ValueError(
            f"profile {args.profile!r} is not canonical for YiDu {stage}"
        )
    teacher_values = (
        args.teacher_cache,
        args.teacher_namespace,
    )
    if stage == "B0":
        # B0 never constructs the detached SAM3/C4 observer.  Recording a
        # default teacher here made an inactive input look like part of the
        # frozen parent and caused otherwise equivalent B0 manifests to
        # disagree with train-only observer runs.
        teacher_cache: Optional[Path] = None
        teacher_namespace: Optional[str] = None
    else:
        if not all(teacher_values):
            raise ValueError(
                f"YiDu {stage} requires teacher cache and namespace"
            )
        teacher_cache = Path(args.teacher_cache).resolve()
        teacher_namespace = str(args.teacher_namespace)
    gate_values = (
        args.gate_checkpoint,
        args.gate_training_archive,
        args.gate_train_scene_list,
        args.gate_forbidden_scene_list,
    )
    if any(gate_values) and not all(gate_values):
        raise ValueError(
            "gate checkpoint, training archive, train scene list, and "
            "forbidden scene list must be provided together"
        )
    output = Path(args.output).resolve()
    root = output.parents[2]
    # Default layout is <root>/logs/yidu_ablation/<tag>/manifest.  Custom log
    # roots may not follow it, so prefer the script's own project location.
    script_root = PROJECT_ROOT
    if not (root / "boxfusion").is_dir():
        root = script_root
    payload: Dict[str, object] = {
        "schema": "boxfusion.yidu.run_manifest.v3",
        "stage": args.stage,
        "profile": args.profile,
        "project_root": os.fspath(root),
        "code_sha256": code_sha256(root),
        "config": os.fspath(Path(args.config).resolve()),
        "config_sha256": sha256_file(Path(args.config).resolve()),
        "scene_list": os.fspath(Path(args.scene_list).resolve()),
        "scene_list_sha256": sha256_file(
            Path(args.scene_list).resolve()
        ),
        "b6_checkpoint": os.fspath(Path(args.b6_checkpoint).resolve()),
        "b6_checkpoint_sha256": sha256_file(
            Path(args.b6_checkpoint).resolve()
        ),
        "yoloe_checkpoint": os.fspath(
            Path(args.yoloe_checkpoint).resolve()
        ),
        "yoloe_checkpoint_sha256": sha256_file(
            Path(args.yoloe_checkpoint).resolve()
        ),
        "teacher_cache": (
            None if teacher_cache is None else os.fspath(teacher_cache)
        ),
        "teacher_namespace": teacher_namespace,
        "teacher_metadata_sha256": (
            None
            if teacher_cache is None
            else cache_metadata_hash(teacher_cache)
        ),
        "cache_missing_policy": (
            None if teacher_cache is None else args.cache_missing_policy
        ),
        "live_root": os.fspath(Path(args.live_root).resolve()),
        "frames_root": os.fspath(Path(args.frames_root).resolve()),
        "prediction_root": os.fspath(
            Path(args.prediction_root).resolve()
        ),
        "log_root": os.fspath(Path(args.log_root).resolve()),
        "diagnostics_root": os.fspath(
            Path(args.diagnostics_root).resolve()
        ),
        "evaluation_root": os.fspath(
            Path(args.evaluation_root).resolve()
        ),
        "minimum_extent": float(args.minimum_extent),
        "post_minimum_extent": args.post_minimum_extent,
        "gate_checkpoint": (
            None
            if not args.gate_checkpoint
            else os.fspath(Path(args.gate_checkpoint).resolve())
        ),
        "gate_checkpoint_sha256": optional_hash(args.gate_checkpoint),
        "gate_training_archive": (
            None
            if not args.gate_training_archive
            else os.fspath(Path(args.gate_training_archive).resolve())
        ),
        "gate_training_archive_sha256": optional_hash(
            args.gate_training_archive
        ),
        "gate_train_scene_list": (
            None
            if not args.gate_train_scene_list
            else os.fspath(Path(args.gate_train_scene_list).resolve())
        ),
        "gate_train_scene_list_sha256": optional_hash(
            args.gate_train_scene_list
        ),
        "gate_forbidden_scene_list": (
            None
            if not args.gate_forbidden_scene_list
            else os.fspath(
                Path(args.gate_forbidden_scene_list).resolve()
            )
        ),
        "gate_forbidden_scene_list_sha256": optional_hash(
            args.gate_forbidden_scene_list
        ),
        "inference_seed": int(args.inference_seed),
        "evaluation_seed": int(args.evaluation_seed),
        "runtime_python": os.fspath(
            Path(args.python_executable).resolve()
        ),
        "runtime_python_version": args.python_version,
        "runtime_torch_version": args.torch_version,
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene-list", required=True)
    parser.add_argument("--b6-checkpoint", required=True)
    parser.add_argument("--yoloe-checkpoint", required=True)
    parser.add_argument("--teacher-cache")
    parser.add_argument("--teacher-namespace")
    parser.add_argument("--cache-missing-policy", required=True)
    parser.add_argument("--live-root", required=True)
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument("--diagnostics-root", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument("--minimum-extent", type=float, required=True)
    parser.add_argument("--post-minimum-extent")
    parser.add_argument("--gate-checkpoint")
    parser.add_argument("--gate-training-archive")
    parser.add_argument("--gate-train-scene-list")
    parser.add_argument("--gate-forbidden-scene-list")
    parser.add_argument("--inference-seed", type=int, required=True)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--verify-existing", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = Path(args.output).resolve()
    payload = build(args)
    if args.verify_existing:
        if not destination.is_file():
            raise SystemExit(f"Missing existing manifest: {destination}")
        existing = json.loads(destination.read_text(encoding="utf-8"))
        if existing != payload:
            raise SystemExit(
                "Refusing resume: YiDu run manifest fingerprint changed"
            )
        print(f"Verified YiDu run manifest: {payload['fingerprint']}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"Refusing to overwrite manifest: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    print(f"Wrote YiDu run manifest: {payload['fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
