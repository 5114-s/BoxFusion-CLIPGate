#!/usr/bin/env python3
"""Capture every historical BoxFusion route as a source-only snapshot.

The capture is deliberately non-destructive. It never follows symlinks and
never copies datasets, weights, environments, caches, predictions, or logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_PARENT = Path("/data/ZhaoX/OVM3D-Dett")
DEFAULT_ARCHIVE_ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_BYTES = 5 * 1024 * 1024

ROUTE_FAMILIES = {
    "early_repro": (
        "boxfusion_repro",
        "boxfusion_stage2_work",
        "boxfusion_stage2_patch",
        "boxfusion_stage2_patch_score04",
        "boxfusion_stage3_dev",
    ),
    "memory_quality": (
        "boxfusion_b3_dev",
        "boxfusion_b5_ap50_dev",
        "boxfusion_b6_dev",
        "boxfusion_joint_b356_dev",
    ),
    "mask_proposal": (
        "boxfusion_maskgraph_dev",
        "boxfusion_trifusion_dev",
        "boxfusion_yidu_dev",
    ),
    "residual_p": (
        "boxfusion_p1_dev",
        "boxfusion_p1g_dev",
        "boxfusion_p1v2_sparse_dev",
        "boxfusion_p2_dev",
        "boxfusion_p2v2_dev",
        "boxfusion_p2v3_dev",
    ),
    "boxer_sgcdet": (
        "boxfusion_boxer_dev",
        "boxfusion_b6_boxer_uncertainty_dev",
        "boxfusion_b6_boxer_uncertainty_final_dev",
        "boxfusion_b6_selective_boxer_dev",
        "boxfusion_b6_sgcdet_local_refiner_dev",
        "boxfusion_b6_selective_boxer_sgcdet_dev",
    ),
    "tr3d": (
        "boxfusion_tr3d_dev",
        "boxfusion_tr3d_epoch12_observer_dev",
        "boxfusion_tr3d_r2_verifier_dev",
        "boxfusion_tr3d_smov_verifier_dev",
        "boxfusion_tr3d_spgroup_dev",
        "boxfusion_tr3d_residual_track_dev",
    ),
}

ROUTES = tuple(
    route
    for family_routes in ROUTE_FAMILIES.values()
    for route in family_routes
)
FAMILY_BY_ROUTE = {
    route: family
    for family, family_routes in ROUTE_FAMILIES.items()
    for route in family_routes
}

SOURCE_DIRS = {
    "boxfusion",
    "config",
    "configs",
    "data_process",
    "docs",
    "evaluation",
    "manifests",
    "scripts",
    "tests",
    "tools",
    "tr3d_plugin",
}
EXCLUDED_NAMES = {
    ".conda",
    ".dist_test",
    ".git",
    ".pytest_cache",
    ".runtime",
    "__pycache__",
    "artifacts",
    "backups",
    "cache",
    "ckpts",
    "data",
    "datasets",
    "diagnostics",
    "dist",
    "eval_outputs",
    "logs",
    "models",
    "official_downloads",
    "reports",
    "results",
    "scannet_sens_rgb",
    "third_party",
    "wheels",
    "work_dirs",
}
TEXT_SUFFIXES = {
    ".bash",
    ".cfg",
    ".conf",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".jsonl",
    ".md",
    ".patch",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
ROOT_SPECIAL_FILES = {
    ".env.example",
    ".gitignore",
    ".gitmodules",
    "Dockerfile",
    "LICENSE",
    "LICENSE.txt",
    "Makefile",
}
EVALUATION_META_BINARY = Path(
    "evaluation/data_util/meta_data/scannet_means.npz"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_text_payload(path: Path) -> bool:
    with path.open("rb") as stream:
        sample = stream.read(8192)
    return b"\x00" not in sample


def evaluation_path_allowed(relative: Path) -> bool:
    parts = relative.parts
    if not parts or parts[0] != "evaluation":
        return False
    if len(parts) == 2:
        return True
    if len(parts) >= 3 and parts[1] == "utils":
        return True
    if len(parts) == 3 and parts[1] == "data_util":
        return True
    if (
        len(parts) >= 4
        and parts[1] == "data_util"
        and parts[2] == "meta_data"
    ):
        return True
    return False


def path_allowed(relative: Path, source: Path) -> bool:
    if not relative.parts:
        return False
    if any(part in EXCLUDED_NAMES for part in relative.parts):
        return False
    if any(part.startswith("scannet_train_detection_data") for part in relative.parts):
        return False
    if any(part.startswith("scannet_eval_output") for part in relative.parts):
        return False

    if len(relative.parts) == 1:
        if relative.name not in ROOT_SPECIAL_FILES and relative.suffix.lower() not in TEXT_SUFFIXES:
            return False
    else:
        if relative.parts[0] not in SOURCE_DIRS:
            return False
        if relative.parts[0] == "evaluation" and not evaluation_path_allowed(relative):
            return False

    if relative == EVALUATION_META_BINARY:
        return source.stat().st_size <= 1024 * 1024
    if relative.suffix.lower() not in TEXT_SUFFIXES and relative.name not in ROOT_SPECIAL_FILES:
        return False
    size = source.stat().st_size
    if size > MAX_SOURCE_BYTES:
        return False
    return is_text_payload(source)


def walk_source_root(root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    candidates: list[Path] = []
    symlinks: list[dict[str, str]] = []

    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            symlinks.append(
                {"path": child.name, "target": os.readlink(child)}
            )
            continue
        if child.is_file():
            relative = Path(child.name)
            if path_allowed(relative, child):
                candidates.append(relative)

    for directory_name in sorted(SOURCE_DIRS):
        start = root / directory_name
        if not start.exists() or start.is_symlink() or not start.is_dir():
            if start.is_symlink():
                symlinks.append(
                    {"path": directory_name, "target": os.readlink(start)}
                )
            continue
        for current, dirnames, filenames in os.walk(start, followlinks=False):
            current_path = Path(current)
            current_relative = current_path.relative_to(root)

            kept_dirs: list[str] = []
            for name in sorted(dirnames):
                child = current_path / name
                relative = child.relative_to(root)
                if child.is_symlink():
                    symlinks.append(
                        {
                            "path": relative.as_posix(),
                            "target": os.readlink(child),
                        }
                    )
                    continue
                if name in EXCLUDED_NAMES:
                    continue
                if directory_name == "evaluation":
                    parts = relative.parts
                    if len(parts) == 2 and parts[1] not in {"data_util", "utils"}:
                        continue
                    if (
                        len(parts) == 3
                        and parts[1] == "data_util"
                        and parts[2] != "meta_data"
                    ):
                        continue
                    if len(parts) >= 3 and parts[1] not in {"data_util", "utils"}:
                        continue
                kept_dirs.append(name)
            dirnames[:] = kept_dirs

            for name in sorted(filenames):
                source = current_path / name
                relative = source.relative_to(root)
                if source.is_symlink():
                    symlinks.append(
                        {
                            "path": relative.as_posix(),
                            "target": os.readlink(source),
                        }
                    )
                    continue
                if source.is_file() and path_allowed(relative, source):
                    candidates.append(relative)

    return sorted(set(candidates), key=lambda item: item.as_posix()), sorted(
        symlinks, key=lambda item: item["path"]
    )


def scan_source(root: Path) -> tuple[dict[str, dict[str, object]], list[dict[str, str]]]:
    relative_paths, symlinks = walk_source_root(root)
    records: dict[str, dict[str, object]] = {}
    for relative in relative_paths:
        source = root / relative
        records[relative.as_posix()] = {
            "bytes": source.stat().st_size,
            "mode": oct(stat.S_IMODE(source.stat().st_mode)),
            "sha256": sha256_file(source),
        }
    return records, symlinks


def scan_digest(records: dict[str, dict[str, object]]) -> str:
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(canonical)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def excluded_top_level(root: Path) -> list[str]:
    names = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.name in EXCLUDED_NAMES:
            names.append(child.name)
        elif child.is_dir() and child.name not in SOURCE_DIRS:
            names.append(child.name)
    return names


def capture_route(
    source_parent: Path,
    archive_root: Path,
    route: str,
) -> dict[str, object]:
    source_root = source_parent / route
    if not source_root.is_dir() or source_root.is_symlink():
        raise RuntimeError(f"missing or unsafe source route: {source_root}")

    snapshot_root = archive_root / "snapshots" / route
    destination_root = snapshot_root / "source"
    if snapshot_root.exists():
        raise RuntimeError(
            f"snapshot already exists; refusing overwrite: {snapshot_root}"
        )
    destination_root.mkdir(parents=True)

    before, symlinks_before = scan_source(source_root)
    for relative, expected in before.items():
        source = source_root / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
        if destination.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"copy size mismatch: {route}/{relative}")
        if sha256_file(destination) != expected["sha256"]:
            raise RuntimeError(f"copy hash mismatch: {route}/{relative}")

    after, symlinks_after = scan_source(source_root)
    if before != after or symlinks_before != symlinks_after:
        raise RuntimeError(
            f"source code changed during capture; retry route: {route}"
        )

    captured_utc = datetime.now(timezone.utc).isoformat()
    total_bytes = sum(int(row["bytes"]) for row in before.values())
    manifest = {
        "schema": "boxfusion.experiment_source_snapshot.v1",
        "route": route,
        "family": FAMILY_BY_ROUTE[route],
        "source_root": str(source_root),
        "captured_utc": captured_utc,
        "source_scan_sha256": scan_digest(before),
        "file_count": len(before),
        "total_bytes": total_bytes,
        "files": before,
    }
    excluded = {
        "schema": "boxfusion.experiment_source_exclusions.v1",
        "route": route,
        "excluded_top_level": excluded_top_level(source_root),
        "recorded_symlinks_not_copied": symlinks_before,
        "policy": {
            "max_source_bytes": MAX_SOURCE_BYTES,
            "never_follow_symlinks": True,
            "excluded_names": sorted(EXCLUDED_NAMES),
            "large_binary_suffixes_excluded": [
                ".ckpt",
                ".npy",
                ".npz (except scannet_means.npz)",
                ".pkl",
                ".pt",
                ".pth",
                ".safetensors",
            ],
        },
    }
    write_json(snapshot_root / "MANIFEST.json", manifest)
    write_json(snapshot_root / "EXCLUDED.json", excluded)
    return {
        "route": route,
        "family": FAMILY_BY_ROUTE[route],
        "source_root": str(source_root),
        "snapshot": str(snapshot_root.relative_to(archive_root)),
        "file_count": len(before),
        "total_bytes": total_bytes,
        "source_scan_sha256": manifest["source_scan_sha256"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-parent", type=Path, default=DEFAULT_SOURCE_PARENT)
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument(
        "--only",
        action="append",
        choices=ROUTES,
        help="Capture only the named route; may be repeated.",
    )
    args = parser.parse_args()
    source_parent = args.source_parent.resolve()
    archive_root = args.archive_root.resolve()
    selected = tuple(args.only) if args.only else ROUTES

    archive_root.mkdir(parents=True, exist_ok=True)
    (archive_root / "snapshots").mkdir(exist_ok=True)
    summaries = []
    for route in selected:
        summary = capture_route(source_parent, archive_root, route)
        summaries.append(summary)
        print(
            f"captured {route}: files={summary['file_count']} "
            f"bytes={summary['total_bytes']}"
        )

    existing_summaries = []
    for route in ROUTES:
        manifest_path = archive_root / "snapshots" / route / "MANIFEST.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_summaries.append(
            {
                "route": route,
                "family": manifest["family"],
                "source_root": manifest["source_root"],
                "snapshot": f"snapshots/{route}",
                "file_count": manifest["file_count"],
                "total_bytes": manifest["total_bytes"],
                "source_scan_sha256": manifest["source_scan_sha256"],
            }
        )
    catalog = {
        "schema": "boxfusion.experiment_archive_catalog.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_parent": str(source_parent),
        "route_count": len(existing_summaries),
        "expected_route_count": len(ROUTES),
        "families": {key: list(value) for key, value in ROUTE_FAMILIES.items()},
        "routes": existing_summaries,
    }
    write_json(archive_root / "CATALOG.json", catalog)
    print(
        f"catalogued {len(existing_summaries)}/{len(ROUTES)} routes at "
        f"{archive_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

