#!/usr/bin/env python3
"""Validate the portable, source-only BoxFusion migration snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MIGRATION_MANIFEST.json"
SOURCE_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_residual_track_dev"
)
SOURCE_PARENT_COMMIT = "374029e87ba82420387d5e6ef6fc3ceea934f0fc"

REQUIRED = (
    "demo.py",
    "demo_tr3d_terminal_active.py",
    "boxfusion/online_refinement.py",
    "boxfusion/boxer_lifter.py",
    "boxfusion/tr3d_terminal_active.py",
    "boxfusion/tr3d_c2_maskrgbd_observer.py",
    "boxfusion/tr3d_c3_online_identity.py",
    "boxfusion/tr3d_r4_smov_observer.py",
    "boxfusion/tr3d_r5_spgroup_observer.py",
    "scripts/run_scannet_b6_g0_tr3d_terminal_active.sh",
    "scripts/run_scannet_tr3d_c3_online_identity.sh",
    "scripts/run_tr3d_c3_online_shadow.sh",
    "tools/audit_tr3d_terminal_active.py",
    "tools/materialize_tr3d_c3_online_shadow.py",
    "evaluation/eval_scannet.py",
    "evaluation/data_util/meta_data/scannetv2_val.txt",
    "models/scannet_b6_iou_mlp.npz",
    "external_overlays/third_party/boxer/boxernet/boxernet.py",
    "external_overlays/third_party/boxer/boxernet/alehead_autograd_safe.patch",
)

FORBIDDEN_ROOTS = (
    "artifacts",
    "cache",
    "data",
    "diagnostics",
    "eval_outputs",
    "logs",
    "reports",
    "results",
    "work_dirs",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(ROOT)
        if relative == Path(MANIFEST.name):
            continue
        if ".pytest_cache" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.suffix in {".pyc", ".pyo"} or relative == Path(".env"):
            continue
        files.append(relative)
    return sorted(files, key=lambda item: item.as_posix())


def validate_structure() -> list[str]:
    issues: list[str] = []
    for relative in REQUIRED:
        path = ROOT / relative
        if not path.is_file() or path.is_symlink():
            issues.append(f"missing or non-regular required file: {relative}")
    for name in FORBIDDEN_ROOTS:
        if (ROOT / name).exists():
            issues.append(f"runtime/data directory must not be committed: {name}/")
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            issues.append(f"symlink is not portable: {path.relative_to(ROOT)}")
    return issues


def build_manifest() -> dict[str, object]:
    rows = []
    for relative in tracked_files():
        path = ROOT / relative
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {
        "schema": "boxfusion.source_migration.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_root": str(SOURCE_ROOT),
        "source_parent_commit": SOURCE_PARENT_COMMIT,
        "scope": "source/config/scripts/tests/docs/evaluator; no data or runtime artifacts",
        "files": rows,
    }


def verify_manifest() -> list[str]:
    if not MANIFEST.is_file():
        return [f"missing manifest: {MANIFEST.name}; run with --write-manifest"]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_rows = manifest.get("files")
    if not isinstance(expected_rows, list):
        return ["manifest files field is invalid"]
    expected = {row["path"]: row for row in expected_rows}
    actual_paths = {path.as_posix() for path in tracked_files()}
    issues: list[str] = []
    for missing in sorted(set(expected) - actual_paths):
        issues.append(f"manifest file missing: {missing}")
    for extra in sorted(actual_paths - set(expected)):
        issues.append(f"unmanifested file: {extra}")
    for relative in sorted(actual_paths & set(expected)):
        path = ROOT / relative
        row = expected[relative]
        actual_size = path.stat().st_size
        if actual_size != row.get("bytes"):
            issues.append(f"size mismatch: {relative}")
            continue
        if sha256(path) != row.get("sha256"):
            issues.append(f"SHA256 mismatch: {relative}")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Create or replace MIGRATION_MANIFEST.json after intentional edits.",
    )
    args = parser.parse_args()

    issues = validate_structure()
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1

    if args.write_manifest:
        payload = build_manifest()
        MANIFEST.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {MANIFEST} with {len(payload['files'])} files")
        return 0

    issues = verify_manifest()
    if issues:
        for issue in issues:
            print(f"ERROR: {issue}")
        return 1

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    print(
        "migration snapshot OK: "
        f"files={len(payload['files'])}, root={ROOT}, "
        f"source={payload['source_root']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
