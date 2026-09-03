#!/usr/bin/env python3
"""Verify hashes, layout and syntax of the historical BoxFusion archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "CATALOG.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_digest(records: dict[str, dict[str, object]]) -> str:
    canonical = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_route(entry: dict[str, object], syntax: bool) -> tuple[int, int, list[str]]:
    route = str(entry["route"])
    snapshot = ROOT / str(entry["snapshot"])
    source = snapshot / "source"
    issues: list[str] = []
    manifest_path = snapshot / "MANIFEST.json"
    excluded_path = snapshot / "EXCLUDED.json"
    if not manifest_path.is_file() or not excluded_path.is_file():
        return 0, 0, [f"{route}: missing MANIFEST.json or EXCLUDED.json"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in (
        "route",
        "family",
        "source_root",
        "source_scan_sha256",
        "file_count",
        "total_bytes",
    ):
        if manifest.get(key) != entry.get(key):
            issues.append(f"{route}: catalog/manifest mismatch: {key}")
    expected = manifest.get("files", {})
    if not isinstance(expected, dict):
        return 0, 0, [f"{route}: invalid files map"]

    actual: set[str] = set()
    python_files: list[Path] = []
    shell_files: list[Path] = []
    for current, dirnames, filenames in os.walk(source, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            path = current_path / name
            if path.is_symlink():
                issues.append(f"{route}: archived symlink directory: {path.relative_to(source)}")
                dirnames.remove(name)
        for name in filenames:
            path = current_path / name
            relative = path.relative_to(source).as_posix()
            actual.add(relative)
            if path.is_symlink() or not path.is_file():
                issues.append(f"{route}: archived non-regular file: {relative}")
                continue
            row = expected.get(relative)
            if row is None:
                issues.append(f"{route}: unmanifested file: {relative}")
                continue
            if path.stat().st_size != row.get("bytes"):
                issues.append(f"{route}: size mismatch: {relative}")
                continue
            if sha256_file(path) != row.get("sha256"):
                issues.append(f"{route}: SHA256 mismatch: {relative}")
            archived_mode = oct(stat.S_IMODE(path.stat().st_mode))
            if archived_mode != row.get("mode"):
                issues.append(f"{route}: mode mismatch: {relative}")
            if path.suffix == ".py":
                python_files.append(path)
            if path.suffix in {".sh", ".bash"}:
                shell_files.append(path)

    for relative in sorted(set(expected) - actual):
        issues.append(f"{route}: missing archived file: {relative}")
    if scan_digest(expected) != manifest.get("source_scan_sha256"):
        issues.append(f"{route}: manifest scan digest mismatch")
    if len(expected) != manifest.get("file_count"):
        issues.append(f"{route}: manifest file_count mismatch")
    if sum(int(row["bytes"]) for row in expected.values()) != manifest.get("total_bytes"):
        issues.append(f"{route}: manifest total_bytes mismatch")

    if syntax and not issues:
        for path in python_files:
            try:
                compile(path.read_bytes(), str(path), "exec")
            except SyntaxError as error:
                issues.append(f"{route}: Python syntax error: {path.relative_to(source)}: {error}")
        for path in shell_files:
            result = subprocess.run(
                ["bash", "-n", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode:
                issues.append(
                    f"{route}: shell syntax error: {path.relative_to(source)}: "
                    f"{result.stderr.strip()}"
                )
    return len(python_files), len(shell_files), issues


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--syntax", action="store_true")
    args = parser.parse_args()
    if args.syntax and sys.version_info < (3, 10):
        print(
            "--syntax requires Python >= 3.10 because historical routes use "
            "parenthesized context managers",
            file=sys.stderr,
        )
        return 2
    if not CATALOG.is_file():
        print(f"missing catalog: {CATALOG}", file=sys.stderr)
        return 1
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    vendor_manifest = ROOT / "vendors" / "vendors.json"
    if not vendor_manifest.is_file():
        print(f"missing vendor manifest: {vendor_manifest}", file=sys.stderr)
        return 1
    vendors = json.loads(vendor_manifest.read_text(encoding="utf-8"))
    for vendor in vendors.get("vendors", []):
        vendor_assets = {
            key: vendor.get(key)
            for key in ("local_patch", "local_overlay")
        }
        for state in vendor.get("local_states", []):
            if "local_deletion_manifest" in state:
                vendor_assets[
                    f"{state.get('route')}:local_deletion_manifest"
                ] = state["local_deletion_manifest"]
        for key, relative in vendor_assets.items():
            if relative is None:
                continue
            path = ROOT / relative
            if not path.is_file() or path.is_symlink():
                print(
                    f"missing/unsafe vendor asset: {relative}",
                    file=sys.stderr,
                )
                return 1
    routes = catalog.get("routes", [])
    if len(routes) != catalog.get("expected_route_count"):
        print(
            f"route count mismatch: {len(routes)} != "
            f"{catalog.get('expected_route_count')}",
            file=sys.stderr,
        )
        return 1

    all_issues: list[str] = []
    python_count = 0
    shell_count = 0
    for entry in routes:
        py_count, sh_count, issues = verify_route(entry, args.syntax)
        python_count += py_count
        shell_count += sh_count
        all_issues.extend(issues)
        status = "OK" if not issues else "FAIL"
        print(
            f"{entry['route']}: {status}, files={entry['file_count']}, "
            f"bytes={entry['total_bytes']}"
        )
    if all_issues:
        for issue in all_issues:
            print(f"ERROR: {issue}", file=sys.stderr)
        return 1
    print(
        f"archive OK: routes={len(routes)}, python={python_count}, "
        f"shell={shell_count}, syntax={int(args.syntax)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
