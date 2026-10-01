#!/usr/bin/env python3
"""Resume-safe exact-100 driver for CA-1M-native B6 train scenes.

Default mode is preflight and never builds a scene.  Explicit ``--mode run``
first requires all 100 frozen train tars to be complete.  It then audits and
skips existing canonical scene directories, or builds and audits a missing
scene.  Hidden/partial/quarantine artifacts are recorded but never removed or
treated as completion.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA = "boxfusion.ca1m_native_b6_train100_driver.v1"
COMPLETION_SCHEMA = "boxfusion.ca1m_native_b6_train100_completion.v1"
SCENE_SCHEMA = "boxfusion.ca1m_native_b6_train_scene.v1"
SUBSET_SCHEMA = "boxfusion.ca1m_native_b6_train_subset.v1"
EXPECTED_SCENES = 100
SCENE_RE = re.compile(r"[0-9]{8}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def require_regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing {label}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {path}")


def atomic_replace(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlink report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def freeze(path: Path, payload: bytes) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists() or path.is_symlink():
        require_regular(path, "frozen completion report")
        if path.read_bytes() != payload:
            raise ValueError(f"refusing to change frozen completion report: {path}")
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def acquire_lock(path: Path):
    if path.is_symlink():
        raise ValueError(f"refusing symlink lock path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError(f"another CA-1M train100 driver holds the lock: {path}")
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def load_contract(args: argparse.Namespace) -> tuple[dict[str, Any], list[str]]:
    require_regular(args.subset_manifest, "frozen subset manifest")
    require_regular(args.scene_ids, "frozen scene-ID list")
    require_regular(args.val_url_list, "official validation URL list")
    manifest = json.loads(args.subset_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != SUBSET_SCHEMA:
        raise ValueError("unsupported subset manifest schema")
    safety = manifest.get("safety_contract", {})
    if (
        safety.get("train_only") is not True
        or safety.get("validation_ground_truth_access") is not False
        or int(safety.get("validation_scene_overlap_count", -1)) != 0
    ):
        raise ValueError("frozen subset train/validation safety contract failed")
    if manifest.get("source", {}).get("val_url_list_sha256") != sha256_file(
        args.val_url_list
    ):
        raise ValueError("validation URL list differs from frozen manifest")
    ids = [row.strip() for row in args.scene_ids.read_text().splitlines() if row.strip()]
    if (
        len(ids) != EXPECTED_SCENES
        or len(ids) != len(set(ids))
        or any(SCENE_RE.fullmatch(value) is None for value in ids)
    ):
        raise ValueError("frozen scene list must contain exactly 100 unique 8-digit IDs")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_SCENES:
        raise ValueError("frozen subset manifest must contain exactly 100 entries")
    manifest_ids = [str(row.get("scene_id")) for row in entries]
    if manifest_ids != ids:
        raise ValueError("frozen scene list order differs from subset manifest")
    if len(set(manifest_ids)) != EXPECTED_SCENES:
        raise ValueError("subset manifest scene IDs are not unique")
    return manifest, ids


def tar_header_audit(path: Path, scene_id: str) -> dict[str, Any]:
    require_regular(path, f"train tar {scene_id}")
    size = path.stat().st_size
    if size < 1536 or size % 512:
        raise ValueError(f"{scene_id}: train tar size is not a complete tar block stream")
    try:
        with tarfile.open(path, mode="r:") as archive:
            first = archive.next()
            if first is None or first.name.lstrip("./").split("/", 1)[0] != scene_id:
                raise ValueError("first tar member has wrong scene prefix")
        # A fixed-cost EOF check prevents preflight from scanning hundreds of
        # GB while still rejecting interrupted downloads. The official
        # downloader performs a full `tar -tf` before atomically renaming
        # `.part` to `.tar`; the per-scene builder later validates every member
        # and requires world.gt before producing output.
        with path.open("rb") as handle:
            handle.seek(-1024, os.SEEK_END)
            if handle.read(1024) != bytes(1024):
                raise ValueError("missing canonical two-zero-block tar terminator")
    except (tarfile.TarError, OSError, ValueError) as exc:
        raise ValueError(f"{scene_id}: incomplete/invalid train tar: {exc}") from exc
    return {
        "bytes": size,
        "first_header_scene_prefix_valid": True,
        "two_zero_block_terminator": True,
        "full_member_validation_deferred_to_single_scene_builder": True,
    }


def readiness(args: argparse.Namespace, ids: list[str]) -> dict[str, Any]:
    if args.tar_root.is_symlink():
        raise ValueError(f"refusing symlink tar root: {args.tar_root}")
    rows: list[dict[str, Any]] = []
    complete = absent = invalid = 0
    partial_artifacts = sorted(
        path.name for path in args.tar_root.glob("*.part") if path.is_file()
    ) if args.tar_root.is_dir() else []
    hidden_artifacts = sorted(
        path.name for path in args.tar_root.iterdir() if path.name.startswith(".")
    ) if args.tar_root.is_dir() else []
    for scene_id in ids:
        path = args.tar_root / f"ca1m-train-{scene_id}.tar"
        row: dict[str, Any] = {"scene_id": scene_id, "path": str(path)}
        if not path.exists():
            absent += 1
            row.update({"ready": False, "error": "absent"})
        else:
            try:
                row.update(tar_header_audit(path, scene_id))
                row.update({"ready": True, "error": None})
                complete += 1
            except ValueError as exc:
                invalid += 1
                row.update({"ready": False, "error": str(exc)})
        rows.append(row)
    return {
        "ready": complete == EXPECTED_SCENES and absent == 0 and invalid == 0,
        "counts": {
            "expected": EXPECTED_SCENES,
            "complete": complete,
            "absent": absent,
            "invalid": invalid,
            "partial_artifacts_ignored": len(partial_artifacts),
            "hidden_tar_artifacts_ignored": len(hidden_artifacts),
        },
        "partial_artifacts_ignored": partial_artifacts,
        "hidden_tar_artifacts_ignored": hidden_artifacts,
        "entries": rows,
    }


def parse_json_stdout(result: subprocess.CompletedProcess[str], label: str) -> dict[str, Any]:
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed ({result.returncode})\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} did not emit one JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} JSON is not an object")
    return value


def audit_scene(args: argparse.Namespace, scene_id: str) -> dict[str, Any]:
    command = [
            str(args.python),
            str(args.auditor),
            "--scene-dir",
            str(args.output_root / scene_id),
            "--geometry-check",
            "full",
            "--pixel-check",
            args.pixel_check,
        ]
    orientation_policy = getattr(args, "orientation_policy", None)
    if orientation_policy is not None:
        command.extend(("--orientation-policy", str(orientation_policy)))
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )
    audit = parse_json_stdout(result, f"audit {scene_id}")
    if (
        audit.get("ok") is not True
        or audit.get("scene_id") != scene_id
        or audit.get("train_only") is not True
        or audit.get("validation_scene_overlap") is not False
        or audit.get("validation_ground_truth_access") is not False
    ):
        raise RuntimeError(f"{scene_id}: audit safety contract failed")
    return audit


def build_scene(args: argparse.Namespace, scene_id: str) -> None:
    command = [
            str(args.python),
            str(args.builder),
            "--tar",
            str(args.tar_root / f"ca1m-train-{scene_id}.tar"),
            "--scene-id",
            scene_id,
            "--subset-manifest",
            str(args.subset_manifest),
            "--val-url-list",
            str(args.val_url_list),
            "--output-root",
            str(args.output_root),
            "--mode",
            "build",
        ]
    if scene_id in getattr(args, "orientation_override_scenes", set()):
        command.extend(("--orientation-policy", str(args.orientation_policy)))
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
    )
    manifest = parse_json_stdout(result, f"build {scene_id}")
    if manifest.get("schema") != SCENE_SCHEMA or manifest.get("scene_id") != scene_id:
        raise RuntimeError(f"{scene_id}: builder emitted wrong scene/schema")


def canonical_scene_status(output_root: Path, scene_id: str) -> str:
    scene = output_root / scene_id
    if not scene.exists():
        return "absent"
    if not scene.is_dir() or scene.is_symlink():
        raise ValueError(f"canonical output path is not a regular directory: {scene}")
    manifest_path = scene / "derived_train_gt_manifest.json"
    require_regular(manifest_path, f"canonical scene manifest {scene_id}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCENE_SCHEMA or manifest.get("scene_id") != scene_id:
        raise ValueError(f"canonical scene has wrong schema/ID: {scene}")
    return "present"


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.time()
    orientation_policy = getattr(args, "orientation_policy", None)
    orientation_override_scenes = getattr(args, "orientation_override_scenes", set())
    manifest, ids = load_contract(args)
    ready = readiness(args, ids)
    base: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": args.mode,
        "train_only": True,
        "validation_ground_truth_access": False,
        "automatic_download": False,
        "automatic_batch_start": False,
        "expected_scenes": EXPECTED_SCENES,
        "subset_manifest": str(args.subset_manifest),
        "subset_manifest_sha256": sha256_file(args.subset_manifest),
        "scene_ids_sha256": sha256_file(args.scene_ids),
        "tar_root": str(args.tar_root),
        "output_root": str(args.output_root),
        "orientation_policy": (
            None
            if orientation_policy is None
            else {
                "path": str(orientation_policy),
                "sha256": sha256_file(orientation_policy),
                "override_scenes": sorted(orientation_override_scenes),
            }
        ),
        "readiness": ready,
        "elapsed_s": time.time() - started,
    }
    if args.mode == "preflight":
        base.update({"ok": ready["ready"], "build_started": False})
        return base, 0 if ready["ready"] else 3
    if not ready["ready"]:
        base.update({"ok": False, "build_started": False, "error": "exact100 tars are not ready"})
        return base, 3
    if args.output_root.is_symlink():
        raise ValueError(f"refusing symlink output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    numeric_entries = sorted(
        path.name
        for path in args.output_root.iterdir()
        if SCENE_RE.fullmatch(path.name)
    )
    unexpected_numeric = sorted(set(numeric_entries) - set(ids))
    if unexpected_numeric:
        raise ValueError(
            "output root contains numeric scenes outside frozen100: "
            + ",".join(unexpected_numeric)
        )
    hidden_outputs = sorted(
        path.name for path in args.output_root.iterdir() if path.name.startswith(".")
    )
    rows: list[dict[str, Any]] = []
    built = skipped = 0
    for ordinal, scene_id in enumerate(ids, 1):
        status = canonical_scene_status(args.output_root, scene_id)
        print(
            f"[{ordinal:03d}/{EXPECTED_SCENES}] {scene_id}: "
            + ("full audit then skip" if status == "present" else "build then full audit"),
            file=sys.stderr,
            flush=True,
        )
        if status == "absent":
            build_scene(args, scene_id)
            action = "built_and_audited"
            built += 1
        else:
            action = "existing_full_audited_skip"
            skipped += 1
        audit = audit_scene(args, scene_id)
        rows.append(
            {
                "ordinal": ordinal,
                "scene_id": scene_id,
                "action": action,
                "source_tar_sha256": audit["source_tar_sha256"],
                "derived_train_gt_sha256": audit["derived_train_gt_sha256"],
                "counts": audit["counts"],
            }
        )
    canonical_numeric = sorted(
        path.name
        for path in args.output_root.iterdir()
        if path.is_dir() and not path.is_symlink() and SCENE_RE.fullmatch(path.name)
    )
    if canonical_numeric != sorted(ids):
        raise ValueError("output root canonical numeric directories are not exact frozen100")
    completion_rows = [
        {
            "ordinal": row["ordinal"],
            "scene_id": row["scene_id"],
            "source_tar_sha256": row["source_tar_sha256"],
            "derived_train_gt_sha256": row["derived_train_gt_sha256"],
            "counts": row["counts"],
        }
        for row in rows
    ]
    completion = {
        "schema": COMPLETION_SCHEMA,
        "ok": True,
        "train_only": True,
        "validation_ground_truth_access": False,
        "subset_manifest_sha256": sha256_file(args.subset_manifest),
        "scene_ids_sha256": sha256_file(args.scene_ids),
        "orientation_policy_sha256": (
            None
            if orientation_policy is None
            else sha256_file(orientation_policy)
        ),
        # This frozen report describes dataset identity only. Per-run resume
        # actions and hidden diagnostics belong in latest_run.json so a second
        # all-skip audit reproduces these bytes exactly.
        "counts": {"exact_scenes": len(completion_rows)},
        "scenes": completion_rows,
    }
    args.report_root.mkdir(parents=True, exist_ok=True)
    completion_path = args.report_root / "exact100_completion.json"
    completion_hash = freeze(completion_path, canonical_json(completion))
    base.update(
        {
            "ok": True,
            "build_started": True,
            "counts": {
                "exact_scenes": len(rows),
                "built_this_run": built,
                "existing_full_audited_skip": skipped,
                "hidden_output_artifacts_ignored": len(hidden_outputs),
            },
            "hidden_output_artifacts_ignored": hidden_outputs,
            "completion_report": str(completion_path),
            "completion_report_sha256": completion_hash,
            "scenes": rows,
            "elapsed_s": time.time() - started,
        }
    )
    atomic_replace(args.report_root / "latest_run.json", canonical_json(base))
    return base, 0


def parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[1]
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("preflight", "run"), default="preflight")
    result.add_argument(
        "--subset-manifest",
        type=Path,
        default=root / "manifests/ca1m_native_b6_train100_v1/subset_manifest.json",
    )
    result.add_argument(
        "--scene-ids",
        type=Path,
        default=root / "manifests/ca1m_native_b6_train100_v1/scene_ids.txt",
    )
    result.add_argument("--val-url-list", type=Path, default=Path("/data/ZhaoX/BoxFusion/data/val.txt"))
    result.add_argument("--tar-root", type=Path, default=Path("/extra/ZhaoX/ca1m_apple_train_tars"))
    result.add_argument(
        "--output-root", type=Path, default=Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1")
    )
    result.add_argument(
        "--report-root", type=Path, default=root / "reports/ca1m_native_b6_train100_v1"
    )
    result.add_argument(
        "--lock", type=Path, default=Path("/tmp/boxfusion_ca1m_native_b6_train100_v1.lock")
    )
    result.add_argument("--python", type=Path, default=Path(sys.executable))
    result.add_argument(
        "--builder", type=Path, default=root / "tools/build_ca1m_native_b6_train_scene.py"
    )
    result.add_argument(
        "--auditor", type=Path, default=root / "tools/audit_ca1m_native_b6_train_scene.py"
    )
    result.add_argument("--pixel-check", choices=("sample", "all", "none"), default="sample")
    result.add_argument("--orientation-policy", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    for name in ("subset_manifest", "scene_ids", "val_url_list", "python", "builder", "auditor"):
        value = getattr(args, name).resolve()
        setattr(args, name, value)
        require_regular(value, name.replace("_", " "))
    args.orientation_override_scenes = set()
    if args.orientation_policy is not None:
        args.orientation_policy = args.orientation_policy.resolve()
        require_regular(args.orientation_policy, "orientation policy")
        policy = json.loads(args.orientation_policy.read_text(encoding="utf-8"))
        overrides = policy.get("scene_overrides")
        if not isinstance(overrides, dict):
            raise ValueError("orientation policy scene_overrides must be an object")
        args.orientation_override_scenes = set(overrides)
    args.tar_root = args.tar_root.resolve()
    args.output_root = args.output_root.resolve()
    args.report_root = args.report_root.resolve()
    args.lock = args.lock.resolve()
    lock_handle = acquire_lock(args.lock)
    try:
        report, status = run(args)
        print(json.dumps(report, indent=2, sort_keys=True))
        return status
    finally:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        lock_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
