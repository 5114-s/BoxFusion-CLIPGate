#!/usr/bin/env python3
"""Freeze and audit a deterministic, train-only CA-1M scene subset.

This tool intentionally does not download data, inspect validation ground truth, or
start training.  It turns Apple's official train/validation URL lists into an
immutable scene-level selection manifest and a refreshable local-tar readiness
report.  The companion shell wrapper is preflight-only unless ``--download`` is
passed explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit


SCHEMA = "boxfusion.ca1m_native_b6_train_subset.v1"
READINESS_SCHEMA = "boxfusion.ca1m_native_b6_train_readiness.v1"
DEFAULT_NAMESPACE = "boxfusion.ca1m-native-b6.train100.v1"
OFFICIAL_HOST = "ml-site.cdn-apple.com"
TRAIN_PATH = re.compile(r"^/datasets/ca1m/train/ca1m-train-([0-9]{8})\.tar$")
VAL_PATH = re.compile(r"^/datasets/ca1m/val/ca1m-val-([0-9]{8})\.tar$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _require_regular_file(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"missing {description}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(mode) or path.stat().st_size <= 0:
        raise ValueError(f"{description} must be a non-empty regular file: {path}")


def _parse_url_list(path: Path, split: str) -> list[tuple[str, str]]:
    _require_regular_file(path, f"official CA-1M {split} URL list")
    pattern = TRAIN_PATH if split == "train" else VAL_PATH
    rows: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    seen_urls: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        url = raw.strip()
        if not url:
            continue
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != OFFICIAL_HOST
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"{path}:{line_number}: URL is not an exact official HTTPS URL"
            )
        matched = pattern.fullmatch(parsed.path)
        if matched is None:
            raise ValueError(f"{path}:{line_number}: invalid {split} URL path")
        scene_id = matched.group(1)
        if scene_id in seen_ids or url in seen_urls:
            raise ValueError(f"{path}:{line_number}: duplicate {split} scene or URL")
        seen_ids.add(scene_id)
        seen_urls.add(url)
        rows.append((scene_id, url))
    if not rows:
        raise ValueError(f"official CA-1M {split} URL list is empty: {path}")
    return rows


def _selection_key(namespace: str, scene_id: str) -> tuple[str, str]:
    digest = sha256_bytes(f"{namespace}\0{scene_id}".encode("utf-8"))
    return digest, scene_id


def select_train_rows(
    train_rows: Iterable[tuple[str, str]], subset_size: int, namespace: str
) -> list[tuple[str, str, str]]:
    rows = list(train_rows)
    if subset_size <= 0 or subset_size > len(rows):
        raise ValueError(
            f"subset size must be in [1,{len(rows)}], received {subset_size}"
        )
    ranked = sorted(rows, key=lambda row: _selection_key(namespace, row[0]))
    return [
        (scene_id, url, _selection_key(namespace, scene_id)[0])
        for scene_id, url in ranked[:subset_size]
    ]


def _freeze(path: Path, data: bytes) -> None:
    if path.exists() or path.is_symlink():
        _require_regular_file(path, "frozen subset artifact")
        if path.read_bytes() != data:
            raise ValueError(f"refusing to change frozen subset artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        path.chmod(0o444)
    finally:
        temporary.unlink(missing_ok=True)


def _replace(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise ValueError(f"refusing symlink report path: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _tar_header_status(path: Path, scene_id: str) -> tuple[bool, str | None]:
    try:
        with tarfile.open(path, mode="r:") as archive:
            first = archive.next()
    except (tarfile.TarError, OSError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if first is None:
        return False, "empty tar archive"
    first_component = first.name.lstrip("./").split("/", 1)[0]
    if first_component != scene_id:
        return False, f"first tar member belongs to {first_component!r}, not {scene_id!r}"
    return True, None


def build_manifest(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    train_rows = _parse_url_list(args.train_url_list, "train")
    val_rows = _parse_url_list(args.val_url_list, "val")
    train_ids = {row[0] for row in train_rows}
    val_ids = {row[0] for row in val_rows}
    source_overlap = sorted(train_ids & val_ids)
    if source_overlap:
        raise ValueError(
            "official train and validation URL lists overlap: " + ",".join(source_overlap)
        )
    selected = select_train_rows(train_rows, args.subset_size, args.namespace)
    selected_ids = [row[0] for row in selected]
    selected_overlap = sorted(set(selected_ids) & val_ids)
    if selected_overlap:
        raise ValueError(
            "selected train subset overlaps validation IDs: " + ",".join(selected_overlap)
        )
    entries = [
        {
            "rank": rank,
            "scene_id": scene_id,
            "selection_key_sha256": key,
            "url": url,
            "url_sha256": sha256_bytes(url.encode("utf-8")),
            "tar_name": f"ca1m-train-{scene_id}.tar",
        }
        for rank, (scene_id, url, key) in enumerate(selected)
    ]
    selection_bytes = ("\n".join(selected_ids) + "\n").encode("ascii")
    manifest = {
        "schema": SCHEMA,
        "purpose": "CA-1M-native B6 train-only data readiness; no training performed",
        "selection": {
            "algorithm": "ascending_sha256(namespace + NUL + scene_id), scene_id_tiebreak",
            "namespace": args.namespace,
            "subset_size": len(entries),
            "scene_ids_sha256": sha256_bytes(selection_bytes),
        },
        "source": {
            "train_url_list": str(args.train_url_list.resolve()),
            "train_url_list_sha256": sha256_file(args.train_url_list),
            "train_scene_count": len(train_rows),
            "val_url_list": str(args.val_url_list.resolve()),
            "val_url_list_sha256": sha256_file(args.val_url_list),
            "val_scene_count": len(val_rows),
            "train_val_overlap": source_overlap,
        },
        "safety_contract": {
            "train_only": True,
            "validation_scene_overlap_count": len(selected_overlap),
            "validation_ground_truth_access": False,
            "training_started": False,
            "automatic_download": False,
        },
        "entries": entries,
    }
    return manifest, entries


def freeze_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    if output_dir.is_symlink():
        raise ValueError(f"refusing symlink output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_bytes = canonical_json(manifest)
    entries = manifest["entries"]
    ids = "".join(f"{row['scene_id']}\n" for row in entries).encode("ascii")
    urls = "".join(f"{row['url']}\n" for row in entries).encode("utf-8")
    tsv = "rank\tscene_id\tselection_key_sha256\turl_sha256\ttar_name\turl\n"
    tsv += "".join(
        f"{row['rank']}\t{row['scene_id']}\t{row['selection_key_sha256']}\t"
        f"{row['url_sha256']}\t{row['tar_name']}\t{row['url']}\n"
        for row in entries
    )
    artifacts = {
        "subset_manifest.json": manifest_bytes,
        "scene_ids.txt": ids,
        "urls.txt": urls,
        "subset_manifest.tsv": tsv.encode("utf-8"),
    }
    for name, data in artifacts.items():
        _freeze(output_dir / name, data)
        _freeze(
            output_dir / f"{name}.sha256",
            f"{sha256_bytes(data)}  {name}\n".encode("ascii"),
        )


def build_readiness(
    manifest: dict[str, Any], download_root: Path, hash_existing: bool
) -> dict[str, Any]:
    if download_root.is_symlink():
        raise ValueError(f"refusing symlink download root: {download_root}")
    rows: list[dict[str, Any]] = []
    complete = 0
    partial = 0
    absent = 0
    for entry in manifest["entries"]:
        final = download_root / entry["tar_name"]
        part = final.with_name(f"{final.name}.part")
        if final.is_symlink() or part.is_symlink():
            raise ValueError(f"refusing symlink local artifact for {entry['scene_id']}")
        final_exists = final.is_file()
        part_exists = part.is_file()
        tar_ok = False
        tar_error: str | None = None
        actual_sha256: str | None = None
        if final_exists:
            tar_ok, tar_error = _tar_header_status(final, entry["scene_id"])
            if tar_ok and hash_existing:
                actual_sha256 = sha256_file(final)
        ready = bool(final_exists and final.stat().st_size > 0 and tar_ok)
        if ready:
            complete += 1
        elif final_exists or part_exists:
            partial += 1
        else:
            absent += 1
        rows.append(
            {
                "scene_id": entry["scene_id"],
                "url_sha256": entry["url_sha256"],
                "path": str(final),
                "exists": final_exists,
                "bytes": final.stat().st_size if final_exists else 0,
                "partial_path": str(part),
                "partial_bytes": part.stat().st_size if part_exists else 0,
                "tar_header_readable": tar_ok,
                "tar_header_error": tar_error,
                "file_sha256": actual_sha256,
                "ready": ready,
            }
        )
    return {
        "schema": READINESS_SCHEMA,
        "manifest_schema": manifest["schema"],
        "manifest_scene_ids_sha256": manifest["selection"]["scene_ids_sha256"],
        "download_root": str(download_root.resolve(strict=False)),
        "hash_existing": bool(hash_existing),
        "counts": {
            "expected": len(rows),
            "complete": complete,
            "partial": partial,
            "absent": absent,
        },
        "ready": complete == len(rows),
        "entries": rows,
    }


def write_readiness(output_dir: Path, report: dict[str, Any]) -> None:
    _replace(output_dir / "readiness.json", canonical_json(report))
    # A cheap preflight must never erase a checksum inventory produced by an
    # earlier explicit full-file hash audit.  Create/refresh that inventory only
    # when this run actually hashed every locally ready tar.
    if not report["hash_existing"]:
        return
    lines = ["scene_id\tsha256\tbytes\tpath\n"]
    for row in report["entries"]:
        if row["ready"] and row["file_sha256"] is not None:
            lines.append(
                f"{row['scene_id']}\t{row['file_sha256']}\t{row['bytes']}\t{row['path']}\n"
            )
    _replace(output_dir / "downloaded_sha256.tsv", "".join(lines).encode("utf-8"))


def default_paths() -> tuple[Path, Path, Path]:
    pipeline_root = Path(__file__).resolve().parents[1]
    boxfusion_root = pipeline_root.parents[1]
    return (
        boxfusion_root / "data" / "train.txt",
        boxfusion_root / "data" / "val.txt",
        pipeline_root / "manifests" / "ca1m_native_b6_train100_v1",
    )


def parser() -> argparse.ArgumentParser:
    train_default, val_default, output_default = default_paths()
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-url-list", type=Path, default=train_default)
    result.add_argument("--val-url-list", type=Path, default=val_default)
    result.add_argument("--subset-size", type=int, default=100)
    result.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    result.add_argument("--output-dir", type=Path, default=output_default)
    result.add_argument(
        "--download-root",
        type=Path,
        default=Path("/extra/ZhaoX/ca1m_apple_train_tars"),
    )
    result.add_argument(
        "--hash-existing",
        action="store_true",
        help="SHA256 complete local tars (may be slow); URL/manifest hashes are always written",
    )
    result.add_argument(
        "--require-complete",
        action="store_true",
        help="return status 3 unless every selected tar is locally ready",
    )
    return result


def main() -> int:
    args = parser().parse_args()
    manifest, _ = build_manifest(args)
    freeze_manifest(args.output_dir, manifest)
    readiness = build_readiness(manifest, args.download_root, args.hash_existing)
    write_readiness(args.output_dir, readiness)
    summary = {
        "schema": SCHEMA,
        "output_dir": str(args.output_dir.resolve()),
        "subset_size": manifest["selection"]["subset_size"],
        "scene_ids_sha256": manifest["selection"]["scene_ids_sha256"],
        "train_scene_count": manifest["source"]["train_scene_count"],
        "val_scene_count": manifest["source"]["val_scene_count"],
        "validation_overlap": manifest["safety_contract"][
            "validation_scene_overlap_count"
        ],
        "readiness": readiness["counts"],
        "ready": readiness["ready"],
        "tool_download_started": False,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_complete and not readiness["ready"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
