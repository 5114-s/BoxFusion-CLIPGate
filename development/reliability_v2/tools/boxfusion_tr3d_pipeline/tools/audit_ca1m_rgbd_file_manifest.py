#!/usr/bin/env python3
"""Create a strict ordered SHA256 manifest for CA-1M RGB-D frame inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_ids(path: Path) -> list[str]:
    rows = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not rows or len(rows) != len(set(rows)) or any(not row.isdigit() for row in rows):
        raise ValueError("scene list must contain unique numeric CA-1M IDs")
    return rows


def numbered_pngs(root: Path) -> list[Path]:
    rows = list(root.glob("*.png"))
    if any(not path.stem.isdigit() or path.is_symlink() for path in rows):
        raise ValueError(f"non-numeric or symlinked PNG input in {root}")
    return sorted(rows, key=lambda path: int(path.stem))


def write_json_exclusive(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = scene_ids(args.scene_list)
    collection = hashlib.sha256()
    scene_reports: dict[str, Any] = {}
    total_files = 0
    total_bytes = 0
    for scene in scenes:
        root = args.data_root / scene
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"missing regular CA-1M scene root: {root}")
        rgb = numbered_pngs(root / "rgb")
        depth = numbered_pngs(root / "depth")
        if not rgb or [path.stem for path in rgb] != [path.stem for path in depth]:
            raise ValueError(f"{scene}: RGB/depth frame IDs disagree")
        files = []
        for modality, paths in (("rgb", rgb), ("depth", depth)):
            for path in paths:
                relative = f"{scene}/{modality}/{path.name}"
                size = path.stat().st_size
                digest = sha256(path)
                row = f"{relative}\t{size}\t{digest}\n".encode()
                collection.update(row)
                files.append({"path": relative, "bytes": size, "sha256": digest})
                total_files += 1
                total_bytes += size
        scene_reports[scene] = {"frames": len(rgb), "files": files}
    report = {
        "schema": "boxfusion.ca1m_rgbd_file_manifest.v1",
        "ok": True,
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256(args.scene_list),
        "scenes": len(scenes),
        "files": total_files,
        "bytes": total_bytes,
        "collection_sha256": collection.hexdigest(),
        "per_scene": scene_reports,
        "tool_sha256": sha256(Path(__file__)),
    }
    write_json_exclusive(args.output, report)
    print(json.dumps({key: report[key] for key in ("scenes", "files", "bytes", "collection_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
