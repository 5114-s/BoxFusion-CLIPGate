#!/usr/bin/env python3
"""Convert 3RScan RGB-D scans to the existing BoxFusion stream contract.

The conversion keeps 3RScan world coordinates and camera-to-world poses. RGB
and aligned depth are resized to 640x360 and letterboxed to 640x480 so image
geometry is preserved while satisfying the current online provider contract.
Ground-truth OBBs are converted to enclosing AABBs for the same class-agnostic
AABB AP protocol used by the current ScanNet experiments.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import zipfile

import cv2
import numpy as np


SCANNET18_NYU40_IDS = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39}
SIGNS = np.asarray(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
    dtype=np.float64,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_info(text: str) -> dict[str, str]:
    result = {}
    for raw in text.splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def matrix4(value: str) -> np.ndarray:
    values = np.fromstring(value, sep=" ", dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"expected 16 matrix values, received {values.size}")
    return values.reshape(4, 4)


def load_label_map(path: Path) -> dict[str, tuple[int, str]]:
    rows = list(csv.reader(path.open(newline="", encoding="utf-8-sig")))
    mapping: dict[str, tuple[int, str]] = {}
    for row in rows:
        if len(row) < 5 or not row[0].strip().isdigit():
            continue
        mapping[row[1].strip().casefold()] = (int(row[2]), row[3].strip())
    if not mapping:
        raise ValueError(f"no category mappings parsed from {path}")
    return mapping


class SequenceSource:
    def __init__(self, scan_dir: Path):
        self.scan_dir = scan_dir
        self.directory = scan_dir / "sequence"
        self.archive_path = scan_dir / "sequence.zip"
        self.archive = None
        if self.directory.is_dir():
            self.names = [p.name for p in self.directory.iterdir() if p.is_file()]
        elif self.archive_path.is_file():
            self.archive = zipfile.ZipFile(self.archive_path)
            self.names = [n for n in self.archive.namelist() if not n.endswith("/")]
        else:
            raise FileNotFoundError(f"missing sequence directory/archive: {scan_dir}")
        self.by_base = {Path(name).name: name for name in self.names}

    def read(self, basename: str) -> bytes:
        name = self.by_base.get(basename)
        if name is None:
            raise FileNotFoundError(f"{self.scan_dir}: missing {basename}")
        if self.archive is not None:
            return self.archive.read(name)
        return (self.directory / name).read_bytes()

    def frame_ids(self) -> list[int]:
        patterns = {
            "color": re.compile(r"^frame-(\d+)\.color\.jpg$"),
            "depth": re.compile(r"^frame-(\d+)\.depth\.pgm$"),
            "pose": re.compile(r"^frame-(\d+)\.pose\.txt$"),
        }
        groups = {}
        for key, pattern in patterns.items():
            groups[key] = {
                int(match.group(1))
                for base in self.by_base
                if (match := pattern.match(base)) is not None
            }
        shared = sorted(groups["color"] & groups["depth"] & groups["pose"])
        if not shared:
            raise ValueError(f"{self.scan_dir}: no complete RGB-D-pose frames")
        return shared

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()


def decode_image(blob: bytes, flags: int) -> np.ndarray:
    value = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), flags)
    if value is None:
        raise ValueError("OpenCV failed to decode an image")
    return value


def aabb_corners(group: dict) -> np.ndarray:
    obb = group["obb"]
    center = np.asarray(obb["centroid"], dtype=np.float64)
    lengths = np.asarray(obb["axesLengths"], dtype=np.float64)
    axes = np.asarray(obb["normalizedAxes"], dtype=np.float64).reshape(3, 3)
    if center.shape != (3,) or lengths.shape != (3,) or np.any(lengths <= 0):
        raise ValueError("invalid 3RScan OBB")
    if not np.isfinite(center).all() or not np.isfinite(lengths).all() or not np.isfinite(axes).all():
        raise ValueError("non-finite 3RScan OBB")
    oriented = center + (SIGNS * (0.5 * lengths)) @ axes
    low, high = oriented.min(axis=0), oriented.max(axis=0)
    return np.asarray(
        [[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])],
        dtype=np.float64,
    )


def select_scan_ids(raw_root: Path, scan_list: Path | None) -> list[str]:
    if scan_list is not None:
        ids = [line.strip() for line in scan_list.read_text().splitlines() if line.strip() and not line.startswith("#")]
    else:
        ids = sorted(path.name for path in raw_root.iterdir() if path.is_dir())
    missing = [scan_id for scan_id in ids if not (raw_root / scan_id).is_dir()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} requested scans; first: {missing[:3]}")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--scan-list", type=Path)
    parser.add_argument("--alias-start", type=int, default=8000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip scans whose alias directory already has a readable metadata.json; "
        "aliases stay bound to the full scan-list order so partial reruns are stable",
    )
    args = parser.parse_args()

    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    mapping = load_label_map(args.mapping.resolve())
    scan_ids = select_scan_ids(raw_root, args.scan_list)
    if args.limit is not None:
        scan_ids = scan_ids[: args.limit]
    manifest_rows = []

    for index, scan_id in enumerate(scan_ids):
        alias = f"scene{args.alias_start + index:04d}_00"
        scan_dir = raw_root / scan_id
        target = output_root / alias
        frames = target / "frames"
        if target.exists() and args.resume:
            metadata_path = target / "metadata.json"
            if metadata_path.is_file():
                try:
                    row = json.loads(metadata_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    row = None
                if isinstance(row, dict) and row.get("alias") == alias and row.get("scan_id") == scan_id:
                    manifest_rows.append(row)
                    print(f"{alias} <- {scan_id}: resumed, frames={row.get('frames')} gt={row.get('gt_scannet18')}", flush=True)
                    continue
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {target}")
        if target.exists():
            shutil.rmtree(target)
        for name in ("color", "depth", "pose", "intrinsic"):
            (frames / name).mkdir(parents=True, exist_ok=True)

        source = SequenceSource(scan_dir)
        try:
            info = parse_info(source.read("_info.txt").decode("utf-8"))
            color_width = int(info["m_colorWidth"])
            color_height = int(info["m_colorHeight"])
            depth_width = int(info["m_depthWidth"])
            depth_height = int(info["m_depthHeight"])
            depth_shift = float(info.get("m_depthShift", "1000"))
            color_k = matrix4(info["m_calibrationColorIntrinsic"])
            scale = min(640.0 / color_width, 480.0 / color_height)
            resized_width = int(round(color_width * scale))
            resized_height = int(round(color_height * scale))
            x0 = (640 - resized_width) // 2
            y0 = (480 - resized_height) // 2
            target_k = np.eye(4, dtype=np.float64)
            target_k[0, 0] = color_k[0, 0] * scale
            target_k[1, 1] = color_k[1, 1] * scale
            target_k[0, 2] = color_k[0, 2] * scale + x0
            target_k[1, 2] = color_k[1, 2] * scale + y0
            np.savetxt(frames / "intrinsic" / "intrinsic_depth.txt", target_k, fmt="%.10g")
            # BoxFusion's non-ScanNet path reads this companion 3x3 file.
            np.savetxt(frames / "K_depth.txt", target_k[:3, :3], fmt="%.10g")

            frame_ids = source.frame_ids()
            for output_id, frame_id in enumerate(frame_ids):
                stem = f"frame-{frame_id:06d}"
                color = decode_image(source.read(stem + ".color.jpg"), cv2.IMREAD_COLOR)
                depth = decode_image(source.read(stem + ".depth.pgm"), cv2.IMREAD_UNCHANGED)
                if color.shape[:2] != (color_height, color_width):
                    raise ValueError(f"{scan_id}/{stem}: unexpected color shape {color.shape}")
                if depth.shape[:2] != (depth_height, depth_width):
                    raise ValueError(f"{scan_id}/{stem}: unexpected depth shape {depth.shape}")
                color_small = cv2.resize(color, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
                depth_small = cv2.resize(depth, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
                color_out = np.zeros((480, 640, 3), dtype=np.uint8)
                depth_out = np.zeros((480, 640), dtype=np.uint16)
                color_out[y0 : y0 + resized_height, x0 : x0 + resized_width] = color_small
                depth_out[y0 : y0 + resized_height, x0 : x0 + resized_width] = depth_small.astype(np.uint16)
                if not cv2.imwrite(str(frames / "color" / f"{output_id}.jpg"), color_out, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError("failed to write color frame")
                if not cv2.imwrite(str(frames / "depth" / f"{output_id}.png"), depth_out):
                    raise RuntimeError("failed to write depth frame")
                pose = np.loadtxt(io.BytesIO(source.read(stem + ".pose.txt"))).reshape(4, 4)
                if not np.isfinite(pose).all():
                    raise ValueError(f"{scan_id}/{stem}: invalid camera pose")
                np.savetxt(frames / "pose" / f"{output_id}.txt", pose, fmt="%.10g")
        finally:
            source.close()

        semseg_path = scan_dir / "semseg.v2.json"
        semseg = json.loads(semseg_path.read_text(encoding="utf-8"))
        selected, ignored, unmapped = [], [], []
        selected_labels = []
        for group in semseg.get("segGroups", []):
            label = str(group.get("label", "")).strip().casefold()
            mapped = mapping.get(label)
            if mapped is None:
                unmapped.append(label)
                continue
            if mapped[0] not in SCANNET18_NYU40_IDS:
                ignored.append(label)
                continue
            selected.append(aabb_corners(group))
            selected_labels.append({"raw": label, "nyu40_id": mapped[0], "nyu40": mapped[1]})
        gt = np.stack(selected) if selected else np.zeros((0, 8, 3), dtype=np.float64)
        np.save(target / "gt_aabb_scannet18.npy", gt)
        row = {
            "alias": alias,
            "scan_id": scan_id,
            "frames": len(frame_ids),
            "raw_color_size": [color_width, color_height],
            "raw_depth_size": [depth_width, depth_height],
            "target_size": [640, 480],
            "letterbox_xy": [x0, y0],
            "depth_shift": depth_shift,
            "gt_scannet18": len(gt),
            "gt_labels": selected_labels,
            "ignored_labels": sorted(set(ignored)),
            "unmapped_labels": sorted(set(unmapped)),
            "semseg_sha256": sha256(semseg_path),
        }
        (target / "metadata.json").write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        manifest_rows.append(row)
        print(f"{alias} <- {scan_id}: frames={len(frame_ids)} gt={len(gt)}", flush=True)

    manifest = {
        "protocol": "3RScan ScanNet18-mapped class-agnostic AABB AP",
        "coordinate_frame": "native 3RScan world frame",
        "image_conversion": "aspect-preserving 640x360 resize plus 60px vertical letterbox to 640x480",
        "depth_unit": "millimetres on disk; current ScannetDataset divides by 1000",
        "mapping": str(args.mapping.resolve()),
        "mapping_sha256": sha256(args.mapping.resolve()),
        "scans": manifest_rows,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_root / "scenes.txt").write_text("".join(row["alias"] + "\n" for row in manifest_rows))
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
