#!/usr/bin/env python3
"""Regenerate native candidate figure assets with a blue-only box palette.

The clean RGB frame is used as the immutable background.  For each original
overlay pixel, the script estimates its anti-aliasing opacity against the
known orange/green/blue stroke colors, then redraws that pixel with the native
blue.  Scene pixels are copied from the clean frame, so furniture colors are
never recolored.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "figure_assets/pipeline_candidate_visuals_scene0432_01"
BASE = ROOT / "figure_assets/rgbd_input_scannet_scene0432_01/frame_0000_rgb.png"
SOURCES = {
    "2d": ASSET_DIR / "01_single_frame_candidates.png",
    "3d": ASSET_DIR / "03_boxer_3d_candidates_on_rgb.png",
}
OUTPUTS = {
    "2d": ASSET_DIR / "01_single_frame_candidates_native_blue.png",
    "3d": ASSET_DIR / "03_boxer_3d_candidates_on_rgb_native_blue.png",
}

# Exact flat colors used by the original data-driven renderer.
ORIGINAL_STROKES = np.asarray([
    [246.0, 130.0, 59.0],
    [94.0, 197.0, 34.0],
    [11.0, 158.0, 245.0],
])
NATIVE_BLUE = np.asarray([21.0, 101.0, 255.0])  # #1565FF


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redraw(source_path: Path, output_path: Path) -> dict:
    base = np.asarray(Image.open(BASE).convert("RGB"), dtype=np.float64)
    source = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float64)
    if source.shape != base.shape:
        raise ValueError(f"shape mismatch: {source.shape} != {base.shape}")

    delta = source - base
    changed = np.max(np.abs(delta), axis=2) > 10.0
    best_error = np.full(base.shape[:2], np.inf, dtype=np.float64)
    best_alpha = np.zeros(base.shape[:2], dtype=np.float64)

    for stroke in ORIGINAL_STROKES:
        direction = stroke[None, None, :] - base
        denom = np.sum(direction * direction, axis=2)
        alpha = np.divide(
            np.sum(delta * direction, axis=2), denom,
            out=np.zeros_like(denom), where=denom > 1e-8,
        )
        alpha = np.clip(alpha, 0.0, 1.0)
        reconstructed = base + alpha[..., None] * direction
        error = np.linalg.norm(source - reconstructed, axis=2)
        select = error < best_error
        best_error[select] = error[select]
        best_alpha[select] = alpha[select]

    # The known-stroke fit prevents ordinary RGB texture from entering the
    # overlay mask.  The small tolerance retains antialiased boundary pixels.
    overlay = changed & (best_error < 12.0) & (best_alpha > 0.035)
    result = base.copy()
    alpha = best_alpha[..., None]
    blue_overlay = base * (1.0 - alpha) + NATIVE_BLUE[None, None, :] * alpha
    result[overlay] = blue_overlay[overlay]
    Image.fromarray(np.rint(result).clip(0, 255).astype(np.uint8)).save(output_path)
    return {
        "source": str(source_path),
        "source_sha256": sha256(source_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "overlay_pixels": int(np.count_nonzero(overlay)),
        "unmodified_scene_pixels": int(overlay.size - np.count_nonzero(overlay)),
    }


def main() -> None:
    records = {key: redraw(SOURCES[key], OUTPUTS[key]) for key in SOURCES}

    # Smaller versions fit the compact method panel without downstream resampling.
    for key, output in OUTPUTS.items():
        with Image.open(output) as image:
            compact = image.resize((864, 648), Image.Resampling.LANCZOS)
            compact_path = output.with_name(output.stem + "_compact.png")
            compact.save(compact_path)
        records[key]["compact_output"] = str(compact_path)
        records[key]["compact_output_sha256"] = sha256(compact_path)

    manifest = {
        "schema": "boxfusion.figure.native_candidate_blue.v1",
        "scene_id": "scene0432_01",
        "background": str(BASE),
        "background_sha256": sha256(BASE),
        "native_blue_rgb": NATIVE_BLUE.astype(int).tolist(),
        "semantic": "all strokes are native-branch candidates",
        "records": records,
    }
    manifest_path = ASSET_DIR / "native_candidate_blue_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for record in records.values():
        print(record["output"])
        print(record["compact_output"])
    print(manifest_path)


if __name__ == "__main__":
    main()
