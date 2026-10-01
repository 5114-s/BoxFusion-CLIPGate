#!/usr/bin/env python3
"""Download only frozen CA-1M metadata files without listing the HF repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError


FILES = (
    "K_depth.txt",
    "K_rgb.txt",
    "all_poses.npy",
    "T_gravity.npy",
    "after_filter_boxes.npy",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Kevin1804/BoxFusion")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    scenes = tuple(x.strip() for x in args.scene_list.read_text().splitlines() if x.strip())
    if len(scenes) != 40 or len(set(scenes)) != 40:
        raise ValueError("frozen recovery list must contain exactly 40 unique scenes")
    args.output_root.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    downloaded: list[str] = []
    for scene in scenes:
        if not (scene.isdigit() and len(scene) == 8):
            raise ValueError(f"invalid scene ID: {scene}")
        print(f"Downloading frozen metadata for {scene}", flush=True)
        for name in FILES:
            filename = f"{scene}/{name}"
            try:
                path = Path(
                    hf_hub_download(
                        repo_id=args.repo,
                        filename=filename,
                        repo_type="dataset",
                        revision=args.revision,
                        local_dir=args.output_root,
                    )
                )
            except EntryNotFoundError:
                print(f"Frozen repository is missing {filename}", flush=True)
                missing.append(filename)
                continue
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"missing downloaded metadata: {path}")
            downloaded.append(filename)
    report = {
        "schema": "boxfusion.ca1m_frozen_metadata_download.v1",
        "repo": args.repo,
        "revision": args.revision,
        "scenes": len(scenes),
        "downloaded": downloaded,
        "missing": missing,
        "complete": not missing,
    }
    report_path = args.output_root / "metadata_download_report.json"
    temporary = report_path.with_name(f".{report_path.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(report_path)
    if missing:
        print(f"Frozen metadata inventory has {len(missing)} missing files: {report_path}")
        return 2
    print(f"Frozen metadata complete: {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
