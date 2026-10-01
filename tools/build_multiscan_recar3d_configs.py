#!/usr/bin/env python3
"""Build isolated original and locked ReCaR-3D configs for MultiScan."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
LOCKED_CONFIG = ROOT / "development" / "reliability_v2" / "config" / "scannet_t05_boxer_reliability_v2_official100.yaml"


def read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def common(cfg: dict, prepared: Path) -> None:
    cfg.setdefault("experiment", {})["seed"] = 0
    cfg["dataset"] = "scannet"
    cfg["data"]["datadir"] = str(prepared / "placeholder" / "frames")
    cfg["data"]["start"] = 0
    # The adapter already samples the 60 Hz stream at one-second intervals.
    cfg["data"]["gap"] = 1
    cfg["cam"].update({"H": 480, "W": 640, "png_depth_scale": 1000.0})
    cfg["vis"].update({"rerun": False, "show_class": False, "show_label": False})
    cfg["eval"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    prepared, run_root = args.prepared_root.resolve(), args.run_root.resolve()

    original = read(ROOT / "config" / "scannet.yaml")
    common(original, prepared)
    original["data"]["output_dir"] = str(run_root / "predictions" / "original_boxfusion")
    original["box_fusion"]["pst_path"] = str(ROOT / "data" / "pst_1024_0.tiff")

    locked = read(LOCKED_CONFIG)
    common(locked, prepared)
    locked["data"]["output_dir"] = str(run_root / "predictions" / "recar3d_native")
    locked["lifting"]["boxer"]["diagnostics_dir"] = str(run_root / "diagnostics" / "recar3d_native")
    locked["association"]["pvq_ar"]["diagnostics_dir"] = str(run_root / "diagnostics" / "recar3d_pvq")
    candidate = locked["online_candidate_map"]
    candidate["output_root"] = str(run_root / "predictions" / "recar3d_online_host")
    candidate["diagnostics_root"] = str(run_root / "diagnostics" / "recar3d_online")
    candidate["audit_output_root"] = str(run_root / "components")
    candidate["provider"]["diagnostics_root"] = str(run_root / "diagnostics" / "recar3d_provider")

    for directory in (
        run_root / "predictions" / "original_boxfusion",
        run_root / "predictions" / "recar3d_native",
        run_root / "predictions" / "recar3d_online_host",
        run_root / "components",
        run_root / "logs" / "original_boxfusion",
        run_root / "logs" / "recar3d",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    write(run_root / "configs" / "original_boxfusion.yaml", original)
    write(run_root / "configs" / "recar3d_locked.yaml", locked)
    print(run_root / "configs" / "original_boxfusion.yaml")
    print(run_root / "configs" / "recar3d_locked.yaml")


if __name__ == "__main__":
    main()
