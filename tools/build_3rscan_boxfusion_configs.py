#!/usr/bin/env python3
"""Build isolated original/latest BoxFusion configs for prepared 3RScan."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def common(cfg: dict, prepared: Path) -> None:
    cfg.setdefault("experiment", {})["seed"] = 0
    cfg["dataset"] = "scannet"  # reuse the validated ScanNet-format stream adapter
    cfg["data"]["datadir"] = str(prepared / "placeholder" / "frames")
    cfg["data"]["start"] = 0
    cfg["data"]["gap"] = 25
    cfg["cam"].update({"H": 480, "W": 640, "png_depth_scale": 1000.0})
    cfg["vis"].update({"rerun": False, "show_class": False, "show_label": False})
    cfg["eval"] = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    prepared = args.prepared_root.resolve()
    run_root = args.run_root.resolve()

    original = read(ROOT / "config" / "scannet.yaml")
    common(original, prepared)
    original["data"]["output_dir"] = str(run_root / "predictions" / "original_boxfusion")
    original["box_fusion"]["pst_path"] = str(ROOT / "data" / "pst_1024_0.tiff")

    latest = read(ROOT / "config" / "scannet_t05_boxer_strict_causal_nochild_single_gpu_fps.yaml")
    common(latest, prepared)
    latest["data"]["output_dir"] = str(run_root / "predictions" / "latest_native_host")
    latest["lifting"]["proposal_cache"]["mode"] = "disabled"
    latest["lifting"]["boxer"]["diagnostics_dir"] = str(run_root / "diagnostics" / "native_boxer")
    latest["association"]["appearance_gate"]["enabled"] = False
    latest["association"]["pvq_ar"]["enabled"] = False
    latest["online_candidate_map"]["enabled"] = True
    latest["online_candidate_map"]["output_root"] = str(run_root / "predictions" / "latest_full")
    latest["online_candidate_map"]["diagnostics_root"] = str(run_root / "diagnostics" / "latest_full")
    latest["online_candidate_map"]["write_every_keyframe"] = False
    latest["online_candidate_map"]["provider"]["device"] = "cuda:0"
    latest["online_candidate_map"]["provider"]["diagnostics_root"] = str(run_root / "diagnostics" / "provider")
    latest["online_candidate_map"]["state"]["m1p"]["use_children"] = False

    for directory in (
        run_root / "predictions" / "original_boxfusion",
        run_root / "predictions" / "latest_native_host",
        run_root / "predictions" / "latest_full",
        run_root / "logs" / "original_boxfusion",
        run_root / "logs" / "latest_full",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    write(run_root / "configs" / "original_boxfusion.yaml", original)
    write(run_root / "configs" / "latest_full.yaml", latest)
    print(run_root / "configs" / "original_boxfusion.yaml")
    print(run_root / "configs" / "latest_full.yaml")


if __name__ == "__main__":
    main()
