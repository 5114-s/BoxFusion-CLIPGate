#!/usr/bin/env python3
"""Materialize one create-only CA-1M train collection staging config."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml


NAMESPACE = "ca1m-native-b6-train100-score04-gap20-cutr-v1"


def create(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--phase", choices=("record", "observer"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--native-diagnostics-root", type=Path)
    parser.add_argument("--boxer-diagnostics-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.template.read_text())
    if (
        cfg.get("dataset") != "CA1M"
        or float(cfg["detection"]["score_thresh"]) != 0.4
        or int(cfg["data"]["gap"]) != 20
        or cfg.get("online_refinement") != {"enabled": False}
        or cfg.get("eval") is not True
        or cfg["lifting"]["proposal_cache"]["namespace"] != NAMESPACE
    ):
        raise ValueError("train collection template violates frozen common contract")
    cfg["data"]["datadir"] = str(args.data_root / "_placeholder" / "00000000")
    cfg["data"]["output_dir"] = str(args.output_root)
    cache = cfg["lifting"]["proposal_cache"]
    cache["root"] = str(args.cache_root)
    if args.phase == "record":
        if (
            cfg["lifting"].get("backend") != "cutr"
            or cache.get("mode") != "record"
            or cfg.get("ca1m_native_b6_observer", {}).get("enabled") is not False
        ):
            raise ValueError("record template violates CuTR-only contract")
        if any(value is not None for value in (
            args.baseline_root, args.native_diagnostics_root, args.boxer_diagnostics_root
        )):
            raise ValueError("record phase received observer-only paths")
    else:
        observer = cfg.get("ca1m_native_b6_observer", {})
        gate = cfg.get("lifting", {}).get("boxer", {}).get("selective_gate", {})
        if (
            cfg["lifting"].get("backend") != "boxer"
            or cache.get("mode") != "replay"
            or args.baseline_root is None
            or args.native_diagnostics_root is None
            or args.boxer_diagnostics_root is None
            or observer.get("enabled") is not True
            or observer.get("observer_only") is not True
            or gate != {
                "enabled": True,
                "max_center_shift_m": 0.10,
                "min_volume_ratio": 0.50,
                "max_volume_ratio": 2.00,
            }
        ):
            raise ValueError("observer template violates replay/G0/native-B6 contract")
        cache["baseline_prediction_root"] = str(args.baseline_root)
        observer.setdefault("diagnostics", {})["root"] = str(args.native_diagnostics_root)
        cfg["lifting"]["boxer"]["diagnostics_dir"] = str(args.boxer_diagnostics_root)
    create(args.output, yaml.safe_dump(cfg, sort_keys=False))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
