#!/usr/bin/env python3
"""Create an isolated ScanNet configuration for the MVSR design ablation."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.base.read_text())
    root = args.run_root.resolve()
    online = config["online_candidate_map"]
    config["data"]["output_dir"] = str(root / "native")
    config["lifting"]["boxer"]["diagnostics_dir"] = str(root / "diagnostics" / "boxer")
    config["association"]["pvq_ar"]["diagnostics_dir"] = str(root / "diagnostics" / "pvq")
    online["output_root"] = str(root / "online")
    online["diagnostics_root"] = str(root / "diagnostics" / "online")
    online["audit_output_root"] = str(root / "components")
    online["provider"]["diagnostics_root"] = str(root / "diagnostics" / "provider")
    online["provider"]["evidence_cache_root"] = str(root / "evidence")
    online["state"]["audit_mvsr_factors"] = True
    online["state"]["audit_controls"] = False
    online["state"]["audit_direct_plr"] = False
    online["state"]["audit_plr_controls"] = False
    online["state"]["audit_raw_iou_max"] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
