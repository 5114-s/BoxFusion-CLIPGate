#!/usr/bin/env python3
"""Create an isolated one-factor MVSR no-visibility audit configuration."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    root = args.run_root.resolve()
    config["data"]["output_dir"] = str(root / "predictions" / "native")
    config["lifting"]["boxer"]["diagnostics_dir"] = str(root / "diagnostics" / "native")
    # PVQ-AR is an unrelated shadow observer and is not part of the frozen
    # ReCaR route evaluated here.  Disable it so missing diagnostic appearance
    # features cannot abort an otherwise valid cross-dataset sequence.
    config["association"]["pvq_ar"]["enabled"] = False
    config["association"]["pvq_ar"]["diagnostics_dir"] = str(root / "diagnostics" / "pvq")
    online = config["online_candidate_map"]
    online["output_root"] = str(root / "predictions" / "online_host")
    online["diagnostics_root"] = str(root / "diagnostics" / "online")
    online["audit_output_root"] = str(root / "components")
    online["provider"]["diagnostics_root"] = str(root / "diagnostics" / "provider")
    online["provider"]["evidence_cache_root"] = str(root / "evidence")
    state = online["state"]
    state["audit_controls"] = True
    state["audit_mvsr_factors"] = True
    state["mvsr_factor_variants"] = ["no_visibility"]
    state["audit_direct_plr"] = False
    state["audit_plr_controls"] = False
    state["audit_raw_iou_max"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
