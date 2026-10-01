#!/usr/bin/env python3
"""Create matched MVSR aggregation configs without changing source code."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


MODES = ("native", "first", "mean", "max", "ema", "diverse_max", "reliability")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        row = copy.deepcopy(base)
        state = row["online_candidate_map"]["state"]
        if state.get("variant") != "reliability_v2":
            raise ValueError("base config is not reliability_v2")
        state["m2"]["support_mode"] = mode
        tag = f"scannet_reliability_v2_mvsr_{mode}"
        row["data"]["output_dir"] = str(args.output / f"{tag}_native")
        online = row["online_candidate_map"]
        online["output_root"] = str(args.output / tag)
        online["diagnostics_root"] = str(args.output / f"{tag}_diagnostics")
        online["provider"]["diagnostics_root"] = str(
            args.output / f"{tag}_provider"
        )
        path = args.output / f"{mode}.yaml"
        path.write_text(yaml.safe_dump(row, sort_keys=False), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
