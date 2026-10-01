#!/usr/bin/env python3
"""Summarize extra50 official-evaluator logs for the ReCaR-3D factorial."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


ORDER = ("base", "plr", "calr", "mvsr", "plr_calr", "plr_mvsr", "calr_mvsr", "full")
LABEL = {
    "base": "Base",
    "plr": "PLR",
    "calr": "CALR",
    "mvsr": "MVSR",
    "plr_calr": "PLR + CALR",
    "plr_mvsr": "PLR + MVSR",
    "calr_mvsr": "CALR + MVSR",
    "full": "PLR + CALR + MVSR",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(path: Path) -> list[float]:
    values = [float(row) * 100.0 for row in re.findall(r"eval mAP:\s*([0-9.]+)", path.read_text(errors="ignore"))]
    if len(values) != 3:
        raise RuntimeError(f"expected three official AP values in {path}, found {len(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "boxfusion.recar3d.extra50_factorial.v1":
        raise RuntimeError("unexpected extra50 factorial manifest")
    results = {}
    log_hashes = {}
    for arm in ORDER:
        path = args.log_root / f"recar3d_extra50_factorial_{arm}.log"
        results[arm] = metrics(path)
        log_hashes[str(path.resolve())] = digest(path)

    payload = {
        "schema": "boxfusion.recar3d.extra50_factorial.metrics.v1",
        "thresholds": [0.15, 0.25, 0.50],
        "scene_count": manifest["scene_count"],
        "results": results,
        "delta_from_base": {
            arm: [value - base for value, base in zip(results[arm], results["base"])]
            for arm in ORDER
        },
        "conditional_plr_on_calr_mvsr": [
            value - prefix for value, prefix in zip(results["full"], results["calr_mvsr"])
        ],
        "manifest_sha256": digest(args.manifest),
        "official_log_sha256": log_hashes,
        "note": "Full uses the runtime assembly order native, PLR, CALR; this only affects deterministic ordering among exact score ties.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# ReCaR-3D extra50 holdout factorial",
        "",
        "All arms are materialized from one strictly causal online pass over the 50 pre-registered unseen scenes. MVSR changes only native scores; native geometry, labels and row count are fixed.",
        "",
        "| Configuration | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|",
    ]
    for arm in ORDER:
        values = results[arm]
        lines.append(f"| {LABEL[arm]} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |")
    gain = payload["conditional_plr_on_calr_mvsr"]
    lines.extend([
        "",
        f"PLR conditioned on CALR+MVSR contributes **{gain[0]:+.4f}/{gain[1]:+.4f}/{gain[2]:+.4f}** AP.",
        "",
        "The Full arm follows the implementation's terminal row order `native -> PLR -> CALR`, identical to the locked official100 materialization.",
    ])
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
