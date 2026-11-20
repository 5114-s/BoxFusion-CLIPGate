#!/usr/bin/env python3
"""Combine official prefix AP logs with causal replay delay statistics."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(r"eval mAP:\s+([0-9.]+)")
TAGS = ("p25", "p50", "p75", "p100")


def metrics(path: Path) -> list[float]:
    values = [100.0 * float(value) for value in PATTERN.findall(path.read_text())]
    if len(values) != 3:
        raise RuntimeError(f"expected AP15/AP25/AP50 in {path}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-results", type=Path, required=True)
    parser.add_argument("--eval-log-root", type=Path, required=True)
    parser.add_argument("--label-prefix", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay = json.loads(args.replay_results.read_text())
    aps = {
        system: {tag: metrics(args.eval_log_root / f"{args.label_prefix}_{system}_{tag}.log") for tag in TAGS}
        for system in ("base", "recar")
    }
    payload = {"replay": replay, "ap": aps}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    terminal_readout = replay.get("readout_protocol") == (
        "prefix_terminal_native_extent_filter_and_state_readout"
    )
    protocol_text = (
        "At every checkpoint, input is stopped at the observed prefix, the same "
        "deterministic native-map extent filter used by the ScanNet terminal "
        "output is applied, and the causal candidate state is read on the "
        "surviving native identities. No future observation or history replay "
        "is used."
        if terminal_readout
        else
        "Rows are provisional keyframe snapshots before the host mapper's "
        "terminal native-map filter."
    )
    lines = [
        "# ScanNet-100 causal online-prefix quality and first-correct-output delay", "",
        protocol_text,
        "Prefix AP is evaluated against each scene's full GT; therefore early-prefix values measure partial-map quality, not a standard final-map benchmark.",
        "Delay starts at the first keyframe where frozen auxiliary proposal-or-anchor evidence overlaps that GT at the evaluated IoU.",
        "", "| Prefix | Base AP15 | AP25 | AP50 | ReCaR AP15 | AP25 | AP50 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tag in TAGS:
        b, r = aps["base"][tag], aps["recar"][tag]
        lines.append(f"| {tag[1:]}% | {b[0]:.4f} | {b[1]:.4f} | {b[2]:.4f} | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} |")
    if "checkpoints" in replay:
        lines += [
            "",
            "| Prefix | Keyframes | Base records | ReCaR records |",
            "|---:|---:|---:|---:|",
        ]
        for fraction, tag in ((0.25, "25"), (0.50, "50"), (0.75, "75"), (1.00, "100")):
            row = replay["checkpoints"][str(fraction)]
            lines.append(
                f"| {tag}% | {row['keyframes']:,} | {row['native_records']:,} | "
                f"{row['recar_records']:,} |"
            )
    lines += ["", "| IoU | Eligible GT | Detected | Rate | Median delay (KF) | P90 | P95 |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for threshold in (0.15, 0.25, 0.50):
        row = replay["delay_summary"][str(threshold)]
        lines.append(f"| {threshold:.2f} | {row['eligible_gt']} | {row['detected_eligible_gt']} | {100*row['eligible_detection_rate']:.2f}% | {row['median_delay_keyframes']:.2f} | {row['p90_delay_keyframes']:.2f} | {row['p95_delay_keyframes']:.2f} |")
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
