#!/usr/bin/env python3
"""Summarize U0/U1/U2 metrics and apply the fixed-10 promotion gate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


MAP_PATTERN = re.compile(r"eval mAP:\s*([0-9]+(?:[.][0-9]+)?)")
THRESHOLDS = (0.15, 0.25, 0.50)


def load_map(path: Path):
    values = [float(value) * 100.0 for value in MAP_PATTERN.findall(
        path.read_text(encoding="utf-8")
    )]
    if len(values) != 3:
        raise ValueError(f"Expected three eval mAP rows in {path}, got {len(values)}")
    return {
        f"AP{int(threshold * 100)}": value
        for threshold, value in zip(THRESHOLDS, values)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-log", type=Path, required=True)
    parser.add_argument("--observer-log", type=Path, required=True)
    parser.add_argument("--active-log", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = {
        "U0": load_map(args.control_log),
        "U1": load_map(args.observer_log),
        "U2": load_map(args.active_log),
    }
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    delta = {
        key: metrics["U2"][key] - metrics["U0"][key]
        for key in metrics["U0"]
    }
    observer_delta = {
        key: metrics["U1"][key] - metrics["U0"][key]
        for key in metrics["U0"]
    }
    coverage = float(audit.get("observer_weight_coverage", 0.0))
    invalid = int(audit.get("observer", {}).get(
        "invalid_boxer_confidence", -1
    )) + int(audit.get("active", {}).get(
        "invalid_boxer_confidence", -1
    ))
    accuracy_gate = (
        delta["AP50"] >= 0.5
        and delta["AP25"] >= 0.0
        and delta["AP15"] >= -0.3
    )
    observer_metric_identity = all(
        abs(value) <= 1e-9 for value in observer_delta.values()
    )
    promote = bool(
        audit.get("ok")
        and observer_metric_identity
        and invalid == 0
        and coverage >= 0.05
        and accuracy_gate
    )
    report = {
        "schema": "boxfusion.b6_boxer_uncertainty.effectiveness.v1",
        "metrics_percent": metrics,
        "U2_minus_U0_percent_points": delta,
        "U1_minus_U0_percent_points": observer_delta,
        "structural_audit_ok": bool(audit.get("ok")),
        "observer_metric_identity": observer_metric_identity,
        "invalid_confidence_total": invalid,
        "effective_fusion_coverage": coverage,
        "fixed10_gate": {
            "minimum_effective_fusion_coverage": 0.05,
            "minimum_delta_AP50": 0.5,
            "minimum_delta_AP25": 0.0,
            "minimum_delta_AP15": -0.3,
            "accuracy_gate_pass": accuracy_gate,
        },
        "promote_to_full100": promote,
        "interpretation": (
            "Run the frozen U0/U2 full-100 comparison once."
            if promote
            else "Stop this route at fixed-10 and record a negative ablation."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
