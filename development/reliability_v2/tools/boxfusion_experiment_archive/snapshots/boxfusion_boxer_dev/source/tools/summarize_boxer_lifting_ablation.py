#!/usr/bin/env python3
"""Summarize paired X0/X1/X2 ScanNet evaluation logs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


METRIC_PATTERN = re.compile(
    r"eval (mAP|APrec|ARecall):\s+([0-9eE+.-]+)"
)


def parse_metrics(path: Path):
    values = {"mAP": [], "APrec": [], "ARecall": []}
    for name, value in METRIC_PATTERN.findall(
        path.read_text(encoding="utf-8", errors="replace")
    ):
        values[name].append(float(value))
    for name, sequence in values.items():
        if len(sequence) != 3:
            raise ValueError(
                f"{path}: expected three {name} values, found {len(sequence)}"
            )
    return {
        "AP15": values["mAP"][0] * 100.0,
        "AP25": values["mAP"][1] * 100.0,
        "AP50": values["mAP"][2] * 100.0,
        "precision15": values["APrec"][0] * 100.0,
        "precision25": values["APrec"][1] * 100.0,
        "precision50": values["APrec"][2] * 100.0,
        "recall15": values["ARecall"][0] * 100.0,
        "recall25": values["ARecall"][1] * 100.0,
        "recall50": values["ARecall"][2] * 100.0,
    }


def read_scene_list(path: Path):
    scenes = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes:
        raise ValueError(f"Empty scene list: {path}")
    if len(scenes) != len(set(scenes)):
        raise ValueError(f"Duplicate scene IDs: {path}")
    return scenes


def load_runtime_rows(root: Path, scenes):
    runtimes = []
    proposals = 0
    observed_calls = 0
    forward_calls = 0
    for scene in scenes:
        path = root / f"{scene}_boxer_lifting.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            count = int(row["count"])
            proposals += count
            observed_calls += 1
            if count > 0:
                runtimes.append(float(row["runtime_ms"]))
                forward_calls += 1
    return runtimes, proposals, observed_calls, forward_calls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-log", type=Path, required=True)
    parser.add_argument("--observer-log", type=Path, required=True)
    parser.add_argument("--active-log", type=Path, required=True)
    parser.add_argument("--contract-report", type=Path, required=True)
    parser.add_argument("--active-diagnostics", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--route-revision", required=True)
    parser.add_argument(
        "--phase",
        choices=("fixed10", "full100"),
        default="fixed10",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = parse_metrics(args.baseline_log)
    observer = parse_metrics(args.observer_log)
    active = parse_metrics(args.active_log)
    contract = json.loads(args.contract_report.read_text(encoding="utf-8"))
    scenes = read_scene_list(args.scene_list)
    runtimes, proposals, observed_calls, forward_calls = load_runtime_rows(
        args.active_diagnostics,
        scenes,
    )
    runtime_array = np.asarray(runtimes, dtype=np.float64)

    delta = {
        name: active[name] - baseline[name]
        for name in ("AP15", "AP25", "AP50")
    }
    observer_delta = {
        name: observer[name] - baseline[name]
        for name in ("AP15", "AP25", "AP50")
    }
    identity_metrics = all(
        abs(value) < 1e-9 for value in observer_delta.values()
    )
    recall_delta = {
        name: active[name] - baseline[name]
        for name in ("recall15", "recall25", "recall50")
    }
    fixed10_positive = (
        delta["AP25"] >= 0.5
        and delta["AP50"] >= 0.5
        and delta["AP15"] > -0.5
        and (
            recall_delta["recall25"] >= 2.0
            or recall_delta["recall50"] >= 1.0
        )
    )
    full100_positive = (
        delta["AP25"] >= 1.0
        and delta["AP50"] >= 1.0
        and delta["AP15"] > -0.5
    )
    report = {
        "schema": "boxfusion.boxer_lifting.ablation_summary.v1",
        "route_revision": str(args.route_revision),
        "scene_list_sha256": hashlib.sha256(
            args.scene_list.read_bytes()
        ).hexdigest(),
        "phase": args.phase,
        "contract_ok": bool(contract.get("ok", False)),
        "observer_metric_identity": identity_metrics,
        "baseline": baseline,
        "observer": observer,
        "active": active,
        "delta_active_minus_cutr": delta,
        "recall_delta_active_minus_cutr": recall_delta,
        "runtime": {
            "observed_calls": observed_calls,
            "forward_calls": forward_calls,
            "proposals": proposals,
            "median_ms_per_keyframe": (
                float(np.median(runtime_array)) if runtime_array.size else None
            ),
            "p95_ms_per_keyframe": (
                float(np.quantile(runtime_array, 0.95))
                if runtime_array.size
                else None
            ),
            "mean_ms_per_proposal": (
                float(runtime_array.sum() / proposals)
                if proposals > 0
                else None
            ),
        },
        "fixed10_pass_rule": (
            "contract_ok and observer identity and "
            "delta AP25>=0.5, AP50>=0.5, AP15>-0.5, and "
            "(delta recall25>=2 or delta recall50>=1)"
        ),
        "recommend_full100": bool(
            args.phase == "fixed10"
            and contract.get("ok", False)
            and identity_metrics
            and fixed10_positive
        ),
        "full100_retention_rule": (
            "contract_ok and observer identity and delta AP25>=1.0, "
            "AP50>=1.0, AP15>-0.5"
        ),
        "retain_module": bool(
            args.phase == "full100"
            and contract.get("ok", False)
            and identity_metrics
            and full100_positive
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if contract.get("ok", False) and identity_metrics else 1


if __name__ == "__main__":
    raise SystemExit(main())
