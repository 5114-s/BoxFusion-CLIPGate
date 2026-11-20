#!/usr/bin/env python3
"""Create one canonical ledger for the final submission-completion runs."""
from __future__ import annotations

import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/submission_completion_20261006"
PATTERN = re.compile(r"eval mAP:\s+([0-9.]+)")


def load(path: Path):
    if not path.is_file():
        raise RuntimeError(f"missing required result: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def scannet_metrics() -> dict[str, list[float]]:
    labels = {
        "base": "base", "plr": "plr", "calr": "calr", "mvsr": "mvsr",
        "plr_calr": "plr_calr", "plr_mvsr": "plr_mvsr",
        "calr_mvsr": "calr_mvsr", "full": "full",
    }
    root = ROOT / "logs/scannet_official100_real_score"
    result = {}
    for name, suffix in labels.items():
        path = root / f"recar3d_novis_factorial_{suffix}_20261006.log"
        values = [100.0 * float(value) for value in PATTERN.findall(path.read_text())]
        if len(values) != 3:
            raise RuntimeError(f"invalid ScanNet AP log: {path}")
        result[name] = values
    return result


def main() -> None:
    efficiency = load(
        ROOT / "reports/original_boxfusion_vs_final_recar_efficiency100_prewarmed_20261006/summary.json"
    )
    ca1m = load(
        ROOT / "development/final_controls/experiments/ca1m_novis107_20261006/report_no_visibility/results.json"
    )
    temporal = load(
        ROOT / "development/final_controls/experiments/mvsr_novis_temporal_20261006/report/results.json"
    )
    scan_bootstrap = load(
        ROOT / "reports/final_bootstrap_scannet_novis_20261006/results.json"
    )
    ca_bootstrap = load(
        ROOT / "reports/final_bootstrap_ca1m_novis_20261006/results.json"
    )
    mvsr_stats = load(
        ROOT / "figure_assets/mvsr_support_ranking/mvsr_support_ranking_optimized.json"
    )
    multiscan = load(
        ROOT / "development/final_controls/experiments/multiscan50_novis_20261006/report/metrics.json"
    )
    multiscan_budget = load(
        ROOT / "development/final_controls/experiments/multiscan50_novis_20261006/budget_report/results.json"
    )
    scannet = scannet_metrics()
    result = {
        "schema": "boxfusion.submission_completion.final_novis.v1",
        "final_mvsr": {
            "factors": ["proposal_score", "2d_overlap", "3d_consistency"],
            "visibility": False,
            "temporal_aggregation": "historical_maximum",
        },
        "scannet_factorial": scannet,
        "ca1m_factorial": ca1m["results"],
        "efficiency": efficiency,
        "temporal_aggregation": temporal,
        "bootstrap": {"scannet": scan_bootstrap, "ca1m": ca_bootstrap},
        "multiscan50": multiscan,
        "multiscan50_budget": multiscan_budget,
        "mvsr_mechanism": mvsr_stats,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = {
        "base": "Base", "plr": "PLR", "calr": "CALR", "mvsr": "MVSR",
        "plr_calr": "PLR+CALR", "plr_mvsr": "PLR+MVSR",
        "calr_mvsr": "CALR+MVSR", "full": "PLR+CALR+MVSR",
    }
    ca_keys = {
        "base": "base", "plr": "p", "calr": "a", "mvsr": "m2",
        "plr_calr": "p_a", "plr_mvsr": "p_m2",
        "calr_mvsr": "a_m2", "full": "p_a_m2",
    }
    lines = [
        "# Final submission-completion ledger", "",
        "Final MVSR: proposal score + 2D overlap + 3D geometry consistency; ",
        "visibility excluded; historical-maximum aggregation.", "",
        "## Final paired 2^3 factorial", "",
        "| Configuration | ScanNet AP15 | AP25 | AP50 | CA-1M AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in labels:
        scan = scannet[name]
        ca = ca1m["results"][ca_keys[name]]
        lines.append(
            f"| {labels[name]} | {scan[0]:.4f} | {scan[1]:.4f} | {scan[2]:.4f} | "
            f"{ca['ap15']:.4f} | {ca['ap25']:.4f} | {ca['ap50']:.4f} |"
        )

    original = efficiency["arms"]["original"]
    recar = efficiency["arms"]["recar"]
    keyframe = efficiency.get(
        "recar_keyframe",
        efficiency.get("recar_keyframes", efficiency.get("keyframe", {})),
    )
    lines.extend([
        "", "## Final matched efficiency", "",
        "| Method | FPS | Peak allocated memory (GiB) |",
        "|---|---:|---:|",
        f"| Original BoxFusion | {original['frame_weighted_fps']:.4f} | {original['peak_allocated_gib_max']:.3f} |",
        f"| ReCaR-3D | {recar['frame_weighted_fps']:.4f} | {recar['peak_allocated_gib_max']:.3f} |",
        "",
        f"Throughput retention: {100.0 * recar['frame_weighted_fps'] / original['frame_weighted_fps']:.2f}%.",
    ])
    if keyframe:
        lines.append(f"Keyframe diagnostics: `{json.dumps(keyframe, sort_keys=True)}`.")

    lines.extend([
        "", "## Final three-factor MVSR temporal aggregation", "",
        "| Aggregation | AP15 | AP25 | AP50 |",
        "|---|---:|---:|---:|",
    ])
    for name, values in temporal["metrics"].items():
        lines.append(
            f"| {name} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} |"
        )

    lines.extend(["", "## MultiScan-50 final no-visibility results", ""])
    multiscan_order = (
        "base", "plr", "calr", "mvsr", "plr_calr", "plr_mvsr",
        "calr_mvsr", "full_no_visibility",
    )
    for protocol, protocol_result in multiscan["protocols"].items():
        lines.extend([
            f"### {protocol}", "",
            "| Configuration | AP15 | AP25 | AP50 | Predictions |",
            "|---|---:|---:|---:|---:|",
        ])
        for name in multiscan_order:
            arm = protocol_result["arms"][name]
            values = [arm["metrics"][str(t)]["ap"] for t in (0.15, 0.25, 0.5)]
            lines.append(
                f"| {name} | {values[0]:.4f} | {values[1]:.4f} | "
                f"{values[2]:.4f} | {arm['prediction_boxes']:,} |"
            )
        lines.append("")
    lines.append(
        "Final MultiScan budget/bootstrap: "
        "`development/final_controls/experiments/multiscan50_novis_20261006/budget_report/REPORT.md`."
    )

    lines.extend(["", "## Bootstrap and mechanism outputs", ""])
    lines.append(
        "- ScanNet paired bootstrap: `reports/final_bootstrap_scannet_novis_20261006/REPORT.md`."
    )
    lines.append(
        "- CA-1M paired bootstrap: `reports/final_bootstrap_ca1m_novis_20261006/REPORT.md`."
    )
    summary = mvsr_stats["tp_fp_summary"]
    lines.append(
        f"- MVSR native rows: n={mvsr_stats['native_box_count']}; TP/FP mean effective support "
        f"{summary['TP']['mean_effective_support']:.4f}/{summary['FP']['mean_effective_support']:.4f}; "
        f"positive-correction fractions {summary['TP']['positive_correction_fraction']:.4f}/"
        f"{summary['FP']['positive_correction_fraction']:.4f}."
    )
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "REPORT.md")


if __name__ == "__main__":
    main()
