#!/usr/bin/env python3
"""Select genuine final-config PLR/CALR/MVSR success and failure cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

MAIN = Path("/data/ZhaoX/BoxFusion")
sys.path.insert(0, str(MAIN))
from tools.audit_ca1m_nms_child_headroom import pairwise_iou
from tools.audit_final_ledger import scannet_inputs


def read(path: Path):
    with path.open("rb") as handle:
        rows = pickle.load(handle)[0]
    return [
        (int(row[0]), np.asarray(row[1], dtype=np.float64).reshape(8, 3), float(row[2]))
        for row in rows
    ]


def ranks(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    result = np.empty(len(scores), dtype=np.int64)
    result[order] = np.arange(1, len(scores) + 1)
    return result


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--factorial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = [r.strip() for r in args.scene_list.read_text().splitlines()
              if r.strip() and not r.lstrip().startswith("#")]
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise RuntimeError("expected ScanNet official100")
    manifest_path = args.factorial / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("children_enabled") is not False or not manifest.get("strictly_causal"):
        raise RuntimeError("qualitative source is not final strict-online no-child run")
    gts, aligns, transform = scannet_inputs()
    pools = {name: [] for name in (
        "plr_recovery", "calr_recovery", "mvsr_reranking",
        "failure_mvsr_fp", "failure_plr_localization", "failure_calr_background",
    )}

    for scene in scenes:
        paths = {
            "native": args.components / "native_native" / f"{scene}_boxes.pkl",
            "mvsr": args.components / "native_max" / f"{scene}_boxes.pkl",
            "plr": args.components / "births_plr_v1" / f"{scene}_boxes.pkl",
            "calr": args.components / "births_calr_v1" / f"{scene}_boxes.pkl",
        }
        rows = {name: read(path) for name, path in paths.items()}
        boxes = {
            name: np.asarray([r[1] for r in values], dtype=np.float64).reshape(-1, 8, 3)
            for name, values in rows.items()
        }
        aligned = {
            name: transform(value, aligns[scene]) if len(value) else value
            for name, value in boxes.items()
        }
        gt = gts[scene]
        iou = {
            name: pairwise_iou(value, gt) if len(value) and len(gt)
            else np.zeros((len(value), len(gt)), dtype=np.float64)
            for name, value in aligned.items()
        }
        max_gt = {
            name: value.max(axis=0) if len(value) else np.zeros(len(gt), dtype=np.float64)
            for name, value in iou.items()
        }
        for source in ("plr", "calr"):
            for index, row in enumerate(rows[source]):
                gt_index = int(np.argmax(iou[source][index])) if len(gt) else -1
                value = float(iou[source][index, gt_index]) if gt_index >= 0 else 0.0
                item = {
                    "scene": scene, "source": source, "source_index": index,
                    "gt_index": gt_index, "iou": value, "score": row[2],
                    "box": row[1].round(7).tolist(),
                    "box_aligned": aligned[source][index].round(7).tolist(),
                    "gt_box_aligned": gt[gt_index].round(7).tolist() if gt_index >= 0 else None,
                    "native_gt_iou": float(max_gt["native"][gt_index]) if gt_index >= 0 else 0.0,
                    "plr_gt_iou": float(max_gt["plr"][gt_index]) if gt_index >= 0 else 0.0,
                    "calr_gt_iou": float(max_gt["calr"][gt_index]) if gt_index >= 0 else 0.0,
                }
                if source == "plr":
                    if item["native_gt_iou"] < .15 and value >= .25 and item["calr_gt_iou"] < .25:
                        pools["plr_recovery"].append(item)
                    if item["native_gt_iou"] < .15 and .25 <= value < .50:
                        pools["failure_plr_localization"].append(item | {
                            "failure_reason": "PLR covers the missed target but remains below IoU 0.50"
                        })
                else:
                    if item["native_gt_iou"] < .15 and item["plr_gt_iou"] < .15 and value >= .25:
                        pools["calr_recovery"].append(item)
                    if value < .05:
                        pools["failure_calr_background"].append(item | {
                            "failure_reason": "persistent anchor consensus has no GT overlap at IoU 0.05"
                        })

        original = np.asarray([r[2] for r in rows["native"]], dtype=np.float64)
        updated = np.asarray([r[2] for r in rows["mvsr"]], dtype=np.float64)
        before, after = ranks(original), ranks(updated)
        for index, row in enumerate(rows["native"]):
            gt_index = int(np.argmax(iou["native"][index])) if len(gt) else -1
            value = float(iou["native"][index, gt_index]) if gt_index >= 0 else 0.0
            item = {
                "scene": scene, "source": "native", "source_index": index,
                "gt_index": gt_index, "iou": value,
                "original_score": float(original[index]),
                "reranked_score": float(updated[index]),
                "original_rank": int(before[index]), "reranked_rank": int(after[index]),
                "rank_gain": int(before[index] - after[index]),
                "box": row[1].round(7).tolist(),
                "box_aligned": aligned["native"][index].round(7).tolist(),
                "gt_box_aligned": gt[gt_index].round(7).tolist() if gt_index >= 0 else None,
            }
            if value >= .25 and item["rank_gain"] > 0 and updated[index] > original[index]:
                pools["mvsr_reranking"].append(item)
            if value < .15 and item["rank_gain"] > 0 and updated[index] > original[index]:
                pools["failure_mvsr_fp"].append(item | {
                    "failure_reason": "high-support native false positive moves forward after reranking"
                })

    pools["plr_recovery"].sort(key=lambda r: (-r["iou"], r["scene"], r["source_index"]))
    pools["calr_recovery"].sort(key=lambda r: (-r["iou"], r["scene"], r["source_index"]))
    pools["mvsr_reranking"].sort(key=lambda r: (-r["rank_gain"], -r["iou"], r["scene"]))
    pools["failure_mvsr_fp"].sort(key=lambda r: (-r["rank_gain"], r["scene"]))
    pools["failure_plr_localization"].sort(key=lambda r: (-r["iou"], r["scene"]))
    pools["failure_calr_background"].sort(key=lambda r: (-r["score"], r["scene"]))
    for required in ("plr_recovery", "calr_recovery", "mvsr_reranking"):
        if not pools[required]:
            raise RuntimeError(f"no genuine case found for {required}")
    failure_kind = next((name for name in (
        "failure_mvsr_fp", "failure_plr_localization", "failure_calr_background"
    ) if pools[name]), None)
    if failure_kind is None:
        raise RuntimeError("no genuine failure case found")
    selected = {
        "plr_recovery": pools["plr_recovery"][0],
        "calr_recovery": pools["calr_recovery"][0],
        "mvsr_reranking": pools["mvsr_reranking"][0],
        "failure": pools[failure_kind][0] | {"failure_type": failure_kind},
    }
    payload = {
        "schema": "boxfusion.recar3d.qualitative_selection.v1",
        "source_factorial": str(args.factorial.resolve()),
        "source_manifest_sha256": digest(manifest_path),
        "strictly_causal": True, "children_enabled": False,
        "cases": selected,
        "candidate_counts": {name: len(value) for name, value in pools.items()},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "selected_cases.json").write_text(json.dumps(payload, indent=2) + "\n")
    selected_scenes = sorted({row["scene"] for row in selected.values()})
    (args.output / "selected_scenes.txt").write_text("\n".join(selected_scenes) + "\n")
    fields = sorted({k for row in selected.values() for k in row
                     if k not in ("box", "box_aligned", "gt_box_aligned")})
    with (args.output / "case_audit.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields); writer.writeheader()
        for panel, row in selected.items():
            writer.writerow({**{key: row.get(key) for key in fields}, "source": row["source"]})
    print(json.dumps({panel: row["scene"] for panel, row in selected.items()}, indent=2))


if __name__ == "__main__":
    main()
