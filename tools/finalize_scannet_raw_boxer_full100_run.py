#!/usr/bin/env python3
"""Finalize the sharded, no-GT Raw Boxer full100 inference ledger.

This utility never opens ScanNet annotations or an evaluator.  It verifies
that every official scene completed, seals the frozen model/schedule inputs,
and records the unchanged T05 prediction bytes required by the generic Boxer
candidate sealer.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "logs" / "scannet_raw_boxer_full100_score05_v1"
DEFAULT_SCENE_LIST = (
    ROOT / "evaluation" / "data_util" / "meta_data" / "scannetv2_val.txt"
)
DEFAULT_BASELINE_ROOT = ROOT / "results" / "scannet_topk_fusion_score05"
DEFAULT_CACHE_ROOT = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/"
    "scannet-score05-gap25-postfilter-v2"
)
BOXER_ROOT = Path("/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/third_party/boxer")
OWL_CHECKPOINT = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/"
    "owlv2-base-patch16-ensemble.pt"
)
OWL_TEXT_CACHE = Path(
    "/data/ZhaoX/OVM3D-Dett/third_party/boxer/ckpts/"
    "owlv2-base-patch16-ensemble_textemb_878186d327b0.pt"
)
EXPECTED_SCENE_LIST_SHA256 = (
    "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
)
GT_GUARD = "BOXFUSION_SHADOW_GT_ACCESS=forbidden annotation_path=None"


class FinalizeError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_scenes(path: Path) -> tuple[str, ...]:
    if not path.is_file() or _sha256(path) != EXPECTED_SCENE_LIST_SHA256:
        raise FinalizeError("official full100 scene-list identity mismatch")
    scenes = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(scenes) != 100 or len(set(scenes)) != 100:
        raise FinalizeError("official scene list is not 100 unique scenes")
    return scenes


def _atomic_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def finalize(
    *,
    run_root: Path,
    scene_list: Path,
    baseline_root: Path,
    cache_root: Path,
) -> dict[str, object]:
    run_root = run_root.resolve()
    scene_list = scene_list.resolve()
    baseline_root = baseline_root.resolve()
    cache_root = cache_root.resolve()
    scenes = _official_scenes(scene_list)

    frozen_paths = [
        BOXER_ROOT / "run_boxer.py",
        BOXER_ROOT / "owl" / "owl_wrapper.py",
        BOXER_ROOT / "boxernet" / "boxernet.py",
        BOXER_ROOT / "ckpts" / "boxernet_hw960in2x6d768-c88128f8.ckpt",
        BOXER_ROOT / "ckpts" / "dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        OWL_CHECKPOINT,
        OWL_TEXT_CACHE,
        scene_list,
        ROOT / "config" / "scannet_topk_fusion_score05.yaml",
    ]
    audit_by_scene: dict[str, list[str]] = {}
    total_raw_candidates = 0
    for scene in scenes:
        manifest = cache_root / scene / "manifest.json"
        prediction = baseline_root / f"{scene}_boxes.pkl"
        log = run_root / "scenes" / f"{scene}.log"
        raw_csv = run_root / "boxer_raw" / scene / "boxer_3dbbs.csv"
        owl_csv = run_root / "boxer_raw" / scene / "owl_2dbbs.csv"
        for path in (manifest, prediction, log, raw_csv, owl_csv):
            if not path.is_file() or path.stat().st_size == 0:
                raise FinalizeError(f"missing completed full100 artifact: {path}")
        log_lines = log.read_text(encoding="utf-8", errors="strict").splitlines()
        if not log_lines or log_lines[0].strip() != GT_GUARD:
            raise FinalizeError(f"no-GT guard missing from {scene} log")
        log_text = "\n".join(log_lines)
        if "Traceback" in log_text or "Exception in thread" in log_text:
            raise FinalizeError(f"inference exception found in {scene} log")
        if "Saved 3D BBs to" not in log_text or "Saved 2D BBs to" not in log_text:
            raise FinalizeError(f"incomplete Raw Boxer completion markers for {scene}")
        with raw_csv.open("r", encoding="utf-8", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        total_raw_candidates += count
        frozen_paths.append(manifest)

    for audit_path in sorted(run_root.glob("schedule_audit_worker*_of_*.tsv")):
        with audit_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            header = next(reader, None)
            if header != [
                "scene",
                "manifest_keyframes",
                "valid_keyframes",
                "invalid_pose_keyframes",
                "raw_candidate_frames",
                "raw_candidates",
            ]:
                raise FinalizeError(f"invalid worker schedule audit: {audit_path}")
            for row in reader:
                if len(row) != 6 or row[0] in audit_by_scene:
                    raise FinalizeError(f"duplicate/invalid schedule audit row: {row}")
                audit_by_scene[row[0]] = row
    if set(audit_by_scene) != set(scenes):
        raise FinalizeError("worker schedule audits do not cover official full100")

    frozen_rows = "".join(
        f"{_sha256(path)}  {path.resolve()}\n"
        for path in frozen_paths
        if path.is_file()
    )
    native_rows = "".join(
        f"{_sha256(baseline_root / f'{scene}_boxes.pkl')}  "
        f"{(baseline_root / f'{scene}_boxes.pkl').resolve()}\n"
        for scene in scenes
    )
    audit_header = (
        "scene\tmanifest_keyframes\tvalid_keyframes\tinvalid_pose_keyframes\t"
        "raw_candidate_frames\traw_candidates\n"
    )
    audit_payload = audit_header + "".join(
        "\t".join(audit_by_scene[scene]) + "\n" for scene in scenes
    )
    _atomic_text(run_root / "frozen_inputs_sha256.txt", frozen_rows)
    _atomic_text(run_root / "native_before_sha256.txt", native_rows)
    _atomic_text(run_root / "native_after_sha256.txt", native_rows)
    _atomic_text(run_root / "schedule_audit.tsv", audit_payload)

    report: dict[str, object] = {
        "schema": "boxfusion.scannet_raw_boxer_full100_finalize.v1",
        "scene_count": len(scenes),
        "raw_candidate_count": total_raw_candidates,
        "gt_access": False,
        "evaluator_access": False,
        "training": False,
        "native_prediction_identity": True,
        "scene_list_sha256": _sha256(scene_list),
        "frozen_input_ledger_sha256": _sha256(
            run_root / "frozen_inputs_sha256.txt"
        ),
        "native_ledger_sha256": _sha256(run_root / "native_before_sha256.txt"),
    }
    _atomic_text(
        run_root / "finalize_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--scene-list", type=Path, default=DEFAULT_SCENE_LIST)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize(
                run_root=args.run_root,
                scene_list=args.scene_list,
                baseline_root=args.baseline_root,
                cache_root=args.cache_root,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
