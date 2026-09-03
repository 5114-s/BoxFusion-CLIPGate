#!/usr/bin/env python3
"""Create an output-identical real-score Cbest C1 observer replay.

C1 replays the frozen per-keyframe Raw-Boxer stream through the existing
query-before-commit S3R memory and submits every newly committed three-view
receipt to a bounded asynchronous diagnostic scheduler.  It copies each Cbest
prediction byte-for-byte and has no geometry, score, semantic or birth API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from boxfusion.causal_async_observer import (  # noqa: E402
    BoundedCausalAsyncObserver,
    CausalAsyncObserverConfig,
)
from boxfusion.s3r_receipt_tracker import S3RReceiptTracker  # noqa: E402
from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    CSV_NAME,
    CSV_RELATIVE,
    PREDICTION_SUFFIX,
    TOP_K_PER_FRAME,
    _completion_rows,
    _load_native_prediction,
    _load_raw_observations,
    _regular_file,
    _scene_list,
    _valid_schedule,
    _write_json,
    wait_for_raw_completion,
)


SCHEMA = "boxfusion.scannet_c1_causal_async_observer_full100.v1"
MANIFEST_NAME = "C1_CAUSAL_ASYNC_OBSERVER_FULL100.json"
OFFICIAL_SCENE_LIST_SHA256 = (
    "4b18fc586f7ad60cb17f41ee7d1d8b0ab1a0782917b5ae3519dd8ec90e7744d5"
)


class C1MaterializationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_summary(rows: Sequence[Any]) -> tuple[int, float | None, float | None]:
    scores = [float(row[2]) for row in rows]
    if not scores:
        return 0, None, None
    return len(set(scores)), min(scores), max(scores)


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    scenes = _scene_list(scene_list, args.expected_scene_count)
    scene_list_sha = _sha256(scene_list)
    if args.expected_scene_count == 100 and scene_list_sha != OFFICIAL_SCENE_LIST_SHA256:
        raise C1MaterializationError("official100 scene-list hash mismatch")

    baseline_root = args.baseline_root.resolve()
    raw_log_root = args.raw_log_root.resolve()
    schedule_root = args.schedule_root.resolve()
    scene_rgbd_root = args.scene_rgbd_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise C1MaterializationError(f"refusing to overwrite output root: {output_root}")
    for path, label in (
        (baseline_root, "real-score Cbest root"),
        (raw_log_root, "Raw-Boxer log root"),
        (schedule_root, "schedule root"),
        (scene_rgbd_root, "RGB-D root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise C1MaterializationError(f"{label} must be a regular directory: {path}")

    if args.expected_scene_count == 100:
        completions = wait_for_raw_completion(
            log_root=raw_log_root,
            scenes=scenes,
            timeout_seconds=0.0,
            poll_seconds=1.0,
        )
    else:
        all_completions = _completion_rows(raw_log_root)
        missing = sorted(set(scenes) - set(all_completions))
        if missing:
            raise C1MaterializationError(f"missing completion rows: {missing}")
        completions = {scene: all_completions[scene] for scene in scenes}

    config = CausalAsyncObserverConfig(
        max_workers=2,
        max_pending_tasks=32,
        max_results=1024,
        max_result_lag_keyframes=4,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    diagnostics_root = stage / "observer_diagnostics"
    diagnostics_root.mkdir()
    scene_reports: dict[str, Any] = {}
    baseline_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    diagnostic_hashes: dict[str, str] = {}
    total_rows = total_receipts = total_submitted = total_completed = total_dropped = 0
    all_scores: list[float] = []

    try:
        for position, scene in enumerate(scenes, 1):
            completion = completions[scene]
            schedule, schedule_sha, world_offset, _ = _valid_schedule(
                schedule_root=schedule_root,
                scene_rgbd_root=scene_rgbd_root,
                scene=scene,
                completion=completion,
            )
            raw_path = raw_log_root / CSV_RELATIVE / scene / CSV_NAME
            by_frame, raw_sha = _load_raw_observations(
                path=raw_path,
                schedule=schedule,
                completion=completion,
                world_offset=world_offset,
            )
            tracker = S3RReceiptTracker()
            observer = BoundedCausalAsyncObserver(scene, config)
            selected_observations = 0
            last_history_frame: int | None = None
            for keyframe_step, frame_id in enumerate(schedule):
                observer.poll(keyframe_step)
                selected = tuple(
                    sorted(
                        by_frame.get(frame_id, ()),
                        key=lambda row: (-row.score, row.source_row, row.sealed_npz_row),
                    )[:TOP_K_PER_FRAME]
                )
                selected_observations += len(selected)
                query = tracker.query(frame_id, selected)
                if (
                    query.history_max_frame_id != last_history_frame
                    or (
                        query.history_max_frame_id is not None
                        and query.history_max_frame_id >= frame_id
                    )
                    or query.selected_source_rows
                    != tuple(row.source_row for row in selected)
                    or query.observation_capacity_dropped_source_rows
                    or not query.audit_complete
                ):
                    raise C1MaterializationError(
                        f"causal memory query audit failed: {scene}/{frame_id}"
                    )
                commit = tracker.commit(query)
                if not commit.audit_complete or commit.output_mutation_applied:
                    raise C1MaterializationError(
                        f"causal memory commit audit failed: {scene}/{frame_id}"
                    )
                memory_version = keyframe_step + 1
                for receipt in commit.newly_frozen_receipts:
                    observer.submit(
                        receipt,
                        keyframe_step=keyframe_step,
                        memory_version=memory_version,
                    )
                last_history_frame = frame_id
            observer.close(max(len(schedule) - 1, 0))
            tracker_summary = tracker.summary()
            observer_summary = observer.summary()
            result_rows = observer.result_rows()
            drop_rows = observer.drop_rows()
            if (
                tracker_summary["audit_complete"] is not True
                or tracker_summary["pending_frame_id"] is not None
                or int(tracker_summary["receipts"])
                != int(observer_summary["completed_results"])
                + int(observer_summary["dropped_tasks"])
                or any(
                    max(row["evidence_frame_ids"]) > row["enqueue_frame_id"]
                    for row in result_rows
                )
            ):
                raise C1MaterializationError(
                    f"terminal C1 observer audit failed: {scene}; "
                    f"tracker_receipts={tracker_summary['receipts']} "
                    f"tracker_audit={tracker_summary['audit_complete']} "
                    f"pending_frame={tracker_summary['pending_frame_id']} "
                    f"async={observer_summary}"
                )

            native_path = _regular_file(
                baseline_root / f"{scene}{PREDICTION_SUFFIX}",
                "real-score Cbest prediction",
            )
            native_hash = _sha256(native_path)
            native = _load_native_prediction(native_path)
            unique_scores, min_score, max_score = _score_summary(native.rows)
            all_scores.extend(float(row[2]) for row in native.rows)
            output_path = stage / f"{scene}{PREDICTION_SUFFIX}"
            shutil.copyfile(native_path, output_path)
            output_hash = _sha256(output_path)
            if output_hash != native_hash:
                raise C1MaterializationError(f"prediction identity failed: {scene}")

            diagnostic_path = diagnostics_root / f"{scene}.json"
            _write_json(
                diagnostic_path,
                {
                    "schema": SCHEMA + ".scene",
                    "scene_id": scene,
                    "observer_only": True,
                    "output_mutation_applied": False,
                    "query_before_commit": True,
                    "schedule_sha256": schedule_sha,
                    "raw_boxer_csv_sha256": raw_sha,
                    "baseline_prediction_sha256": native_hash,
                    "output_prediction_sha256": output_hash,
                    "selected_observation_count": selected_observations,
                    "tracker": tracker_summary,
                    "async_observer": observer_summary,
                    "results": list(result_rows),
                    "drops": list(drop_rows),
                },
            )
            diagnostic_hash = _sha256(diagnostic_path)
            baseline_hashes[scene] = native_hash
            output_hashes[scene] = output_hash
            diagnostic_hashes[scene] = diagnostic_hash
            scene_reports[scene] = {
                "native_rows": len(native.rows),
                "unique_native_scores": unique_scores,
                "minimum_native_score": min_score,
                "maximum_native_score": max_score,
                "keyframes": tracker_summary["keyframes"],
                "receipts": tracker_summary["receipts"],
                "async_completed": observer_summary["completed_results"],
                "async_dropped": observer_summary["dropped_tasks"],
                "prediction_identity": True,
                "diagnostic_sha256": diagnostic_hash,
            }
            total_rows += len(native.rows)
            total_receipts += int(tracker_summary["receipts"])
            total_submitted += int(observer_summary["submitted_tasks"])
            total_completed += int(observer_summary["completed_results"])
            total_dropped += int(observer_summary["dropped_tasks"])
            print(
                f"[{position}/{len(scenes)}] {scene}: rows={len(native.rows)} "
                f"receipts={tracker_summary['receipts']} "
                f"completed={observer_summary['completed_results']} "
                f"dropped={observer_summary['dropped_tasks']} identity=1",
                flush=True,
            )

        if not all_scores or len(set(all_scores)) <= 1:
            raise C1MaterializationError("baseline does not contain preserved real scores")
        manifest = {
            "schema": SCHEMA,
            "mode": "c1_output_inert_causal_async_observer",
            "scene_count": len(scenes),
            "prediction_rows": total_rows,
            "unique_real_scores": len(set(all_scores)),
            "minimum_real_score": min(all_scores),
            "maximum_real_score": max(all_scores),
            "causal_receipts": total_receipts,
            "async_submitted_tasks": total_submitted,
            "async_completed_results": total_completed,
            "async_dropped_tasks": total_dropped,
            "observer_only": True,
            "output_inert": True,
            "native_predictions_byte_identical": True,
            "output_mutation_applied": False,
            "geometry_changed": False,
            "score_changed": False,
            "label_changed": False,
            "row_order_changed": False,
            "row_count_changed": False,
            "birth": False,
            "overlay": False,
            "training_free": True,
            "online_learning": False,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "past_only": True,
            "query_before_commit": True,
            "asynchronous_scheduler": config.as_dict(),
            "inputs": {
                "scene_list": os.fspath(scene_list),
                "scene_list_sha256": scene_list_sha,
                "baseline_root": os.fspath(baseline_root),
                "raw_log_root": os.fspath(raw_log_root),
                "schedule_root": os.fspath(schedule_root),
                "scene_rgbd_root": os.fspath(scene_rgbd_root),
                "memory_module": os.fspath((ROOT / "boxfusion/s3r_receipt_tracker.py").resolve()),
                "memory_module_sha256": _sha256(ROOT / "boxfusion/s3r_receipt_tracker.py"),
                "async_module": os.fspath((ROOT / "boxfusion/causal_async_observer.py").resolve()),
                "async_module_sha256": _sha256(ROOT / "boxfusion/causal_async_observer.py"),
                "materializer": os.fspath(Path(__file__).resolve()),
                "materializer_sha256": _sha256(Path(__file__).resolve()),
            },
            "baseline_prediction_sha256": baseline_hashes,
            "output_prediction_sha256": output_hashes,
            "diagnostic_sha256": diagnostic_hashes,
            "scenes": scene_reports,
        }
        _write_json(stage / MANIFEST_NAME, manifest)
        if output_root.exists() or output_root.is_symlink():
            raise C1MaterializationError(f"refusing existing output root: {output_root}")
        os.rename(stage, output_root)
        return manifest
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-list", type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
    parser.add_argument("--expected-scene-count", type=int, default=100)
    parser.add_argument(
        "--baseline-root", type=Path,
        default=ROOT / "results/scannet_t05_boxer_replay_active_score05",
    )
    parser.add_argument(
        "--raw-log-root", type=Path,
        default=ROOT / "logs/scannet_raw_boxer_full100_score05_v1",
    )
    parser.add_argument(
        "--schedule-root", type=Path,
        default=Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_boxer_dev/cache/cutr_proposals/"
            "scannet-score05-gap25-postfilter-v2"
        ),
    )
    parser.add_argument(
        "--scene-rgbd-root", type=Path,
        default=ROOT / "upstream_clean/scannet_readme_frames",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results/scannet_cbest_real_score_c1_causal_async_observer_score05",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = materialize(args)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "schema", "scene_count", "prediction_rows",
                    "unique_real_scores", "causal_receipts",
                    "async_completed_results", "async_dropped_tasks",
                    "native_predictions_byte_identical",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
