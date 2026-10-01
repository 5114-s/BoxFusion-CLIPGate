#!/usr/bin/env python3
"""Materialize past-only Diverse-TopK SAM3+depth+CLIP high-confidence births.

Candidate generation, native novelty and the three-view receipt are consumed
from the frozen Raw-Boxer Past3 v2 audit.  Only candidates that reached its
pre-NMS insertion point are eligible.  This program adds a causal SAM3
mask/depth confirmation using frames no later than the receipt confirmation,
then applies the frozen CLIP-vocabulary decision, support counting, self-NMS
and a two-birth scene cap.  It has no annotation or evaluator API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if os.fspath(ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(ROOT))

from boxfusion.s3r_receipt_tracker import S3RReceipt, S3RReceiptTracker  # noqa: E402
from boxfusion.sam3_diverse_maskdepth_birth import (  # noqa: E402
    SAM3BirthConfig,
    SAM3TeacherView,
    confirm_candidate,
)
from tools.materialize_scannet_raw_boxer_past3_birth_full100 import (  # noqa: E402
    APPENDED_SCORE,
    CSV_NAME,
    CSV_RELATIVE,
    PREDICTION_SUFFIX,
    TOP_K_PER_FRAME,
    V2_MAX_BIRTHS_PER_SCENE,
    V2_SCHEMA,
    V2_SELF_NMS_AABB_IOU,
    V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT,
    _aabb_overlap_matrices,
    _assert_native_prefix,
    _augmented_payload,
    _completion_rows,
    _load_clip_gate_sidecar,
    _load_native_prediction,
    _load_raw_observations,
    _regular_file,
    _scene_list,
    _valid_schedule,
    _write_json,
    _write_pickle,
    wait_for_raw_completion,
)


SCHEMA = "boxfusion.scannet_sam3_diverse_clip_birth_full100.v1"
TEACHER_SCHEMA = "boxfusion_scannet_sam3_teacher_provenance_v1"
RUNTIME_SCHEMA = "boxfusion_scannet_runtime_rgb_v1"
ELIGIBLE_V2_DECISIONS = frozenset(("accepted", "self_nms", "scene_cap"))
MANIFEST_NAME = "SAM3_DIVERSE_CLIP_BIRTH_FULL100.json"


class MaterializationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise MaterializationError(f"JSON must be a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MaterializationError(f"JSON root must be an object: {path}")
    return value


def _load_matrix(path: Path) -> np.ndarray:
    value = np.loadtxt(path, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise MaterializationError(f"expected a finite 4x4 matrix: {path}")
    return value


def _teacher_views(
    teacher_root: Path,
    scenes: Sequence[str],
    scene_list_sha256: str,
) -> tuple[dict[str, tuple[SAM3TeacherView, ...]], dict[str, Any]]:
    teacher_paths = sorted((teacher_root / "manifests").glob("provenance_*.json"))
    runtime_paths = sorted(
        (teacher_root / "runtime_rgb" / "manifests").glob("runtime_rgb_*.json")
    )
    if not teacher_paths or not runtime_paths:
        raise MaterializationError("SAM3 teacher provenance/runtime manifests are missing")
    teachers = [_read_json(path) for path in teacher_paths]
    runtimes = [_read_json(path) for path in runtime_paths]
    for payload in teachers:
        if payload.get("schema") != TEACHER_SCHEMA:
            raise MaterializationError("unsupported SAM3 teacher manifest")
    for payload in runtimes:
        if payload.get("schema") != RUNTIME_SCHEMA or payload.get("complete") is not True:
            raise MaterializationError("unsupported or incomplete SAM3 runtime manifest")
    namespaces = {str(row.get("namespace")) for row in teachers + runtimes}
    if len(namespaces) != 1:
        raise MaterializationError("SAM3 teacher namespace mismatch")
    for payload in teachers + runtimes:
        if str(payload["scene_list"]["sha256"]) != scene_list_sha256:
            raise MaterializationError("SAM3 cache belongs to a different scene list")

    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    for payload in runtimes:
        for row in payload["frames"]:
            if row.get("orientation") != "upright":
                raise MaterializationError("SAM3 cache frame is not upright")
            rows_by_scene.setdefault(str(row["scene_id"]), []).append(row)
    missing = sorted(set(scenes) - set(rows_by_scene))
    if missing:
        raise MaterializationError(
            f"SAM3 frame coverage is missing requested scenes: {missing}"
        )

    result: dict[str, tuple[SAM3TeacherView, ...]] = {}
    for scene in scenes:
        scene_rows = sorted(rows_by_scene[scene], key=lambda row: int(row["frame_index"]))
        first_depth = Path(scene_rows[0]["sources"]["depth"]["path"])
        scan_root = first_depth.parents[1]
        if scan_root.name != scene:
            raise MaterializationError(f"malformed SAM3 source root for {scene}")
        intrinsic = _load_matrix(scan_root / "intrinsic" / "intrinsic_depth.txt")
        extrinsic = _load_matrix(scan_root / "intrinsic" / "extrinsic_depth.txt")
        views: list[SAM3TeacherView] = []
        for row in scene_rows:
            frame_id = int(row["frame_index"])
            pose_path = Path(row["sources"]["pose"]["path"])
            depth_path = Path(row["sources"]["depth"]["path"])
            if _sha256(pose_path) != str(row["sources"]["pose"]["sha256"]):
                raise MaterializationError(f"pose hash mismatch: {scene}/{frame_id}")
            if _sha256(depth_path) != str(row["sources"]["depth"]["sha256"]):
                raise MaterializationError(f"depth hash mismatch: {scene}/{frame_id}")
            pose = np.loadtxt(pose_path, dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                continue
            key = str(row["proposal_cache_key"])
            proposal_path = teacher_root / (hashlib.sha256(key.encode()).hexdigest() + ".npz")
            if proposal_path.is_symlink() or not proposal_path.is_file():
                raise MaterializationError(f"missing SAM3 proposal cache: {proposal_path}")
            shape = tuple(int(value) for value in row["shape"][:2])
            views.append(
                SAM3TeacherView(
                    frame_id=frame_id,
                    intrinsics=intrinsic,
                    camera_to_world=pose @ extrinsic,
                    depth_path=depth_path,
                    proposal_path=proposal_path,
                    image_shape=shape,
                )
            )
        if not views:
            raise MaterializationError(f"no valid SAM3 views for {scene}")
        result[scene] = tuple(views)
    provenance = {
        "namespace": namespaces.pop(),
        "teacher_manifests": {os.fspath(path): _sha256(path) for path in teacher_paths},
        "runtime_manifests": {os.fspath(path): _sha256(path) for path in runtime_paths},
        "teacher_root": os.fspath(teacher_root),
    }
    return result, provenance


def _replay_receipts(
    scene: str,
    schedule: Sequence[int],
    by_frame: Mapping[int, Sequence[Any]],
) -> dict[int, S3RReceipt]:
    tracker = S3RReceiptTracker()
    for frame_id in schedule:
        selected = tuple(
            sorted(
                by_frame.get(frame_id, ()),
                key=lambda row: (-row.score, row.source_row, row.sealed_npz_row),
            )[:TOP_K_PER_FRAME]
        )
        query = tracker.query(frame_id, selected)
        commit = tracker.commit(query)
        if (
            query.selected_source_rows != tuple(row.source_row for row in selected)
            or query.observation_capacity_dropped_source_rows
            or not query.audit_complete
            or not commit.audit_complete
        ):
            raise MaterializationError(f"Past3 replay was incomplete for {scene}/{frame_id}")
    if not tracker.summary()["audit_complete"]:
        raise MaterializationError(f"Past3 tracker audit failed for {scene}")
    return {receipt.track_id: receipt for receipt in tracker.receipts()}


def _eligible_candidates(
    scene: str,
    v2_scene: Mapping[str, Any],
    receipts: Mapping[int, S3RReceipt],
) -> list[tuple[S3RReceipt, Mapping[str, Any]]]:
    decisions = v2_scene.get("receipt_decisions")
    if not isinstance(decisions, list):
        raise MaterializationError(f"missing v2 receipt decisions for {scene}")
    result = []
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get("decision") not in ELIGIBLE_V2_DECISIONS:
            continue
        track_id = int(decision["track_id"])
        receipt = receipts.get(track_id)
        if receipt is None:
            raise MaterializationError(f"missing replayed receipt {scene}/{track_id}")
        if tuple(decision["evidence_source_rows"]) != receipt.evidence_source_rows:
            raise MaterializationError(f"receipt identity mismatch {scene}/{track_id}")
        result.append((receipt, decision))
    return result


def _select_scene_births(
    scene: str,
    candidates: Sequence[tuple[S3RReceipt, Mapping[str, Any]]],
    views: Sequence[SAM3TeacherView],
    clip_records: Mapping[tuple[int, tuple[int, int, int]], Mapping[str, Any]],
    config: SAM3BirthConfig,
) -> tuple[tuple[S3RReceipt, ...], list[dict[str, Any]]]:
    selected: list[S3RReceipt] = []
    decisions: list[dict[str, Any]] = []
    for receipt, v2_decision in candidates:
        mask_depth = confirm_candidate(
            receipt.corners, receipt.confirmation_frame_id, views, config
        )
        key = (receipt.track_id, tuple(receipt.evidence_source_rows))
        clip_record = clip_records.get(key)
        if clip_record is None:
            raise MaterializationError(f"missing CLIP receipt {scene}/{receipt.track_id}")
        decision = "accepted"
        if mask_depth["mask_depth_pass"] is not True:
            decision = "mask_depth"
        elif clip_record.get("gate_pass") is not True:
            decision = "clip_gate"
        elif selected:
            current = np.asarray(receipt.corners, dtype=np.float64)[None]
            prior = np.stack([item.corners for item in selected])
            iou, current_in_prior, prior_in_current = _aabb_overlap_matrices(current, prior)
            if (
                float(iou.max()) >= V2_SELF_NMS_AABB_IOU
                or float(current_in_prior.max())
                >= V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                or float(prior_in_current.max())
                >= V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT
            ):
                decision = "self_nms"
        if decision == "accepted" and len(selected) >= V2_MAX_BIRTHS_PER_SCENE:
            decision = "scene_cap"
        if decision == "accepted":
            selected.append(receipt)
        decisions.append(
            {
                "track_id": receipt.track_id,
                "decision": decision,
                "confirmation_frame_id": receipt.confirmation_frame_id,
                "evidence_frame_ids": list(receipt.evidence_frame_ids),
                "evidence_source_rows": list(receipt.evidence_source_rows),
                "v2_pre_nms_decision": v2_decision["decision"],
                "sam3_mask_depth": mask_depth,
                "clip_gate_pass": bool(clip_record.get("gate_pass") is True),
                "clip_summary": clip_record.get("clip_summary"),
            }
        )
    return tuple(selected), decisions


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    scene_list = args.scene_list.resolve()
    scenes = _scene_list(scene_list, args.expected_scene_count)
    scene_list_sha = _sha256(scene_list)
    teacher_scene_list = args.teacher_scene_list.resolve()
    teacher_scenes = _scene_list(teacher_scene_list, 100)
    teacher_scene_list_sha = _sha256(teacher_scene_list)
    if not set(scenes).issubset(teacher_scenes):
        raise MaterializationError("requested scenes are outside the SAM3 teacher universe")
    baseline_root = args.baseline_root.resolve()
    raw_log_root = args.raw_log_root.resolve()
    schedule_root = args.schedule_root.resolve()
    scene_rgbd_root = args.scene_rgbd_root.resolve()
    teacher_root = args.teacher_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise MaterializationError(f"refusing to overwrite output root: {output_root}")
    for path, label in (
        (baseline_root, "baseline root"), (raw_log_root, "raw log root"),
        (schedule_root, "schedule root"), (scene_rgbd_root, "RGB-D root"),
        (teacher_root, "SAM3 teacher root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise MaterializationError(f"{label} must be a regular directory: {path}")

    v2_manifest = _read_json(args.v2_manifest.resolve())
    if (
        v2_manifest.get("schema") != V2_SCHEMA
        or v2_manifest.get("gt_access") is not False
        or v2_manifest.get("evaluator_access") is not False
        or v2_manifest.get("past_only_confirmation") is not True
        or int(v2_manifest.get("scene_count", -1)) != 100
    ):
        raise MaterializationError("v2 candidate manifest contract mismatch")
    clip = _load_clip_gate_sidecar(args.clip_sidecar.resolve())
    config = SAM3BirthConfig()
    teacher_views, teacher_provenance = _teacher_views(
        teacher_root, scenes, teacher_scene_list_sha
    )
    if args.expected_scene_count == 100:
        completions = wait_for_raw_completion(
            log_root=raw_log_root,
            scenes=scenes,
            timeout_seconds=0.0,
            poll_seconds=1.0,
        )
    else:
        all_completions = _completion_rows(raw_log_root)
        missing_completions = sorted(set(scenes) - set(all_completions))
        if missing_completions:
            raise MaterializationError(
                f"missing raw completion rows for smoke scenes: {missing_completions}"
            )
        completions = {scene: all_completions[scene] for scene in scenes}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.stage-", dir=output_root.parent))
    scene_reports: dict[str, Any] = {}
    native_hashes: dict[str, str] = {}
    output_hashes: dict[str, str] = {}
    total_native = total_candidates = total_births = 0
    total_mask_pass = total_clip_pass_after_mask = 0
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
            receipts = _replay_receipts(scene, schedule, by_frame)
            v2_scene = v2_manifest["scenes"].get(scene)
            if not isinstance(v2_scene, dict):
                raise MaterializationError(f"missing v2 scene {scene}")
            candidates = _eligible_candidates(scene, v2_scene, receipts)
            selected, decisions = _select_scene_births(
                scene,
                candidates,
                teacher_views[scene],
                clip.records.get(scene, {}),
                config,
            )

            native_path = _regular_file(
                baseline_root / f"{scene}{PREDICTION_SUFFIX}", "Cbest prediction"
            )
            native_hash = _sha256(native_path)
            expected_hash = v2_manifest["native_prediction_sha256"].get(scene)
            if native_hash != expected_hash:
                raise MaterializationError(f"Cbest/v2 baseline hash mismatch for {scene}")
            native = _load_native_prediction(native_path)
            payload = _augmented_payload(native, selected)
            output_path = stage / f"{scene}{PREDICTION_SUFFIX}"
            _write_pickle(output_path, payload)
            reloaded = _load_native_prediction(output_path)
            _assert_native_prefix(native.rows, reloaded.rows, scene)
            if len(reloaded.rows) != len(native.rows) + len(selected):
                raise MaterializationError(f"birth suffix count changed for {scene}")
            if _sha256(native_path) != native_hash or _sha256(raw_path) != raw_sha:
                raise MaterializationError(f"frozen input changed while processing {scene}")

            native_hashes[scene] = native_hash
            output_hashes[scene] = _sha256(output_path)
            mask_pass = sum(
                row["sam3_mask_depth"]["mask_depth_pass"] is True for row in decisions
            )
            clip_after_mask = sum(
                row["sam3_mask_depth"]["mask_depth_pass"] is True
                and row["clip_gate_pass"] is True
                for row in decisions
            )
            scene_reports[scene] = {
                "native_count": len(native.rows),
                "eligible_candidate_count": len(candidates),
                "mask_depth_pass_count": mask_pass,
                "clip_pass_after_mask_count": clip_after_mask,
                "birth_count": len(selected),
                "schedule_sha256": schedule_sha,
                "raw_boxer_csv_sha256": raw_sha,
                "decisions": decisions,
                "suffix": [
                    {
                        "suffix_index": index,
                        "track_id": receipt.track_id,
                        "score": APPENDED_SCORE,
                        "confirmation_frame_id": receipt.confirmation_frame_id,
                        "evidence_frame_ids": list(receipt.evidence_frame_ids),
                        "evidence_source_rows": list(receipt.evidence_source_rows),
                        "corners_world": np.asarray(receipt.corners).tolist(),
                    }
                    for index, receipt in enumerate(selected)
                ],
            }
            total_native += len(native.rows)
            total_candidates += len(candidates)
            total_births += len(selected)
            total_mask_pass += mask_pass
            total_clip_pass_after_mask += clip_after_mask
            print(
                f"[{position}/{len(scenes)}] {scene}: candidates={len(candidates)} "
                f"mask={mask_pass} clip_after_mask={clip_after_mask} births={len(selected)}",
                flush=True,
            )

        manifest = {
            "schema": SCHEMA,
            "mode": "active_high_confidence_birth",
            "scene_count": len(scenes),
            "native_count": total_native,
            "eligible_candidate_count": total_candidates,
            "mask_depth_pass_count": total_mask_pass,
            "clip_pass_after_mask_count": total_clip_pass_after_mask,
            "birth_count": total_births,
            "training_free": True,
            "target_dataset_training": False,
            "online_learning": False,
            "gt_access": False,
            "annotation_access": False,
            "evaluator_access": False,
            "past_only_confirmation": True,
            "native_rows_are_unchanged_prefix": True,
            "score_mode": "constant_1.0",
            "pipeline": [
                "native-unmatched frozen Raw-Boxer Past3 receipt",
                "past-only Diverse Top-K visible SAM3 teacher views",
                "SAM3 mask-depth consistency",
                "frozen CLIP vocabulary consistency",
                "multi-view strong support count",
                "high-confidence append-only birth",
            ],
            "frozen_policy": {
                "candidate_insertion_point": (
                    "after_v2_score_geometry_semantic_native_novelty_before_self_nms"
                ),
                "eligible_v2_decisions": sorted(ELIGIBLE_V2_DECISIONS),
                "sam3_mask_depth": config.as_dict(),
                "clip_gate_required_value": True,
                "self_nms_aabb_iou_gte_reject": V2_SELF_NMS_AABB_IOU,
                "self_nms_bidirectional_containment_gte_reject": (
                    V2_SELF_NMS_BIDIRECTIONAL_CONTAINMENT
                ),
                "max_births_per_scene": V2_MAX_BIRTHS_PER_SCENE,
                "future_teacher_frames_allowed": False,
            },
            "inputs": {
                "scene_list": os.fspath(scene_list),
                "scene_list_sha256": scene_list_sha,
                "teacher_scene_list": os.fspath(teacher_scene_list),
                "teacher_scene_list_sha256": teacher_scene_list_sha,
                "baseline_root": os.fspath(baseline_root),
                "raw_log_root": os.fspath(raw_log_root),
                "schedule_root": os.fspath(schedule_root),
                "scene_rgbd_root": os.fspath(scene_rgbd_root),
                "v2_manifest": os.fspath(args.v2_manifest.resolve()),
                "v2_manifest_sha256": _sha256(args.v2_manifest.resolve()),
                "clip_sidecar": os.fspath(clip.path),
                "clip_sidecar_sha256": clip.sha256,
                "teacher": teacher_provenance,
                "module": os.fspath(
                    (ROOT / "boxfusion/sam3_diverse_maskdepth_birth.py").resolve()
                ),
                "module_sha256": _sha256(
                    ROOT / "boxfusion/sam3_diverse_maskdepth_birth.py"
                ),
                "materializer": os.fspath(Path(__file__).resolve()),
                "materializer_sha256": _sha256(Path(__file__).resolve()),
            },
            "native_prediction_sha256": native_hashes,
            "output_prediction_sha256": output_hashes,
            "scenes": scene_reports,
        }
        _write_json(stage / MANIFEST_NAME, manifest)
        if output_root.exists() or output_root.is_symlink():
            raise MaterializationError(f"refusing existing output root: {output_root}")
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
        "--teacher-scene-list", type=Path,
        default=ROOT / "evaluation/data_util/meta_data/scannetv2_val.txt",
    )
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
        "--teacher-root", type=Path,
        default=Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_maskgraph_dev/cache/sam3_teacher/"
            "sam3_teacher_full100_c050_frozen_v1"
        ),
    )
    parser.add_argument(
        "--v2-manifest", type=Path,
        default=(
            ROOT / "results/scannet_cbest_raw_boxer_past3_birth_v2_m50_score05/"
            "RAW_BOXER_PAST3_BIRTH_FULL100.json"
        ),
    )
    parser.add_argument(
        "--clip-sidecar", type=Path,
        default=(
            ROOT / "logs/scannet_cbest_raw_boxer_clip_vocab_shadow_score05/"
            "CLIP_VOCAB_SHADOW_FULL100.json"
        ),
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "results/scannet_cbest_sam3_diverse_clip_birth_score05",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = materialize(args)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "scene_count": result["scene_count"],
                "native_count": result["native_count"],
                "eligible_candidate_count": result["eligible_candidate_count"],
                "mask_depth_pass_count": result["mask_depth_pass_count"],
                "clip_pass_after_mask_count": result["clip_pass_after_mask_count"],
                "birth_count": result["birth_count"],
                "output_root": os.fspath(args.output_root.resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
