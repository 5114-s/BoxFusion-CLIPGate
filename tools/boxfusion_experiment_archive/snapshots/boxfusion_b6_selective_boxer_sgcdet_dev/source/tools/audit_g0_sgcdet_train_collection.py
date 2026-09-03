#!/usr/bin/env python3
"""Fail-closed audit for G0+B6 train-only sparse-refiner diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCENE_RE = re.compile(r"scene\d{4}_\d{2}")
EXPECTED_B6_SHA256 = (
    "d60abf798edbfa3d7902b42651be7d6053727948f740e05795de6feed60a7071"
)
EXPECTED_YOLOE_SHA256 = (
    "292bdf157a9ec7315f34b567cb93467c5043cd1889a1cc18abbfdeb88d7a948d"
)
EXPECTED_BOXER_SHA256 = (
    "d5a30b348a8f5b0e5990ff3aa0e8f473ce77d860da22586322e7f47abc83ca6f"
)
EXPECTED_BOXER_COMMIT = "1f86542dc342a4b1d474c87c97c5d1d6566d9148"
EXPECTED_DINO_SHA256 = (
    "4057cbaaad8c16657adb09d6815f28d4164eeba30532fde23f0d17313124caea"
)
EXPECTED_CONFIG_SHA256 = (
    "54c4e7686edfc0ecd7bbe1e21e7fba79063e6ea52dedf39ba1dbc95a127d6b36"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_scenes(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    scenes = [
        line.strip().split()[0]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not scenes or len(scenes) != len(set(scenes)):
        raise ValueError(f"scene list is empty or contains duplicates: {path}")
    invalid = [scene for scene in scenes if SCENE_RE.fullmatch(scene) is None]
    if invalid:
        raise ValueError(f"invalid ScanNet scene ids: {invalid[:5]}")
    return scenes


def scalar(array: Any, name: str) -> Any:
    value = np.asarray(array)
    if value.ndim != 0 or value.dtype.hasobject:
        raise TypeError(f"{name} must be a safe scalar")
    result = value.item()
    if isinstance(result, bytes):
        result = result.decode("utf-8")
    return result


def load_prediction(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f"unexpected prediction container: {path}")
    rows = payload[0]
    if not isinstance(rows, list):
        raise TypeError(f"prediction rows must be a list: {path}")
    labels: list[int] = []
    corners: list[np.ndarray] = []
    scores: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, tuple) or len(row) != 3:
            raise TypeError(f"{path}: prediction row {index} is malformed")
        label, raw_corners, raw_score = row
        if isinstance(label, bool) or not isinstance(label, (int, np.integer)):
            raise TypeError(f"{path}: prediction row {index} label is not integer")
        corner_array = np.asarray(raw_corners)
        if corner_array.shape != (8, 3) or corner_array.dtype != np.float32:
            raise TypeError(
                f"{path}: prediction row {index} corners must be float32 (8, 3)"
            )
        if not np.isfinite(corner_array).all():
            raise ValueError(f"{path}: prediction row {index} corners are non-finite")
        score = float(raw_score)
        if not np.isfinite(score):
            raise ValueError(f"{path}: prediction row {index} score is non-finite")
        labels.append(int(label))
        corners.append(corner_array.copy())
        scores.append(score)
    label_array = np.asarray(labels, dtype=np.int64)
    corner_array = (
        np.stack(corners).astype(np.float32, copy=False)
        if corners
        else np.empty((0, 8, 3), dtype=np.float32)
    )
    score_array = np.asarray(scores, dtype=np.float32)
    if score_array.ndim != 1 or not np.isfinite(score_array).all():
        raise ValueError(f"invalid prediction scores: {path}")
    return label_array, corner_array, score_array


def combined_digest(paths: Iterable[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def require_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name}={actual!r}, expected {expected!r}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    scenes = read_scenes(args.scene_list)
    forbidden = set(read_scenes(args.forbidden_scene_list))
    overlap = sorted(set(scenes) & forbidden)
    if overlap:
        raise ValueError(f"validation leakage: {overlap[:5]}")

    for path in (
        args.prediction_root,
        args.diagnostics_root,
        args.boxer_diagnostics_root,
        args.log_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    for path, expected, label in (
        (args.config, EXPECTED_CONFIG_SHA256, "combined config"),
        (args.b6_checkpoint, EXPECTED_B6_SHA256, "B6 checkpoint"),
        (args.yoloe_checkpoint, EXPECTED_YOLOE_SHA256, "YOLOE checkpoint"),
    ):
        require_equal(f"{label} SHA256", sha256_file(path), expected)

    import yaml

    with args.config.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    lifting = config["lifting"]
    boxer = lifting["boxer"]
    gate = boxer["selective_gate"]
    require_equal("detection.score_thresh", config["detection"]["score_thresh"], 0.4)
    require_equal("lifting.backend", lifting["backend"], "boxer")
    require_equal("boxer.mode", boxer["mode"], "active")
    require_equal("boxer.apply_stage", boxer["apply_stage"], "post_filter")
    require_equal("boxer.expected_commit", boxer["expected_commit"], EXPECTED_BOXER_COMMIT)
    require_equal("boxer.checkpoint_sha256", boxer["checkpoint_sha256"], EXPECTED_BOXER_SHA256)
    require_equal("boxer.dinov3_sha256", boxer["dinov3_sha256"], EXPECTED_DINO_SHA256)
    require_equal("selective_gate.enabled", gate["enabled"], True)
    require_equal("selective_gate.max_center_shift_m", gate["max_center_shift_m"], 0.10)
    require_equal("selective_gate.min_volume_ratio", gate["min_volume_ratio"], 0.50)
    require_equal("selective_gate.max_volume_ratio", gate["max_volume_ratio"], 2.00)

    driver_log = args.log_root / "driver.log"
    text = driver_log.read_text(encoding="utf-8")
    for token in (
        "Online ablation profile: sgcdet_sparse_observer",
        "Proposal cache: disabled",
        "Skip evaluation: 1",
        "Selective Boxer gate: center<=0.10 m; volume=[0.50,2.00]",
    ):
        if token not in text:
            raise ValueError(f"driver log is missing frozen contract: {token}")

    prediction_paths: list[Path] = []
    diagnostic_paths: list[Path] = []
    boxer_paths: list[Path] = []
    total_rows = 0
    sparse_rows = 0
    boxer_calls = 0
    boxer_inference_calls = 0
    boxer_empty_calls = 0
    identity_rows = 0
    for scene in scenes:
        prediction = args.prediction_root / f"{scene}_boxes.pkl"
        diagnostic = args.diagnostics_root / f"{scene}_tracks.npz"
        boxer_path = args.boxer_diagnostics_root / f"{scene}_boxer_lifting.jsonl"
        for path in (prediction, diagnostic, boxer_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        prediction_labels, prediction_corners, scores = load_prediction(prediction)
        total_rows += int(scores.size)
        with np.load(diagnostic, allow_pickle=False) as archive:
            required = {
                "online_ablation_profile",
                "mutation_quality_enabled",
                "mutation_refit_enabled",
                "mutation_box_refiner_enabled",
                "mutation_sparse_geometry_enabled",
                "mutation_supplemental_output_enabled",
                "mutation_soft_nms_enabled",
                "box_refiner_coordinate_frame",
                "candidate_ttl_clock",
                "candidate_track_ttl",
                "archive_confirmed_tracks",
                "top_k_views",
                "sparse_collect_diagnostics",
                "sparse_pair_source_indices",
                "sparse_points_local",
                "sparse_point_mask",
                "sparse_view_features",
                "sparse_view_mask",
                "sparse_local_boxes",
                "sparse_quality_features",
                "sparse_final_b6_scores",
                "joint_points_local",
                "joint_point_mask",
                "joint_view_features",
                "joint_view_mask",
                "joint_local_boxes",
                "joint_quality_features",
                "result_indices",
                "source_indices",
                "track_ids",
                "sparse_original_corners",
                "sparse_active_corners",
                "refit_applied",
                "labels",
                "output_geometry_schema",
                "output_pre_geometry_boxes",
                "output_pre_geometry_corners",
                "output_post_geometry_boxes",
                "output_post_geometry_corners",
                "output_source_indices",
                "output_stable_ids",
                "output_refit_applied",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"{diagnostic} missing fields: {sorted(missing)}")
            expected_scalars = {
                "online_ablation_profile": "sgcdet_sparse_observer",
                "mutation_quality_enabled": True,
                "mutation_refit_enabled": False,
                "mutation_box_refiner_enabled": False,
                "mutation_sparse_geometry_enabled": False,
                "mutation_supplemental_output_enabled": False,
                "mutation_soft_nms_enabled": False,
                "box_refiner_coordinate_frame": "world_aabb",
                "candidate_ttl_clock": "provider_call",
                "candidate_track_ttl": 3,
                "archive_confirmed_tracks": False,
                "top_k_views": 5,
                "sparse_collect_diagnostics": True,
            }
            for name, expected in expected_scalars.items():
                require_equal(f"{scene}:{name}", scalar(archive[name], name), expected)
            result_indices = np.asarray(archive["result_indices"])
            if (
                result_indices.ndim != 1
                or not np.issubdtype(result_indices.dtype, np.integer)
                or len(np.unique(result_indices)) != len(result_indices)
                or (result_indices < 0).any()
                or (result_indices >= len(scores)).any()
                or (
                    len(result_indices) > 1
                    and not np.all(np.diff(result_indices) > 0)
                )
            ):
                raise ValueError(f"{scene}: invalid result_indices")
            final_scores = np.asarray(archive["sparse_final_b6_scores"])
            if not np.array_equal(final_scores, scores[result_indices]):
                raise ValueError(f"{scene}: sparse_final_b6_scores disagree with predictions")

            # Prove observer identity inside the same execution.  Comparing two
            # independent GPU runs is not byte-stable, so the runtime records the
            # complete output immediately before and after the disabled geometry
            # mutation point.  Every final output row, including rows not sampled
            # for sparse training, must be exact identity and must be the exported
            # prediction in the same order.
            require_equal(
                f"{scene}:output_geometry_schema",
                scalar(archive["output_geometry_schema"], "output_geometry_schema"),
                "boxfusion.full_output_geometry_prepost.v1",
            )
            full_arrays = {
                name: np.asarray(archive[name])
                for name in (
                    "output_pre_geometry_boxes",
                    "output_pre_geometry_corners",
                    "output_post_geometry_boxes",
                    "output_post_geometry_corners",
                    "output_source_indices",
                    "output_stable_ids",
                    "output_refit_applied",
                )
            }
            expected_shapes = {
                "output_pre_geometry_boxes": (len(scores), 6),
                "output_pre_geometry_corners": (len(scores), 8, 3),
                "output_post_geometry_boxes": (len(scores), 6),
                "output_post_geometry_corners": (len(scores), 8, 3),
                "output_source_indices": (len(scores),),
                "output_stable_ids": (len(scores),),
                "output_refit_applied": (len(scores),),
            }
            expected_dtypes = {
                "output_pre_geometry_boxes": np.dtype(np.float32),
                "output_pre_geometry_corners": np.dtype(np.float32),
                "output_post_geometry_boxes": np.dtype(np.float32),
                "output_post_geometry_corners": np.dtype(np.float32),
                "output_source_indices": np.dtype(np.int64),
                "output_stable_ids": np.dtype(np.int64),
                "output_refit_applied": np.dtype(np.bool_),
            }
            for name, array in full_arrays.items():
                if array.shape != expected_shapes[name] or array.dtype != expected_dtypes[name]:
                    raise TypeError(
                        f"{scene}:{name} shape/dtype={array.shape}/{array.dtype}, "
                        f"expected {expected_shapes[name]}/{expected_dtypes[name]}"
                    )
            for before, after in (
                ("output_pre_geometry_boxes", "output_post_geometry_boxes"),
                ("output_pre_geometry_corners", "output_post_geometry_corners"),
            ):
                if not np.array_equal(full_arrays[before], full_arrays[after], equal_nan=True):
                    raise ValueError(f"{scene}: observer changed full output geometry")
            if not np.array_equal(
                full_arrays["output_post_geometry_corners"],
                prediction_corners,
                equal_nan=True,
            ):
                raise ValueError(f"{scene}: exported prediction order/geometry drifted")
            sources = full_arrays["output_source_indices"]
            stable_ids = full_arrays["output_stable_ids"]
            if (
                (sources < 0).any()
                or len(np.unique(sources)) != len(sources)
                or (len(sources) > 1 and not np.all(np.diff(sources) > 0))
            ):
                raise ValueError(f"{scene}: output source order is not strict identity")
            if len(np.unique(stable_ids)) != len(stable_ids):
                raise ValueError(f"{scene}: duplicate output stable ids")
            if full_arrays["output_refit_applied"].any():
                raise ValueError(f"{scene}: observer applied a full-output geometry mutation")

            observed_sources = np.asarray(archive["source_indices"])
            observed_ids = np.asarray(archive["track_ids"])
            observed_labels = np.asarray(archive["labels"])
            if observed_labels.shape != result_indices.shape or observed_labels.dtype.hasobject:
                raise TypeError(f"{scene}: observed label sequence is unsafe")
            for full_name, observed in (
                ("output_pre_geometry_corners", archive["sparse_original_corners"]),
                ("output_post_geometry_corners", archive["sparse_active_corners"]),
                ("output_source_indices", observed_sources),
                ("output_stable_ids", observed_ids),
                ("output_refit_applied", archive["refit_applied"]),
            ):
                if not np.array_equal(
                    full_arrays[full_name][result_indices],
                    np.asarray(observed),
                    equal_nan=True,
                ):
                    raise ValueError(f"{scene}: {full_name} observed-row mapping drifted")
            # Force label parsing/count validation even though class labels are
            # not mutated by this geometry-only observer profile.
            if prediction_labels.shape != (len(scores),):
                raise ValueError(f"{scene}: exported prediction labels are malformed")
            identity_rows += int(len(scores))
            for sparse_name, joint_name in (
                ("sparse_points_local", "joint_points_local"),
                ("sparse_point_mask", "joint_point_mask"),
                ("sparse_view_features", "joint_view_features"),
                ("sparse_view_mask", "joint_view_mask"),
                ("sparse_quality_features", "joint_quality_features"),
            ):
                if not np.array_equal(archive[sparse_name], archive[joint_name]):
                    raise ValueError(f"{scene}: {sparse_name} is not runtime-exact")
            if not np.array_equal(
                archive["sparse_local_boxes"],
                archive["joint_local_boxes"],
                equal_nan=True,
            ):
                raise ValueError(f"{scene}: sparse/joint local boxes disagree")
            sparse_rows += int(len(result_indices))

        with boxer_path.open("r", encoding="utf-8") as handle:
            scene_calls = 0
            scene_diagnostic_keys: set[tuple[int, str]] = set()
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                require_equal(
                    f"{scene}:{line_number}:schema",
                    row.get("schema"),
                    "boxfusion.boxer_lifting.frame.v1",
                )
                require_equal(f"{scene}:{line_number}:scene_id", row["scene_id"], scene)
                if row.get("attempt_id") not in {"primary", "retry"}:
                    raise ValueError(
                        f"{scene}:{line_number}: invalid Boxer attempt_id"
                    )
                frame_id = row.get("frame_id")
                if (
                    isinstance(frame_id, bool)
                    or not isinstance(frame_id, int)
                    or frame_id < 0
                ):
                    raise TypeError(
                        f"{scene}:{line_number}:frame_id must be a non-negative integer"
                    )
                diagnostic_key = (frame_id, row["attempt_id"])
                if diagnostic_key in scene_diagnostic_keys:
                    raise ValueError(
                        f"{scene}:{line_number}: duplicate Boxer diagnostic key "
                        f"{diagnostic_key}"
                    )
                scene_diagnostic_keys.add(diagnostic_key)
                require_equal(f"{scene}:{line_number}:mode", row["mode"], "active")
                require_equal(f"{scene}:{line_number}:apply_stage", row["apply_stage"], "post_filter")
                if row.get("mutation_enabled") is not True:
                    raise TypeError(
                        f"{scene}:{line_number}:mutation_enabled must be Boolean true"
                    )
                if row.get("selective_gate_enabled") is not True:
                    raise TypeError(
                        f"{scene}:{line_number}:selective_gate_enabled must be Boolean true"
                    )
                require_equal(f"{scene}:{line_number}:selective_gate", row["selective_gate"], {
                    "max_center_shift_m": 0.1,
                    "max_volume_ratio": 2.0,
                    "min_volume_ratio": 0.5,
                })
                count_names = (
                    "count",
                    "eligible_count",
                    "applied_count",
                    "fallback_count",
                )
                for name in count_names:
                    value = row.get(name)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise TypeError(
                            f"{scene}:{line_number}:{name} must be a non-negative integer"
                        )
                proposal_count = row["count"]
                if row["eligible_count"] + row["fallback_count"] != proposal_count:
                    raise ValueError(
                        f"{scene}:{line_number}: selective Boxer counts do not partition proposals"
                    )
                if row["applied_count"] != row["eligible_count"]:
                    raise ValueError(
                        f"{scene}:{line_number}: active Boxer did not apply every eligible proposal"
                    )
                if proposal_count == 0:
                    # The lifter intentionally returns before model inference for
                    # an empty CuTR proposal set, so frame rows of this subtype do
                    # not carry runtime Boxer checkpoint/commit fields.  The
                    # frozen config above proves the declared backend; at least
                    # one non-empty row in this same scene file is required below
                    # to prove runtime checkpoint/commit provenance.
                    for name, expected in (
                        ("boxer_commit", EXPECTED_BOXER_COMMIT),
                        ("boxer_checkpoint_sha256", EXPECTED_BOXER_SHA256),
                    ):
                        if name in row:
                            require_equal(
                                f"{scene}:{line_number}:{name}",
                                row[name],
                                expected,
                            )
                    boxer_empty_calls += 1
                else:
                    require_equal(
                        f"{scene}:{line_number}:boxer_commit",
                        row.get("boxer_commit"),
                        EXPECTED_BOXER_COMMIT,
                    )
                    require_equal(
                        f"{scene}:{line_number}:boxer_checkpoint_sha256",
                        row.get("boxer_checkpoint_sha256"),
                        EXPECTED_BOXER_SHA256,
                    )
                    boxer_inference_calls += 1
                scene_calls += 1
            if scene_calls == 0:
                raise ValueError(f"{scene}: Boxer diagnostics contain no calls")
            boxer_calls += scene_calls
        prediction_paths.append(prediction)
        diagnostic_paths.append(diagnostic)
        boxer_paths.append(boxer_path)

    if boxer_inference_calls == 0:
        raise ValueError("collection contains no actual Boxer inference calls")
    if total_rows == 0 or sparse_rows == 0:
        raise ValueError("collection contains no usable prediction/sparse rows")

    expected_prediction_names = {f"{scene}_boxes.pkl" for scene in scenes}
    actual_prediction_names = {path.name for path in args.prediction_root.glob("scene*_boxes.pkl")}
    if actual_prediction_names != expected_prediction_names:
        raise ValueError("prediction root has a missing or extra scene")

    report = {
        "schema": "boxfusion.g0_sgcdet_train_collection_manifest.v1",
        "scene_count": len(scenes),
        "scene_list": str(args.scene_list.resolve()),
        "scene_list_sha256": sha256_file(args.scene_list),
        "forbidden_scene_list": str(args.forbidden_scene_list.resolve()),
        "forbidden_scene_list_sha256": sha256_file(args.forbidden_scene_list),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "b6_checkpoint_sha256": sha256_file(args.b6_checkpoint),
        "yoloe_checkpoint_sha256": sha256_file(args.yoloe_checkpoint),
        "boxer_checkpoint_sha256": EXPECTED_BOXER_SHA256,
        "boxer_commit": EXPECTED_BOXER_COMMIT,
        "dinov3_sha256": EXPECTED_DINO_SHA256,
        "profile": "sgcdet_sparse_observer",
        "proposal_cache_mode": "disabled",
        "skip_evaluation": True,
        "score_thresh": 0.4,
        "minimum_extent": 0.4,
        "b6_detector_blend": 0.4,
        "g0_gate": {
            "max_center_shift_m": 0.1,
            "min_volume_ratio": 0.5,
            "max_volume_ratio": 2.0,
        },
        "prediction_rows": total_rows,
        "sparse_rows": sparse_rows,
        "full_output_identity_rows": identity_rows,
        "full_output_identity_scenes": len(scenes),
        "boxer_calls": boxer_calls,
        "boxer_inference_calls": boxer_inference_calls,
        "boxer_empty_calls": boxer_empty_calls,
        "prediction_bundle_sha256": combined_digest(prediction_paths, args.prediction_root),
        "diagnostic_bundle_sha256": combined_digest(diagnostic_paths, args.diagnostics_root),
        "boxer_bundle_sha256": combined_digest(boxer_paths, args.boxer_diagnostics_root),
    }
    if args.verify_existing:
        if not args.output.is_file():
            raise FileNotFoundError(f"immutable collection manifest is missing: {args.output}")
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing != report:
            raise ValueError(
                "immutable collection manifest disagrees with the current artifacts; "
                "refusing to re-sign changed training data"
            )
    else:
        if args.output.exists():
            raise FileExistsError(
                f"immutable collection manifest already exists: {args.output}; "
                "use --verify-existing or a fresh run tag"
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, args.output)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--forbidden-scene-list", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--diagnostics-root", type=Path, required=True)
    parser.add_argument("--boxer-diagnostics-root", type=Path, required=True)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--b6-checkpoint", type=Path, required=True)
    parser.add_argument("--yoloe-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="verify the immutable manifest instead of rewriting it",
    )
    return parser.parse_args()


def main() -> int:
    report = audit(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
