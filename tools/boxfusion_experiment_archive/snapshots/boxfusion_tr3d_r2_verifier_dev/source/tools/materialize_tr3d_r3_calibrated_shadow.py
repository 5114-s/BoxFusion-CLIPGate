#!/usr/bin/env python3
"""Materialize an isolated train-gated R3 veto-calibrated prediction tree.

The command deliberately reuses the raw R3 shadow materializer's immutable
lineage loader and byte-level output invariants.  The only new operation is a
train-only, veto-only risk gate: it may suppress a replacement selected by the
frozen R3 primary rule, but it cannot select another proposal or mutate labels,
scores, output order, or output count.

No ground truth, validation counterfactual report, or CLIP feature is read.
The calibrator must carry ``activation_authorized=true`` from its train-only
gate; an unauthorized model fails before an output namespace is claimed.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from boxfusion.tr3d_r2_provenance import (  # noqa: E402
    canonical_json_sha256,
    code_artifact_tree_sha256,
    load_prefix_manifest,
    sha256_file,
)
from boxfusion.tr3d_r3_calibrator import (  # noqa: E402
    calibrator_sha256,
    load_calibrator,
    materialize_calibrated_prediction,
)
from boxfusion.tr3d_r3_active import active_config_sha256  # noqa: E402
from boxfusion.tr3d_r3_cache import tr3d_r3_cache_path  # noqa: E402
from tools import materialize_tr3d_r3_shadow_active as raw  # noqa: E402
from tools.tr3d_data import read_scene_list  # noqa: E402


MANIFEST_SCHEMA = "boxfusion.tr3d_r3_calibrated_shadow_manifest.v1"
CONFIG_SCHEMA = "boxfusion.tr3d_r3_calibrated_shadow_config.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--r3-export-report", type=Path, required=True)
    parser.add_argument("--r3-cache-root", type=Path, required=True)
    parser.add_argument("--calibrator-model", type=Path, required=True)
    parser.add_argument("--scene-list", type=Path, required=True)
    parser.add_argument("--scans-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prefix-id", default="p100")
    parser.add_argument("--resume", action="store_true")
    return parser


def _plain(value: Any, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: Any = dict(value)
    elif hasattr(value, "as_dict"):
        result = value.as_dict()
    elif is_dataclass(value):
        result = asdict(value)
    else:
        raise TypeError(f"{name} must be a mapping/dataclass")
    encoded = json.dumps(result, sort_keys=True, allow_nan=False)
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise TypeError(f"{name} must resolve to a JSON mapping")
    return normalized


def _load_authorized_calibrator(path: Path):
    unresolved = path.expanduser().absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ValueError(f"calibrator must be a regular non-symlink file: {unresolved}")
    if unresolved.stat().st_mode & 0o222:
        raise ValueError(f"calibrator must be immutable/read-only: {unresolved}")
    model = load_calibrator(unresolved)
    if not model.activation_authorized:
        raise PermissionError("R3 calibrator did not pass its train-only gate")
    if model.metadata.get("train_gate_pass") is not True:
        raise PermissionError("R3 calibrator lacks a positive train-gate attestation")
    return unresolved.resolve(), model


def _config(model: Any) -> dict[str, Any]:
    base = raw._config()
    return {
        "schema": CONFIG_SCHEMA,
        "ground_truth_access": False,
        "counterfactual_report_access": False,
        "clip_access": False,
        "clip_semantics_unchanged": True,
        "base_r3_config": base,
        "base_r3_config_sha256": canonical_json_sha256(base),
        "candidate_population": "frozen_r3_primary_replacements_only",
        "gate": "accept_if_max_gain_or_neutral_probability_strictly_gt_harm",
        "veto_only": True,
        "may_add_primary_replacements": False,
        "output_mutation": "geometry_only",
        "preserved_fields": ["label", "score", "order", "count", "container_types"],
        "activation_authorized": bool(model.activation_authorized),
        "calibrator_sha256": calibrator_sha256(model),
    }


def _validate_inference_lineage(
    model: Any,
    export: Mapping[str, Any],
    frozen_anchor: Mapping[str, Any],
    prefix_id: str,
) -> dict[str, Any]:
    """Require train/validation compatibility before claiming output paths."""

    contract = model.metadata.get("inference_lineage_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("calibrator lacks an inference lineage contract")
    expected = {
        "prefix_id": prefix_id,
        "parent_checkpoint_sha256": str(
            export["expected_parent_checkpoint_sha256"]
        ),
        "parent_config_sha256": str(export["expected_parent_config_sha256"]),
        "r3_config_sha256": str(export["r3_config_sha256"]),
        "r3_code_sha256": str(export["r3_code_sha256"]),
        "primary_active_config_sha256": active_config_sha256(),
    }
    for name, value in expected.items():
        if contract.get(name) != value:
            raise ValueError(f"calibrator/inference lineage mismatch: {name}")
    anchor_contract = contract.get("anchor_distribution_contract")
    if not isinstance(anchor_contract, Mapping):
        raise ValueError("calibrator lacks an anchor-distribution contract")
    metadata = frozen_anchor.get("metadata")
    artifacts = frozen_anchor.get("artifacts")
    if not isinstance(metadata, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("validation anchor lacks distribution provenance")
    quality = artifacts.get("quality_checkpoint")
    yoloe = artifacts.get("yoloe_checkpoint")
    if not isinstance(quality, Mapping) or not isinstance(yoloe, Mapping):
        raise ValueError("validation anchor lacks quality/YOLOE checkpoint hashes")
    observed_anchor = {
        "score_threshold": float(metadata["score_threshold"]),
        "minimum_extent_m": float(metadata["minimum_extent_m"]),
        "quality_detector_blend": float(metadata["quality_detector_blend"]),
        "selective_gate": dict(metadata["selective_gate"]),
        "quality_checkpoint_sha256": str(quality["sha256"]),
        "yoloe_checkpoint_sha256": str(yoloe["sha256"]),
    }
    if json.loads(json.dumps(dict(anchor_contract), sort_keys=True)) != json.loads(
        json.dumps(observed_anchor, sort_keys=True)
    ):
        raise ValueError("calibrator/validation G0 distribution contract mismatch")
    return {
        "compatible": True,
        "checked": expected,
        "anchor_distribution_contract": observed_anchor,
        "training_prefix_manifest_sha256": contract.get(
            "training_prefix_manifest_sha256"
        ),
        "online_scope": "p100_end_of_scene_only_not_true_streaming_prefixes",
    }


def _code_hash() -> str:
    from boxfusion import tr3d_r3_active, tr3d_r3_calibrator

    return code_artifact_tree_sha256(
        (
            Path(__file__),
            Path(raw.__file__),
            Path(tr3d_r3_active.__file__),
            Path(tr3d_r3_calibrator.__file__),
        )
    )


def _materialize_payload(source: object, cache: Any, model: Any):
    result = materialize_calibrated_prediction(
        source, cache, model, require_authorized=True
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError("materialize_calibrated_prediction returned an unsupported result")
    payload, summary = result
    normalized = _plain(summary, "calibrated materialization summary")
    if normalized.get("schema") != "boxfusion.tr3d_r3_calibrated_summary.v1":
        raise ValueError("calibrated materialization summary schema changed")
    if not normalized.get("veto_only"):
        raise ValueError("calibrated materialization is not veto-only")
    if normalized.get("calibrator_sha256") != calibrator_sha256(model):
        raise ValueError("calibrated materialization used a different model")
    primary = int(normalized.get("primary_count", -1))
    accepted = int(normalized.get("accepted_count", -1))
    vetoed = int(normalized.get("vetoed_count", -1))
    active = normalized.get("active_summary")
    if (
        primary < 0
        or accepted < 0
        or vetoed < 0
        or accepted + vetoed != primary
        or not isinstance(active, dict)
        or int(active.get("selected_count", -1)) != accepted
    ):
        raise ValueError("calibrated materialization summary counts are inconsistent")
    return payload, normalized


def _normalized_resume_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["counts"] = dict(result.get("counts", {}), resumed_scenes=0)
    result["scenes"] = [
        dict(row, resumed=False) for row in result.get("scenes", [])
    ]
    return result


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    frozen_manifest = args.frozen_manifest.resolve()
    export_path = args.r3_export_report.resolve()
    r3_root = args.r3_cache_root.resolve()
    scene_list = args.scene_list.resolve()
    scans_root = args.scans_root.resolve()
    output_root = args.output_root.expanduser().absolute()
    manifest_path = args.manifest.expanduser().absolute()
    scenes = read_scene_list(scene_list)
    if len(scenes) not in (10, 100):
        raise ValueError("calibrated shadow materialization requires fixed10 or full100")

    # Authorization is checked before creating or claiming any output path.
    model_path, model = _load_authorized_calibrator(args.calibrator_model)
    model_file_sha = sha256_file(model_path)
    model_sha = calibrator_sha256(model)
    config = _config(model)
    config_sha = canonical_json_sha256(config)
    code_sha = _code_hash()

    before = raw.verify_frozen_anchor_manifest(frozen_manifest)
    before_snapshot = raw._snapshot(before)
    frozen_positions = {
        scene_id: index for index, scene_id in enumerate(before["scene_ids"])
    }
    if any(scene_id not in frozen_positions for scene_id in scenes) or [
        frozen_positions[scene_id] for scene_id in scenes
    ] != sorted(frozen_positions[scene_id] for scene_id in scenes):
        raise ValueError("materialization scenes are not an ordered frozen-manifest subset")

    export = raw._load_export(export_path, scenes)
    for name, expected in {
        "frozen_manifest": frozen_manifest,
        "r3_cache_root": r3_root,
        "scene_list": scene_list,
        "scans_root": scans_root,
    }.items():
        if Path(str(export.get(name, ""))).resolve() != expected:
            raise ValueError(f"R3 export {name} path mismatch")
    if export.get("prefix_id") != args.prefix_id:
        raise ValueError("R3 export prefix mismatch")
    if export.get("frozen_manifest_sha256") != sha256_file(frozen_manifest):
        raise ValueError("R3 export frozen manifest bytes mismatch")
    if export.get("frozen_prediction_tree_sha256") != before["prediction_tree_sha256"]:
        raise ValueError("R3 export frozen prediction tree mismatch")
    lineage_compatibility = _validate_inference_lineage(
        model, export, before, args.prefix_id
    )

    export_rows = {str(row["scene_id"]): row for row in export["scenes"]}
    prefix_rows: Mapping[str, Mapping[str, Any]] = {}
    if bool(export["r3_config"].get("r2a_enabled")):
        prefix_rows = load_prefix_manifest(
            Path(str(export["prefix_manifest"])).resolve(), prefix_id=args.prefix_id
        )

    frozen_root = Path(str(before["reference_result_root"])).resolve()
    protected_roots = (frozen_root, r3_root)
    if any(
        output_root == protected
        or output_root in protected.parents
        or protected in output_root.parents
        for protected in protected_roots
    ):
        raise ValueError("output root must be isolated from frozen/R3 input trees")
    if manifest_path in {frozen_manifest, export_path, scene_list, model_path}:
        raise ValueError("manifest path aliases an immutable input")
    if output_root == model_path.parent or output_root in model_path.parents:
        raise ValueError("output root must be isolated from the calibrator model")

    raw._claim_namespace(output_root, manifest_path, resume=bool(args.resume))
    existing = raw._existing_names(output_root, scenes)
    expected_names = {f"{scene}_boxes.pkl" for scene in scenes}
    if manifest_path.exists() and existing != expected_names:
        raise ValueError("complete manifest exists but prediction tree is incomplete")

    per_scene: list[dict[str, Any]] = []
    output_hashes: dict[str, str] = {}
    total_rows = total_changed = total_primary = total_accepted = total_vetoed = 0
    resumed = 0
    for scene_id in scenes:
        source_path = frozen_root / f"{scene_id}_boxes.pkl"
        source = raw._load_prediction(source_path)
        sidecar_path = tr3d_r3_cache_path(r3_root, scene_id, args.prefix_id)
        if sha256_file(sidecar_path) != export_rows[scene_id]["r3_sidecar_sha256"]:
            raise ValueError(f"{scene_id}: immutable R3 sidecar SHA mismatch")
        cache = raw._load_bound_cache(
            scene_id=scene_id,
            prefix_id=args.prefix_id,
            export=export,
            frozen_manifest=frozen_manifest,
            frozen_root=frozen_root,
            r3_root=r3_root,
            scans_root=scans_root,
            prefix_rows=prefix_rows,
        )
        active, api_summary = _materialize_payload(source, cache, model)
        invariant = raw._validate_invariants(source, active)
        accepted = int(api_summary["accepted_count"])
        if accepted < int(invariant["changed_rows"]):
            raise ValueError(f"{scene_id}: changed rows exceed accepted replacements")
        encoded = pickle.dumps(active, protocol=raw.PICKLE_PROTOCOL)
        round_trip = pickle.loads(encoded)  # noqa: S301 - bytes created above
        raw._validate_invariants(active, round_trip)
        target = output_root / f"{scene_id}_boxes.pkl"
        was_resumed = target.name in existing
        if was_resumed:
            observed = raw._load_prediction(target)
            raw._validate_invariants(active, observed)
            if any(
                not raw._geometry_equal(expected[1], actual[1])
                for expected, actual in zip(active[0], observed[0])
            ):
                raise ValueError(f"{scene_id}: resumed calibrated geometry differs")
            if target.read_bytes() != encoded:
                raise ValueError(f"{scene_id}: resumed calibrated pickle bytes differ")
            resumed += 1
        else:
            raw._write_bytes_create_only(target, encoded)
        output_sha = sha256_file(target)
        output_hashes[target.name] = output_sha
        per_scene.append(
            {
                "scene_id": scene_id,
                "source_prediction_sha256": sha256_file(source_path),
                "r3_sidecar_sha256": sha256_file(sidecar_path),
                "calibrated_prediction_sha256": output_sha,
                "calibrator_sha256": model_sha,
                "rows": int(invariant["rows"]),
                "changed_rows": int(invariant["changed_rows"]),
                "primary_count": int(api_summary["primary_count"]),
                "accepted_count": accepted,
                "vetoed_count": int(api_summary["vetoed_count"]),
                "resumed": was_resumed,
                "api_summary": api_summary,
            }
        )
        total_rows += int(invariant["rows"])
        total_changed += int(invariant["changed_rows"])
        total_primary += int(api_summary["primary_count"])
        total_accepted += accepted
        total_vetoed += int(api_summary["vetoed_count"])

    if raw._existing_names(output_root, scenes) != expected_names:
        raise RuntimeError("calibrated shadow output set is incomplete")
    after = raw.verify_frozen_anchor_manifest(frozen_manifest)
    after_snapshot = raw._snapshot(after)
    if before_snapshot != after_snapshot:
        raise RuntimeError("frozen G0 changed during calibrated materialization")

    parent_lineage = {
        "parent_cache_root": str(Path(str(export["parent_cache_root"])).resolve()),
        "prefix_manifest": str(Path(str(export["prefix_manifest"])).resolve()),
        "expected_parent_checkpoint_sha256": export[
            "expected_parent_checkpoint_sha256"
        ],
        "expected_parent_config_sha256": export["expected_parent_config_sha256"],
        "parent_evidence_hashes": dict(export.get("parent_evidence_hashes", {})),
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "complete": True,
        "shadow_only": True,
        "formal_active_authorized": False,
        "train_gate_activation_authorized": True,
        "veto_only": True,
        "ground_truth_access": False,
        "counterfactual_report_access": False,
        "clip_access": False,
        "labels_scores_order_count_unchanged": True,
        "replacement_geometry_frame": "scannet_unaligned_world",
        "config": config,
        "config_sha256": config_sha,
        "code_sha256": code_sha,
        "calibrator_model": str(model_path),
        "calibrator_model_file_sha256": model_file_sha,
        "calibrator_sha256": model_sha,
        "calibrator_dataset_sha256": model.dataset_sha256,
        "calibrator_train_scene_list_sha256": model.scene_list_sha256,
        "calibrator_metadata": dict(model.metadata),
        "frozen_manifest": str(frozen_manifest),
        "frozen_manifest_sha256": sha256_file(frozen_manifest),
        "frozen_prediction_tree_sha256": before["prediction_tree_sha256"],
        "r3_export_report": str(export_path),
        "r3_export_report_sha256": sha256_file(export_path),
        "r3_cache_root": str(r3_root),
        "r3_config_sha256": export["r3_config_sha256"],
        "r3_code_sha256": export["r3_code_sha256"],
        "parent_lineage": parent_lineage,
        "calibrator_lineage_compatibility": lineage_compatibility,
        "scene_list": str(scene_list),
        "scene_list_sha256": sha256_file(scene_list),
        "scans_root": str(scans_root),
        "prefix_id": args.prefix_id,
        "output_root": str(output_root),
        "output_prediction_tree_sha256": raw._sha_tree(output_hashes),
        "counts": {
            "scenes": len(scenes),
            "rows": total_rows,
            "primary_replacements": total_primary,
            "accepted_replacements": total_accepted,
            "vetoed_replacements": total_vetoed,
            "byte_changed_rows": total_changed,
            "resumed_scenes": resumed,
        },
        "frozen_anchor": {"before": before_snapshot, "after": after_snapshot},
        "scenes": per_scene,
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _normalized_resume_manifest(existing_manifest) != _normalized_resume_manifest(
            manifest
        ):
            raise ValueError("existing calibrated shadow manifest disagrees with replay")
        return existing_manifest
    raw._write_json_create_only(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    manifest = materialize(build_parser().parse_args(argv))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
