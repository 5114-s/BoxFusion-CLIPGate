"""Fail-closed contracts for the E961 CA-only outer-dev continuation gate.

The runtime instance preregistration is deliberately sealed before this
module is allowed to inspect the expanded-training receipt, checkpoint, or
fold-0 ground truth.  Fold 1 and official validation are outside every path
contract.  Numerical AP/oracle primitives are reused byte-for-byte from the
previously sealed exact-20 xfit-R2 evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping

from .ca1m_tr3d_xfit_r2_eval import (
    continuation_gate as _sealed_continuation_gate,
    create_or_verify_json,
    match_targets,
    metric_delta,
    official_ca_ap,
    read_json,
    regular_directory,
    regular_file,
    same_gt_oracle_scene,
    sha256_bytes,
    sha256_file,
)


CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_outer_dev_eval_config.v2"
BASE_CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_e961_outer_dev_eval_config.v1"
PROTOCOL_PREREGISTRATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_outer_dev_protocol_preregistration.v2"
)
PREREGISTRATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_outer_dev_preregistration.v2"
)
BINDING_SCHEMA = "boxfusion.ca1m_tr3d_e961_outer_dev_checkpoint_binding.v1"
COLLECTION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_outer_dev_proposal_collection.v1"
)
REPORT_SCHEMA = "boxfusion.ca1m_tr3d_e961_outer_dev_report.v1"
CONTINUATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_outer_dev_continuation_receipt.v1"
)
INNER_AUTHORIZATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_e961_inner_training_authorization.v1"
)
STOP_SCHEMA = "boxfusion.ca1m_tr3d_e961_inner_training_stop.v1"
NAMESPACE = "ca1m_tr3d_e961_outer_dev_eval_v1"

PIPELINE_ROOT = Path("/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline")
OVM_ROOT = Path("/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev")
BASE_CONFIG_PATH = PIPELINE_ROOT / "config/ca1m_tr3d_e961_outer_dev_eval_v1.json"
BASE_CONFIG_SHA256 = "d8d364a7ceff5c8c4fe492d02c9c0104be15c3eb7c6e59f8bf4badc398128938"
CONFIG_PATH = PIPELINE_ROOT / "config/ca1m_tr3d_e961_outer_dev_eval_v2.json"
MANIFEST_ROOT = PIPELINE_ROOT / f"manifests/{NAMESPACE}"
REPORT_ROOT = PIPELINE_ROOT / f"reports/{NAMESPACE}"
DIAGNOSTIC_ROOT = Path(f"/extra/ZhaoX/{NAMESPACE}")
TRAIN_RUN_ROOT = Path("/extra/ZhaoX/tr3d_ca1m_e961_outer_train_r2/runs")
TRAIN_WORK_ROOT = Path("/extra/ZhaoX/tr3d_ca1m_work_dirs/ca1m_e961_outer_train_r2")
PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION.json"
V1_PROTOCOL_PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION_PROTOCOL.json"
V1_PROTOCOL_PREREGISTRATION_SHA256 = (
    "9c8fdc2689636584fd55bcc536a51ccc8f208e3e077d1f9b8f1479858b8dc05f"
)
V1_PROTOCOL_INVALID_PATH = MANIFEST_ROOT / "PREREGISTRATION_PROTOCOL_V1_INVALID.json"
V1_PROTOCOL_INVALID_SHA256 = (
    "31d39340015df4101725d475310ec09b5daa19751c677ae0d2e51f75ad5ad3d8"
)
PROTOCOL_PREREGISTRATION_PATH = MANIFEST_ROOT / "PREREGISTRATION_PROTOCOL_V2.json"
BINDING_PATH = MANIFEST_ROOT / "CHECKPOINT_BINDING.json"
COLLECTION_PATH = REPORT_ROOT / "PROPOSAL_COLLECTION.json"
REPORT_PATH = REPORT_ROOT / "EVALUATION_REPORT.json"
CONTINUATION_PATH = REPORT_ROOT / "CONTINUATION_RECEIPT.json"
INNER_AUTHORIZATION_PATH = REPORT_ROOT / "INNER_TRAINING_AUTHORIZATION.json"
STOP_PATH = REPORT_ROOT / "STOP_WITHOUT_INNER_TRAINING.json"

FOLD0_SHA256 = "9c886ca85ba599881797b25a49d2fc72dd136d255a245a09fe1cf17cbce735a7"
SELECTION_SHA256 = "eceeb29aa0a4a7c7f8548d1a3e09c25b2b5d3eeb6d486481e63d32a3c0e97791"
POINT_PARITY_SHA256 = "35d9dfafc7272d92d98c97c6ef23f4323432e9bd0af5045bc5f78b1ae9afa00d"
POINT_CONFIG_SHA256 = "479f7e61eff9fd23fc086ebc2603e161caa876defe73c556a0e671a8fd35c052"
ANCHOR_SHA256 = "202340ff97c3de5f49a969ec983222cae0772913b831f808f870aea71a639c88"
ANCHOR_MANIFEST_SHA256 = "b574d029adcf3d24735869d314a7e2f2dd39e6ad05ea16a6a193d2171dec1669"
V1_COMPARISON_SHA256 = "b38d55b42eceef21639943a41c689ce45c1eb698dfe46e0f4e43280500e967f1"
GT_INVENTORY_SHA256 = "6c3bdfd666ca49558ac390197abeec588949f05e33d9c8a18d1b5c8326d9e9a7"
INNER_ROLES = ("inner_holdout2", "inner_holdout3", "inner_holdout4")
RUN_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,79}$")
SCENE_ID = re.compile(r"^[0-9]{8}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


def validate_run_tag(value: str) -> str:
    tag = str(value)
    if RUN_TAG.fullmatch(tag) is None or ".." in tag:
        raise ValueError("outer training run tag is not a safe fixed token")
    return tag


def expected_training_receipt(run_tag: str) -> Path:
    tag = validate_run_tag(run_tag)
    return TRAIN_RUN_ROOT / tag / "RUN_RECEIPT.json"


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if SHA256.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def guard_fixed_path(path: Path, allowed_root: Path, name: str) -> Path:
    """Reject lexical escape and every existing symlink in a fixed path chain."""

    target = Path(path)
    root = Path(allowed_root)
    if not target.is_absolute() or not root.is_absolute():
        raise ValueError(f"{name} requires absolute paths")
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name} escapes fixed root {root}: {target}") from error
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise ValueError(f"{name} fixed root is unsafe: {root}")
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{name} parent must not be a symlink: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"{name} parent must be a directory: {current}")
    if target.is_symlink():
        raise ValueError(f"{name} target must not be a symlink: {target}")
    return target


def _paths(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        rows: list[str] = []
        for key, child in value.items():
            if key in {"path", "root", "output_root", "receipt_root"}:
                rows.append(str(child))
            rows.extend(_paths(child))
        return rows
    if isinstance(value, list):
        rows = []
        for child in value:
            rows.extend(_paths(child))
        return rows
    return []


def _record(value: Any, *, path: Path, sha256: str, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} record is missing")
    if Path(str(value.get("path", ""))).resolve() != path.resolve():
        raise ValueError(f"{name} path differs")
    if value.get("sha256") != sha256:
        raise ValueError(f"{name} frozen SHA256 differs")


def load_config(path: Path = CONFIG_PATH) -> tuple[Path, dict[str, Any]]:
    source, revision = read_json(path, "E961 outer evaluation config", immutable=True)
    if source != CONFIG_PATH:
        raise ValueError("E961 evaluation requires the fixed v2 config path")
    if set(revision) != {
        "schema", "base_config", "predecessor_invalidation",
        "access_updates", "training_updates", "evaluation_updates",
        "implementation_updates",
    } or revision.get("schema") != CONFIG_SCHEMA:
        raise ValueError("E961 evaluation v2 revision keys differ")
    base_record = revision.get("base_config") or {}
    base_path, base = read_json(
        Path(str(base_record.get("path", ""))),
        "superseded E961 evaluation config v1", immutable=True,
    )
    if (
        base_path != BASE_CONFIG_PATH
        or base_record.get("sha256") != BASE_CONFIG_SHA256
        or sha256_file(base_path) != BASE_CONFIG_SHA256
        or base.get("schema") != BASE_CONFIG_SCHEMA
    ):
        raise ValueError("E961 evaluation v1 predecessor differs")
    invalidation = revision.get("predecessor_invalidation") or {}
    if invalidation != {
        "protocol_v1_path": str(V1_PROTOCOL_PREREGISTRATION_PATH),
        "protocol_v1_sha256": V1_PROTOCOL_PREREGISTRATION_SHA256,
        "invalid_receipt_path": str(V1_PROTOCOL_INVALID_PATH),
        "invalid_receipt_sha256": V1_PROTOCOL_INVALID_SHA256,
        "formal_v1_authorized": False,
    }:
        raise ValueError("E961 protocol-v1 invalidation binding differs")
    evaluation_updates = revision.get("evaluation_updates") or {}
    if evaluation_updates != {
        "preregistration_protocol": str(PROTOCOL_PREREGISTRATION_PATH),
    }:
        raise ValueError("E961 protocol-v2 output revision differs")
    access_updates = revision.get("access_updates") or {}
    training_updates = revision.get("training_updates") or {}
    implementation_updates = revision.get("implementation_updates") or {}
    if set(implementation_updates) != {
        "e961_contract", "e961_runner", "e961_runner_entrypoint",
        "single_command_wrapper",
        "protocol_preregistration_sealer", "e961_outer_train_contract",
        "e961_outer_train_trainer", "e961_outer_train_driver",
        "e961_outer_train_tests",
    }:
        raise ValueError("E961 v2 implementation update inventory differs")
    cfg = json.loads(json.dumps(base))
    cfg["schema"] = CONFIG_SCHEMA
    cfg["access"] = access_updates
    cfg["training"] = training_updates
    cfg["evaluation_stage"].update(evaluation_updates)
    cfg["implementation"].update(implementation_updates)
    expected_keys = {
        "schema", "namespace", "access", "training", "scene_contract",
        "selection_contract", "point_lineage", "point_inference", "runtime",
        "proposal_stage", "evaluation_stage", "implementation",
    }
    if set(cfg) != expected_keys:
        raise ValueError("E961 evaluation config keys differ")
    if cfg.get("schema") != CONFIG_SCHEMA or cfg.get("namespace") != NAMESPACE:
        raise ValueError("E961 evaluation config identity differs")
    if cfg.get("access") != {
        "official_train_only": True,
        "fold0_reused_dev_only": True,
        "fold0_gt_after_exact20_proposal_seal_only": True,
        "fold1_access": False,
        "official_validation_access": False,
        "scannet_training_weights_loaded": False,
        "scannet_training_data_configured_or_opened": False,
        "plugin_imports_scannet_adapter_class_definition": True,
        "scannet_adapter_instantiated": False,
    }:
        raise ValueError("E961 evaluation access contract differs")
    training = cfg.get("training") or {}
    if training != {
        "receipt_root": str(TRAIN_RUN_ROOT),
        "receipt_schema": "boxfusion.tr3d.ca1m_e961_outer_train_run.r2",
        "authorization_consumption_schema": (
            "boxfusion.tr3d.ca1m_e961_outer_auth_consumption.r2"
        ),
        "training_started_claim_schema": (
            "boxfusion.tr3d.ca1m_e961_outer_training_started.r2"
        ),
        "launch_schema": "boxfusion.tr3d.ca1m_e961_outer_launch.r2",
        "checkpoint_audit_schema": (
            "boxfusion.tr3d.mmengine_terminal_checkpoint_audit.r2"
        ),
        "role": "outer_dev",
        "train_scene_count": 1001,
        "heldout_fold": 0,
        "checkpoint_name": "iter_11268.pth",
        "optimizer_updates": 11268,
        "global_batch": 16,
        "world_size": 2,
        "initialization": "random_scratch_ca_only",
        "checkpoint_selection": False,
        "unique_final_checkpoint": True,
        "authorization_consumed_once_by_sha256": True,
        "training_start_claim_permanent_before_spawn": True,
        "full_config_equality_required": True,
        "deep_checkpoint_terminal_state_required": True,
    }:
        raise ValueError("E961 outer training receipt contract differs")
    scene_path = OVM_ROOT / "data/tr3d_ca1m_e961_v1/splits/predict_fold0.txt"
    _record(
        cfg.get("scene_contract"), path=scene_path, sha256=FOLD0_SHA256,
        name="exact fold0 scene list",
    )
    if cfg["scene_contract"] != {
        "path": str(scene_path), "sha256": FOLD0_SHA256,
        "count": 20, "fold": 0, "role": "reused_dev", "exact": True,
    }:
        raise ValueError("E961 fold0 scene contract differs")
    selection_path = OVM_ROOT / "data/tr3d_ca1m_e961_v1/SELECTION_CONTRACT.json"
    _record(
        cfg.get("selection_contract"), path=selection_path,
        sha256=SELECTION_SHA256, name="E961 selection contract",
    )
    lineage = cfg.get("point_lineage") or {}
    parity_path = PIPELINE_ROOT / (
        "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/"
        "lineage_training_point_parity_v4.json"
    )
    _record(
        {"path": lineage.get("receipt_path"), "sha256": lineage.get("receipt_sha256")},
        path=parity_path, sha256=POINT_PARITY_SHA256, name="same-path point parity",
    )
    if lineage.get("processed_rgbd") != {
        "root": "/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1",
        "depth_scale": 1000.0,
    } or lineage.get("fold0_exact_scene_subset") is not True:
        raise ValueError("E961 same point-path lineage differs")
    point_path = OVM_ROOT / (
        "config/tr3d/tr3d_ca1m_foreground_point_inference_xfit_r2.py"
    )
    _record(
        cfg.get("point_inference"), path=point_path,
        sha256=POINT_CONFIG_SHA256, name="point-only inference config",
    )
    point = cfg["point_inference"]
    if any(point.get(key) is not expected for key, expected in (
        ("point_input_only", True), ("standalone", True),
        ("ground_truth_access", False), ("validation_access", False),
        ("scannet_config_access", False),
    )):
        raise ValueError("E961 point-only inference isolation differs")
    if cfg.get("runtime") != {
        "worker_python": str(OVM_ROOT / ".conda/boxfusion-tr3d/bin/python"),
        "worker_script": str(PIPELINE_ROOT / "tools/ca1m_tr3d_terminal_worker.py"),
        "runtime_root": "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev",
        "project_root": str(OVM_ROOT),
        "vendor_root": str(OVM_ROOT / "third_party/mmdetection3d"),
        "startup_timeout_s": 600,
    }:
        raise ValueError("E961 proposal runtime differs")
    protocol = (cfg.get("proposal_stage") or {}).get("protocol")
    if protocol != {
        "pixel_stride": 4, "voxel_size_m": 0.01,
        "min_depth_m": 0.1, "max_depth_m": 6.0,
        "score_threshold": 0.01, "max_proposals": 256,
        "near_iou": 0.15, "prefix_id": "p100_gap20",
    }:
        raise ValueError("E961 proposal point-path protocol differs")
    proposal = cfg["proposal_stage"]
    if (
        proposal.get("scene_count") != 20
        or proposal.get("fold0_only") is not True
        or proposal.get("create_only") is not True
        or proposal.get("gpu_required") is not True
        or any(proposal.get(key) is not False for key in (
            "ground_truth_access", "anchor_access", "b6_access"
        ))
        or Path(str(proposal.get("output_root", ""))).resolve()
        != DIAGNOSTIC_ROOT / "proposals"
        or Path(str(proposal.get("collection_manifest", ""))).resolve()
        != COLLECTION_PATH
    ):
        raise ValueError("E961 proposal-stage isolation differs")
    evaluation = cfg.get("evaluation_stage") or {}
    if (
        evaluation.get("scene_count") != 20
        or evaluation.get("heldout_fold") != 0
        or evaluation.get("fold0_role") != "reused_dev"
        or evaluation.get("requires_sealed_proposal_collection") is not True
        or evaluation.get("cpu_only") is not True
        or evaluation.get("official_metric")
        != "CA class-agnostic global-score AP over world-axis-aligned AABBs"
        or evaluation.get("iou_thresholds") != [0.15, 0.25, 0.50]
        or evaluation.get("oracle") != "same_best_gt_geometry_replacement"
        or evaluation.get("oracle_deployable") is not False
        or evaluation.get("raw_detector_ap_role")
        != "diagnostic_only_no_checkpoint_selection"
        or evaluation.get("continuation_gate") != {
            "same_gt_min_iou_gain": 0.05,
            "min_replacements": 10,
            "min_replacement_scenes": 5,
            "min_delta_ap15": 0.0,
            "min_delta_ap25": 0.0,
            "min_delta_ap50": 0.005,
            "pass_authorizes_inner_roles": list(INNER_ROLES),
        }
    ):
        raise ValueError("E961 evaluation science contract differs")
    fixed_records = (
        (
            "anchor_shadow",
            PIPELINE_ROOT / (
                "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
                "fold0_final_base_b6_v2_oof.npz"
            ),
            ANCHOR_SHA256,
        ),
        (
            "anchor_shadow_manifest",
            PIPELINE_ROOT / (
                "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
                "fold0_final_base_b6_v2_oof.manifest.json"
            ),
            ANCHOR_MANIFEST_SHA256,
        ),
        (
            "v1_fold0_comparison_manifest",
            PIPELINE_ROOT / (
                "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
                "v1_fold0_proposal_comparison_manifest.json"
            ),
            V1_COMPARISON_SHA256,
        ),
        (
            "gt_shadow_inventory",
            PIPELINE_ROOT / (
                "manifests/ca1m_tr3d_benefit_gate_final_base_v4/"
                "derived_train_gt_shadow_inventory_v1.json"
            ),
            GT_INVENTORY_SHA256,
        ),
    )
    for name, expected_path, expected_sha in fixed_records:
        _sha((evaluation.get(name) or {}).get("sha256"), name)
        if (
            Path(str((evaluation.get(name) or {}).get("path", ""))).resolve()
            != expected_path
            or (evaluation.get(name) or {}).get("sha256") != expected_sha
        ):
            raise ValueError(f"E961 evaluation {name} path/SHA256 differs")
    if Path(str(evaluation.get("gt_shadow_root", ""))).resolve() != PIPELINE_ROOT / (
        "inputs/ca1m_tr3d_benefit_gate_final_base_v4/derived_train_gt_fitdev80"
    ):
        raise ValueError("E961 fold0 GT shadow root differs")
    expected_outputs = {
        "preregistration_protocol": PROTOCOL_PREREGISTRATION_PATH,
        "preregistration": PREREGISTRATION_PATH,
        "binding": BINDING_PATH,
        "report": REPORT_PATH,
        "continuation_receipt": CONTINUATION_PATH,
        "inner_authorization": INNER_AUTHORIZATION_PATH,
        "stop_receipt": STOP_PATH,
    }
    for name, expected in expected_outputs.items():
        if Path(str(evaluation.get(name, ""))).resolve() != expected:
            raise ValueError(f"E961 evaluation {name} namespace differs")
    implementation = cfg.get("implementation")
    if not isinstance(implementation, dict) or set(implementation) != {
        "e961_contract", "e961_runner", "e961_runner_entrypoint",
        "single_command_wrapper",
        "protocol_preregistration_sealer",
        "e961_outer_train_contract", "e961_outer_train_trainer",
        "e961_outer_train_driver", "e961_outer_train_tests",
        "sealed_r2_eval_contract", "sealed_r2_eval_runner",
        "point_builder", "proposal_contract", "terminal_geometry",
        "rgbd_backprojection", "worker_client", "worker",
        "point_inference_contract", "point_inference_config",
        "official_adapter",
    }:
        raise ValueError("E961 implementation inventory differs")
    for name, value in implementation.items():
        record = value or {}
        implementation_path = Path(str(record.get("path", "")))
        expected = _sha(record.get("sha256"), f"implementation {name}")
        actual = sha256_file(regular_file(implementation_path, f"implementation {name}"))
        if actual != expected:
            raise ValueError(f"E961 implementation drift: {name}")
    forbidden = ("fold1", "official_val", "official-validation", "/val/")
    for candidate in _paths(cfg):
        lowered = candidate.lower()
        if any(token in lowered for token in forbidden):
            raise ValueError(f"forbidden evaluation path in config: {candidate}")
    return source, cfg


def scene_ids(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    record = cfg.get("scene_contract") or {}
    source = regular_file(Path(str(record.get("path", ""))), "exact fold0 scene list")
    if sha256_file(source) != FOLD0_SHA256:
        raise ValueError("exact fold0 scene-list SHA256 differs")
    scenes = tuple(line.strip() for line in source.read_text().splitlines() if line.strip())
    if (
        len(scenes) != 20 or len(set(scenes)) != 20
        or any(SCENE_ID.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError("fold0 reused-dev list must contain exact 20 scene IDs")
    return scenes


def validate_preregisterable_static(
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate only code, selection, and fold-0 identifiers.

    In particular this function deliberately does not stat/hash/open the
    expanded training receipt/checkpoint, anchor NPZ, GT inventory, or GT
    arrays.  It is safe to call before the runtime preregistration exists.
    """

    scenes = scene_ids(cfg)
    selection_record = cfg["selection_contract"]
    selection_path = regular_file(
        Path(selection_record["path"]), "E961 selection contract", immutable=True
    )
    if sha256_file(selection_path) != SELECTION_SHA256:
        raise ValueError("E961 selection contract changed")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    artifact = (selection.get("artifacts") or {}).get("splits/predict_fold0.txt") or {}
    if (
        selection.get("fold1_gt_opened") is not False
        or selection.get("fold1_scene_list_opened") is not False
        or selection.get("official_validation_gt_opened") is not False
        or selection.get("gt_array_content_loaded") is not False
        or artifact.get("sha256") != FOLD0_SHA256
    ):
        raise ValueError("E961 selection does not bind isolated exact20 fold0")
    return {
        "scene_count": len(scenes),
        "scene_list_sha256": FOLD0_SHA256,
        "selection_contract_sha256": SELECTION_SHA256,
        "expanded_training_receipt_access": False,
        "expanded_checkpoint_access": False,
        "anchor_array_access": False,
        "ground_truth_access": False,
        "fold1_access": False,
        "official_validation_access": False,
    }


def _expected_preregistration(
    source: Path, cfg: Mapping[str, Any], run_tag: str,
) -> dict[str, Any]:
    tag = validate_run_tag(run_tag)
    evaluation = cfg["evaluation_stage"]
    inputs = {
        "fold0_scene_list": cfg["scene_contract"],
        "selection_contract": cfg["selection_contract"],
        "point_parity": {
            "path": cfg["point_lineage"]["receipt_path"],
            "sha256": cfg["point_lineage"]["receipt_sha256"],
        },
        "point_inference_config": cfg["point_inference"],
        "final_base_b6_oof_anchor": evaluation["anchor_shadow"],
        "final_base_b6_oof_anchor_manifest": evaluation["anchor_shadow_manifest"],
        "same_point_path_v1_comparison": evaluation[
            "v1_fold0_comparison_manifest"
        ],
        "opaque_fold0_gt_inventory": evaluation["gt_shadow_inventory"],
    }
    return {
        "schema": PREREGISTRATION_SCHEMA,
        "complete": True,
        "create_only": True,
        "namespace": NAMESPACE,
        "outer_training_run_tag": tag,
        "outer_training_receipt_expected_path": str(expected_training_receipt(tag)),
        "protocol_preregistration": {
            "path": str(PROTOCOL_PREREGISTRATION_PATH),
            "sha256": sha256_file(PROTOCOL_PREREGISTRATION_PATH),
            "schema": PROTOCOL_PREREGISTRATION_SCHEMA,
        },
        "sealed_before_expanded_training_receipt_access": True,
        "sealed_before_expanded_checkpoint_access": True,
        "sealed_before_fold0_gt_access": True,
        "expanded_training_receipt_access_at_seal": False,
        "expanded_checkpoint_access_at_seal": False,
        "anchor_array_access_at_seal": False,
        "fold0_gt_access_at_seal": False,
        "fold1_access": False,
        "official_validation_access": False,
        "partition": "official_train_fold0_reused_dev_exact20",
        "fold0_prior_exposure": [
            "v1_checkpoint_diagnostic", "terminal_v4_gate", "xfit_r2_outer_gate"
        ],
        "checkpoint_policy": {
            "one_explicit_outer_run_receipt": True,
            "checkpoint_name": "iter_11268.pth",
            "optimizer_updates": 11268,
            "checkpoint_selection": False,
            "raw_detector_ap_checkpoint_selection": False,
        },
        "metric": {
            "class_mode": "CA_class_agnostic",
            "coordinate_frame": "world",
            "box_geometry": "axis_aligned_AABB_from_8_corners",
            "ranking": "global_prediction_score",
            "duplicate_matching": "one_detection_per_scene_gt_per_threshold",
            "iou_comparison": "strict_greater_than_threshold",
            "iou_thresholds": [0.15, 0.25, 0.50],
        },
        "paired_comparison": {
            "same_exact20_scene_ids": True,
            "same_rgbd_point_builder": True,
            "same_point_array_sha256_per_scene": True,
            "same_point_inference_config": True,
            "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        },
        "terminal_oracle": {
            "anchor": "final_base_plus_B6_fold0_OOF",
            "each_anchor_score_model_excludes_scene": True,
            "candidate_source": "expanded_E961_outer_dev_TR3D",
            "matching": "candidate_and_anchor_must_share_best_GT",
            "near_anchor_iou": 0.15,
            "minimum_same_gt_iou_gain": 0.05,
            "replacement": "geometry_only_preserve_anchor_score_and_row_order",
            "deployable": False,
        },
        "continuation_gate": {
            "proposal_exact20_finite_ca_only": True,
            "min_replacements": 10,
            "min_replacement_scenes": 5,
            "min_delta_ap15": 0.0,
            "min_delta_ap25": 0.0,
            "min_delta_ap50": 0.005,
            "pass_authorizes_inner_roles": list(INNER_ROLES),
            "failure_action": (
                "stop_without_inner_training_or_fold1_or_official_validation_access"
            ),
        },
        "inputs": inputs,
        "evaluation_config": {
            "path": str(source), "sha256": sha256_file(source),
        },
        "implementation": cfg["implementation"],
        "outputs": {
            "proposal_collection": str(COLLECTION_PATH),
            "evaluation_report": str(REPORT_PATH),
            "continuation_receipt": str(CONTINUATION_PATH),
            "pass_authorization": str(INNER_AUTHORIZATION_PATH),
            "failure_stop_receipt": str(STOP_PATH),
        },
    }


def _expected_protocol_preregistration(
    source: Path, cfg: Mapping[str, Any],
) -> dict[str, Any]:
    evaluation = cfg["evaluation_stage"]
    return {
        "schema": PROTOCOL_PREREGISTRATION_SCHEMA,
        "complete": True,
        "create_only": True,
        "namespace": NAMESPACE,
        "sealed_before_any_expanded_outer_checkpoint_access": True,
        "sealed_before_any_formal_fold0_anchor_or_gt_access": True,
        "expanded_training_receipt_access_at_seal": False,
        "expanded_checkpoint_access_at_seal": False,
        "anchor_array_access_at_seal": False,
        "fold0_gt_access_at_seal": False,
        "fold1_access": False,
        "official_validation_access": False,
        "invalidated_predecessor": {
            "protocol_v1": {
                "path": str(V1_PROTOCOL_PREREGISTRATION_PATH),
                "sha256": V1_PROTOCOL_PREREGISTRATION_SHA256,
            },
            "invalid_receipt": {
                "path": str(V1_PROTOCOL_INVALID_PATH),
                "sha256": V1_PROTOCOL_INVALID_SHA256,
            },
            "formal_v1_authorized": False,
        },
        "runtime_instance_rule": {
            "explicit_outer_run_tag_required": True,
            "receipt_pattern": (
                "/extra/ZhaoX/tr3d_ca1m_e961_outer_train_r2/runs/"
                "<outer_run_tag>/RUN_RECEIPT.json"
            ),
            "create_only_instance_preregistration_required_before_verify_run": True,
            "checkpoint_name": "iter_11268.pth",
            "optimizer_updates": 11268,
            "checkpoint_selection": False,
            "receipt_schema": "boxfusion.tr3d.ca1m_e961_outer_train_run.r2",
            "authorization_consumption_schema": (
                "boxfusion.tr3d.ca1m_e961_outer_auth_consumption.r2"
            ),
            "training_started_claim_schema": (
                "boxfusion.tr3d.ca1m_e961_outer_training_started.r2"
            ),
            "deep_checkpoint_audit_schema": (
                "boxfusion.tr3d.mmengine_terminal_checkpoint_audit.r2"
            ),
        },
        "outer_train_r2_frozen_review": {
            "independent_review_pass": True,
            "tool": cfg["implementation"]["e961_outer_train_contract"],
            "trainer": cfg["implementation"]["e961_outer_train_trainer"],
            "driver": cfg["implementation"]["e961_outer_train_driver"],
            "tests": cfg["implementation"]["e961_outer_train_tests"],
            "run_receipt_schema": "boxfusion.tr3d.ca1m_e961_outer_train_run.r2",
            "authorization_consumption_schema": (
                "boxfusion.tr3d.ca1m_e961_outer_auth_consumption.r2"
            ),
            "training_started_claim_schema": (
                "boxfusion.tr3d.ca1m_e961_outer_training_started.r2"
            ),
        },
        "partition": "official_train_fold0_reused_dev_exact20",
        "metric": {
            "class_mode": "CA_class_agnostic",
            "coordinate_frame": "world",
            "box_geometry": "axis_aligned_AABB_from_8_corners",
            "ranking": "global_prediction_score",
            "duplicate_matching": "one_detection_per_scene_gt_per_threshold",
            "iou_comparison": "strict_greater_than_threshold",
            "iou_thresholds": [0.15, 0.25, 0.50],
        },
        "paired_comparison": {
            "same_exact20_scene_ids": True,
            "same_rgbd_point_builder": True,
            "same_point_array_sha256_per_scene": True,
            "same_point_inference_config": True,
            "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        },
        "terminal_oracle": {
            "anchor": "final_base_plus_B6_fold0_OOF",
            "each_anchor_score_model_excludes_scene": True,
            "candidate_source": "expanded_E961_outer_dev_TR3D",
            "matching": "candidate_and_anchor_must_share_best_GT",
            "near_anchor_iou": 0.15,
            "minimum_same_gt_iou_gain": 0.05,
            "replacement": "geometry_only_preserve_anchor_score_and_row_order",
            "deployable": False,
        },
        "continuation_gate": {
            "proposal_exact20_finite_ca_only": True,
            "min_replacements": 10,
            "min_replacement_scenes": 5,
            "min_delta_ap15": 0.0,
            "min_delta_ap25": 0.0,
            "min_delta_ap50": 0.005,
            "pass_authorizes_inner_roles": list(INNER_ROLES),
            "failure_action": (
                "stop_without_inner_training_or_fold1_or_official_validation_access"
            ),
        },
        "frozen_inputs_without_array_access": {
            "fold0_scene_list": cfg["scene_contract"],
            "selection_contract": cfg["selection_contract"],
            "point_parity": {
                "path": cfg["point_lineage"]["receipt_path"],
                "sha256": cfg["point_lineage"]["receipt_sha256"],
            },
            "point_inference_config": cfg["point_inference"],
            "final_base_b6_oof_anchor": evaluation["anchor_shadow"],
            "final_base_b6_oof_anchor_manifest": evaluation[
                "anchor_shadow_manifest"
            ],
            "same_point_path_v1_comparison": evaluation[
                "v1_fold0_comparison_manifest"
            ],
            "opaque_fold0_gt_inventory": evaluation["gt_shadow_inventory"],
        },
        "evaluation_config": {
            "path": str(source), "sha256": sha256_file(source),
        },
        "implementation": cfg["implementation"],
    }


def validate_invalidated_protocol_v1() -> dict[str, Any]:
    predecessor_path, predecessor = read_json(
        V1_PROTOCOL_PREREGISTRATION_PATH,
        "invalidated E961 protocol preregistration v1", immutable=True,
    )
    invalid_path, invalid = read_json(
        V1_PROTOCOL_INVALID_PATH,
        "E961 protocol-v1 invalidation receipt", immutable=True,
    )
    if (
        sha256_file(predecessor_path) != V1_PROTOCOL_PREREGISTRATION_SHA256
        or predecessor.get("schema")
        != "boxfusion.ca1m_tr3d_e961_outer_dev_protocol_preregistration.v1"
        or sha256_file(invalid_path) != V1_PROTOCOL_INVALID_SHA256
        or invalid.get("schema")
        != "boxfusion.ca1m_tr3d_e961_outer_dev_protocol_preregistration_invalid.v1"
        or invalid.get("invalid") is not True
        or invalid.get("superseded") is not True
        or invalid.get("formal_evaluation_authorized") is not False
        or (invalid.get("predecessor") or {}).get("sha256")
        != V1_PROTOCOL_PREREGISTRATION_SHA256
        or (invalid.get("replacement_requirement") or {}).get("v1_must_be_rejected")
        is not True
    ):
        raise ValueError("E961 protocol-v1 invalidation chain differs")
    return {
        "protocol_v1_sha256": V1_PROTOCOL_PREREGISTRATION_SHA256,
        "invalid_receipt_sha256": V1_PROTOCOL_INVALID_SHA256,
        "formal_v1_authorized": False,
    }


def seal_protocol_preregistration(
    config_path: Path,
) -> tuple[Path, dict[str, Any]]:
    source, cfg = load_config(config_path)
    validate_invalidated_protocol_v1()
    validate_preregisterable_static(cfg)
    payload = _expected_protocol_preregistration(source, cfg)
    guard_fixed_path(
        PROTOCOL_PREREGISTRATION_PATH, PIPELINE_ROOT,
        "protocol preregistration",
    )
    path = create_or_verify_json(
        PROTOCOL_PREREGISTRATION_PATH, payload,
        "E961 outer protocol preregistration",
    )
    return path, payload


def validate_protocol_preregistration(
    config_path: Path,
) -> tuple[Path, dict[str, Any]]:
    source, cfg = load_config(config_path)
    validate_invalidated_protocol_v1()
    expected = _expected_protocol_preregistration(source, cfg)
    path, actual = read_json(
        PROTOCOL_PREREGISTRATION_PATH,
        "E961 outer protocol preregistration", immutable=True,
    )
    if actual != expected:
        raise ValueError("E961 outer protocol preregistration differs")
    return path, actual


def seal_preregistration(
    config_path: Path, run_tag: str,
) -> tuple[Path, dict[str, Any]]:
    source, cfg = load_config(config_path)
    validate_preregisterable_static(cfg)
    validate_protocol_preregistration(config_path)
    payload = _expected_preregistration(source, cfg, run_tag)
    guard_fixed_path(PREREGISTRATION_PATH, PIPELINE_ROOT, "preregistration")
    path = create_or_verify_json(
        PREREGISTRATION_PATH, payload, "E961 outer evaluation preregistration"
    )
    return path, payload


def validate_preregistration(
    config_path: Path, run_tag: str,
) -> tuple[Path, dict[str, Any]]:
    source, cfg = load_config(config_path)
    validate_protocol_preregistration(config_path)
    expected = _expected_preregistration(source, cfg, run_tag)
    path, actual = read_json(
        PREREGISTRATION_PATH, "E961 outer evaluation preregistration", immutable=True
    )
    if actual != expected:
        raise ValueError("E961 outer evaluation preregistration differs")
    return path, actual


def continuation_gate(
    *, proposal_integrity_pass: bool, scene_count: int,
    replacement_count: int, replacement_scene_count: int,
    oracle_ap_delta: Mapping[str, float],
) -> dict[str, Any]:
    gate = _sealed_continuation_gate(
        proposal_integrity_pass=proposal_integrity_pass,
        scene_count=scene_count,
        replacement_count=replacement_count,
        replacement_scene_count=replacement_scene_count,
        oracle_ap_delta=oracle_ap_delta,
    )
    if gate.get("authorized_inner_roles") not in ([], list(INNER_ROLES)):
        raise ValueError("sealed continuation primitive returned unexpected roles")
    values = [
        gate["checks"]["proposal_exact20_finite_ca_only"],
        gate["checks"]["same_gt_gain_ge_0_05_replacements_ge_10"],
        gate["checks"]["same_gt_gain_ge_0_05_scenes_ge_5"],
        gate["checks"]["oracle_delta_ap15_nonnegative"],
        gate["checks"]["oracle_delta_ap25_nonnegative"],
        gate["checks"]["oracle_delta_ap50_at_least_0_005"],
    ]
    if gate.get("pass") is not all(value is True for value in values):
        raise ValueError("continuation gate is internally inconsistent")
    return gate


def validate_inner_authorization(role: str) -> dict[str, Any]:
    if role not in INNER_ROLES:
        raise ValueError("only the three preregistered inner roles are eligible")
    if STOP_PATH.exists() or STOP_PATH.is_symlink():
        raise PermissionError("E961 outer gate stopped inner training")
    source, value = read_json(
        INNER_AUTHORIZATION_PATH, "E961 inner training authorization", immutable=True
    )
    if (
        value.get("schema") != INNER_AUTHORIZATION_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("pass") is not True
        or value.get("authorized_roles") != list(INNER_ROLES)
        or role not in value.get("authorized_roles", [])
        or value.get("fold1_access") is not False
        or value.get("official_validation_access") is not False
    ):
        raise PermissionError("E961 inner authorization contract differs")
    receipt = value.get("continuation_receipt") or {}
    receipt_path = regular_file(
        Path(str(receipt.get("path", ""))), "E961 continuation receipt", immutable=True
    )
    if receipt_path != CONTINUATION_PATH or sha256_file(receipt_path) != receipt.get("sha256"):
        raise PermissionError("E961 continuation receipt binding differs")
    return {"authorized": True, "role": role, "path": str(source), "sha256": sha256_file(source)}


__all__ = [
    "ANCHOR_MANIFEST_SHA256", "ANCHOR_SHA256", "BINDING_PATH",
    "BINDING_SCHEMA", "COLLECTION_PATH", "COLLECTION_SCHEMA", "CONFIG_PATH",
    "CONFIG_SCHEMA", "CONTINUATION_PATH", "CONTINUATION_SCHEMA",
    "DIAGNOSTIC_ROOT", "FOLD0_SHA256", "GT_INVENTORY_SHA256",
    "INNER_AUTHORIZATION_PATH", "INNER_AUTHORIZATION_SCHEMA", "INNER_ROLES",
    "MANIFEST_ROOT", "NAMESPACE", "POINT_CONFIG_SHA256",
    "POINT_PARITY_SHA256", "PREREGISTRATION_PATH", "PREREGISTRATION_SCHEMA",
    "PROTOCOL_PREREGISTRATION_PATH", "PROTOCOL_PREREGISTRATION_SCHEMA",
    "REPORT_PATH", "REPORT_SCHEMA", "REPORT_ROOT", "SELECTION_SHA256",
    "STOP_PATH", "STOP_SCHEMA", "TRAIN_RUN_ROOT", "TRAIN_WORK_ROOT",
    "V1_COMPARISON_SHA256", "continuation_gate", "create_or_verify_json",
    "expected_training_receipt", "guard_fixed_path", "load_config", "match_targets", "metric_delta",
    "official_ca_ap", "read_json", "regular_directory", "regular_file",
    "same_gt_oracle_scene", "scene_ids", "seal_preregistration", "sha256_bytes",
    "seal_protocol_preregistration",
    "sha256_file", "validate_inner_authorization", "validate_preregisterable_static",
    "validate_invalidated_protocol_v1",
    "validate_preregistration", "validate_protocol_preregistration", "validate_run_tag",
]
