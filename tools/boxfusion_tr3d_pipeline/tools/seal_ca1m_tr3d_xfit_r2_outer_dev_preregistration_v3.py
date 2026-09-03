#!/usr/bin/env python3
"""Seal eval-config v2 and its fixed outer-wrapper-log contract before GT."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_xfit_r2_eval import (  # noqa: E402
    NAMESPACE,
    OUTER_WRAPPER_LOG,
    PREREGISTRATION_V2_SCHEMA,
    PREREGISTRATION_V3_SCHEMA,
    create_or_verify_json,
    evaluation_config_contract_sha256,
    load_config,
    preregistration_input_contract,
    read_json,
    regular_file,
    sha256_file,
)


CONFIG = ROOT / "config/ca1m_tr3d_xfit_r2_outer_dev_eval_v2.json"
V2 = ROOT / (
    "manifests/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/preregistration_v2.json"
)
V2_SHA256 = "ac432705669efad65da7337c9f083eeb9e8ac93c7b2da279f77af929c358d347"
OUTPUT = ROOT / (
    "manifests/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/preregistration_v3.json"
)


def seal() -> dict:
    config_path, cfg = load_config(CONFIG)
    predecessor_path, predecessor = read_json(
        V2, "outer-dev preregistration-v2", immutable=True
    )
    if (
        sha256_file(predecessor_path) != V2_SHA256
        or predecessor.get("schema") != PREREGISTRATION_V2_SCHEMA
        or predecessor.get("complete") is not True
        or predecessor.get("sealed_before_r2_fold0_gt_access") is not True
        or predecessor.get("r2_fold0_gt_access_at_seal") is not False
        or predecessor.get("r2_proposal_access_at_seal") is not False
    ):
        raise ValueError("preregistration-v2 predecessor differs")

    inputs = preregistration_input_contract(cfg)
    for name, record in inputs.items():
        path = regular_file(Path(record["path"]), f"preregistration-v3 {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"preregistration-v3 input SHA256 drift: {name}")

    forbidden = (
        Path(cfg["training"]["binding_path"]),
        Path(cfg["proposal_stage"]["collection_manifest"]),
        Path(cfg["evaluation_stage"]["report"]),
        Path(cfg["evaluation_stage"]["continuation_receipt"]),
    )
    if any(path.exists() for path in forbidden):
        raise FileExistsError("preregistration-v3 must precede R2 evaluation outputs")
    proposal_root = Path(cfg["proposal_stage"]["output_root"])
    if proposal_root.exists() and any(proposal_root.iterdir()):
        raise FileExistsError("R2 proposal namespace is non-empty before preregistration-v3")

    gate = dict(predecessor["continuation_gate"])
    outer_contract = {
        "path": str(OUTER_WRAPPER_LOG),
        "regular_non_symlink_required": True,
        "terminal_line": "TRAIN_EXIT=0",
        "terminal_line_must_be_last_and_unique": True,
        "r2_runner_preamble_required": True,
        "role": "outer_dev",
        "optimizer_updates": 11268,
        "error_markers_forbidden": True,
        "binding_snapshot_required": True,
    }
    payload = {
        "schema": PREREGISTRATION_V3_SCHEMA,
        "complete": True,
        "create_only": True,
        "namespace": NAMESPACE,
        "train_only": True,
        "partition": "threshold_dev_fold0",
        "fold0_role": "reused_dev",
        "fold0_prior_exposure": [
            "v1_checkpoint_diagnostic", "terminal_v4_gate"
        ],
        "fold0_scene_count": 20,
        "fold1_access": False,
        "official_validation_access": False,
        "sealed_before_r2_fold0_gt_access": True,
        "r2_fold0_gt_access_at_seal": False,
        "r2_proposal_access_at_seal": False,
        "gpu_started_by_preregistration": False,
        "gt_array_content_access_at_seal": False,
        "gt_inventory_binding_is_opaque_metadata_only": True,
        "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        "checkpoint_selection_authorized": False,
        "oracle_role": "non_deployable_same_best_gt_geometry_headroom",
        "continuation_gate": gate,
        "failure_action": (
            "stop_without_training_inner_models_or_opening_fold1_or_"
            "official_validation"
        ),
        "outer_wrapper_log_contract": outer_contract,
        "outer_wrapper_log_complete_at_preregistration": False,
        "predecessor": {
            "path": str(predecessor_path),
            "sha256": V2_SHA256,
            "schema": PREREGISTRATION_V2_SCHEMA,
        },
        "evaluation_config_contract": {
            "path": str(config_path),
            "normalizer": (
                "canonical_sorted_json_with_evaluation_stage."
                "preregistration_replaced_by_schema_self_marker"
            ),
            "semantic_sha256": evaluation_config_contract_sha256(cfg),
            "binds_complete_implementation_inventory": True,
        },
        "implementation": cfg["implementation"],
        "inputs": inputs,
    }
    create_or_verify_json(OUTPUT, payload, "xfit-R2 outer-dev preregistration-v3")
    return payload


if __name__ == "__main__":
    print(json.dumps(seal(), indent=2, sort_keys=True))
