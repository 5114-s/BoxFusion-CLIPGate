#!/usr/bin/env python3
"""Seal the final self-excluding R2 outer-dev evaluation contract.

This is run before any R2 fold0 proposal or GT access.  It extends the earlier
science preregistration with the final evaluation-config semantics, complete
implementation inventory, and exact path/hash binding for every allowed input.
The GT inventory is treated as opaque metadata bytes; no GT array is opened.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_xfit_r2_eval import (  # noqa: E402
    NAMESPACE,
    PREREGISTRATION_SCHEMA,
    PREREGISTRATION_V2_SCHEMA,
    create_or_verify_json,
    evaluation_config_contract_sha256,
    load_config,
    preregistration_input_contract,
    read_json,
    regular_file,
    sha256_file,
)


CONFIG = ROOT / "config/ca1m_tr3d_xfit_r2_outer_dev_eval_v1.json"
V1 = ROOT / (
    "manifests/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/preregistration.json"
)
V1_SHA256 = "f215ed1ef22c0e167911694a2416c949379febce682310b26d2a97208b46b244"
OUTPUT = ROOT / (
    "manifests/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/preregistration_v2.json"
)


def seal() -> dict:
    config_path, cfg = load_config(CONFIG)
    v1_path, v1 = read_json(V1, "outer-dev preregistration-v1", immutable=True)
    if (
        sha256_file(v1_path) != V1_SHA256
        or v1.get("schema") != PREREGISTRATION_SCHEMA
        or v1.get("complete") is not True
        or v1.get("sealed_before_r2_fold0_gt_access") is not True
        or v1.get("r2_fold0_gt_access_at_seal") is not False
        or v1.get("r2_proposal_access_at_seal") is not False
    ):
        raise ValueError("preregistration-v1 predecessor differs")

    inputs = preregistration_input_contract(cfg)
    for name, record in inputs.items():
        path = regular_file(Path(record["path"]), f"preregistration-v2 {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"preregistration-v2 input SHA256 drift: {name}")

    forbidden_outputs = {
        "checkpoint_binding": Path(cfg["training"]["binding_path"]),
        "proposal_collection": Path(cfg["proposal_stage"]["collection_manifest"]),
        "evaluation_report": Path(cfg["evaluation_stage"]["report"]),
        "continuation_receipt": Path(
            cfg["evaluation_stage"]["continuation_receipt"]
        ),
    }
    present = [name for name, path in forbidden_outputs.items() if path.exists()]
    if present:
        raise FileExistsError(
            f"preregistration-v2 must precede R2 evaluation artifacts: {present}"
        )
    proposal_root = Path(cfg["proposal_stage"]["output_root"])
    if proposal_root.exists() and any(proposal_root.iterdir()):
        raise FileExistsError("R2 proposal namespace is non-empty before preregistration-v2")

    gate = {
        "proposal_exact20_finite_ca_only": True,
        "same_gt_min_iou_gain": 0.05,
        "min_replacements": 10,
        "min_replacement_scenes": 5,
        "min_delta_ap15": 0.0,
        "min_delta_ap25": 0.0,
        "min_delta_ap50": 0.005,
        "pass_authorizes_inner_roles": [
            "inner_holdout2", "inner_holdout3", "inner_holdout4"
        ],
    }
    payload = {
        "schema": PREREGISTRATION_V2_SCHEMA,
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
        "predecessor": {
            "path": str(v1_path),
            "sha256": V1_SHA256,
            "schema": PREREGISTRATION_SCHEMA,
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
    create_or_verify_json(OUTPUT, payload, "xfit-R2 outer-dev preregistration-v2")
    return payload


if __name__ == "__main__":
    print(json.dumps(seal(), indent=2, sort_keys=True))
