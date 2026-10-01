#!/usr/bin/env python3
"""Create-only preregistration for the R2 outer-dev continuation decision."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_xfit_r2_eval import (  # noqa: E402
    FOLD0_SHA256,
    POINT_PARITY_SHA256,
    PREREGISTRATION_SCHEMA,
    R2_AUTHORIZATION_SHA256,
    XFIT_CONTRACT_SHA256,
    create_or_verify_json,
    regular_file,
    sha256_file,
)


OUTPUT = ROOT / (
    "manifests/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
    "preregistration.json"
)
ARTIFACTS = {
    "fold0_scene_list": (
        Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/"
            "tr3d_ca1m_visible_xfit_v2_formal/splits/predict_fold0.txt"
        ), FOLD0_SHA256,
    ),
    "xfit_contract": (
        Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/"
            "tr3d_ca1m_visible_xfit_v2_formal/XFIT_FORMAL_CONTRACT.json"
        ), XFIT_CONTRACT_SHA256,
    ),
    "r2_training_authorization": (
        Path(
            "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/data/"
            "tr3d_ca1m_visible_xfit_v2_formal/TRAINING_AUTHORIZATION_R2.json"
        ), R2_AUTHORIZATION_SHA256,
    ),
    "point_parity": (
        ROOT / (
            "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/"
            "lineage_training_point_parity_v4.json"
        ), POINT_PARITY_SHA256,
    ),
    "anchor_shadow": (
        ROOT / (
            "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
            "fold0_final_base_b6_v2_oof.npz"
        ), "202340ff97c3de5f49a969ec983222cae0772913b831f808f870aea71a639c88",
    ),
    "anchor_shadow_manifest": (
        ROOT / (
            "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
            "fold0_final_base_b6_v2_oof.manifest.json"
        ), "b574d029adcf3d24735869d314a7e2f2dd39e6ad05ea16a6a193d2171dec1669",
    ),
    "v1_fold0_comparison_manifest": (
        ROOT / (
            "inputs/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
            "v1_fold0_proposal_comparison_manifest.json"
        ), "b38d55b42eceef21639943a41c689ce45c1eb698dfe46e0f4e43280500e967f1",
    ),
    "v1_sealed_raw_diagnostic": (
        ROOT / (
            "manifests/ca1m_tr3d_terminal_ca_native_train100_v3/"
            "checkpoint_dev_diagnostic_v4.receipt.json"
        ), "ab097945760157b824a04075e825df9981e9c6af52754f72c020ff54981f9b33",
    ),
    "gt_shadow_inventory": (
        ROOT / (
            "manifests/ca1m_tr3d_benefit_gate_final_base_v4/"
            "derived_train_gt_shadow_inventory_v1.json"
        ), "6c3bdfd666ca49558ac390197abeec588949f05e33d9c8a18d1b5c8326d9e9a7",
    ),
}


def seal() -> dict:
    records = {}
    for name, (path, expected) in ARTIFACTS.items():
        source = regular_file(path, name)
        if sha256_file(source) != expected:
            raise ValueError(f"preregistration input SHA256 drift: {name}")
        records[name] = {"path": str(source), "sha256": expected}
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "complete": True,
        "create_only": True,
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
        "raw_detector_ap_role": "diagnostic_only_no_checkpoint_selection",
        "checkpoint_selection_authorized": False,
        "oracle_role": "non_deployable_same_best_gt_geometry_headroom",
        "continuation_gate": {
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
        },
        "failure_action": (
            "stop_without_training_inner_models_or_opening_fold1_or_official_validation"
        ),
        "inputs": records,
    }
    create_or_verify_json(OUTPUT, payload, "xfit-R2 outer-dev preregistration")
    return payload


if __name__ == "__main__":
    print(json.dumps(seal(), indent=2, sort_keys=True))
