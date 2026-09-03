#!/usr/bin/env python3
"""Seal the CA-1M terminal-gate v4 science contract before any GT join."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_terminal_gate_v4 import (  # noqa: E402
    BENEFIT_TARGET,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    GATE_TRAIN_FOLDS,
    LOCKED_INTERNAL_FOLDS,
    PREREGISTRATION_SCHEMA,
    QUALITY_TARGET,
    SELECTION_RULE,
    THRESHOLD_DEV_FOLDS,
    preregistration_code_records,
    preregistration_science_contract,
    preregistration_upstream_records,
    validate_static_config,
    write_binding_create_only,
)
from boxfusion.ca1m_tr3d_terminal_v4 import sha256_file  # noqa: E402


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    _, cfg = validate_static_config(args.config)
    if (
        cfg.get("state") != "ready_after_all_train100_seals"
        or cfg.get("run_authorized") is not True
        or cfg.get("train_only") is not True
        or cfg.get("ground_truth_used_only_after_candidate_seal") is not True
        or cfg.get("validation_ground_truth_access") is not False
        or cfg.get("validation_prediction_access") is not False
    ):
        raise PermissionError("terminal gate v4 is not ready for preregistration")
    payload = {
        "schema": PREREGISTRATION_SCHEMA,
        "complete": True,
        "train_only": True,
        "sealed_before_first_gt_join": True,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "validation_prediction_access": False,
        "official_validation_comparable": False,
        "locked_internal_fold1_gt_access": False,
        "fit_fold_ids": list(GATE_TRAIN_FOLDS),
        "threshold_dev_fold_ids": list(THRESHOLD_DEV_FOLDS),
        "locked_internal_fold_ids": list(LOCKED_INTERNAL_FOLDS),
        "anchor_score_source": "ca1m_native_b6_final_base_all_fold_oof_row_scores_v2",
        "deploy_b6_scores_used_for_stacked_training": False,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "quality_target": QUALITY_TARGET,
        "benefit_target": BENEFIT_TARGET,
        "selection_rule": SELECTION_RULE,
        "science": preregistration_science_contract(),
        "code": preregistration_code_records(),
        "upstream": preregistration_upstream_records(cfg),
    }
    write_binding_create_only(args.output, payload)
    output = args.output.resolve()
    print(json.dumps({
        "complete": True,
        "ground_truth_access": False,
        "sealed_before_first_gt_join": True,
        "path": str(output),
        "sha256": sha256_file(output),
        "code": payload["code"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
