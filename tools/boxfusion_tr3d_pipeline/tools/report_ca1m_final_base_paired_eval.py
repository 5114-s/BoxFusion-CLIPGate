#!/usr/bin/env python3
"""Seal the paired fixed10 CA-1M final-base evaluation into one report.

This command only reads frozen inference/evaluation evidence.  The resulting
fixed10 report may authorize train100 evidence collection, but it can never
authorize a canonical active route.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "boxfusion.ca1m_final_base_paired_eval.v1"
IDENTITY_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
RECOVERY_SCHEMA = "boxfusion.ca1m_final_base_inference_recovery_receipt.v1"
THRESHOLDS = (("AP15", "0.150000"), ("AP25", "0.250000"), ("AP50", "0.500000"))
METRIC_NAMES = ("mAP", "APrec", "ARecall")

_BATCH_RE = re.compile(r"^Eval batch: ([0-9]+) scan_idx ([0-9]{8})$")
_PRED_RE = re.compile(r"^pred_path (/.*/([0-9]{8})_boxes[.]pkl)$")
_PRED_COUNT_RE = re.compile(r"^pred_labels Counter[(][{]0: ([0-9]+)[}][)]$")
_IOU_RE = re.compile(r"^-+ iou_thresh: ([0-9]+[.][0-9]+) -+$")
_METRIC_RE = re.compile(r"^eval (mAP|APrec|ARecall): ([0-9]+(?:[.][0-9]+)?)$")
_HASH_LINE_RE = re.compile(r"^([0-9a-f]{64})  ([^\t\r\n]+)$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, label: str) -> Path:
    raw = Path(path)
    if raw.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {raw}")
    resolved = raw.resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise ValueError(f"{label} must be a non-empty regular file: {resolved}")
    return resolved


def load_json(path: Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = regular_file(path, label)
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return resolved, value


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def validate_identity(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == IDENTITY_SCHEMA, "invalid identity-audit schema")
    _require(value.get("ok") is True, "identity audit did not pass")
    _require(value.get("dataset") == "CA1M", "identity audit is not CA1M")
    _require(value.get("split") == "validation_fixed10", "identity audit is not fixed10")
    _require(value.get("scene_count") == 10, "identity audit must contain 10 scenes")
    _require(value.get("ground_truth_access") is False, "inference audit accessed GT")
    _require(value.get("evaluation_invoked") is False, "inference audit invoked evaluation")
    _require(value.get("training_invoked") is False, "inference audit invoked training")
    _require(value.get("clip_appearance_gate_active") is True, "CLIP gate was not active")
    _require(value.get("reliable_view_top_k") == 3, "reliable-view TopK is not 3")
    _require(
        value.get("scannet_learned_b6_or_gate_reused") is False,
        "ScanNet learned B6/gate was reused",
    )
    _require(
        value.get("downstream_native_b6_recollection_required") is True,
        "CA-native B6 recollection is not required by the audit",
    )
    same_run = value.get("same_run")
    _require(isinstance(same_run, Mapping), "identity audit lacks same-run summary")
    for key in ("byte_identity_scenes", "semantic_identity_scenes", "hard_link_identity_scenes"):
        _require(same_run.get(key) == 10, f"same_run.{key} is not 10")
    paired = value.get("paired_g0_control")
    _require(isinstance(paired, Mapping), "identity audit lacks paired G0 control")
    _require(paired.get("identity_expected") is False, "control/active identity was expected")
    _require(paired.get("scenes_with_any_difference") == 10, "not all scenes differ from control")
    _require(paired.get("control_rows") == 674, "unexpected fixed10 control row count")
    _require(paired.get("active_rows") == 690, "unexpected fixed10 active row count")
    _require(paired.get("row_count_delta") == 16, "unexpected fixed10 row-count delta")
    per_scene = value.get("per_scene")
    _require(isinstance(per_scene, Mapping) and len(per_scene) == 10, "invalid per-scene audit")
    for scene, row in per_scene.items():
        _require(re.fullmatch(r"[0-9]{8}", str(scene)) is not None, "invalid audit scene id")
        _require(isinstance(row, Mapping), f"invalid audit row for {scene}")
        for key in ("byte_identity", "semantic_identity", "hard_link_identity"):
            _require(row.get(key) is True, f"{scene}: {key} did not pass")


def validate_recovery(value: Mapping[str, Any]) -> None:
    _require(value.get("schema") == RECOVERY_SCHEMA, "invalid recovery-receipt schema")
    for key in ("complete", "prediction_phase_complete", "all_frozen_source_hashes_match_before_recovery_edits", "zero_proposal_rows_are_control_active_exact"):
        _require(value.get(key) is True, f"recovery receipt field {key} did not pass")
    for key in ("control_scene_count", "active_scene_count", "identity_scene_count", "boxer_diagnostic_scene_count", "inference_log_scene_count"):
        _require(value.get(key) == 10, f"recovery receipt field {key} is not 10")
    _require(value.get("evaluation_started") is False, "evaluation had begun before recovery")
    _require(
        value.get("failure_stage") == "post_inference_pre_evaluation_identity_audit",
        "unexpected recovery stage",
    )
    _require(
        value.get("recovery_policy")
        == "repair audit-only contract; do not rerun or overwrite predictions",
        "unexpected recovery policy",
    )


def parse_evaluator_hashes(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _HASH_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"malformed evaluator hash row: {line!r}")
        digest, relative = match.groups()
        _require(relative not in rows, f"duplicate evaluator source: {relative}")
        source = regular_file(ROOT / relative, f"evaluator source {relative}")
        try:
            source.relative_to(ROOT)
        except ValueError as error:
            raise ValueError(f"evaluator source escapes repository: {relative}") from error
        _require(sha256_file(source) == digest, f"evaluator source hash drift: {relative}")
        rows[relative] = digest
    expected = {
        "evaluation/eval_ca1m.py",
        "evaluation/utils/ap_helper.py",
        "evaluation/utils/eval_det.py",
        "evaluation/utils/box_util.py",
    }
    _require(set(rows) == expected, "evaluator source set differs from the frozen protocol")
    return rows


def parse_eval_log(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8", errors="strict").splitlines()
    _require(lines.count("Num scenes in test dataset: 10") == 1, f"{path}: invalid scene count")
    batches = [match.groups() for line in lines if (match := _BATCH_RE.fullmatch(line.strip()))]
    _require(len(batches) == 10, f"{path}: expected exactly 10 Eval batch rows")
    _require([int(index) for index, _ in batches] == list(range(10)), f"{path}: invalid batch indices")
    scenes = [scene for _, scene in batches]
    _require(len(set(scenes)) == 10, f"{path}: repeated scene ids")

    pred_rows = [match.groups() for line in lines if (match := _PRED_RE.fullmatch(line.strip()))]
    _require(len(pred_rows) == 10, f"{path}: expected exactly 10 pred_path rows")
    pred_paths = [Path(raw) for raw, _ in pred_rows]
    _require([scene for _, scene in pred_rows] == scenes, f"{path}: pred_path scene order differs")
    prediction_roots = {item.parent.resolve() for item in pred_paths}
    _require(len(prediction_roots) == 1, f"{path}: multiple prediction roots")
    prediction_root = next(iter(prediction_roots))
    _require(prediction_root.is_dir() and not prediction_root.is_symlink(), f"{path}: invalid prediction root")
    for item in pred_paths:
        regular_file(item, f"logged prediction {item.name}")

    counts = [int(match.group(1)) for line in lines if (match := _PRED_COUNT_RE.fullmatch(line.strip()))]
    _require(len(counts) == 10, f"{path}: expected exactly 10 prediction counts")

    ious = [match.group(1) for line in lines if (match := _IOU_RE.fullmatch(line.strip()))]
    _require(ious == [raw for _, raw in THRESHOLDS], f"{path}: IoU threshold sequence differs")
    rows = [match.groups() for line in lines if (match := _METRIC_RE.fullmatch(line.strip()))]
    expected_names = [name for _ in THRESHOLDS for name in METRIC_NAMES]
    _require([name for name, _ in rows] == expected_names, f"{path}: expected exactly 3 metric triplets")
    metrics: dict[str, dict[str, float]] = {}
    for index, (label, _) in enumerate(THRESHOLDS):
        triplet = rows[index * 3 : index * 3 + 3]
        metrics[label] = {name: float(raw) for name, raw in triplet}
    return {
        "scenes": scenes,
        "prediction_root": str(prediction_root),
        "prediction_rows": sum(counts),
        "per_scene_prediction_rows": dict(zip(scenes, counts)),
        "metrics": metrics,
    }


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to overwrite report: {target}") from error
        target.chmod(0o444)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    report_path = args.report.resolve()
    if report_path.exists() or report_path.is_symlink():
        raise FileExistsError(f"refusing to overwrite report: {report_path}")

    identity_path, identity = load_json(args.identity_audit, "identity audit")
    recovery_path, recovery = load_json(args.recovery_receipt, "recovery receipt")
    hashes_path = regular_file(args.evaluator_hashes, "evaluator hash manifest")
    control_path = regular_file(args.control_log, "control evaluation log")
    active_path = regular_file(args.active_log, "active evaluation log")
    validate_identity(identity)
    validate_recovery(recovery)
    evaluator_sources = parse_evaluator_hashes(hashes_path)
    control = parse_eval_log(control_path)
    active = parse_eval_log(active_path)

    scenes = control["scenes"]
    _require(active["scenes"] == scenes, "control/active scene order differs")
    _require(set(identity["per_scene"]) == set(scenes), "logs and identity-audit scenes differ")
    _require(control["prediction_rows"] == identity["paired_g0_control"]["control_rows"], "control row count disagrees with identity audit")
    _require(active["prediction_rows"] == identity["paired_g0_control"]["active_rows"], "active row count disagrees with identity audit")
    for scene in scenes:
        audit_row = identity["per_scene"][scene]
        _require(active["per_scene_prediction_rows"][scene] == audit_row["active_rows"], f"{scene}: active row count disagrees")
        _require(control["per_scene_prediction_rows"][scene] == audit_row["paired_g0_control"]["control_rows"], f"{scene}: control row count disagrees")

    baseline = control["metrics"]
    candidate = active["metrics"]
    delta = {
        threshold: {
            name: round(candidate[threshold][name] - baseline[threshold][name], 6)
            for name in METRIC_NAMES
        }
        for threshold, _ in THRESHOLDS
    }
    delta_percentage_points = {
        threshold: {name: round(value * 100.0, 4) for name, value in row.items()}
        for threshold, row in delta.items()
    }
    positive_map_at_all_thresholds = all(delta[label]["mAP"] > 0.0 for label, _ in THRESHOLDS)

    def artifact(path: Path) -> dict[str, str]:
        return {"path": str(path), "sha256": sha256_file(path)}

    report: dict[str, Any] = {
        "schema": SCHEMA,
        "complete": True,
        "dataset": "CA1M",
        "split": "validation_fixed10",
        "scene_count": 10,
        "scene_order": scenes,
        "scene_list_sha256": identity["scene_list_sha256"],
        "paired_official_evaluation": True,
        "ground_truth_access": True,
        "ground_truth_access_scope": "official_evaluator_logs_and_this_report_only",
        "inference_ground_truth_access": False,
        "training_invoked": False,
        "fixed10_is_diagnostic_only": True,
        "inputs": {
            "identity_audit": artifact(identity_path),
            "inference_recovery_receipt": artifact(recovery_path),
            "evaluator_hash_manifest": artifact(hashes_path),
            "control_evaluation_log": artifact(control_path),
            "active_evaluation_log": artifact(active_path),
        },
        "evaluator_sources": evaluator_sources,
        "prediction_roots": {
            "control": control["prediction_root"],
            "active": active["prediction_root"],
        },
        "prediction_rows": {
            "control": control["prediction_rows"],
            "active": active["prediction_rows"],
            "delta": active["prediction_rows"] - control["prediction_rows"],
        },
        "control": baseline,
        "active": candidate,
        "delta": delta,
        "delta_percentage_points": delta_percentage_points,
        "positive_map_at_all_thresholds": positive_map_at_all_thresholds,
        "decision": {
            "train100_final_base_collection_authorized": positive_map_at_all_thresholds,
            "ca1m_native_b6_retraining_required": True,
            "canonical_active_authorized": False,
            "formal_canonical103_authorized": False,
            "live_mutation_authorized": False,
            "basis": "fixed10 diagnostic supports train100 evidence collection only",
        },
    }
    _require(
        report["decision"]["train100_final_base_collection_authorized"] is True,
        "fixed10 mAP did not improve at all three thresholds; train100 is not authorized",
    )
    write_json_create_only(report_path, report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--identity-audit", type=Path, required=True)
    value.add_argument("--recovery-receipt", type=Path, required=True)
    value.add_argument("--evaluator-hashes", type=Path, required=True)
    value.add_argument("--control-log", type=Path, required=True)
    value.add_argument("--active-log", type=Path, required=True)
    value.add_argument("--report", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    report = build_report(parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
