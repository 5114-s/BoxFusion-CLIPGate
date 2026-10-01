"""Fail-closed binding for the CA-native TR3D final checkpoint.

The terminal collection is not allowed to accept a naked checkpoint or model
config.  A completed CA-1M scratch-training run must first be sealed into the
manifest defined here.  Loading the manifest re-hashes every referenced file
and revalidates the effective training config and driver log.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from typing import Any, Mapping


SCHEMA = "boxfusion.tr3d.ca1m_checkpoint_binding.v1"
AUDIT_SCHEMA = "boxfusion.tr3d.ca1m_checkpoint_binding_audit.v1"
DEV_DIAGNOSTIC_SCHEMA = "boxfusion.tr3d.ca1m_train_only_dev_eval.v1"
DEV_RECEIPT_SCHEMA = "boxfusion.tr3d.ca1m_checkpoint_dev_diagnostic_receipt.v1"
EXPECTED_WORK_ROOT = Path(
    "/extra/ZhaoX/tr3d_ca1m_work_dirs/ca1m_fg_scratch_seed0_fp32_gb16_v1"
)
EXPECTED_SOURCE_CONFIG = Path(
    "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev/"
    "config/tr3d/tr3d_ca1m_foreground.py"
)
EXPECTED_SOURCE_CONFIG_SHA256 = (
    "0185b32df0854189d1f5827060810c0ee7056aa1c7c5afbd19b69b255843d567"
)
EXPECTED_CHECKPOINT_NAME = "epoch_12.pth"
EXPECTED_EFFECTIVE_CONFIG_NAME = "tr3d_ca1m_foreground.py"
FORBIDDEN_SCANNET_SHA256 = frozenset(
    {
        # Converted official ScanNet initialization.
        "09f2f650540716556719d2858d9a484dcbf682e2e94576887855b9b637b6492e",
        # Completed local ScanNet foreground checkpoint.
        "a484fd79093aa3004f4d2984e7ad8763c5d5cbec7edd04172d775643a8436448",
        # ScanNet foreground inference/training configs.
        "e74b29335f32baa6595bcc84a9b3e4fdd14b92a7044abd408a44de95fc360dc4",
        "86ffb8d6ff8dcc2f376057e9bd1f7d1f7f87a294eb75922ad5a1e73755c79905",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SCENE_SPLIT_SHA256 = {
    "weights_train": "7f0a22c660f7f9bd44137f5049c694393e038f5ab97ec55053443bfc00967478",
    "threshold_dev": "9c886ca85ba599881797b25a49d2fc72dd136d255a245a09fe1cf17cbce735a7",
    "locked_internal_check": "d6238bae873c98737858ac3a84c0706091fa9a91113321ac9736a8d64de6d6b6",
    "train100": "35321e9942dc5d512db2952b9ca6228b1291127e0c13fd92aa458f2d7eb7f9fd",
}


@dataclass(frozen=True)
class CA1MTR3DCheckpointBinding:
    manifest_path: Path
    manifest_sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    effective_config_path: Path
    effective_config_sha256: str
    source_config_path: Path
    source_config_sha256: str
    training_log_path: Path
    training_log_sha256: str
    work_root: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_file(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    value = path.resolve()
    if not value.is_file() or value.is_symlink() or value.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {value}")
    return value


def regular_directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    value = path.resolve()
    if not value.is_dir() or value.is_symlink():
        raise FileNotFoundError(f"missing regular {name}: {value}")
    return value


def _safe_literal(node: ast.AST) -> Any:
    """Evaluate MMEngine's dumped ``dict(key=value)`` literals only."""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_literal(value) for value in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_literal(value) for value in node.elts)
    if isinstance(node, ast.Set):
        return {_safe_literal(value) for value in node.elts}
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise ValueError("dictionary unpacking is not a safe literal")
        return {
            _safe_literal(key): _safe_literal(value)
            for key, value in zip(node.keys, node.values)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
        and all(keyword.arg is not None for keyword in node.keywords)
    ):
        return {
            str(keyword.arg): _safe_literal(keyword.value)
            for keyword in node.keywords
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_literal(node.operand)
        if not isinstance(value, (int, float, complex)):
            raise ValueError("unary safe literal is not numeric")
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ValueError(f"unsupported safe literal node: {type(node).__name__}")


def _literal_assignments(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=os.fspath(path))
    values: dict[str, Any] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            try:
                values[statement.targets[0].id] = _safe_literal(statement.value)
            except (ValueError, TypeError):
                continue
    return values


def _contains_non_null_initializer(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if key in {"pretrained", "init_cfg"} and nested is not None:
                return True
            if _contains_non_null_initializer(nested):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_non_null_initializer(item) for item in value)
    return False


def validate_effective_config(path: Path, work_root: Path) -> dict[str, Any]:
    source = regular_file(path, "effective CA-1M config")
    values = _literal_assignments(source)
    required = {
        "load_from", "resume", "model", "optim_wrapper", "train_cfg",
        "train_dataloader", "test_cfg", "test_dataloader", "test_evaluator",
        "randomness", "work_dir",
    }
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"effective config lacks literal fields: {missing}")
    if values["load_from"] is not None or values["resume"] is not False:
        raise ValueError("effective config did not start from random scratch")
    if _contains_non_null_initializer(values["model"]):
        raise ValueError("effective config contains pretrained/init_cfg weights")
    head = (values["model"].get("bbox_head") or {})
    if (
        head.get("type") != "TR3DClassAgnosticHead"
        or int(head.get("num_reg_outs", -1)) != 6
    ):
        raise ValueError("effective config is not CA class-agnostic TR3D")
    wrapper = values["optim_wrapper"]
    optimizer = wrapper.get("optimizer") or {}
    if (
        wrapper.get("type") != "OptimWrapper"
        or optimizer.get("type") != "AdamW"
        or float(optimizer.get("lr", -1.0)) != 0.001
        or float(optimizer.get("weight_decay", -1.0)) != 0.0001
    ):
        raise ValueError("effective config is not the frozen FP32 scratch optimizer")
    train_cfg = values["train_cfg"]
    if (
        train_cfg.get("type") != "EpochBasedTrainLoop"
        or int(train_cfg.get("max_epochs", -1)) != 12
        or int(train_cfg.get("val_interval", -1)) != 12
    ):
        raise ValueError("effective config is not the fixed 12-epoch loop")
    loader = values["train_dataloader"]
    dataset = loader.get("dataset") or {}
    leaf = dataset.get("dataset") or {}
    if (
        int(loader.get("batch_size", -1)) < 1
        or int(loader.get("num_workers", -1)) != 4
        or loader.get("persistent_workers") is not True
        or dataset.get("type") != "RepeatDataset"
        or int(dataset.get("times", -1)) != 15
        or leaf.get("type") != "TR3DForegroundCA1MDataset"
        or Path(str(leaf.get("ann_file", ""))).name
        != "ca1m_infos_weights_train_foreground.pkl"
    ):
        raise ValueError("effective config train loader differs from frozen CA protocol")
    if any(values[name] is not None for name in (
        "test_cfg", "test_dataloader", "test_evaluator"
    )):
        raise ValueError("effective config can access a forbidden test dataset")
    if values["randomness"] != {"deterministic": True, "seed": 0}:
        raise ValueError("effective config randomness contract differs")
    if Path(str(values["work_dir"])).resolve() != work_root.resolve():
        raise ValueError("effective config work_dir differs from fixed CA work root")
    return {
        "per_process_batch": int(loader["batch_size"]),
        "workers_per_process": int(loader["num_workers"]),
        "fixed_epochs": int(train_cfg["max_epochs"]),
        "repeat": int(dataset["times"]),
        "precision": "fp32",
        "initialization": "random_scratch",
    }


def validate_training_log(path: Path, per_process_batch: int) -> dict[str, Any]:
    source = regular_file(path, "formal training driver log")
    text = source.read_text(encoding="utf-8", errors="strict")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[-1] != "TRAIN_EXIT=0" or lines.count("TRAIN_EXIT=0") != 1:
        raise ValueError("training log must end with exactly one TRAIN_EXIT=0")
    if any(line.startswith("TRAIN_EXIT=") and line != "TRAIN_EXIT=0" for line in lines):
        raise ValueError("training log contains a failed TRAIN_EXIT marker")
    required_fragments = (
        "initialization: random scratch (no ScanNet checkpoint/module)",
        "resume: 0; AMP: 0",
        "precision/CuBLAS: FP32/:4096:8",
        "protocol: fixed 12 epochs; only epoch_12 checkpoint",
        "Fixed CA-1M epoch-12 checkpoint completed:",
    )
    missing = [fragment for fragment in required_fragments if fragment not in text]
    if missing:
        raise ValueError(f"training log lacks frozen protocol evidence: {missing}")
    batch_matches = re.findall(
        r"per-GPU/global batch:\s*([0-9]+)/([0-9]+);\s*workers:\s*([0-9]+)",
        text,
    )
    if len(batch_matches) != 1:
        raise ValueError("training log must contain one batch/worker declaration")
    per_gpu, global_batch, workers = map(int, batch_matches[0])
    if per_gpu != per_process_batch or global_batch != 16 or workers != 4:
        raise ValueError("training log batch/worker protocol differs")
    world_matches = re.findall(r"GPUs: .* \(([0-9]+) processes\)", text)
    if len(world_matches) != 1:
        raise ValueError("training log must contain one distributed world size")
    world_size = int(world_matches[0])
    if world_size < 1 or per_gpu * world_size != 16:
        raise ValueError("training log does not prove global batch 16")
    return {
        "world_size": world_size,
        "per_process_batch": per_gpu,
        "global_batch": global_batch,
        "workers_per_process": workers,
        "precision": "fp32",
        "cublas_workspace_config": ":4096:8",
        "exit_marker": "TRAIN_EXIT=0",
    }


def build_binding_payload(
    *,
    work_root: Path,
    source_config: Path,
    training_log: Path,
) -> dict[str, Any]:
    root = regular_directory(work_root, "fixed CA-1M work root")
    if root != EXPECTED_WORK_ROOT.resolve():
        raise ValueError(f"unexpected CA-1M work root: {root}")
    source = regular_file(source_config, "CA-1M source config")
    if source != EXPECTED_SOURCE_CONFIG.resolve():
        raise ValueError(f"unexpected CA-1M source config: {source}")
    source_sha = sha256_file(source)
    if source_sha != EXPECTED_SOURCE_CONFIG_SHA256:
        raise ValueError("CA-1M source config SHA256 differs from frozen training config")
    checkpoint = regular_file(root / EXPECTED_CHECKPOINT_NAME, "epoch-12 checkpoint")
    effective = regular_file(
        root / EXPECTED_EFFECTIVE_CONFIG_NAME, "effective work config"
    )
    checkpoint_sha = sha256_file(checkpoint)
    effective_sha = sha256_file(effective)
    if checkpoint_sha in FORBIDDEN_SCANNET_SHA256:
        raise ValueError("checkpoint matches a forbidden ScanNet-trained artifact")
    if effective_sha in FORBIDDEN_SCANNET_SHA256:
        raise ValueError("effective config matches a forbidden ScanNet config")
    config_contract = validate_effective_config(effective, root)
    log_contract = validate_training_log(training_log, config_contract["per_process_batch"])
    log = regular_file(training_log, "formal training driver log")
    return {
        "schema": SCHEMA,
        "complete": True,
        "create_only": True,
        "dataset": "ca1m_train100",
        "model_family": "ca1m_native_class_agnostic_tr3d",
        "work_root": os.fspath(root),
        "checkpoint": {
            "path": os.fspath(checkpoint),
            "filename": checkpoint.name,
            "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size,
            "epoch": 12,
        },
        "config": {
            "path": os.fspath(effective),
            "sha256": effective_sha,
            "bytes": effective.stat().st_size,
            "source_path": os.fspath(source),
            "source_sha256": source_sha,
            "source_bytes": source.stat().st_size,
        },
        "training": {
            "complete": True,
            "fixed_epochs": 12,
            "final_epoch": 12,
            "checkpoint_selection": False,
            "log_path": os.fspath(log),
            "log_sha256": sha256_file(log),
            "log_bytes": log.stat().st_size,
            **log_contract,
        },
        "initialization": {
            "kind": "random_scratch",
            "load_from": None,
            "pretrained_or_init_cfg": False,
            "scannet_trained_module_access": False,
        },
        "isolation": {
            "weights_train": "folds_2_3_4",
            "threshold_development": "fold_0",
            "locked_internal_check": "fold_1",
            "locked_fold1_gt_access": False,
            "official_validation_gt_access": False,
            "split_sha256": dict(_SCENE_SPLIT_SHA256),
        },
        "metric_protocol": {
            "training_dev_evaluator": "mmdet3d.IndoorMetric",
            "training_dev_metric_role": "diagnostic_only",
            "checkpoint_selected_by_dev_metric": False,
            "ca_official_evaluator": (
                "evaluation/eval_ca1m.py -> evaluation/utils/eval_det.py:"
                "get_iou_obb_v2 -> box_util.py:box3d_iou_v2"
            ),
            "ca_official_metric_run_during_training": False,
            "metrics_are_not_interchangeable": True,
        },
        "effective_protocol": config_contract,
    }


def _expect_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} keys differ: missing={sorted(expected-set(value))}, "
            f"extra={sorted(set(value)-expected)}"
        )


def load_checkpoint_binding(path: Path) -> CA1MTR3DCheckpointBinding:
    manifest_path = regular_file(path, "CA-1M TR3D checkpoint binding")
    if manifest_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("CA-1M TR3D checkpoint binding must be read-only")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint binding is not valid JSON") from error
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint binding must be a JSON object")
    _expect_exact_keys(
        payload,
        {
            "schema", "complete", "create_only", "dataset", "model_family",
            "work_root", "checkpoint", "config", "training", "initialization",
            "isolation", "metric_protocol", "effective_protocol",
        },
        "checkpoint binding",
    )
    if (
        payload["schema"] != SCHEMA
        or payload["complete"] is not True
        or payload["create_only"] is not True
        or payload["dataset"] != "ca1m_train100"
        or payload["model_family"] != "ca1m_native_class_agnostic_tr3d"
    ):
        raise ValueError("checkpoint binding scalar contract differs")
    root = Path(str(payload["work_root"])).resolve()
    if root != EXPECTED_WORK_ROOT.resolve():
        raise ValueError("checkpoint binding work root differs")
    checkpoint = payload["checkpoint"]
    config = payload["config"]
    training = payload["training"]
    initialization = payload["initialization"]
    isolation = payload["isolation"]
    protocol = payload["effective_protocol"]
    if not all(isinstance(value, Mapping) for value in (
        checkpoint, config, training, initialization, isolation, protocol
    )):
        raise ValueError("checkpoint binding sections must be JSON objects")
    expected_payload = build_binding_payload(
        work_root=root,
        source_config=Path(str(config.get("source_path", ""))),
        training_log=Path(str(training.get("log_path", ""))),
    )
    if payload != expected_payload:
        raise ValueError("checkpoint binding differs from independent reconstruction")
    checkpoint_path = regular_file(
        Path(str(checkpoint["path"])), "bound CA-1M checkpoint"
    )
    effective_path = regular_file(
        Path(str(config["path"])), "bound CA-1M effective config"
    )
    source_path = regular_file(
        Path(str(config["source_path"])), "bound CA-1M source config"
    )
    log_path = regular_file(
        Path(str(training["log_path"])), "bound CA-1M training log"
    )
    return CA1MTR3DCheckpointBinding(
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=str(checkpoint["sha256"]),
        effective_config_path=effective_path,
        effective_config_sha256=str(config["sha256"]),
        source_config_path=source_path,
        source_config_sha256=str(config["source_sha256"]),
        training_log_path=log_path,
        training_log_sha256=str(training["log_sha256"]),
        work_root=root,
    )


def _verified_path_hash(
    payload: Mapping[str, Any], path_key: str, sha_key: str, name: str
) -> tuple[Path, str]:
    source = regular_file(Path(str(payload.get(path_key, ""))), name)
    expected = str(payload.get(sha_key, ""))
    actual = sha256_file(source)
    if actual != expected:
        raise ValueError(f"{name} SHA256 mismatch")
    return source, actual


def build_dev_diagnostic_receipt(
    *, binding_path: Path, dev_report_path: Path
) -> dict[str, Any]:
    binding = load_checkpoint_binding(binding_path)
    report_path = regular_file(dev_report_path, "CA fold0 dev metric report")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("CA fold0 dev metric report is not valid JSON") from error
    if not isinstance(report, Mapping):
        raise ValueError("CA fold0 dev metric report must be a JSON object")
    _expect_exact_keys(
        report,
        {
            "schema", "complete", "coordinate_export", "evaluation",
            "ground_truth_count", "headline", "inputs", "metrics", "partition",
            "prediction_count", "scene_count", "scientific_isolation", "train_only",
        },
        "CA fold0 dev metric report",
    )
    if (
        report["schema"] != DEV_DIAGNOSTIC_SCHEMA
        or report["complete"] is not True
        or report["train_only"] is not True
        or report["partition"] != "threshold_dev_fold0"
        or int(report["scene_count"]) != 20
        or int(report["ground_truth_count"]) != 3073
        or int(report["prediction_count"]) < 1
    ):
        raise ValueError("CA fold0 dev metric scalar contract differs")
    inputs = report["inputs"]
    evaluation = report["evaluation"]
    coordinate = report["coordinate_export"]
    headline = report["headline"]
    metrics = report["metrics"]
    isolation = report["scientific_isolation"]
    if not all(isinstance(value, Mapping) for value in (
        inputs, evaluation, coordinate, headline, metrics, isolation
    )):
        raise ValueError("CA fold0 dev metric sections must be JSON objects")
    checkpoint, checkpoint_sha = _verified_path_hash(
        inputs, "checkpoint", "checkpoint_sha256", "dev-eval checkpoint"
    )
    if (
        checkpoint != binding.checkpoint_path
        or checkpoint_sha != binding.checkpoint_sha256
    ):
        raise ValueError("dev metric does not evaluate the bound CA checkpoint")
    verified_files: dict[str, dict[str, Any]] = {
        "checkpoint": {
            "path": os.fspath(checkpoint), "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size,
        }
    }
    for name, section, path_key, sha_key in (
        ("dev_config", inputs, "config", "config_sha256"),
        ("dataset_contract", inputs, "dataset_contract", "dataset_contract_sha256"),
        ("dev_annotation", inputs, "dev_annotation", "dev_annotation_sha256"),
        ("dev_split", inputs, "dev_split", "dev_split_sha256"),
        ("prediction_dump", inputs, "prediction_dump", "prediction_dump_sha256"),
        ("evaluation_tool", inputs, "tool", "tool_sha256"),
        ("eval_det", evaluation, "eval_det", "eval_det_sha256"),
        ("box_util", evaluation, "box_util", "box_util_sha256"),
        ("coordinate_npz", coordinate, "npz", "npz_sha256"),
    ):
        source, digest = _verified_path_hash(section, path_key, sha_key, name)
        verified_files[name] = {
            "path": os.fspath(source), "sha256": digest,
            "bytes": source.stat().st_size,
        }
    if (
        str(inputs.get("dev_split_sha256")) != _SCENE_SPLIT_SHA256["threshold_dev"]
        or evaluation.get("iou_function") != "box3d_iou_v2"
        or evaluation.get("ap_semantics")
        != "eval_det.voc_ap_continuous_strict_iou_gt"
        or evaluation.get("thresholds") != [0.15, 0.25, 0.5]
        or evaluation.get("checkpoint_selection") is not False
        or evaluation.get("official_validation_comparable") is not False
        or evaluation.get("post_inference_score_filter") is not None
        or headline.get("indoor_metric") is not False
        or isolation != {
            "indoor_metric_used": False,
            "locked_fold1_access": False,
            "official_validation_access": False,
            "official_validation_prediction_access": False,
        }
    ):
        raise ValueError("CA fold0 evaluator/isolation contract differs")
    expected_metric_names = {"ap15": 0.15, "ap25": 0.25, "ap50": 0.5}
    if set(metrics) != set(expected_metric_names):
        raise ValueError("CA fold0 metric threshold rows differ")
    ap: dict[str, float] = {}
    recall: dict[str, float] = {}
    for name, threshold in expected_metric_names.items():
        row = metrics[name]
        if not isinstance(row, Mapping):
            raise ValueError(f"CA fold0 metric row {name} is not an object")
        _expect_exact_keys(
            row,
            {
                "ap", "false_positives", "ground_truths", "iou_threshold",
                "precision", "predictions", "recall", "strict_iou_comparison",
                "true_positives",
            },
            f"CA fold0 metric row {name}",
        )
        prediction_count = int(report["prediction_count"])
        ground_truth_count = int(report["ground_truth_count"])
        true_positive = int(row["true_positives"])
        false_positive = int(row["false_positives"])
        row_ap = float(row["ap"])
        row_precision = float(row["precision"])
        row_recall = float(row["recall"])
        if (
            float(row["iou_threshold"]) != threshold
            or row["strict_iou_comparison"] is not True
            or int(row["predictions"]) != prediction_count
            or int(row["ground_truths"]) != ground_truth_count
            or true_positive + false_positive != prediction_count
            or not 0.0 <= row_ap <= 1.0
            or abs(row_precision - true_positive / prediction_count) > 1e-15
            # eval_det uses the historical 1e-6 recall denominator epsilon.
            or abs(row_recall - true_positive / (ground_truth_count + 1e-6))
            > 1e-15
            or float(headline[name]) != row_ap
        ):
            raise ValueError(f"CA fold0 metric row {name} is internally inconsistent")
        ap[name] = row_ap
        recall[name] = row_recall
    return {
        "schema": DEV_RECEIPT_SCHEMA,
        "complete": True,
        "create_only": True,
        "preferred_revision": "v4_cpu_safe",
        "checkpoint_binding": {
            "path": os.fspath(binding.manifest_path),
            "sha256": binding.manifest_sha256,
            "checkpoint_sha256": binding.checkpoint_sha256,
        },
        "source_report": {
            "path": os.fspath(report_path),
            "sha256": sha256_file(report_path),
            "bytes": report_path.stat().st_size,
        },
        "role": "train_only_fold0_dev_diagnostic",
        "authorization": {
            "diagnostic_only": True,
            "activation_authorized": False,
            "checkpoint_selection_authorized": False,
            "terminal_collection_authorized": False,
        },
        "partition": "threshold_dev_fold0",
        "scene_count": int(report["scene_count"]),
        "prediction_count": int(report["prediction_count"]),
        "ground_truth_count": int(report["ground_truth_count"]),
        "ap": ap,
        "recall": recall,
        "metric_protocol": {
            "indoor_metric": False,
            "ca_official_equivalent": True,
            "official_validation_comparable": False,
            "iou_function": "box3d_iou_v2",
            "strict_iou_comparison": True,
            "checkpoint_selection": False,
        },
        "scientific_isolation": dict(isolation),
        "verified_files": verified_files,
    }


def load_dev_diagnostic_receipt(path: Path) -> dict[str, Any]:
    receipt_path = regular_file(path, "CA checkpoint dev diagnostic receipt")
    if receipt_path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise ValueError("CA checkpoint dev diagnostic receipt must be read-only")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("CA checkpoint dev diagnostic receipt is not valid JSON") from error
    if not isinstance(payload, Mapping) or payload.get("schema") != DEV_RECEIPT_SCHEMA:
        raise ValueError("CA checkpoint dev diagnostic receipt schema differs")
    binding_section = payload.get("checkpoint_binding") or {}
    report_section = payload.get("source_report") or {}
    expected = build_dev_diagnostic_receipt(
        binding_path=Path(str(binding_section.get("path", ""))),
        dev_report_path=Path(str(report_section.get("path", ""))),
    )
    if payload != expected:
        raise ValueError("CA checkpoint dev diagnostic receipt differs on recomputation")
    return dict(payload)


__all__ = [
    "AUDIT_SCHEMA",
    "DEV_DIAGNOSTIC_SCHEMA",
    "DEV_RECEIPT_SCHEMA",
    "CA1MTR3DCheckpointBinding",
    "EXPECTED_EFFECTIVE_CONFIG_NAME",
    "EXPECTED_SOURCE_CONFIG",
    "EXPECTED_SOURCE_CONFIG_SHA256",
    "EXPECTED_WORK_ROOT",
    "FORBIDDEN_SCANNET_SHA256",
    "SCHEMA",
    "build_binding_payload",
    "build_dev_diagnostic_receipt",
    "load_checkpoint_binding",
    "load_dev_diagnostic_receipt",
    "regular_directory",
    "regular_file",
    "sha256_file",
    "validate_effective_config",
    "validate_training_log",
]
