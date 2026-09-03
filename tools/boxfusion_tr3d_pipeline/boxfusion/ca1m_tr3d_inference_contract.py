"""Static contract checks for the CA-1M-only TR3D point inference config."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "boxfusion.tr3d.ca1m_point_inference_config.v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_value(node: ast.AST, values: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in values:
            raise ValueError(f"literal config references unknown name {node.id!r}")
        return deepcopy(values[node.id])
    if isinstance(node, ast.List):
        return [_safe_value(item, values) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_value(item, values) for item in node.elts)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError("literal config forbids dictionary expansion")
            result[_safe_value(key, values)] = _safe_value(value, values)
        return result
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "dict"
        and not node.args
    ):
        result = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                raise ValueError("literal config forbids dictionary expansion")
            result[keyword.arg] = _safe_value(keyword.value, values)
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _safe_value(node.operand, values)
        if isinstance(operand, bool) or not isinstance(operand, (int, float)):
            raise ValueError("literal config unary operator requires a number")
        return -operand if isinstance(node.op, ast.USub) else operand
    raise ValueError(f"literal config contains executable expression {type(node).__name__}")


def parse_literal_config(path: Path) -> dict[str, Any]:
    """Parse assignment-only MMEngine config syntax without executing Python."""

    source = path.resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"config is not a regular file: {source}")
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    values: dict[str, Any] = {}
    for statement in tree.body:
        if (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Constant)
            and isinstance(statement.value.value, str)
        ):
            continue
        if (
            not isinstance(statement, ast.Assign)
            or len(statement.targets) != 1
            or not isinstance(statement.targets[0], ast.Name)
        ):
            raise ValueError(
                "point-inference config must contain only literal name assignments"
            )
        name = statement.targets[0].id
        if name in values:
            raise ValueError(f"literal config reassigns {name!r}")
        values[name] = _safe_value(statement.value, values)
    return values


def validate_ca1m_point_inference_config(
    *,
    inference_path: Path,
    inference_sha256: str,
    effective_training_path: Path,
    effective_training_sha256: str,
) -> dict[str, Any]:
    """Prove that inference is architecture-identical and has no data source."""

    inference = inference_path.resolve()
    effective = effective_training_path.resolve()
    if sha256_file(inference) != inference_sha256:
        raise ValueError("CA-1M point-inference config SHA256 differs")
    if sha256_file(effective) != effective_training_sha256:
        raise ValueError("sealed effective CA-1M training config SHA256 differs")
    current = parse_literal_config(inference)
    trained = parse_literal_config(effective)
    if "_base_" in current:
        raise ValueError("CA-1M point-inference config must be standalone")
    if current.get("default_scope") != "mmdet3d":
        raise ValueError("CA-1M point-inference default scope differs")
    if current.get("custom_imports") != {
        "imports": ["projects.TR3D.tr3d", "tr3d_plugin"],
        "allow_failed_imports": False,
    }:
        raise ValueError("CA-1M point-inference imports differ")
    if current.get("model") != trained.get("model"):
        raise ValueError("point-inference model architecture differs from CA training")
    for name in (
        "train_cfg",
        "train_dataloader",
        "optim_wrapper",
        "param_scheduler",
        "val_cfg",
        "val_dataloader",
        "val_evaluator",
        "test_cfg",
        "test_evaluator",
        "load_from",
    ):
        if current.get(name, object()) is not None:
            raise ValueError(f"point-inference config field {name} must be None")
    if current.get("resume") is not False:
        raise ValueError("point-inference config cannot resume training")
    pipeline = current.get("point_pipeline")
    if pipeline != trained.get("test_pipeline"):
        raise ValueError("point-inference pipeline differs from CA training test pipeline")
    if not isinstance(pipeline, list) or [row.get("type") for row in pipeline] != [
        "LoadPointsFromFile",
        "GlobalAlignment",
        "MultiScaleFlipAug3D",
        "Pack3DDetInputs",
    ]:
        raise ValueError("point-inference pipeline is not the frozen XYZRGB pipeline")
    if pipeline[-1].get("keys") != ["points"]:
        raise ValueError("point-inference pipeline may pack only points")
    loader = current.get("test_dataloader")
    if not isinstance(loader, dict) or set(loader) != {
        "batch_size", "num_workers", "persistent_workers", "sampler", "dataset"
    }:
        raise ValueError("point-inference lazy dataset shell differs")
    dataset = loader.get("dataset")
    if not isinstance(dataset, dict) or dataset != {
        "type": "TR3DForegroundCA1MDataset",
        "data_root": None,
        "data_prefix": {"pts": ""},
        "ann_file": "",
        "pipeline": pipeline,
        "filter_empty_gt": False,
        "metainfo": {"classes": ("foreground",), "categories": {"foreground": 0}},
        "box_type_3d": "Depth",
        "backend_args": None,
        "test_mode": True,
        "load_eval_anns": False,
    }:
        raise ValueError("point-inference dataset shell could access external data")
    return {
        "schema": SCHEMA,
        "path": str(inference),
        "sha256": inference_sha256,
        "architecture_matches_ca_training": True,
        "pipeline_matches_ca_training_test_pipeline": True,
        "point_input_only": True,
        "dataset_shell_lazy_only": True,
        "data_root": None,
        "ann_file": "",
        "ground_truth_access": False,
        "validation_access": False,
        "evaluator_access": False,
        "scannet_config_access": False,
    }


__all__ = [
    "SCHEMA",
    "parse_literal_config",
    "sha256_file",
    "validate_ca1m_point_inference_config",
]
