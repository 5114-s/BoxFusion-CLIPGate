"""Fail-closed contracts for the CA-1M xfit-R2 outer-dev evaluation.

The protocol has two hard boundaries:

* proposal collection is point-only and GT-free on the exact 20 fold-0
  scenes; and
* ground truth becomes reachable only after the exact proposal collection has
  been sealed read-only.

Fold 1 and official CA-1M validation are outside every accepted scene/path
contract.  The numerical AP implementation mirrors the official CA global
score ordering, strict IoU comparison, and duplicate-aware matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np

from .ca1m_tr3d_checkpoint_binding import FORBIDDEN_SCANNET_SHA256
from .ca1m_tr3d_inference_contract import parse_literal_config
from .ca1m_tr3d_terminal import (
    associate_terminal_candidates,
    pairwise_world_aabb_iou,
    world_aabb,
)


CONFIG_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_eval_config.v2"
BINDING_SCHEMA = "boxfusion.tr3d.ca1m_xfit_r2_outer_dev_checkpoint_binding.v2"
COLLECTION_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_proposal_collection.v1"
REPORT_SCHEMA = "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_report.v1"
CONTINUATION_RECEIPT_SCHEMA = (
    "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_continuation_receipt.v1"
)
PREREGISTRATION_SCHEMA = (
    "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_preregistration.v1"
)
PREREGISTRATION_V2_SCHEMA = (
    "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_preregistration.v2"
)
PREREGISTRATION_V3_SCHEMA = (
    "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_preregistration.v3"
)
NAMESPACE = "ca1m_tr3d_xfit_r2_outer_dev_eval_v1"
PIPELINE_ROOT = Path("/data/ZhaoX/BoxFusion/tools/boxfusion_tr3d_pipeline")
OVM_ROOT = Path("/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_dev")
WORK_ROOT = Path(
    "/extra/ZhaoX/tr3d_ca1m_work_dirs/ca1m_xfit_v2_formal_r2/"
    "ca1m_xfit_v2_formal_outer_dev_seed0_r2"
)
OUTER_WRAPPER_LOG = Path(
    "/extra/ZhaoX/tr3d_ca1m_logs/"
    "train_ca1m_xfit_v2_formal_outer_dev_seed0_r2.log"
)
FOLD0_SHA256 = "9c886ca85ba599881797b25a49d2fc72dd136d255a245a09fe1cf17cbce735a7"
XFIT_CONTRACT_SHA256 = "562b1204e96eed9ce9883b5fcecc65a422104896dc26a87ea988db66d5e01572"
R2_AUTHORIZATION_SHA256 = "46e30060cfbe000b330d50688c0e5534f2f3887622fab1da3138a2f1613fec5c"
POINT_PARITY_SHA256 = "35d9dfafc7272d92d98c97c6ef23f4323432e9bd0af5045bc5f78b1ae9afa00d"
IOU_THRESHOLDS = (0.15, 0.25, 0.50)
_SHA = re.compile(r"^[0-9a-f]{64}$")
_SCENE = re.compile(r"^[0-9]{8}$")
_TIMESTAMP_LOG = re.compile(r"^[0-9]{8}_[0-9]{6}\.log$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: Any, name: str) -> str:
    result = str(value)
    if _SHA.fullmatch(result) is None:
        raise ValueError(f"{name} must be a lowercase SHA256")
    return result


def regular_file(path: Path, name: str, *, immutable: bool = False) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_file() or result.is_symlink() or result.stat().st_size <= 0:
        raise FileNotFoundError(f"missing regular {name}: {result}")
    if immutable and result.stat().st_mode & 0o222:
        raise ValueError(f"{name} must be read-only: {result}")
    return result


def regular_directory(path: Path, name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {path}")
    result = path.resolve()
    if not result.is_dir() or result.is_symlink():
        raise FileNotFoundError(f"missing regular {name}: {result}")
    return result


def read_json(
    path: Path, name: str, *, immutable: bool = False
) -> tuple[Path, dict[str, Any]]:
    source = regular_file(path, name, immutable=immutable)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is not JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return source, value


def _canonical_json(payload: Mapping[str, Any], *, pretty: bool = True) -> bytes:
    if pretty:
        return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()


def create_or_verify_bytes(path: Path, data: bytes, name: str) -> Path:
    """Atomically create a read-only artifact or verify an identical retry."""

    if path.is_symlink():
        raise ValueError(f"{name} target must not be a symlink: {path}")
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        current = regular_file(target, name, immutable=True)
        if current.read_bytes() != data:
            raise FileExistsError(f"refusing to replace differing {name}: {target}")
        return current
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError:
        current = regular_file(target, name, immutable=True)
        if current.read_bytes() != data:
            raise FileExistsError(f"refusing to replace differing {name}: {target}")
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return regular_file(target, name, immutable=True)


def create_or_verify_json(
    path: Path, payload: Mapping[str, Any], name: str
) -> Path:
    return create_or_verify_bytes(path, _canonical_json(payload), name)


def load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    source, revision = read_json(
        path, "xfit-R2 outer-dev evaluation config", immutable=True
    )
    if source != PIPELINE_ROOT / "config/ca1m_tr3d_xfit_r2_outer_dev_eval_v2.json":
        raise ValueError("xfit-R2 evaluation requires the fixed config path")
    if set(revision) != {
        "schema", "base_config", "training_extension",
        "evaluation_extension", "implementation_updates",
    } or revision.get("schema") != CONFIG_SCHEMA:
        raise ValueError("xfit-R2 evaluation-config revision keys differ")
    base_record = revision.get("base_config") or {}
    base_path, base = read_json(
        Path(str(base_record.get("path", ""))),
        "sealed xfit-R2 evaluation config v1", immutable=True,
    )
    if (
        base_path != PIPELINE_ROOT / "config/ca1m_tr3d_xfit_r2_outer_dev_eval_v1.json"
        or base_record.get("sha256")
        != "43bc1617c7a6dd69929f1c9c4e0482d738f5d9349c1131cc4ff2be67e6e0c0ee"
        or sha256_file(base_path) != base_record.get("sha256")
        or base.get("schema")
        != "boxfusion.ca1m_tr3d_xfit_r2_outer_dev_eval_config.v1"
    ):
        raise ValueError("sealed xfit-R2 evaluation config v1 differs")
    cfg = json.loads(json.dumps(base))
    training_extension = revision.get("training_extension") or {}
    evaluation_extension = revision.get("evaluation_extension") or {}
    implementation_updates = revision.get("implementation_updates") or {}
    if set(training_extension) != {
        "outer_wrapper_log_path", "outer_wrapper_log_required",
        "outer_wrapper_terminal_line",
    } or set(evaluation_extension) != {"preregistration"}:
        raise ValueError("xfit-R2 evaluation-config extensions differ")
    if set(implementation_updates) != {
        "xfit_r2_eval_contract", "xfit_r2_outer_dev_runner",
        "single_command_wrapper", "preregistration_v3_sealer",
    }:
        raise ValueError("xfit-R2 evaluation implementation updates differ")
    cfg["schema"] = CONFIG_SCHEMA
    cfg["training"].update(training_extension)
    cfg["evaluation_stage"].update(evaluation_extension)
    cfg["implementation"].update(implementation_updates)
    required = {
        "schema", "namespace", "run_authorized", "train_only",
        "fold0_only", "ground_truth_after_proposal_seal_only",
        "fold1_access", "official_validation_access", "training",
        "scene_contract", "point_lineage", "point_inference", "runtime",
        "proposal_stage", "evaluation_stage", "implementation",
    }
    if set(cfg) != required:
        raise ValueError("xfit-R2 evaluation config keys differ")
    if (
        cfg.get("schema") != CONFIG_SCHEMA
        or cfg.get("namespace") != NAMESPACE
        or cfg.get("run_authorized") is not True
        or cfg.get("train_only") is not True
        or cfg.get("fold0_only") is not True
        or cfg.get("ground_truth_after_proposal_seal_only") is not True
        or cfg.get("fold1_access") is not False
        or cfg.get("official_validation_access") is not False
    ):
        raise ValueError("xfit-R2 evaluation isolation contract differs")
    training = cfg.get("training") or {}
    if (
        training.get("role") != "outer_dev"
        or training.get("train_folds") != [2, 3, 4]
        or training.get("heldout_fold") != 0
        or training.get("checkpoint_name") != "iter_11268.pth"
        or training.get("effective_config_name") != "outer_dev.py"
        or training.get("optimizer_updates") != 11268
        or training.get("global_batch") != 16
        or training.get("world_size") != 2
        or training.get("initialization") != "random_scratch_ca_only"
        or training.get("scannet_checkpoint_or_module_access") is not False
        or Path(str(training.get("outer_wrapper_log_path", ""))).resolve()
        != OUTER_WRAPPER_LOG
        or training.get("outer_wrapper_terminal_line") != "TRAIN_EXIT=0"
        or training.get("outer_wrapper_log_required") is not True
        or Path(str(training.get("work_root", ""))).resolve() != WORK_ROOT
        or Path(str(training.get("binding_path", ""))).resolve()
        != PIPELINE_ROOT / (
            "manifests/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
            "checkpoint_binding.json"
        )
    ):
        raise ValueError("xfit-R2 training binding contract differs")
    for key, expected in (
        ("xfit_contract_sha256", XFIT_CONTRACT_SHA256),
        ("r2_authorization_sha256", R2_AUTHORIZATION_SHA256),
    ):
        if training.get(key) != expected:
            raise ValueError(f"xfit-R2 training {key} differs")
    scene = cfg.get("scene_contract") or {}
    if Path(str(scene.get("path", ""))).resolve() != OVM_ROOT / (
        "data/tr3d_ca1m_visible_xfit_v2_formal/splits/predict_fold0.txt"
    ):
        raise ValueError("xfit-R2 fold0 scene path differs")
    proposal = cfg.get("proposal_stage") or {}
    if (
        proposal.get("scene_count") != 20
        or proposal.get("ground_truth_access") is not False
        or proposal.get("anchor_access") is not False
        or proposal.get("b6_access") is not False
        or proposal.get("create_only") is not True
        or proposal.get("gpu_required") is not True
        or proposal.get("protocol") != {
            "pixel_stride": 4,
            "voxel_size_m": 0.01,
            "min_depth_m": 0.1,
            "max_depth_m": 6.0,
            "score_threshold": 0.01,
            "max_proposals": 256,
            "near_iou": 0.15,
            "prefix_id": "p100_gap20",
        }
    ):
        raise ValueError("xfit-R2 proposal-stage contract differs")
    point = cfg.get("point_inference") or {}
    if (
        point.get("point_input_only") is not True
        or point.get("standalone") is not True
        or point.get("ground_truth_access") is not False
        or point.get("validation_access") is not False
        or point.get("scannet_config_access") is not False
        or Path(str(point.get("path", ""))).resolve() != OVM_ROOT / (
            "config/tr3d/tr3d_ca1m_foreground_point_inference_xfit_r2.py"
        )
        or point.get("sha256")
        != "479f7e61eff9fd23fc086ebc2603e161caa876defe73c556a0e671a8fd35c052"
    ):
        raise ValueError("xfit-R2 point-inference isolation differs")
    processed = (cfg.get("point_lineage") or {}).get("processed_rgbd") or {}
    if (
        Path(str(processed.get("root", ""))).resolve()
        != Path("/extra/ZhaoX/boxfusion_ca1m_native_b6_train100_v1")
        or processed.get("depth_scale") != 1000.0
    ):
        raise ValueError("xfit-R2 point-lineage source differs")
    runtime = cfg.get("runtime") or {}
    if runtime != {
        "worker_python": os.fspath(
            OVM_ROOT / ".conda/boxfusion-tr3d/bin/python"
        ),
        "worker_script": os.fspath(
            PIPELINE_ROOT / "tools/ca1m_tr3d_terminal_worker.py"
        ),
        "runtime_root": "/data/ZhaoX/OVM3D-Dett/boxfusion_tr3d_r2_verifier_dev",
        "project_root": os.fspath(OVM_ROOT),
        "vendor_root": os.fspath(OVM_ROOT / "third_party/mmdetection3d"),
        "startup_timeout_s": 600,
    }:
        raise ValueError("xfit-R2 proposal runtime differs")
    if (
        Path(str(proposal.get("output_root", ""))).resolve()
        != PIPELINE_ROOT / "diagnostics/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/proposals"
        or Path(str(proposal.get("collection_manifest", ""))).resolve()
        != PIPELINE_ROOT / (
            "reports/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/"
            "proposal_collection_manifest.json"
        )
    ):
        raise ValueError("xfit-R2 proposal namespace differs")
    evaluation = cfg.get("evaluation_stage") or {}
    if (
        evaluation.get("scene_count") != 20
        or evaluation.get("requires_sealed_proposal_collection") is not True
        or evaluation.get("cpu_only") is not True
        or evaluation.get("heldout_fold") != 0
        or evaluation.get("b6_score_source")
        != "ca1m_native_b6_final_base_fold0_oof_v2"
        or evaluation.get("oracle") != "same_best_gt_geometry_replacement"
        or evaluation.get("oracle_deployable") is not False
    ):
        raise ValueError("xfit-R2 evaluation-stage contract differs")
    if (
        Path(str(evaluation.get("report", ""))).resolve()
        != PIPELINE_ROOT / (
            "reports/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/evaluation_report.json"
        )
        or Path(str(evaluation.get("continuation_receipt", ""))).resolve()
        != PIPELINE_ROOT / (
            "reports/ca1m_tr3d_xfit_r2_outer_dev_eval_v1/continuation_receipt.json"
        )
    ):
        raise ValueError("xfit-R2 evaluation output namespace differs")
    expected_implementation = {
        "xfit_r2_eval_contract", "xfit_r2_outer_dev_runner",
        "single_command_wrapper", "v4_point_builder", "v4_proposal_contract",
        "terminal_geometry", "rgbd_backprojection", "worker_client", "worker",
        "point_inference_contract", "point_inference_config",
        "preregistration_sealer", "preregistration_v2_sealer",
        "preregistration_v3_sealer",
        "official_adapter",
    }
    if set(cfg.get("implementation") or {}) != expected_implementation:
        raise ValueError("xfit-R2 implementation inventory differs")
    return source, cfg


def evaluation_config_contract_sha256(cfg: Mapping[str, Any]) -> str:
    """Hash the final config while excluding the preregistration self-record.

    Preregistration-v3 binds every other semantic JSON field, including the
    complete implementation inventory.  Replacing its own path/hash avoids a
    circular digest while still allowing the final config to bind the v2 file.
    """

    normalized = json.loads(json.dumps(cfg))
    stage = normalized.get("evaluation_stage")
    if not isinstance(stage, dict) or "preregistration" not in stage:
        raise ValueError("evaluation config lacks preregistration self-record")
    stage["preregistration"] = {
        "self_bound_by": PREREGISTRATION_V3_SCHEMA,
    }
    return sha256_bytes(_canonical_json(normalized, pretty=False))


def preregistration_input_contract(
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    training = cfg["training"]
    lineage = cfg["point_lineage"]
    evaluation = cfg["evaluation_stage"]
    records = {
        "fold0_scene_list": cfg["scene_contract"],
        "xfit_contract": {
            "path": training["xfit_contract_path"],
            "sha256": training["xfit_contract_sha256"],
        },
        "r2_training_authorization": {
            "path": training["r2_authorization_path"],
            "sha256": training["r2_authorization_sha256"],
        },
        "point_parity": {
            "path": lineage["receipt_path"],
            "sha256": lineage["receipt_sha256"],
        },
        "point_inference_config": cfg["point_inference"],
        "anchor_shadow": evaluation["anchor_shadow"],
        "anchor_shadow_manifest": evaluation["anchor_shadow_manifest"],
        "v1_fold0_comparison_manifest": evaluation[
            "v1_fold0_comparison_manifest"
        ],
        "v1_sealed_raw_diagnostic": evaluation["v1_sealed_raw_diagnostic"],
        "gt_shadow_inventory": evaluation["gt_shadow_inventory"],
    }
    return {
        name: {
            "path": os.fspath(Path(str(record["path"])).resolve()),
            "sha256": _sha(record["sha256"], f"preregistration {name} SHA256"),
        }
        for name, record in records.items()
    }


def scene_ids(cfg: Mapping[str, Any]) -> tuple[str, ...]:
    record = cfg.get("scene_contract") or {}
    source = regular_file(Path(str(record.get("path", ""))), "fold0 scene list")
    if (
        record.get("sha256") != FOLD0_SHA256
        or sha256_file(source) != FOLD0_SHA256
        or record.get("count") != 20
        or record.get("fold") != 0
        or record.get("exact") is not True
    ):
        raise ValueError("fold0 scene contract differs")
    scenes = tuple(row.strip() for row in source.read_text().splitlines() if row.strip())
    if (
        len(scenes) != 20
        or len(set(scenes)) != 20
        or any(_SCENE.fullmatch(scene) is None for scene in scenes)
    ):
        raise ValueError("fold0 scene list must contain exact 20 numeric IDs")
    return scenes


def validate_static_inputs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    scenes = scene_ids(cfg)
    training = cfg["training"]
    records = {
        "xfit_contract": (
            Path(training["xfit_contract_path"]), XFIT_CONTRACT_SHA256
        ),
        "r2_authorization": (
            Path(training["r2_authorization_path"]), R2_AUTHORIZATION_SHA256
        ),
        "point_parity": (
            Path(cfg["point_lineage"]["receipt_path"]), POINT_PARITY_SHA256
        ),
    }
    implementation = cfg.get("implementation") or {}
    if not isinstance(implementation, dict) or not implementation:
        raise ValueError("evaluation implementation inventory is empty")
    for name, record in implementation.items():
        if not isinstance(record, dict):
            raise ValueError(f"implementation record {name} is invalid")
        records[f"implementation.{name}"] = (
            Path(str(record.get("path", ""))),
            _sha(record.get("sha256"), f"implementation {name} SHA256"),
        )
    verified: dict[str, Any] = {}
    for name, (path, expected) in records.items():
        source = regular_file(path, name)
        actual = sha256_file(source)
        if actual != expected:
            raise ValueError(f"static input SHA256 drift: {name}")
        verified[name] = {"path": os.fspath(source), "sha256": actual}
    parity = json.loads(
        Path(verified["point_parity"]["path"]).read_text(encoding="utf-8")
    )
    if (
        parity.get("schema")
        != "boxfusion.ca1m_tr3d_v4_lineage_training_point_parity.v1"
        or parity.get("complete") is not True
        or parity.get("ground_truth_access") is not False
        or parity.get("point_array_parity_scene_count") != 100
        or parity.get("point_byte_parity_scene_count") != 100
        or not set(scenes).issubset(set((parity.get("scenes") or {}).keys()))
    ):
        raise ValueError("point-lineage receipt does not cover exact fold0")
    return {"scene_count": len(scenes), "records": verified}


def _contains_initializer(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (key in {"pretrained", "init_cfg"} and child is not None)
            or _contains_initializer(child)
            for key, child in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_initializer(item) for item in value)
    return False


def validate_effective_config(path: Path, cfg: Mapping[str, Any]) -> dict[str, Any]:
    source = regular_file(path, "effective outer-dev config")
    values = parse_literal_config(source)
    training = cfg["training"]
    if (
        values.get("xfit_role") != "outer_dev"
        or values.get("xfit_train_folds") != [2, 3, 4]
        or values.get("xfit_heldout_fold") != 0
        or values.get("load_from") is not None
        or values.get("resume") is not False
        or _contains_initializer(values.get("model"))
    ):
        raise ValueError("effective config is not CA-only outer-dev random scratch")
    if values.get("train_cfg") != {
        "type": "IterBasedTrainLoop", "max_iters": 11268
    }:
        raise ValueError("effective outer-dev train loop differs")
    checkpoint = (values.get("default_hooks") or {}).get("checkpoint")
    if checkpoint != {
        "type": "CheckpointHook", "by_epoch": False,
        "interval": 11268, "max_keep_ckpts": 1,
    }:
        raise ValueError("effective outer-dev checkpoint hook differs")
    loader = values.get("train_dataloader") or {}
    repeated = loader.get("dataset") or {}
    leaf = repeated.get("dataset") or {}
    if (
        loader.get("batch_size") != 8
        or loader.get("num_workers") != 4
        or loader.get("persistent_workers") is not True
        or loader.get("sampler") != {"type": "InfiniteSampler", "shuffle": True}
        or repeated.get("type") != "RepeatDataset"
        or repeated.get("times") != 1
        or leaf.get("type") != "TR3DForegroundCA1MDataset"
        or Path(str(leaf.get("ann_file", ""))).name
        != "ca1m_infos_train_folds234_visible_foreground_xfit_v2_formal.pkl"
    ):
        raise ValueError("effective outer-dev dataloader/split differs")
    if any(values.get(name) is not None for name in (
        "val_cfg", "val_dataloader", "val_evaluator",
        "test_cfg", "test_dataloader", "test_evaluator",
    )):
        raise ValueError("effective outer-dev config can access val/test")
    wrapper = values.get("optim_wrapper") or {}
    if (
        wrapper.get("type") != "OptimWrapper"
        or wrapper.get("optimizer")
        != {"type": "AdamW", "lr": 0.001, "weight_decay": 0.0001}
        or wrapper.get("clip_grad") != {"max_norm": 10, "norm_type": 2}
    ):
        raise ValueError("effective outer-dev FP32 optimizer differs")
    if values.get("param_scheduler") != [{
        "type": "MultiStepLR", "begin": 0, "end": 11268,
        "by_epoch": False, "milestones": [7512, 10329], "gamma": 0.1,
    }]:
        raise ValueError("effective outer-dev LR schedule differs")
    if (
        values.get("randomness") != {"seed": 0, "deterministic": True}
        or values.get("launcher") != "pytorch"
        or Path(str(values.get("work_dir", ""))).resolve()
        != Path(training["work_root"]).resolve()
    ):
        raise ValueError("effective outer-dev runtime contract differs")
    if "scannet" in source.read_text(encoding="utf-8").lower():
        raise ValueError("effective outer-dev config names ScanNet")
    return {
        "role": "outer_dev", "train_folds": [2, 3, 4], "heldout_fold": 0,
        "optimizer_updates": 11268, "per_process_batch": 8,
        "world_size": 2, "global_batch": 16, "fp32": True,
        "initialization": "random_scratch_ca_only",
    }


def discover_training_log(work_root: Path) -> Path:
    root = regular_directory(work_root, "outer-dev work root")
    rows = [
        item for item in root.glob("*/*.log")
        if item.parent.name == item.stem and _TIMESTAMP_LOG.fullmatch(item.name)
        and item.is_file() and not item.is_symlink()
    ]
    if len(rows) != 1:
        raise ValueError(f"expected one MMEngine training log, found {len(rows)}")
    return regular_file(rows[0], "outer-dev MMEngine training log")


def validate_training_log(path: Path) -> dict[str, Any]:
    source = regular_file(path, "outer-dev MMEngine training log")
    text = source.read_text(encoding="utf-8", errors="strict")
    required = (
        "Distributed training: True",
        "GPU number: 2",
        "IterBasedTrainLoop",
        "max_iters=11268",
        "Saving checkpoint at 11268 iterations",
        "ca1m_infos_train_folds234_visible_foreground_xfit_v2_formal.pkl",
    )
    missing = [fragment for fragment in required if fragment not in text]
    if missing:
        raise ValueError(f"outer-dev training log is incomplete: {missing}")
    if text.count("Saving checkpoint at 11268 iterations") != 1:
        raise ValueError("outer-dev log must contain one final checkpoint event")
    if any(token in text for token in ("Traceback (most recent call last)", "Iter(val)", "Iter(test)")):
        raise ValueError("outer-dev log contains failure or forbidden evaluation")
    iterations = [int(value) for value in re.findall(
        r"Iter\(train\) \[\s*([0-9]+)/11268\]", text
    )]
    if not iterations or max(iterations) > 11268:
        raise ValueError("outer-dev training iteration evidence differs")
    return {
        "world_size": 2, "global_batch": 16, "optimizer_updates": 11268,
        "final_checkpoint_event_count": 1,
        "max_logged_training_iteration": max(iterations),
        "validation_or_test_iteration_count": 0,
    }


def validate_outer_wrapper_log(
    path: Path, *, require_fixed_source_path: bool = True
) -> dict[str, Any]:
    source = regular_file(path, "fixed outer training wrapper log")
    if require_fixed_source_path and source != OUTER_WRAPPER_LOG:
        raise ValueError("outer training wrapper log path differs")
    text = source.read_text(encoding="utf-8", errors="strict")
    terminal_rows = re.findall(r"(?m)^TRAIN_EXIT=.*$", text)
    if terminal_rows != ["TRAIN_EXIT=0"] or not text.endswith("TRAIN_EXIT=0\n"):
        raise ValueError("outer wrapper log lacks unique terminal TRAIN_EXIT=0")
    required_once = (
        "Formal CA-only TR3D asymmetric xfit-v2 R2: outer_dev",
        "exact clean train_cfg: IterBasedTrainLoop(max_iters=11268)",
        "LR milestones 7512,10329; global batch16; FP32; random scratch",
        "no val/test loader; fold1 and official validation unopened",
        f"Completed formal R2 outer_dev: {WORK_ROOT}/iter_11268.pth",
    )
    bad_counts = {
        token: text.count(token) for token in required_once if text.count(token) != 1
    }
    if bad_counts:
        raise ValueError(f"outer wrapper R2 completion evidence differs: {bad_counts}")
    error_markers = (
        "Traceback (most recent call last)", "RuntimeError:", "ValueError:",
        "TRAIN_EXIT=1", "TRAIN_EXIT=2", "FAILED", "Killed",
    )
    present = [token for token in error_markers if token in text]
    if present or "Iter(val)" in text or "Iter(test)" in text:
        raise ValueError(f"outer wrapper log contains failure/forbidden markers: {present}")
    return {
        "fixed_path": os.fspath(OUTER_WRAPPER_LOG),
        "role": "outer_dev",
        "runner": "train_tr3d_ca1m_foreground_xfit_v2_formal_r2.sh",
        "optimizer_updates": 11268,
        "terminal_line": "TRAIN_EXIT=0",
        "terminal_line_count": 1,
        "r2_preamble_count": 1,
        "completed_runner_count": 1,
        "failure_marker_count": 0,
        "validation_or_test_iteration_count": 0,
    }


@dataclass(frozen=True)
class XFitR2Binding:
    path: Path
    sha256: str
    checkpoint_path: Path
    checkpoint_sha256: str
    effective_config_snapshot_path: Path
    effective_config_sha256: str
    training_log_snapshot_path: Path
    training_log_sha256: str
    outer_wrapper_log_source_path: Path
    outer_wrapper_log_snapshot_path: Path
    outer_wrapper_log_sha256: str
    evaluation_config_path: Path
    evaluation_config_sha256: str


def seal_binding(config_path: Path) -> XFitR2Binding:
    config_source, cfg = load_config(config_path)
    validate_static_inputs(cfg)
    training = cfg["training"]
    root = regular_directory(Path(training["work_root"]), "outer-dev work root")
    checkpoint = regular_file(
        root / training["checkpoint_name"], "outer-dev iter-11268 checkpoint"
    )
    effective = regular_file(
        root / training["effective_config_name"], "effective outer-dev config"
    )
    effective_contract = validate_effective_config(effective, cfg)
    log = discover_training_log(root)
    log_contract = validate_training_log(log)
    outer_log = regular_file(
        Path(training["outer_wrapper_log_path"]), "fixed outer training wrapper log"
    )
    outer_stat_before = (outer_log.stat().st_size, outer_log.stat().st_mtime_ns)
    outer_contract = validate_outer_wrapper_log(outer_log)
    outer_bytes = outer_log.read_bytes()
    outer_stat_after = (outer_log.stat().st_size, outer_log.stat().st_mtime_ns)
    if outer_stat_before != outer_stat_after:
        raise ValueError("outer training wrapper log changed during sealing")
    last_checkpoint = regular_file(root / "last_checkpoint", "MMEngine last_checkpoint")
    if Path(last_checkpoint.read_text().strip()).resolve() != checkpoint:
        raise ValueError("MMEngine last_checkpoint does not name iter_11268.pth")
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha in FORBIDDEN_SCANNET_SHA256:
        raise ValueError("R2 checkpoint matches a forbidden ScanNet artifact")
    output = Path(training["binding_path"]).resolve()
    snapshot_root = output.parent / "sealed_training_evidence"
    effective_snapshot = create_or_verify_bytes(
        snapshot_root / "effective_outer_dev.py", effective.read_bytes(),
        "effective config snapshot",
    )
    log_snapshot = create_or_verify_bytes(
        snapshot_root / "mmengine_training.log", log.read_bytes(),
        "training log snapshot",
    )
    outer_log_snapshot = create_or_verify_bytes(
        snapshot_root / "outer_training_wrapper.log", outer_bytes,
        "outer training wrapper log snapshot",
    )
    if validate_outer_wrapper_log(
        outer_log_snapshot, require_fixed_source_path=False
    ) != outer_contract:
        raise ValueError("outer training wrapper snapshot contract differs")
    payload = {
        "schema": BINDING_SCHEMA,
        "complete": True,
        "create_only": True,
        "namespace": NAMESPACE,
        "role": "outer_dev",
        "train_only": True,
        "checkpoint_selection": False,
        "checkpoint": {
            "path": os.fspath(checkpoint), "sha256": checkpoint_sha,
            "bytes": checkpoint.stat().st_size, "optimizer_updates": 11268,
        },
        "effective_config": {
            "source_path": os.fspath(effective),
            "snapshot_path": os.fspath(effective_snapshot),
            "sha256": sha256_file(effective_snapshot),
            "bytes": effective_snapshot.stat().st_size,
            "contract": effective_contract,
        },
        "training_log": {
            "source_path": os.fspath(log),
            "snapshot_path": os.fspath(log_snapshot),
            "sha256": sha256_file(log_snapshot),
            "bytes": log_snapshot.stat().st_size,
            "contract": log_contract,
        },
        "outer_wrapper_log": {
            "source_path": os.fspath(outer_log),
            "snapshot_path": os.fspath(outer_log_snapshot),
            "sha256": sha256_file(outer_log_snapshot),
            "bytes": outer_log_snapshot.stat().st_size,
            "contract": outer_contract,
        },
        "training_protocol": {
            "xfit_contract_path": training["xfit_contract_path"],
            "xfit_contract_sha256": XFIT_CONTRACT_SHA256,
            "r2_authorization_path": training["r2_authorization_path"],
            "r2_authorization_sha256": R2_AUTHORIZATION_SHA256,
            "train_folds": [2, 3, 4], "heldout_fold": 0,
            "initialization": "random_scratch_ca_only",
            "scannet_checkpoint_or_module_access": False,
        },
        "evaluation_isolation": {
            "scene_list_path": cfg["scene_contract"]["path"],
            "scene_list_sha256": FOLD0_SHA256, "scene_count": 20,
            "fold1_access": False, "official_validation_access": False,
            "ground_truth_access_during_binding": False,
        },
        "evaluation_config": {
            "path": os.fspath(config_source),
            "sha256": sha256_file(config_source),
        },
        "implementation": cfg["implementation"],
    }
    create_or_verify_json(output, payload, "xfit-R2 checkpoint binding")
    return load_binding(output)


def load_binding(path: Path) -> XFitR2Binding:
    source, value = read_json(path, "xfit-R2 checkpoint binding", immutable=True)
    if (
        value.get("schema") != BINDING_SCHEMA
        or value.get("complete") is not True
        or value.get("create_only") is not True
        or value.get("namespace") != NAMESPACE
        or value.get("role") != "outer_dev"
        or value.get("train_only") is not True
        or value.get("checkpoint_selection") is not False
    ):
        raise ValueError("xfit-R2 checkpoint binding contract differs")
    checkpoint_record = value.get("checkpoint") or {}
    effective_record = value.get("effective_config") or {}
    log_record = value.get("training_log") or {}
    outer_log_record = value.get("outer_wrapper_log") or {}
    checkpoint = regular_file(Path(str(checkpoint_record.get("path", ""))), "R2 checkpoint")
    effective = regular_file(
        Path(str(effective_record.get("snapshot_path", ""))),
        "sealed effective config", immutable=True,
    )
    log = regular_file(
        Path(str(log_record.get("snapshot_path", ""))),
        "sealed training log", immutable=True,
    )
    outer_log_source = regular_file(
        Path(str(outer_log_record.get("source_path", ""))),
        "bound outer training wrapper log source",
    )
    outer_log = regular_file(
        Path(str(outer_log_record.get("snapshot_path", ""))),
        "sealed outer training wrapper log", immutable=True,
    )
    for artifact, record, name in (
        (checkpoint, checkpoint_record, "checkpoint"),
        (effective, effective_record, "effective config"),
        (log, log_record, "training log"),
        (outer_log, outer_log_record, "outer wrapper log"),
    ):
        if (
            sha256_file(artifact) != _sha(record.get("sha256"), f"{name} SHA256")
            or artifact.stat().st_size != int(record.get("bytes", -1))
        ):
            raise ValueError(f"sealed R2 {name} changed")
    if (
        outer_log_source != OUTER_WRAPPER_LOG
        or sha256_file(outer_log_source) != outer_log_record.get("sha256")
        or validate_outer_wrapper_log(
            outer_log, require_fixed_source_path=False
        ) != outer_log_record.get("contract")
    ):
        raise ValueError("sealed R2 outer wrapper source/contract changed")
    protocol = value.get("training_protocol") or {}
    if (
        protocol.get("xfit_contract_sha256") != XFIT_CONTRACT_SHA256
        or protocol.get("r2_authorization_sha256") != R2_AUTHORIZATION_SHA256
        or protocol.get("train_folds") != [2, 3, 4]
        or protocol.get("heldout_fold") != 0
        or protocol.get("scannet_checkpoint_or_module_access") is not False
    ):
        raise ValueError("sealed R2 training protocol differs")
    isolation = value.get("evaluation_isolation") or {}
    if (
        isolation.get("scene_list_sha256") != FOLD0_SHA256
        or isolation.get("scene_count") != 20
        or isolation.get("fold1_access") is not False
        or isolation.get("official_validation_access") is not False
        or isolation.get("ground_truth_access_during_binding") is not False
    ):
        raise ValueError("sealed R2 evaluation isolation differs")
    config_record = value.get("evaluation_config") or {}
    evaluation_config = regular_file(
        Path(str(config_record.get("path", ""))), "bound evaluation config"
    )
    if sha256_file(evaluation_config) != _sha(
        config_record.get("sha256"), "bound evaluation config SHA256"
    ):
        raise ValueError("bound evaluation config changed")
    return XFitR2Binding(
        path=source, sha256=sha256_file(source),
        checkpoint_path=checkpoint,
        checkpoint_sha256=sha256_file(checkpoint),
        effective_config_snapshot_path=effective,
        effective_config_sha256=sha256_file(effective),
        training_log_snapshot_path=log,
        training_log_sha256=sha256_file(log),
        outer_wrapper_log_source_path=outer_log_source,
        outer_wrapper_log_snapshot_path=outer_log,
        outer_wrapper_log_sha256=sha256_file(outer_log),
        evaluation_config_path=evaluation_config,
        evaluation_config_sha256=sha256_file(evaluation_config),
    )


def match_targets(corners: Any, gt_corners: Any) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(corners)
    gt = np.asarray(gt_corners)
    world_aabb(boxes)
    world_aabb(gt)
    if not len(boxes):
        return np.empty((0,), np.float64), np.empty((0,), np.int64)
    if not len(gt):
        return np.zeros(len(boxes), np.float64), np.full(len(boxes), -1, np.int64)
    iou = pairwise_world_aabb_iou(boxes, gt)
    matched = np.argmax(iou, axis=1).astype(np.int64)
    return iou[np.arange(len(boxes)), matched], matched


def _voc_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(mpre.size - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    changed = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[changed + 1] - mrec[changed]) * mpre[changed + 1]))


def official_ca_ap(
    *, scene_ids: Any, scores: Any, best_iou: Any, best_gt: Any,
    ground_truth_count: int,
) -> dict[str, dict[str, float | int]]:
    scenes = np.asarray(scene_ids).astype(str)
    rank_score = np.asarray(scores, dtype=np.float64)
    iou = np.asarray(best_iou, dtype=np.float64)
    gt = np.asarray(best_gt, dtype=np.int64)
    if (
        scenes.shape != rank_score.shape or iou.shape != rank_score.shape
        or gt.shape != rank_score.shape or rank_score.ndim != 1
        or not np.isfinite(rank_score).all() or not np.isfinite(iou).all()
        or isinstance(ground_truth_count, bool) or int(ground_truth_count) < 0
    ):
        raise ValueError("official CA AP rows are invalid")
    order = np.argsort(-rank_score)
    positives = int(ground_truth_count)
    result: dict[str, dict[str, float | int]] = {}
    for threshold in IOU_THRESHOLDS:
        tp = np.zeros(len(order), np.float64)
        fp = np.zeros(len(order), np.float64)
        detected: set[tuple[str, int]] = set()
        for rank, row in enumerate(order.tolist()):
            gt_index = int(gt[row])
            key = (str(scenes[row]), gt_index)
            if iou[row] > threshold and gt_index >= 0 and key not in detected:
                tp[rank] = 1.0
                detected.add(key)
            else:
                fp[rank] = 1.0
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recall = cumulative_tp / float(positives + 1.0e-6)
        precision = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, np.finfo(np.float64).eps
        )
        final_tp = int(cumulative_tp[-1]) if len(order) else 0
        final_fp = int(cumulative_fp[-1]) if len(order) else 0
        result[f"iou_{threshold:.2f}"] = {
            "ap": _voc_ap(recall, precision),
            "precision": float(precision[-1]) if len(order) else 0.0,
            "recall": float(recall[-1]) if len(order) else 0.0,
            "tp": final_tp, "fp": final_fp, "fn": positives - final_tp,
        }
    return result


def same_gt_oracle_scene(
    *, anchor_corners: Any, anchor_scores: Any, candidate_corners: Any,
    candidate_scores: Any, gt_corners: Any, near_iou: float = 0.15,
    min_gain: float = 0.05,
) -> tuple[np.ndarray, dict[str, Any]]:
    anchors = np.asarray(anchor_corners, dtype=np.float32)
    scores = np.asarray(anchor_scores, dtype=np.float32)
    candidates = np.asarray(candidate_corners, dtype=np.float32)
    confidence = np.asarray(candidate_scores, dtype=np.float32)
    gt = np.asarray(gt_corners)
    if not math.isfinite(float(min_gain)) or float(min_gain) != 0.05:
        raise ValueError("formal same-GT oracle freezes min_gain=0.05")
    association = associate_terminal_candidates(
        anchor_corners=anchors, anchor_scores=scores,
        candidate_corners=candidates, candidate_scores=confidence,
        near_iou=near_iou,
    )
    output = anchors.copy()
    anchor_iou, anchor_gt = match_targets(anchors, gt)
    candidate_iou, candidate_gt = match_targets(candidates, gt)
    gains: list[float] = []
    selected_rows: list[int] = []
    selected_anchors: list[int] = []
    same_gt_eligible = 0
    target_switch_near = 0
    for anchor in association.represented_anchor_indices.tolist():
        rows = np.flatnonzero(
            association.near_mask & (association.best_anchor_indices == anchor)
        )
        same = rows[
            (candidate_gt[rows] == anchor_gt[anchor]) & (anchor_gt[anchor] >= 0)
        ]
        target_switch_near += int(np.sum(candidate_gt[rows] != anchor_gt[anchor]))
        same_gt_eligible += len(same)
        if not len(same):
            continue
        row = int(same[int(np.argmax(candidate_iou[same]))])
        gain = float(candidate_iou[row] - anchor_iou[anchor])
        if gain >= float(min_gain):
            output[anchor] = candidates[row]
            gains.append(gain)
            selected_rows.append(row)
            selected_anchors.append(anchor)
    return output, {
        "candidate_count": len(candidates),
        "near_candidate_count": int(association.near_mask.sum()),
        "represented_anchor_count": len(association.represented_anchor_indices),
        "same_gt_eligible_candidate_count": same_gt_eligible,
        "near_target_switch_candidate_count": target_switch_near,
        "min_same_gt_iou_gain": float(min_gain),
        "selected_replacement_count": len(gains),
        "gain_ge_0_05_count": len(gains),
        "gain_ge_0_10_count": sum(gain >= 0.10 for gain in gains),
        "positive_iou_gain_sum": float(sum(gains)),
        "positive_iou_gain_mean": float(np.mean(gains)) if gains else 0.0,
        "selected_candidate_rows": selected_rows,
        "selected_anchor_rows": selected_anchors,
        "scores_preserved": True,
        "row_order_preserved": True,
        "row_count_preserved": True,
        "oracle_deployable": False,
    }


def metric_delta(
    active: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    if set(active) != set(baseline):
        raise ValueError("metric key sets differ")
    return {
        key: float(active[key]["ap"] - baseline[key]["ap"])
        for key in baseline
    }


def continuation_gate(
    *, proposal_integrity_pass: bool, scene_count: int,
    replacement_count: int, replacement_scene_count: int,
    oracle_ap_delta: Mapping[str, float],
) -> dict[str, Any]:
    """Apply the create-only preregistered outer-dev continuation gate."""

    def finite_at_least(key: str, threshold: float) -> bool:
        value = float(oracle_ap_delta.get(key, float("nan")))
        return math.isfinite(value) and value >= threshold

    checks = {
        "proposal_exact20_finite_ca_only": (
            proposal_integrity_pass is True and int(scene_count) == 20
        ),
        "same_gt_gain_ge_0_05_replacements_ge_10": int(replacement_count) >= 10,
        "same_gt_gain_ge_0_05_scenes_ge_5": int(replacement_scene_count) >= 5,
        "oracle_delta_ap15_nonnegative": finite_at_least("iou_0.15", 0.0),
        "oracle_delta_ap25_nonnegative": finite_at_least("iou_0.25", 0.0),
        "oracle_delta_ap50_at_least_0_005": finite_at_least("iou_0.50", 0.005),
    }
    passed = all(checks.values())
    return {
        "preregistered": True,
        "checks": checks,
        "thresholds": {
            "scene_count": 20,
            "same_gt_min_iou_gain": 0.05,
            "min_replacements": 10,
            "min_replacement_scenes": 5,
            "min_delta_ap15": 0.0,
            "min_delta_ap25": 0.0,
            "min_delta_ap50": 0.005,
        },
        "pass": passed,
        "continue_inner_training_authorized": passed,
        "authorized_inner_roles": (
            ["inner_holdout2", "inner_holdout3", "inner_holdout4"]
            if passed else []
        ),
        "failure_action": (
            None if passed else (
                "stop_without_training_inner_models_or_opening_fold1_or_"
                "official_validation"
            )
        ),
    }


__all__ = [
    "BINDING_SCHEMA", "COLLECTION_SCHEMA", "CONFIG_SCHEMA",
    "CONTINUATION_RECEIPT_SCHEMA", "FOLD0_SHA256",
    "IOU_THRESHOLDS", "NAMESPACE", "OUTER_WRAPPER_LOG",
    "POINT_PARITY_SHA256", "REPORT_SCHEMA", "PREREGISTRATION_SCHEMA",
    "PREREGISTRATION_V2_SCHEMA", "PREREGISTRATION_V3_SCHEMA",
    "R2_AUTHORIZATION_SHA256", "XFIT_CONTRACT_SHA256", "XFitR2Binding",
    "continuation_gate", "create_or_verify_bytes", "create_or_verify_json",
    "evaluation_config_contract_sha256", "load_binding",
    "load_config", "match_targets", "metric_delta", "official_ca_ap",
    "preregistration_input_contract",
    "read_json", "regular_directory", "regular_file", "same_gt_oracle_scene",
    "scene_ids", "seal_binding", "sha256_bytes", "sha256_file",
    "validate_effective_config", "validate_outer_wrapper_log",
    "validate_static_inputs", "validate_training_log",
]
