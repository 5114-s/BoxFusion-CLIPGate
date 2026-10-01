#!/usr/bin/env python3
"""Create-only binder for the GT-free, CPU-only terminal-v4 Stage O."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_overlay_binding_v4 import (  # noqa: E402
    AUTHORIZATION_SCHEMA,
    BINDING_SCHEMA,
    PROPOSAL_COLLECTION_SCHEMA,
    sha256_file,
    validate_proposal_collection,
)
from tools.preflight_ca1m_tr3d_terminal_train100_v4 import (  # noqa: E402
    validate_config,
)


FINAL_SCHEMA = "boxfusion.ca1m_final_base_identity_audit.v1"
B6_COLLECTION_SCHEMA = "boxfusion.ca1m_native_b6_final_base_train_collection.v2"
B6_CHECKPOINT_SCHEMA = "boxfusion.ca1m_native_b6_iou_mlp.v1"
B6_CHECKPOINT_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_checkpoint_manifest.v1"
B6_OOF_SCHEMA = "boxfusion.ca1m_native_b6_oof_row_scores.v2"
B6_OOF_MANIFEST_SCHEMA = "boxfusion.ca1m_native_b6_oof_row_scores_manifest.v2"


def _json(path: Path, name: str) -> tuple[Path, dict[str, Any]]:
    if path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    source = path.resolve()
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing {name}: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return source, value


def _record(path: Path, schema: str) -> dict[str, str]:
    source = path.resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing sealed upstream: {source}")
    if source.stat().st_mode & 0o222:
        raise ValueError(f"upstream must be read-only: {source}")
    return {"path": str(source), "sha256": sha256_file(source), "schema": schema}


def _code_record(path: Path) -> dict[str, str]:
    source = path.resolve()
    if source.is_symlink() or not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing Stage-O code: {source}")
    return {"path": str(source), "sha256": sha256_file(source)}


def _payload_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _publish_create_only(path: Path, payload: bytes) -> tuple[Path, tuple[int, int]]:
    target = path.resolve()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"refusing existing Stage-O seal: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        stat = target.stat(follow_symlinks=False)
        target.chmod(0o444)
        return target, (stat.st_dev, stat.st_ino)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _remove_own(path: Path, identity: tuple[int, int], payload: bytes) -> None:
    try:
        stat = path.stat(follow_symlinks=False)
        if (
            not path.is_symlink()
            and (stat.st_dev, stat.st_ino) == identity
            and hashlib.sha256(path.read_bytes()).digest()
            == hashlib.sha256(payload).digest()
        ):
            path.unlink()
    except FileNotFoundError:
        pass


def build(
    *, base_config: Path, proposal_manifest: Path, output_config: Path,
    output_authorization: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_config(base_config)
    base_path, base = _json(base_config, "sealed Stage-P base config")
    if base_path.name != "ca1m_tr3d_terminal_train100_v4_p5.json":
        raise ValueError("Stage-O binding requires the revision-5 Stage-P config")

    final_root = ROOT / "results/ca1m_native_final_base_train100_v1/final_base"
    diagnostic_root = ROOT / "diagnostics/ca1m_native_b6_final_base_train100_v2/native_b6"
    completion_root = (
        ROOT / "reports/ca1m_native_b6_final_base_train100_v2/completion/offline_native_b6"
    )
    for path, name in (
        (final_root, "final-base root"),
        (diagnostic_root, "native-B6 diagnostic root"),
        (completion_root, "native-B6 completion root"),
    ):
        if path.is_symlink() or not path.resolve().is_dir():
            raise FileNotFoundError(f"missing {name}: {path.resolve()}")

    upstream = {
        "proposal_collection": _record(proposal_manifest, PROPOSAL_COLLECTION_SCHEMA),
        "final_base_manifest": _record(
            ROOT / "reports/ca1m_native_final_base_train100_v1/collection_manifest.json",
            FINAL_SCHEMA,
        ),
        "native_b6_v2_collection_manifest": _record(
            ROOT / "reports/ca1m_native_b6_final_base_train100_v2/collection_manifest.json",
            B6_COLLECTION_SCHEMA,
        ),
        "native_b6_v2_deployment_checkpoint": _record(
            ROOT / "models/ca1m_native_b6_final_base_iou_mlp_v2.npz",
            B6_CHECKPOINT_SCHEMA,
        ),
        "native_b6_v2_deployment_checkpoint_manifest": _record(
            ROOT / "models/ca1m_native_b6_final_base_iou_mlp_v2.manifest.json",
            B6_CHECKPOINT_MANIFEST_SCHEMA,
        ),
        "native_b6_v2_oof_row_scores": _record(
            ROOT / "models/ca1m_native_b6_final_base_oof_row_scores_v2.npz",
            B6_OOF_SCHEMA,
        ),
        "native_b6_v2_oof_row_scores_manifest": _record(
            ROOT / "models/ca1m_native_b6_final_base_oof_row_scores_v2.manifest.json",
            B6_OOF_MANIFEST_SCHEMA,
        ),
    }
    validate_proposal_collection(upstream["proposal_collection"], base)

    score_usage = {
        "overlay_anchor_scores": "deployable_ca1m_native_b6_v2_checkpoint",
        "deployment_scores_allowed_for_overlay": True,
        "deployment_scores_allowed_for_stacked_gate_training": False,
        "stacked_gate_training_score_source": "all_fold_oof_row_scores_v2",
        "oof_sidecar_loaded_by_overlay": False,
    }
    config = copy.deepcopy(base)
    config["runner_state"] = "stage_o_cpu_revision2_independently_bound"
    config["full_run_authorized"] = False
    overlay = config["overlay_stage"]
    overlay.update(
        {
            "status": "authorized_by_sealed_stage_o_receipt",
            "run_authorized": True,
            "final_anchor_root": str(final_root.resolve()),
            "final_anchor_manifest": upstream["final_base_manifest"]["path"],
            "final_anchor_manifest_sha256": upstream["final_base_manifest"]["sha256"],
            "native_b6_v2_diagnostics_root": str(diagnostic_root.resolve()),
            "native_b6_v2_collection_manifest":
                upstream["native_b6_v2_collection_manifest"]["path"],
            "native_b6_v2_collection_manifest_sha256":
                upstream["native_b6_v2_collection_manifest"]["sha256"],
            "native_b6_v2_completion_root": str(completion_root.resolve()),
            "native_b6_v2_checkpoint":
                upstream["native_b6_v2_deployment_checkpoint"]["path"],
            "native_b6_v2_checkpoint_sha256":
                upstream["native_b6_v2_deployment_checkpoint"]["sha256"],
            "native_b6_v2_checkpoint_manifest":
                upstream["native_b6_v2_deployment_checkpoint_manifest"]["path"],
            "native_b6_v2_checkpoint_manifest_sha256":
                upstream["native_b6_v2_deployment_checkpoint_manifest"]["sha256"],
        }
    )
    config["stage_o_binding"] = {
        "schema": BINDING_SCHEMA,
        "revision": 2,
        "authorization_path": str(output_authorization.resolve()),
        **upstream,
        "score_usage": score_usage,
        "cpu_only": True,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
    }
    config_payload = _payload_bytes(config)
    config_sha = hashlib.sha256(config_payload).hexdigest()
    code = {
        "overlay_runner": _code_record(ROOT / "tools/overlay_ca1m_tr3d_terminal_v4.py"),
        "overlay_binding": _code_record(ROOT / "boxfusion/ca1m_tr3d_overlay_binding_v4.py"),
        "preflight": _code_record(ROOT / "tools/preflight_ca1m_tr3d_terminal_train100_v4.py"),
        "terminal_overlay_contract": _code_record(ROOT / "boxfusion/ca1m_tr3d_terminal_v4.py"),
        "terminal_association": _code_record(ROOT / "boxfusion/ca1m_tr3d_terminal.py"),
        "native_b6_score": _code_record(ROOT / "boxfusion/ca1m_native_b6_score.py"),
        "native_b6_observer": _code_record(ROOT / "boxfusion/ca1m_native_b6_observer.py"),
        "stage_o_sealer": _code_record(Path(__file__)),
    }
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "revision": 2,
        "complete": True,
        "create_only": True,
        "authorization_decision": "ALLOW_STAGE_O_ONLY",
        "overlay_cpu_execution_authorized": True,
        "proposal_gpu_execution_authorized": False,
        "full_two_stage_run_authorized": False,
        "ground_truth_access": False,
        "validation_ground_truth_access": False,
        "deployment_scores_used_for_overlay": True,
        "deployment_scores_allowed_for_stacked_gate_training": False,
        "oof_scores_required_for_stacked_gate_training": True,
        "oof_sidecar_loaded_by_overlay": False,
        "supersedes": {
            "failed_config": _record(
                ROOT / "config/ca1m_tr3d_terminal_train100_v4_o1.json",
                "boxfusion.ca1m_tr3d_terminal_two_stage_config.v4",
            ),
            "failed_authorization": _record(
                ROOT
                / "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/"
                "overlay_stage_authorization_v1.json",
                AUTHORIZATION_SCHEMA,
            ),
            "reason": (
                "revision1_postflight_rejected_order_sensitive_protocol_validation"
            ),
            "stage_o_execution_started": False,
            "overlay_artifact_count": 0,
            "old_artifacts_overwritten": False,
        },
        "bound_config": {
            "path": str(output_config.resolve()), "sha256": config_sha
        },
        "upstream": upstream,
        "code": code,
    }
    return config, authorization


def seal(
    *, config: Mapping[str, Any], authorization: Mapping[str, Any],
    output_config: Path, output_authorization: Path,
) -> dict[str, Any]:
    config_payload = _payload_bytes(config)
    auth_payload = _payload_bytes(authorization)
    config_path, config_identity = _publish_create_only(output_config, config_payload)
    auth_path: Path | None = None
    auth_identity: tuple[int, int] | None = None
    try:
        auth_path, auth_identity = _publish_create_only(
            output_authorization, auth_payload
        )
        report = validate_config(config_path)
        if report["overlay_stage_runtime_authorized"] is not True:
            raise RuntimeError(
                "sealed Stage-O config did not pass independent authorization"
            )
    except BaseException:
        if auth_path is not None and auth_identity is not None:
            _remove_own(auth_path, auth_identity, auth_payload)
        _remove_own(config_path, config_identity, config_payload)
        raise
    return {
        "schema": "boxfusion.ca1m_tr3d_terminal_overlay_seal.v1",
        "complete": True,
        "create_only": True,
        "gpu_started": False,
        "ground_truth_access": False,
        "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
        "authorization": {"path": str(auth_path), "sha256": sha256_file(auth_path)},
        "proposal_collection": config["stage_o_binding"]["proposal_collection"],
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--base-config", type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v4_p5.json",
    )
    value.add_argument(
        "--proposal-manifest", type=Path,
        default=ROOT / "reports/ca1m_tr3d_terminal_ca_native_train100_v4/proposal_collection_manifest_v5.json",
    )
    value.add_argument(
        "--output-config", type=Path,
        default=ROOT / "config/ca1m_tr3d_terminal_train100_v4_o2.json",
    )
    value.add_argument(
        "--output-authorization", type=Path,
        default=ROOT / "manifests/ca1m_tr3d_terminal_ca_native_train100_v4/overlay_stage_authorization_v2.json",
    )
    value.add_argument("--seal", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    config, authorization = build(
        base_config=args.base_config,
        proposal_manifest=args.proposal_manifest,
        output_config=args.output_config,
        output_authorization=args.output_authorization,
    )
    if not args.seal:
        print(json.dumps({
            "ready": True,
            "create_only": True,
            "gpu_started": False,
            "ground_truth_access": False,
            "output_config": str(args.output_config.resolve()),
            "output_authorization": str(args.output_authorization.resolve()),
            "proposal_collection_sha256":
                config["stage_o_binding"]["proposal_collection"]["sha256"],
        }, indent=2, sort_keys=True))
        return 0
    print(json.dumps(seal(
        config=config,
        authorization=authorization,
        output_config=args.output_config,
        output_authorization=args.output_authorization,
    ), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
