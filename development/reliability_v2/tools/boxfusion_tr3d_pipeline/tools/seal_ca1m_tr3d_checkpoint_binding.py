#!/usr/bin/env python3
"""Create or independently audit the CA-native TR3D checkpoint binding.

``seal`` is permitted only after the fixed formal driver log ends in
``TRAIN_EXIT=0``.  Both the binding and optional audit report are create-only
and made read-only.  ``audit`` never rewrites the binding.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from boxfusion.ca1m_tr3d_checkpoint_binding import (  # noqa: E402
    AUDIT_SCHEMA,
    DEV_RECEIPT_SCHEMA,
    EXPECTED_SOURCE_CONFIG,
    EXPECTED_WORK_ROOT,
    SCHEMA,
    build_binding_payload,
    build_dev_diagnostic_receipt,
    load_checkpoint_binding,
    load_dev_diagnostic_receipt,
    sha256_file,
)


def write_json_create_only(path: Path, payload: dict[str, Any], name: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"{name} target must not be a symlink: {path}")
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", dir=target.parent, delete=False
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
        target.chmod(0o444)
    except FileExistsError as error:
        raise FileExistsError(f"refusing to overwrite {name}: {target}") from error
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return target


def seal(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_binding_payload(
        work_root=EXPECTED_WORK_ROOT,
        source_config=EXPECTED_SOURCE_CONFIG,
        training_log=args.training_log,
    )
    target = write_json_create_only(args.output, payload, "checkpoint binding")
    binding = load_checkpoint_binding(target)
    return {
        "schema": SCHEMA,
        "complete": True,
        "binding_path": os.fspath(binding.manifest_path),
        "binding_sha256": binding.manifest_sha256,
        "checkpoint_path": os.fspath(binding.checkpoint_path),
        "checkpoint_sha256": binding.checkpoint_sha256,
        "effective_config_path": os.fspath(binding.effective_config_path),
        "effective_config_sha256": binding.effective_config_sha256,
        "training_log_path": os.fspath(binding.training_log_path),
        "training_log_sha256": binding.training_log_sha256,
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    binding = load_checkpoint_binding(args.binding)
    report = {
        "schema": AUDIT_SCHEMA,
        "complete": True,
        "read_only_audit": True,
        "binding_path": os.fspath(binding.manifest_path),
        "binding_sha256": binding.manifest_sha256,
        "checkpoint_path": os.fspath(binding.checkpoint_path),
        "checkpoint_sha256": binding.checkpoint_sha256,
        "effective_config_path": os.fspath(binding.effective_config_path),
        "effective_config_sha256": binding.effective_config_sha256,
        "source_config_path": os.fspath(binding.source_config_path),
        "source_config_sha256": binding.source_config_sha256,
        "training_log_path": os.fspath(binding.training_log_path),
        "training_log_sha256": binding.training_log_sha256,
        "initialization": "random_scratch",
        "scannet_trained_module_access": False,
        "locked_fold1_gt_access": False,
        "official_validation_gt_access": False,
    }
    if args.output is not None:
        write_json_create_only(args.output, report, "checkpoint binding audit")
    return report


def attach_dev_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    payload = build_dev_diagnostic_receipt(
        binding_path=args.binding,
        dev_report_path=args.dev_report,
    )
    target = write_json_create_only(
        args.output, payload, "checkpoint dev diagnostic receipt"
    )
    verified = load_dev_diagnostic_receipt(target)
    return {
        "schema": DEV_RECEIPT_SCHEMA,
        "complete": True,
        "receipt_path": os.fspath(target),
        "receipt_sha256": sha256_file(target),
        "checkpoint_sha256": verified["checkpoint_binding"]["checkpoint_sha256"],
        "source_report_sha256": verified["source_report"]["sha256"],
        "ap": verified["ap"],
        "recall": verified["recall"],
        "prediction_count": verified["prediction_count"],
        "activation_authorized": False,
    }


def audit_dev_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    payload = load_dev_diagnostic_receipt(args.receipt)
    report = {
        "schema": "boxfusion.tr3d.ca1m_checkpoint_dev_diagnostic_audit.v1",
        "complete": True,
        "read_only_audit": True,
        "receipt_path": os.fspath(Path(args.receipt).resolve()),
        "receipt": payload,
        "activation_authorized": False,
        "checkpoint_selection_authorized": False,
        "terminal_collection_authorized": False,
    }
    if args.output is not None:
        write_json_create_only(args.output, report, "checkpoint dev diagnostic audit")
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--training-log", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path, required=True)
    seal_parser.set_defaults(handler=seal)
    audit_parser = commands.add_parser("audit")
    audit_parser.add_argument("--binding", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path)
    audit_parser.set_defaults(handler=audit)
    attach_parser = commands.add_parser("attach-dev-diagnostic")
    attach_parser.add_argument("--binding", type=Path, required=True)
    attach_parser.add_argument("--dev-report", type=Path, required=True)
    attach_parser.add_argument("--output", type=Path, required=True)
    attach_parser.set_defaults(handler=attach_dev_diagnostic)
    dev_audit_parser = commands.add_parser("audit-dev-diagnostic")
    dev_audit_parser.add_argument("--receipt", type=Path, required=True)
    dev_audit_parser.add_argument("--output", type=Path)
    dev_audit_parser.set_defaults(handler=audit_dev_diagnostic)
    return value


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(args.handler(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
